# coding=utf-8
# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Stage-1 codec decoder: RVQ codes → 24 kHz waveform."""

from __future__ import annotations

import os
from typing import Any, Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
)
from vllm_omni.outputs import OmniOutput

logger = init_logger(__name__)


class MossTTSCodecDecoder(nn.Module):
    """Stage-1 decoder for all MOSS-TTS variants.

    Consumes ``(T, NQ)`` audio VQ codes emitted by Stage 0 and decodes them
    to a 24 kHz mono waveform using the vendored
    ``MossAudioTokenizerModel``.

    All five variants share the same codec checkpoint
    ``OpenMOSS-Team/MOSS-Audio-Tokenizer``.  The number of quantizers
    (``n_vq``) is passed as a ``num_quantizers`` argument to ``batch_decode``
    so one codec instance handles both ``n_vq=32`` (MOSS-TTS) and ``n_vq=16``
    (all other variants) without swapping weights.

    The codec checkpoint path defaults to the value stored in
    ``vllm_config.model_config.hf_config.codec_model_name_or_path`` but can
    be overridden by setting the environment variable
    ``MOSS_TTS_CODEC_PATH``.
    """

    input_modalities = "audio"

    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = False
    enable_update_additional_information: bool = True
    requires_raw_input_tokens: bool = True

    _OUTPUT_SAMPLE_RATE: int = 24_000

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config

        cfg = vllm_config.model_config.hf_config
        self._n_vq: int = int(getattr(cfg, "n_vq", getattr(cfg, "rvq", 16)))
        self._codec_path: str = str(
            getattr(cfg, "codec_model_name_or_path", "OpenMOSS-Team/MOSS-Audio-Tokenizer")
        )

        # Resolved at load_weights() time
        self._codec: MossAudioTokenizerModel | None = None
        self._sr_tensor = torch.tensor(self._OUTPUT_SAMPLE_RATE, dtype=torch.int32)

    # ------------------------------------------------------------------
    # vLLM-Omni stubs (codec has no AR loop)
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        return None

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode audio VQ codes to waveform.

        Each request's codes arrive in the ``runtime_additional_information``
        list as ``info["codes"]["audio"]`` with shape ``(T, NQ)``.

        Returns
        -------
        OmniOutput with:
          multimodal_outputs["model_outputs"] — list of (T_wav,) float32 tensors
          multimodal_outputs["sr"]            — list of scalar int32 tensors
        """
        sr_tensor = self._sr_tensor
        empty = torch.zeros((0,), dtype=torch.float32)
        info_list: list[dict[str, Any]] = runtime_additional_information or []
        num_req = max(len(info_list), 1)

        if self._codec is None:
            logger.warning("MossTTSCodecDecoder called before load_weights(); returning silence.")
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        audios: list[torch.Tensor] = [empty] * num_req
        srs: list[torch.Tensor] = [sr_tensor] * num_req
        device = next(self._codec.parameters()).device

        for i, info in enumerate(info_list):
            if not isinstance(info, dict):
                continue
            codes_dict = info.get("codes", {}) or {}
            audio_codes = codes_dict.get("audio")
            if audio_codes is None or not isinstance(audio_codes, torch.Tensor):
                continue
            if audio_codes.numel() == 0:
                continue

            # left_context_size: number of left-context frames prepended to
            # this chunk by the stage input processor (overlap-add pattern,
            # same as Qwen3-TTS code2wav).  We decode them together with the
            # real chunk for smooth boundaries, then trim them from the output.
            meta = info.get("meta", {}) or {}
            left_ctx = meta.get("left_context_size", 0)
            if isinstance(left_ctx, (list, torch.Tensor)):
                left_ctx = int(left_ctx[0]) if hasattr(left_ctx, "__len__") else int(left_ctx.item())
            left_ctx = int(left_ctx)

            # audio_codes: (T, NQ) or (NQ, T) or flat — normalise to (NQ, T)
            if audio_codes.dim() == 1:
                t = audio_codes.numel() // self._n_vq
                if t == 0:
                    continue
                codes_nq_t = audio_codes.reshape(self._n_vq, t)
            elif audio_codes.dim() == 2:
                codes_nq_t = (
                    audio_codes.transpose(0, 1).contiguous()
                    if audio_codes.shape[1] == self._n_vq
                    else audio_codes
                )
            else:
                logger.warning("Unexpected audio_codes shape %s; skipping.", audio_codes.shape)
                continue

            codes_nq_t = codes_nq_t.to(device=device, dtype=torch.long)
            out = self._codec.batch_decode(codes_list=[codes_nq_t], num_quantizers=self._n_vq)

            if out.audio is None:
                continue

            wav = out.audio.reshape(-1).to(dtype=torch.float32).cpu()
            if out.audio_lengths is not None:
                wav = wav[:int(out.audio_lengths[0].item())]

            # Trim left-context samples (overlap-add: context was prepended so
            # the causal decoder has enough past signal at chunk boundaries).
            if left_ctx > 0:
                trim = left_ctx * self._codec.downsample_rate
                wav = wav[trim:]

            audios[i] = wav

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Drain the Stage-0 weights iterator, then load the codec from its own checkpoint.

        The codec lives in a separate HuggingFace repo
        (``OpenMOSS-Team/MOSS-Audio-Tokenizer``) and is loaded independently
        of the talker weights.
        """
        # Drain the incoming weights iterator — all Stage-0 weights are
        # irrelevant to this stage.
        for _ in weights:
            pass

        codec_path = os.environ.get("MOSS_TTS_CODEC_PATH", self._codec_path)
        logger.info("Loading MOSS Audio Tokenizer from %s", codec_path)

        codec_cfg = MossAudioTokenizerConfig.from_pretrained(codec_path)
        codec = MossAudioTokenizerModel(codec_cfg)

        model_loader = DefaultModelLoader(self.vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=codec_path,
            revision=None,
            subfolder=None,
        )
        codec_weights = model_loader._get_weights_iterator(source)
        params_dict = dict(codec.named_parameters())
        for name, tensor in codec_weights:
            if name in params_dict:
                default_weight_loader(params_dict[name], tensor)

        device = self.vllm_config.device_config.device
        codec.to(device=device, dtype=torch.float32)
        codec.eval()
        self._codec = codec

        logger.info(
            "MOSS Audio Tokenizer loaded: sampling_rate=%d, n_vq=%d",
            codec_cfg.sampling_rate,
            codec_cfg.num_quantizers,
        )

        # Move sr_tensor to the same device
        self._sr_tensor = self._sr_tensor.to(device=device)

        return set(n for n, _ in codec.named_parameters())


__all__ = ["MossTTSCodecDecoder"]
