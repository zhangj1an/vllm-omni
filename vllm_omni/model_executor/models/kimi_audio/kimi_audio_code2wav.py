# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio code2wav stage.

Wraps Moonshot's ``PrefixStreamingFlowMatchingDetokenizer`` (flow-matching DiT
prefix model + BigVGAN vocoder) as a vLLM-Omni generation stage. Consumes the
audio-token IDs produced by the Slice-2 fused thinker and emits a 24 kHz
mono waveform.

The detokenizer is vendored under ``vllm_omni.model_executor.models.kimi_audio
.kimia_detokenizer`` (originally from ``MoonshotAI/Kimi-Audio``'s
``kimia_infer`` package) and loads its own checkpoints from
``<model_path>/audio_detokenizer/`` and ``<model_path>/vocoder/``. We do
**not** wire those checkpoints through vLLM's weight loader — keeping them
out of vLLM's mapper avoids fighting the upstream PR-36127 hf_to_vllm_mapper
every time it changes.

Slice 2 (this file): non-streaming chunked decode (``detokenize_noref``).
Slice 3 will add an async-chunk path that drains the detokenizer's streaming
chunks lazily; Slice 4 will add a CUDA-graph wrapper around the ODE step.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.kimi_audio.cuda_graph_decoder_wrapper import (
    KimiAudioCudaGraphDecoderWrapper,
)
from vllm_omni.model_executor.models.kimi_audio.kimia_detokenizer import (
    detokenize_noref,
    get_audio_detokenizer,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# From config.json: audio token IDs occupy [kimia_token_offset, vocab_size).
# We subtract the offset before handing tokens to the detokenizer, which
# expects local audio-code IDs in [0, 16384) (= num codebook entries minus
# specials).
KIMIA_TOKEN_OFFSET = 152064
NUM_AUDIO_SPECIAL_TOKENS = 512  # kept aside; see config.num_audio_special_tokens
OUTPUT_SAMPLE_RATE = 24000  # vocoder ships at 24 kHz / 480-sample frame


class KimiAudioCode2Wav(nn.Module):
    """Stage-1 code2wav: audio-token IDs -> 24 kHz waveform.

    Lifecycle mirrors :class:`Qwen3TTSCode2Wav`: lazy load the underlying
    decoder on first forward, expose dummy ``embed_input_ids`` /
    ``compute_logits`` so vLLM's runner is happy, and return audio in the
    multimodal_outputs slot of :class:`OmniOutput`.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._detokenizer: Any | None = None
        self._first_chunk_tokens = 100
        self._chunk_tokens = 150
        self._cuda_graph_wrapper: KimiAudioCudaGraphDecoderWrapper | None = None

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def _ensure_detokenizer_loaded(self) -> None:
        if self._detokenizer is not None:
            return
        # ``get_audio_detokenizer(model_path)`` reads
        # ``<model_path>/{audio_detokenizer,vocoder}/`` and pins the
        # detokenizer onto ``torch.cuda.current_device()`` — make sure the
        # stage process owns the right CUDA device first.
        self._detokenizer = get_audio_detokenizer(self.model_path)
        logger.info("KimiAudioCode2Wav detokenizer loaded from %s", self.model_path)
        self._maybe_install_cuda_graph_wrapper()

    def _maybe_install_cuda_graph_wrapper(self) -> None:
        """Wrap ``vocoder.decode_mel`` with a CUDA graph capture (Slice 4).

        Skipped if the stage runs on CPU, if ``enforce_eager: true`` is set
        on the YAML, or if the streaming-chunk config can't be derived.
        """
        if not torch.cuda.is_available():
            return
        if getattr(self.vllm_config.model_config, "enforce_eager", False):
            logger.info("KimiAudioCode2Wav: enforce_eager=True; skipping CUDA Graph capture")
            return

        vocoder = getattr(self._detokenizer, "vocoder", None)
        if vocoder is None or not hasattr(vocoder, "decode_mel"):
            logger.warning("KimiAudioCode2Wav: detokenizer has no `vocoder.decode_mel`; skipping CUDA Graph")
            return

        # Pull chunk math from the connector config (set by the streaming
        # YAML). Default values come from kimia_infer's defaults.
        codec_chunk_frames = 0
        codec_left = 0
        connector_cfg = getattr(self.vllm_config.model_config, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra = connector_cfg.get("extra", connector_cfg)
            if isinstance(extra, dict):
                codec_chunk_frames = int(extra.get("codec_chunk_frames") or 0)
                codec_left = int(extra.get("codec_left_context_frames") or 0)

        wrapper = KimiAudioCudaGraphDecoderWrapper(vocoder=vocoder, enabled=True)
        try:
            wrapper.warmup(
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                dtype=getattr(self._detokenizer, "dtype", torch.bfloat16),
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left,
            )
        except Exception:
            logger.warning("KimiAudioCode2Wav: CUDA Graph warmup failed; falling back to eager", exc_info=True)
            return

        # Re-route ``vocoder.decode_mel`` through the wrapper. Stash the
        # original so callers (or tests) can still reach the eager path.
        self._original_decode_mel = vocoder.decode_mel
        vocoder.decode_mel = wrapper.decode_mel
        self._cuda_graph_wrapper = wrapper
        logger.info("KimiAudioCode2Wav: CUDA Graph wrapper installed on vocoder.decode_mel")

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Stage ignores token embeddings — keep a stable dummy for the runner."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        return [ids]

    def _decode_one(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Decode one request's audio tokens to waveform (non-streaming).

        Args:
            audio_codes: 1-D LongTensor of local audio-code IDs in [0, 16384).
        Returns:
            1-D FloatTensor [wav_len] at 24 kHz.
        """
        # Drop the reference-audio prefill state every request — Slice 2
        # has no voice cloning and we want each request to start clean.
        self._detokenizer.clear_states()
        tokens = audio_codes.unsqueeze(0)  # [1, T]
        wav = detokenize_noref(self._detokenizer, tokens)  # [1, wav_len]
        return wav.squeeze(0).float()

    def _decode_chunk_streaming(
        self,
        audio_codes: torch.Tensor,
        is_final: bool,
        ode_step: int = 15,
    ) -> torch.Tensor:
        """Decode one chunk of audio tokens via the streaming path.

        Used by Slice 3 when ``async_chunk: true``. The detokenizer keeps
        its per-request state (KV cache, mel/wav history) across calls; the
        first chunk emits a half-chunk's worth of audio while later chunks
        get cross-faded with a Hamming window. Caller must reset state
        between requests.
        """
        assert self._detokenizer is not None
        tokens = audio_codes.unsqueeze(0)  # [1, T]
        wav = self._detokenizer.detokenize_streaming(
            tokens,
            ode_step=ode_step,
            is_final=is_final,
        )
        return wav.squeeze(0).float()

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
        self._ensure_detokenizer_loaded()
        sr_tensor = torch.tensor(OUTPUT_SAMPLE_RATE, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        async_chunk = bool(self.vllm_config.model_config.async_chunk)

        # Per-request "is_final" hint comes from runtime_additional_information
        # populated by kimi2code2wav_async_chunk; absent in the non-streaming
        # path.
        finished_flags: list[bool] = [not async_chunk] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(finished_flags):
                    break
                fin = info.get("finished")
                if isinstance(fin, torch.Tensor):
                    finished_flags[i] = bool(fin.item())
                elif fin is not None:
                    finished_flags[i] = bool(fin)

        wav_outputs: list[torch.Tensor] = []
        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() == 0:
                wav_outputs.append(empty)
                continue
            # Filter to audio tokens, normalize into [0, 16384) code space.
            audio_mask = req_ids >= KIMIA_TOKEN_OFFSET
            audio_codes = req_ids[audio_mask] - KIMIA_TOKEN_OFFSET
            # Drop the 512 special-token region so we don't feed control
            # codes (BOS/EOS/etc.) into the flow-matching model.
            audio_codes = audio_codes[audio_codes < (16896 - NUM_AUDIO_SPECIAL_TOKENS)]
            if audio_codes.numel() == 0:
                wav_outputs.append(empty)
                continue
            if async_chunk:
                wav = self._decode_chunk_streaming(audio_codes, is_final=finished_flags[i])
            else:
                wav = self._decode_one(audio_codes)
            wav_outputs.append(wav)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": [w.reshape(1, -1) for w in wav_outputs],
                "sr": [sr_tensor] * len(wav_outputs),
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """No-op. Detokenizer loads its own checkpoints via ``get_audio_detokenizer``.

        We still drain the iterator so the upstream loader doesn't see
        unconsumed weights and complain.
        """
        loaded: set[str] = set()
        for name, _ in weights:
            loaded.add(name)
        return loaded
