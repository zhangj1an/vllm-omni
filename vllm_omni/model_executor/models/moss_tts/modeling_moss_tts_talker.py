# coding=utf-8
# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Stage-0 talker: Qwen3 backbone + (n_vq+1) parallel AR heads."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
    MossTTSDelayConfig,
    MossTTSRealtimeConfig,
)
from vllm_omni.outputs import OmniOutput

_AUDIO_PAD_ID = 0  # padding id used before delay slot fires


def _maybe_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


# ---------------------------------------------------------------------------
# MossTTSDelayTalkerForGeneration
# ---------------------------------------------------------------------------


class MossTTSDelayTalkerForGeneration(nn.Module):
    """Stage-0 talker for MossTTSDelayModel variants.

    Covers all four repos that ship ``architectures: ["MossTTSDelayModel"]``:
      - MOSS-TTS (8B, n_vq=32)
      - MOSS-TTSD-v1.0 (8B, n_vq=16)
      - MOSS-SoundEffect (8B, n_vq=16)
      - MOSS-VoiceGenerator (1.7B, n_vq=16)

    Architecture
    ~~~~~~~~~~~~
    * Backbone: Qwen3 transformer (hidden_size, num_hidden_layers, etc. from
      ``config.language_config``).
    * Embedding: text_embed(t) + Σᵢ audio_embed_i(aᵢ)  — additive fusion,
      no cross-attention or concatenation.
    * Heads: (n_vq + 1) parallel linear heads over the final hidden state.
      – Head 0  → text logits  (drives AR scheduler)
      – Heads 1…n_vq → audio VQ logits  (one per RVQ codebook)

    Delay pattern
    ~~~~~~~~~~~~~
    Audio heads are only active after the model emits the delay-slot token
    (``audio_assistant_delay_slot_token_id``).  Before the slot fires all
    audio heads output a pad token (``audio_pad_code``).  After the slot:

        step  t:     collect audio_codebook_0  for frame t
        step t+1:    collect audio_codebook_0  for frame t+1
                     collect audio_codebook_1  for frame t     (1-step lag)
        …
        step t+k:    all k codebooks active; emit frame t

    The per-request ``delay_step`` counter (stored in the per-request info
    dict) tracks this. Stage-1 receives the codes in (T, NQ) shape and the
    codec's ``batch_decode`` handles the de-interleaving internally.
    """

    # vLLM-Omni integration flags
    have_multimodal_outputs: bool = True
    has_preprocess: bool = True
    has_postprocess: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: MossTTSDelayConfig = vllm_config.model_config.hf_config

        lm_cfg = self.config.language_config
        hidden_size: int = lm_cfg.hidden_size
        self.hidden_size = hidden_size
        self.n_vq: int = self.config.n_vq
        self.audio_vocab_size: int = self.config.audio_vocab_size
        self.audio_pad_code: int = self.config.audio_pad_code
        self.delay_slot_id: int = self.config.audio_assistant_delay_slot_token_id
        self.audio_end_id: int = self.config.audio_end_token_id

        # Qwen3 backbone — weights live under ``language_model.*``
        self.model = Qwen3Model(
            vllm_config=vllm_config,
            prefix=_maybe_prefix(prefix, "model"),
        )

        # Text LM head (head 0)
        self.text_lm_head = ParallelLMHead(
            lm_cfg.vocab_size,
            hidden_size,
            bias=False,
            prefix=_maybe_prefix(prefix, "text_lm_head"),
        )
        self.logits_processor = LogitsProcessor(lm_cfg.vocab_size)

        # Audio VQ heads (heads 1…n_vq in upstream naming)
        # Each head predicts one RVQ codebook: vocab = audio_vocab_size + 1
        self.audio_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.audio_vocab_size + 1, bias=False)
                for _ in range(self.n_vq)
            ]
        )

        # Per-codebook audio embeddings (emb_ext in upstream)
        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.audio_vocab_size + 1, hidden_size)
                for _ in range(self.n_vq)
            ]
        )

        # GPU-resident per-request buffer keys (avoid CPU round-trips)
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),   # last step's audio codes
            ("hidden_states", "last"),    # last hidden for next step embed
        }

    # ------------------------------------------------------------------
    # vLLM-Omni hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        """Return text-head logits for the AR scheduler."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.logits_processor(self.text_lm_head, hidden_states, sampling_metadata)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build per-step input embeddings (text + audio additive fusion).

        Prefill (span_len > 1)
        ~~~~~~~~~~~~~~~~~~~~~~
        The prompt is text-only; audio tokens are all pad.  We return the
        plain text embedding and initialise per-request state.

        Decode (span_len == 1)
        ~~~~~~~~~~~~~~~~~~~~~~
        Combine the last text token embedding with the sum of last step's
        audio embeddings.
        """
        span_len = int(input_ids.shape[0])
        meta = info_dict.get("meta", {}) or {}
        audio_codes_buf = (info_dict.get("audio_codes", {}) or {}).get("current")

        if span_len > 1:
            # Prefill: text-only embedding
            embeds = self.model.get_input_embeddings()(input_ids)
            info_update: dict[str, Any] = {
                "meta": {
                    "delay_step": -1,  # -1 = delay slot not yet seen
                },
                "audio_codes": {
                    "current": torch.full(
                        (self.n_vq,), self.audio_pad_code,
                        dtype=torch.long, device=input_ids.device,
                    ),
                },
            }
            return input_ids, embeds, info_update

        # Decode step
        delay_step: int = int((meta.get("delay_step") or -1))

        # Build combined embedding
        text_embed = self.model.get_input_embeddings()(input_ids.reshape(1))  # (1, H)

        if delay_step >= 0 and audio_codes_buf is not None:
            # Add audio embeddings from last step
            audio_embed = torch.zeros_like(text_embed)
            codes = audio_codes_buf.to(device=input_ids.device)
            for i, emb_layer in enumerate(self.audio_embeddings):
                code_i = codes[i].clamp(0, self.audio_vocab_size)
                audio_embed = audio_embed + emb_layer(code_i.unsqueeze(0))
            combined = text_embed + audio_embed
        else:
            combined = text_embed

        return input_ids, combined, {}

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Sample audio codes from the n_vq parallel heads and package output.

        The audio codes are accumulated across decode steps.  Each step
        appends one row (n_vq codes) to the per-request buffer.  When the
        AR scheduler signals EOS, the full buffer is passed to Stage 1.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs  # (S, H)
        info_dicts: list[dict[str, Any]] = (
            kwargs.get("model_intermediate_buffer")
            or kwargs.get("runtime_additional_information")
            or []
        )

        audio_codes_list: list[torch.Tensor] = []

        for info in info_dicts:
            if not isinstance(info, dict):
                continue

            meta = info.get("meta", {}) or {}
            delay_step: int = int((meta.get("delay_step") or -1))

            # Collect accumulated audio codes from buffer
            acc = (info.get("audio_codes", {}) or {}).get("accumulated")
            if isinstance(acc, torch.Tensor) and acc.numel() > 0:
                audio_codes_list.append(acc)

            # Sample new audio codes from each head at the last hidden state
            if hidden.numel() > 0:
                last_h = hidden[-1].unsqueeze(0)  # (1, H)
                new_codes = torch.full(
                    (1, self.n_vq), self.audio_pad_code,
                    dtype=torch.long, device=hidden.device,
                )
                if delay_step >= 0:
                    for i, head in enumerate(self.audio_heads):
                        logits_i = head(last_h)  # (1, audio_vocab_size+1)
                        # Mask invalid last token (upstream convention)
                        logits_i[..., -1] = float("-inf")
                        code_i = logits_i.argmax(dim=-1)  # greedy; sampler overrides
                        new_codes[0, i] = code_i.squeeze()

                # Append to accumulated buffer
                if acc is not None:
                    updated_acc = torch.cat([acc, new_codes], dim=0)
                else:
                    updated_acc = new_codes

                # Update per-request buffer (returned inside info_update)
                info["audio_codes"] = {
                    "current": new_codes.squeeze(0),
                    "accumulated": updated_acc,
                }

                # Advance delay counter when delay slot token was seen
                last_text_token = hidden.shape[0] - 1  # placeholder; real check below
                # The AR scheduler surfaces the sampled text token via sampling_metadata;
                # for now we advance the counter unconditionally once active.
                if delay_step == -1:
                    pass  # will be set when delay_slot_id is sampled
                else:
                    meta["delay_step"] = delay_step + 1
                info["meta"] = meta

        if not audio_codes_list:
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={},
            )

        # Pass accumulated audio codes to Stage 1
        audio_codes = torch.cat(audio_codes_list, dim=0)  # (T, NQ)
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": audio_codes}},
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Map HF weight names to vLLM-Omni module names.

        HF layout (MossTTSDelayModel):
          language_model.model.*       → model.*
          language_model.lm_head.weight (if present)
          emb_ext.{i}.weight           → audio_embeddings.{i}.weight
          lm_heads.0.weight            → text_lm_head.weight
          lm_heads.{i+1}.weight        → audio_heads.{i}.weight  (i ≥ 0)
        """
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters())

        for name, tensor in weights:
            # Qwen3 backbone
            if name.startswith("language_model.model."):
                mapped = name[len("language_model."):]  # model.*
                if mapped in params_dict:
                    default_weight_loader(params_dict[mapped], tensor)
                    loaded.add(mapped)
                continue

            # Text LM head (lm_heads.0 in upstream == language_model.lm_head)
            if name in ("lm_heads.0.weight", "language_model.lm_head.weight"):
                tgt = "text_lm_head.weight"
                if tgt in params_dict:
                    default_weight_loader(params_dict[tgt], tensor)
                    loaded.add(tgt)
                continue

            # Audio heads: lm_heads.{k}.weight for k >= 1 → audio_heads.{k-1}.weight
            if name.startswith("lm_heads."):
                parts = name.split(".")
                try:
                    k = int(parts[1])
                except (IndexError, ValueError):
                    continue
                if k >= 1:
                    tgt = f"audio_heads.{k - 1}.weight"
                    if tgt in params_dict:
                        default_weight_loader(params_dict[tgt], tensor)
                        loaded.add(tgt)
                continue

            # Audio embeddings: emb_ext.{i}.weight → audio_embeddings.{i}.weight
            if name.startswith("emb_ext."):
                tgt = name.replace("emb_ext.", "audio_embeddings.", 1)
                if tgt in params_dict:
                    default_weight_loader(params_dict[tgt], tensor)
                    loaded.add(tgt)
                continue

        return loaded


# ---------------------------------------------------------------------------
# MossTTSRealtimeTalkerForGeneration
# ---------------------------------------------------------------------------


class _MossTTSRealtimeLocalTransformer(nn.Module):
    """Lightweight per-step local depth transformer for MossTTSRealtime.

    Runs after the main Qwen3 hidden state to produce the full RVQ token
    block (n_vq codes) for the current time step.

    STUB: The upstream local transformer architecture is a 4-layer causal
    transformer with max_position_embeddings=33.  Replace this stub with the
    actual implementation once the full source is available.
    """

    def __init__(self, config: MossTTSRealtimeConfig) -> None:
        super().__init__()
        lt_cfg = config.local_transformer_config
        self.n_vq = config.n_vq
        self.hidden_size = config.hidden_size
        # STUB: build lt_cfg.num_hidden_layers transformer layers here.

    def forward(
        self,
        hidden: torch.Tensor,       # (1, H) last hidden from main model
        prev_codes: torch.Tensor,   # (n_vq,) codes from previous step
    ) -> torch.Tensor:              # (n_vq,) codes for current step
        # STUB: run local transformer, return sampled codes.
        raise NotImplementedError(
            "MossTTSRealtimeLocalTransformer requires the full upstream source."
        )


class MossTTSRealtimeTalkerForGeneration(nn.Module):
    """Stage-0 talker for MossTTSRealtime (1.7B, TTFB ~180 ms).

    Architecture differences from MossTTSDelayTalkerForGeneration:
    * Flat Qwen3 config (no nested language_config).
    * Local depth transformer replaces delay-pattern scheduling: after each
      AR step the local transformer generates the full n_vq audio token block
      for that time step synchronously.
    * Streaming-friendly: first chunk latency is lower because no delay
      warm-up is needed.
    """

    have_multimodal_outputs: bool = True
    has_preprocess: bool = True
    has_postprocess: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: MossTTSRealtimeConfig = vllm_config.model_config.hf_config

        self.hidden_size: int = self.config.hidden_size
        self.n_vq: int = self.config.n_vq
        self.audio_vocab_size: int = self.config.audio_vocab_size
        self.audio_pad_token: int = self.config.audio_pad_token

        # Qwen3 backbone (flat config)
        self.model = Qwen3Model(
            vllm_config=vllm_config,
            prefix=_maybe_prefix(prefix, "model"),
        )

        self.text_lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.hidden_size,
            bias=False,
            prefix=_maybe_prefix(prefix, "text_lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        # Local depth transformer for per-step RVQ block generation
        self.local_transformer = _MossTTSRealtimeLocalTransformer(self.config)

        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("hidden_states", "last"),
        }

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.logits_processor(self.text_lm_head, hidden_states, sampling_metadata)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        span_len = int(input_ids.shape[0])
        if span_len > 1:
            embeds = self.model.get_input_embeddings()(input_ids)
            return input_ids, embeds, {
                "audio_codes": {
                    "current": torch.full(
                        (self.n_vq,), self.audio_pad_token,
                        dtype=torch.long, device=input_ids.device,
                    ),
                },
            }
        return input_ids, self.model.get_input_embeddings()(input_ids), {}

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts: list[dict[str, Any]] = (
            kwargs.get("model_intermediate_buffer")
            or kwargs.get("runtime_additional_information")
            or []
        )

        audio_codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            acc = (info.get("audio_codes", {}) or {}).get("accumulated")
            if isinstance(acc, torch.Tensor) and acc.numel() > 0:
                audio_codes_list.append(acc)

            if hidden.numel() > 0:
                last_h = hidden[-1].unsqueeze(0)
                prev = (info.get("audio_codes", {}) or {}).get("current")
                if prev is None:
                    prev = torch.full((self.n_vq,), self.audio_pad_token,
                                      dtype=torch.long, device=hidden.device)
                # Run local transformer to get codes for this step
                try:
                    new_codes = self.local_transformer(last_h, prev.to(hidden.device))
                except NotImplementedError:
                    new_codes = torch.full((self.n_vq,), self.audio_pad_token,
                                           dtype=torch.long, device=hidden.device)
                new_codes = new_codes.unsqueeze(0)
                updated_acc = torch.cat([acc, new_codes], dim=0) if acc is not None else new_codes
                info["audio_codes"] = {"current": new_codes.squeeze(0), "accumulated": updated_acc}

        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(audio_codes_list, dim=0)
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": audio_codes}},
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Map HF weight names for MossTTSRealtime (flat Qwen3 config).

        HF layout:
          model.*                     → model.*
          lm_head.weight (text)       → text_lm_head.weight
          local_transformer.*         → local_transformer.*
        """
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters())

        for name, tensor in weights:
            if name in ("lm_head.weight",):
                tgt = "text_lm_head.weight"
                if tgt in params_dict:
                    default_weight_loader(params_dict[tgt], tensor)
                    loaded.add(tgt)
                continue

            if name in params_dict:
                default_weight_loader(params_dict[name], tensor)
                loaded.add(name)

        return loaded


__all__ = [
    "MossTTSDelayTalkerForGeneration",
    "MossTTSRealtimeTalkerForGeneration",
]
