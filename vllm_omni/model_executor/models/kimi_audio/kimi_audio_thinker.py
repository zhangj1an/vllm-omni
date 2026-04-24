# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio fused thinker: extends upstream vLLM's
``KimiAudioForConditionalGeneration`` (Whisper-large-v3 + VQ-Adaptor +
Qwen2-7B, ASR-only) with the MIMO audio-out branch (6 Qwen2 decoder layers
+ RMS norm + shared-vocab linear head) hooked at layer
``kimia_mimo_transformer_from_layer_index`` of the main LLM. The branch is
gated by ``config.kimia_generate_audio`` — text-only YAMLs omit it and skip
the ~600M-param allocation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_audio import (
    KimiAudioForConditionalGeneration as _UpstreamKimiAudio,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Mirrors kimi_audio_code2wav.KIMIA_TOKEN_OFFSET.
KIMIA_TOKEN_OFFSET = 152064


def _build_mimo_layers(config) -> tuple[nn.ModuleList, nn.Module]:
    """Build the MIMO decoder layers with HF's Qwen2DecoderLayer. We skip
    vLLM's optimized layer because it's tied to paged KV cache; the MIMO
    branch is small enough that eager attention is fine.

    Returns the layer stack plus a ``Qwen2RotaryEmbedding`` — since
    transformers>=4.45 moved RoPE out of the attention layer, we must
    precompute ``(cos, sin)`` and pass it to each layer as
    ``position_embeddings``."""
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

    qwen2_cfg = Qwen2Config(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.kimia_mimo_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        hidden_act=config.hidden_act,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    layers = nn.ModuleList([Qwen2DecoderLayer(qwen2_cfg, layer_idx=i) for i in range(config.kimia_mimo_layers)])
    rotary_emb = Qwen2RotaryEmbedding(config=qwen2_cfg)
    return layers, rotary_emb


def _build_mimo_norm(config) -> nn.Module:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    return Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


_HF_TO_VLLM_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "model.encoder.": "audio_tower.",
        "model.vq_adaptor.layers.0.": "multi_modal_projector.vq_adaptor_layers_0.",
        "model.vq_adaptor.layers.3.": "multi_modal_projector.vq_adaptor_layers_3.",
        "model.vq_adaptor.layers.4.": "multi_modal_projector.vq_adaptor_layers_4.",
        "model.layers.": "language_model.model.layers.",
        "model.embed_tokens.": "language_model.model.embed_tokens.",
        "model.norm.": "language_model.model.norm.",
        "lm_head.": "language_model.lm_head.",
        "model.mimo_layers.": "mimo_layers.",
        "model.mimo_norm.": "mimo_norm.",
        "mimo_output.": "mimo_output.",
    },
    # Whisper HF uses .fc1/.fc2; vLLM's WhisperEncoder expects .mlp.fc1/.mlp.fc2.
    orig_to_new_substr={
        ".fc1.": ".mlp.fc1.",
        ".fc2.": ".mlp.fc2.",
    },
)


class KimiAudioFusedThinker(_UpstreamKimiAudio):
    hf_to_vllm_mapper = _HF_TO_VLLM_MAPPER

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        # Used by ``_load_whisper_subbundle`` to locate the subbundle.
        self.model_path = vllm_config.model_config.model

        self._generate_audio = bool(getattr(config, "kimia_generate_audio", False))

        if self._generate_audio:
            self.mimo_layers, self.mimo_rotary_emb = _build_mimo_layers(config)
            self.mimo_norm = _build_mimo_norm(config)
            # Head shares the global vocab with the text head; audio IDs
            # occupy [KIMIA_TOKEN_OFFSET, vocab_size).
            self.mimo_output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            self._mimo_branch_layer_idx = int(config.kimia_mimo_transformer_from_layer_index)
            self._mimo_capture_slot: list[torch.Tensor | None] = [None]

            target_layer = self.language_model.model.layers[self._mimo_branch_layer_idx]
            target_layer.register_forward_hook(self._capture_layer21_output)
            logger.info(
                "KimiAudioFusedThinker: MIMO branch hooked at language_model.model.layers[%d] "
                "(mimo_layers=%d, mimo_output vocab=%d)",
                self._mimo_branch_layer_idx,
                len(self.mimo_layers),
                self.mimo_output.out_features,
            )
        else:
            logger.info(
                "KimiAudioFusedThinker: kimia_generate_audio=False; MIMO audio-out branch not built. Text-out only."
            )

    def _capture_layer21_output(self, module, args, output) -> None:
        self._mimo_capture_slot[0] = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self._generate_audio:
            # Clear stale capture before the hook fires inside super().forward.
            self._mimo_capture_slot[0] = None

        text_hidden = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if not self._generate_audio:
            return text_hidden

        capture = self._mimo_capture_slot[0]
        if capture is None:
            logger.warning(
                "MIMO branch enabled but capture buffer is empty. "
                "Layer-21 hook did not fire (CUDA Graph capture? Compiled "
                "forward path?). Falling back to text-only output."
            )
            return text_hidden
        self._mimo_capture_slot[0] = None

        audio_token_ids = self._run_mimo_branch(capture=capture, positions=positions)
        return OmniOutput(
            text_hidden_states=text_hidden,
            multimodal_outputs={"audio_tokens": audio_token_ids},
        )

    def _run_mimo_branch(
        self,
        capture: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the MIMO branch on the captured hidden state and return a
        1-D LongTensor of audio token IDs (offset into the global vocab)."""
        hidden = capture
        # vLLM flattens to 2-D [N, hidden]; HF's Qwen2DecoderLayer expects
        # [B, T, hidden] — treat the flattened tensor as one batch.
        if hidden.dim() == 2:
            hidden_3d = hidden.unsqueeze(0)
        else:
            hidden_3d = hidden

        seq_len = hidden_3d.shape[1]
        if positions is None:
            position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        else:
            position_ids = positions.reshape(1, -1)
            if position_ids.shape[1] != seq_len:
                position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)

        # Causal mask (additive) for HF Qwen2DecoderLayer.
        attn_mask = torch.zeros((1, 1, seq_len, seq_len), device=hidden.device, dtype=hidden.dtype)
        causal_block = torch.full((seq_len, seq_len), float("-inf"), device=hidden.device, dtype=hidden.dtype).triu(1)
        attn_mask = attn_mask + causal_block

        position_embeddings = self.mimo_rotary_emb(hidden_3d, position_ids)

        h = hidden_3d
        for layer in self.mimo_layers:
            layer_out = layer(
                hidden_states=h,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            # transformers>=4.50 returns a bare Tensor [B, T, H]; older
            # versions return a ``(hidden_states, ...)`` tuple.
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        h = self.mimo_norm(h)
        audio_logits = self.mimo_output(h)

        audio_logits = audio_logits[..., KIMIA_TOKEN_OFFSET:]
        audio_codes_local = audio_logits.argmax(dim=-1)
        audio_codes = audio_codes_local + KIMIA_TOKEN_OFFSET
        return audio_codes.reshape(-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import AutoWeightsLoader

        # Skip the whisper decoder (model.*), the detokenizer
        # (audio_decoder.*), and MIMO keys when the branch is disabled.
        skip_prefixes = ["model.", "audio_decoder."]
        if not self._generate_audio:
            skip_prefixes.extend(["mimo_layers.", "mimo_norm.", "mimo_output."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
