# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio fused thinker.

Slice-2 layout: extends upstream vLLM PR 36127's
``KimiAudioForConditionalGeneration`` (Whisper-large-v3 encoder + VQ-Adaptor
+ Qwen2-7B LLM, ASR-only) with the MIMO audio-out branch:

  - 6 MIMO decoder layers (``mimo_layers.0..5``), each Qwen2-style
    (grouped-query attention with kv_heads=4, SiLU MLP, two RMS norms).
  - A final norm (``mimo_norm``) and an audio-output head
    (``mimo_output``: hidden → vocab_size=168448, sharing the global vocab
    with the text head so that audio token IDs sit at >=152064).
  - The MIMO branch reads its input hidden state from the main LLM at
    layer index ``config.kimia_mimo_transformer_from_layer_index`` (=21
    for Kimi-Audio-7B) via a forward hook.

Why subclass instead of skeleton-copy: upstream's class is now stable
enough that the only thing it actively breaks for us is its
``load_weights``, which explicitly filters out ``mimo_*``. We override
``load_weights`` to keep them and add a thin ``mimo_layers``/``mimo_norm``/
``mimo_output`` weight-mapper extension. Everything else (``embed_input_ids``
with the text+whisper √2 fusion, the Whisper sub-bundle loader, the
``compute_logits`` text path) is inherited unchanged.

Whether the MIMO branch is built and run is gated by
``config.kimia_generate_audio`` (set via ``hf_overrides`` in the stage
YAML). Text-out YAMLs omit the flag → the branch is never built,
saving ~600M params + its forward cost. Audio-out YAMLs set it to True.

Slice 4 will move the MIMO sub-stack onto vLLM's paged KV cache for
performance; for Slice 2 we keep it in eager mode (HF-style attention) so
that correctness is independent of vLLM's compile pipeline.

What's NOT yet wired up in this file (deliberate; flagged inline as
``MIMO-TODO``):
  - The MIMO sub-stack's KV cache is not maintained across decode steps
    (every step recomputes the prefix). Acceptable for short audio
    contexts; revisit when we measure decode latency.
  - The audio token sampling is greedy argmax inside ``_run_mimo_branch``
    rather than going through vLLM's sampler (which only handles one
    distribution per step). The text path keeps using vLLM's sampler,
    so ``default_sampling_params`` in the stage YAML only affect the text
    head. Audio tokens are deterministic for a given text prefix.
"""

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

# Audio token ID range in the global vocabulary; mirrors
# vllm_omni.model_executor.models.kimi_audio.kimi_audio_code2wav.KIMIA_TOKEN_OFFSET.
KIMIA_TOKEN_OFFSET = 152064


def _build_mimo_layers(config) -> nn.ModuleList:
    """Build the 6 MIMO decoder layers.

    Uses Hugging Face's ``Qwen2DecoderLayer`` directly. The MIMO branch in
    Kimi-Audio is structurally identical to a Qwen2 decoder layer
    (grouped-query attention + SiLU MLP + two RMS norms), and HF's version
    has well-tested pre-trained-weight loading semantics. We don't go
    through vLLM's optimized layer because it's tied to vLLM's paged KV
    cache, and the MIMO branch is small enough that eager attention is
    acceptable for Slice 2.
    """
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

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
    )
    return nn.ModuleList([Qwen2DecoderLayer(qwen2_cfg, layer_idx=i) for i in range(config.kimia_mimo_layers)])


def _build_mimo_norm(config) -> nn.Module:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    return Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


# Extended weight mapper: matches upstream PR 36127 (audio tower prefix +
# fc1/fc2 -> mlp substr rename so whisper-large-v3 safetensors keys land
# on vLLM's Whisper modules) plus the Slice-2 audio-out keys that
# upstream's load_weights deliberately filters out.
_HF_TO_VLLM_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        # Audio tower (Whisper encoder loaded from the whisper-large-v3
        # secondary bundle).
        "model.encoder.": "audio_tower.",
        # VQ-Adaptor projector:
        "model.vq_adaptor.layers.0.": "multi_modal_projector.vq_adaptor_layers_0.",
        "model.vq_adaptor.layers.3.": "multi_modal_projector.vq_adaptor_layers_3.",
        "model.vq_adaptor.layers.4.": "multi_modal_projector.vq_adaptor_layers_4.",
        # Language model (Qwen2-7B):
        "model.layers.": "language_model.model.layers.",
        "model.embed_tokens.": "language_model.model.embed_tokens.",
        "model.norm.": "language_model.model.norm.",
        "lm_head.": "language_model.lm_head.",
        # Slice-2 audio-out (MIMO branch):
        "model.mimo_layers.": "mimo_layers.",
        "model.mimo_norm.": "mimo_norm.",
        "mimo_output.": "mimo_output.",
    },
    orig_to_new_substr={
        # Whisper HF layout uses ``.fc1.``/``.fc2.`` for MLP; vLLM's
        # WhisperEncoder expects them under ``.mlp.fc1.``/``.mlp.fc2.``.
        ".fc1.": ".mlp.fc1.",
        ".fc2.": ".mlp.fc2.",
    },
)


class KimiAudioFusedThinker(_UpstreamKimiAudio):
    """Fused thinker with both the text and audio output branches.

    Registered in ``vllm_omni.model_executor.models.registry`` under the
    architecture name ``KimiAudioThinkerForConditionalGeneration`` and
    instantiated by :class:`KimiAudioForConditionalGeneration` (the omni
    dispatcher) when ``model_stage == 'fused_thinker'``.
    """

    hf_to_vllm_mapper = _HF_TO_VLLM_MAPPER

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        # Needed by ``_load_whisper_subbundle`` to locate
        # ``<model_path>/whisper-large-v3/model.safetensors``.
        self.model_path = vllm_config.model_config.model

        # Opt in to the MIMO audio-out branch via hf_overrides in the
        # stage YAML (see kimi_audio_audio_out.yaml). Text-only YAMLs omit
        # the flag so we skip allocating and running the branch entirely.
        self._generate_audio = bool(getattr(config, "kimia_generate_audio", False))

        if self._generate_audio:
            self.mimo_layers = _build_mimo_layers(config)
            self.mimo_norm = _build_mimo_norm(config)
            # Output head shares the global vocab (text + audio tokens). Audio
            # token IDs occupy [KIMIA_TOKEN_OFFSET, vocab_size); when sampling
            # we slice to that range.
            self.mimo_output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            self._mimo_branch_layer_idx = int(config.kimia_mimo_transformer_from_layer_index)
            # Single-slot list so the layer-21 forward hook and the MIMO
            # branch don't fight over an instance attribute. vLLM serializes
            # forwards per model instance, so a single slot is safe.
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
        # Qwen2DecoderLayer returns a tuple; first element is the hidden state.
        self._mimo_capture_slot[0] = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """Forward pass.

        Runs the upstream text path unconditionally. When the MIMO branch
        is enabled (``kimia_generate_audio=True`` in the stage YAML's
        hf_overrides), also runs the 6-layer MIMO branch on the hidden
        state captured at layer ``kimia_mimo_transformer_from_layer_index``
        and packs audio token IDs into the OmniOutput sidecar.
        """
        if self._generate_audio:
            # Clear any stale capture from a previous forward before the
            # layer-21 hook fires again inside super().forward.
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
        # Consume and release the slot for the next step.
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
        """Run the 6-layer MIMO branch on the captured layer-21 hidden state.

        Returns a 1-D LongTensor of audio token IDs (already offset into the
        global vocab, so >=KIMIA_TOKEN_OFFSET) suitable for handoff to the
        code2wav stage.

        MIMO-TODO: this implementation does not maintain a KV cache across
        decode steps — every step recomputes attention over the captured
        prefix. For long audio contexts (>2k tokens of audio output) this
        will be slow; Slice 4's CUDA-graph wrapper will fold the cache in.
        """
        hidden = capture
        # vLLM flattens batch+sequence into a 2-D [N, hidden] tensor. HF's
        # Qwen2DecoderLayer expects [B, T, hidden]. Treat the flattened
        # tensor as a single batch.
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

        # Causal attention mask for HF Qwen2DecoderLayer (additive form).
        attn_mask = torch.zeros((1, 1, seq_len, seq_len), device=hidden.device, dtype=hidden.dtype)
        causal_block = torch.full((seq_len, seq_len), float("-inf"), device=hidden.device, dtype=hidden.dtype).triu(1)
        attn_mask = attn_mask + causal_block

        h = hidden_3d
        for layer in self.mimo_layers:
            layer_out = layer(
                hidden_states=h,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            h = layer_out[0]

        h = self.mimo_norm(h)
        audio_logits = self.mimo_output(h)  # [1, seq, vocab_size]

        # Greedy argmax restricted to the audio token slice. A real Slice 2
        # might use temperature sampling here; staying greedy keeps audio
        # output deterministic for regression testing.
        audio_logits = audio_logits[..., KIMIA_TOKEN_OFFSET:]
        audio_codes_local = audio_logits.argmax(dim=-1)  # [1, seq]
        audio_codes = audio_codes_local + KIMIA_TOKEN_OFFSET
        return audio_codes.reshape(-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, optionally including the MIMO branch.

        The Whisper-large-v3 encoder ships as a secondary safetensors
        bundle under ``<model_path>/whisper-large-v3/``; upstream's
        ``__init__`` registers it via ``self.secondary_weights`` so those
        keys are appended to the ``weights`` iterable we receive here.
        Our ``hf_to_vllm_mapper`` handles the ``model.encoder.`` ->
        ``audio_tower.`` rename and the ``.fc1.``/``.fc2.`` -> ``.mlp.*``
        substring rename so the bundle lands on the right modules.

        We skip:

        - ``model.decoder.*`` and any other ``model.*`` keys that the
          mapper doesn't rename (the whisper decoder isn't used).
        - ``audio_decoder.*`` (the flow-matching detokenizer lives in
          the code2wav stage).
        - ``mimo_layers.*`` / ``mimo_norm.*`` / ``mimo_output.*`` when
          ``kimia_generate_audio=False`` (the MIMO modules weren't
          allocated, so AutoWeightsLoader would otherwise raise "no
          destination").
        """
        from vllm.model_executor.models.utils import AutoWeightsLoader

        skip_prefixes = ["model.", "audio_decoder."]
        if not self._generate_audio:
            skip_prefixes.extend(["mimo_layers.", "mimo_norm.", "mimo_output."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
