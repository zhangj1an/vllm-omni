# coding=utf-8
# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MOSS-TTS model configuration."""

from __future__ import annotations

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3 import Qwen3Config


class MossTTSDelayConfig(PretrainedConfig):
    """Config for MossTTSDelayModel (8B: MOSS-TTS, MOSS-TTSD, MOSS-SoundEffect;
    1.7B: MOSS-VoiceGenerator).

    The HuggingFace checkpoint stores a nested ``language_config`` which is a
    Qwen3 config.  We unwrap it here so that vLLM can use ``get_text_config()``
    to size KV caches and allocate memory correctly.
    """

    model_type = "moss_tts_delay"

    def __init__(
        self,
        language_config: dict | None = None,
        n_vq: int = 32,
        audio_vocab_size: int = 1024,
        sampling_rate: int = 24000,
        audio_pad_code: int = 1024,
        audio_start_token_id: int = 151652,
        audio_end_token_id: int = 151653,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_gen_slot_token_id: int = 151656,
        audio_assistant_delay_slot_token_id: int = 151662,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        codec_model_name_or_path: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        **kwargs: object,
    ) -> None:
        if language_config is None:
            language_config = {}
        if isinstance(language_config, dict):
            language_config.pop("model_type", None)
            self.language_config = Qwen3Config(**language_config)
        else:
            self.language_config = language_config

        super().__init__(**kwargs)

        self.n_vq = n_vq
        self.audio_vocab_size = audio_vocab_size
        self.sampling_rate = sampling_rate
        self.audio_pad_code = audio_pad_code
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.audio_user_slot_token_id = audio_user_slot_token_id
        self.audio_assistant_gen_slot_token_id = audio_assistant_gen_slot_token_id
        self.audio_assistant_delay_slot_token_id = audio_assistant_delay_slot_token_id
        self.pad_token_id = pad_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.codec_model_name_or_path = codec_model_name_or_path

    def get_text_config(self, **_: object) -> Qwen3Config:
        return self.language_config


class MossTTSLocalTransformerConfig(PretrainedConfig):
    """Config for the lightweight local depth transformer in MossTTSRealtime."""

    model_type = "moss_tts_realtime_local_transformer"

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        max_position_embeddings: int = 33,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings


class MossTTSRealtimeConfig(PretrainedConfig):
    """Config for MossTTSRealtime (1.7B, TTFB ~180 ms streaming model).

    Unlike MossTTSDelayConfig, this model has a flat Qwen3 config (no nested
    language_config) plus a local depth transformer for per-step RVQ block
    generation.
    """

    model_type = "moss_tts_realtime"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        max_position_embeddings: int = 40960,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        rvq: int = 16,
        audio_vocab_size: int = 1027,
        audio_pad_token: int = 1024,
        sampling_rate: int = 24000,
        reference_audio_pad: int = 151654,
        text_pad: int = 151655,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        local_transformer_config: dict | None = None,
        codec_model_name_or_path: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        **kwargs: object,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rvq = rvq
        self.n_vq = rvq  # alias for uniform access
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.sampling_rate = sampling_rate
        self.reference_audio_pad = reference_audio_pad
        self.text_pad = text_pad
        self.codec_model_name_or_path = codec_model_name_or_path

        if local_transformer_config is None:
            local_transformer_config = {}
        if isinstance(local_transformer_config, dict):
            self.local_transformer_config = MossTTSLocalTransformerConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                **local_transformer_config,
            )
        else:
            self.local_transformer_config = local_transformer_config

    def get_text_config(self, **_: object) -> "MossTTSRealtimeConfig":
        # The model IS a flat Qwen3-style config; return self for vLLM sizing.
        return self


AutoConfig.register("moss_tts_delay", MossTTSDelayConfig)
AutoConfig.register("moss_tts_realtime", MossTTSRealtimeConfig)

__all__ = ["MossTTSDelayConfig", "MossTTSRealtimeConfig", "MossTTSLocalTransformerConfig"]
