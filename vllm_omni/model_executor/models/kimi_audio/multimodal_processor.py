"""vllm-omni KimiAudio multimodal processor: tokenizes input audio with
GLM-4-Voice and uses the resulting codec IDs as the placeholder
expansion (instead of N copies of ``kimia_text_blank``).

Mirrors upstream Moonshot's dual representation: discrete codec tokens
(GLM-4-Voice) on the audio stream, with continuous Whisper features
fused on top via ``(audio + whisper) * sqrt(2)``."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature
from vllm.model_executor.models.kimi_audio import (
    KimiAudioMultiModalProcessor as _UpstreamProcessor,
    _get_feat_extract_output_lengths,
)
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement
from vllm.transformers_utils.processors.kimi_audio import KimiAudioProcessor

from vllm_omni.model_executor.models.kimi_audio.glm4_voice_tokenizer import tokenize_audio

_GLM4_FIELD = "glm4_audio_token_ids"
_GLM4_LEN_FIELD = "glm4_audio_token_lens"

_OMNI_KIMIAUDIO_FIELD_CONFIG = {
    "whisper_input_features": MultiModalFieldConfig.batched("audio"),
    "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
    _GLM4_FIELD: MultiModalFieldConfig.batched("audio"),
    _GLM4_LEN_FIELD: MultiModalFieldConfig.batched("audio"),
}


class OmniKimiAudioMultiModalProcessor(_UpstreamProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        raw_audios = self._extract_raw_audio(mm_data)
        feature = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        if not raw_audios:
            return feature

        token_arrays = []
        token_lens = []
        for arr, sr in raw_audios:
            ids = tokenize_audio(arr, sr)
            arr_np = np.asarray(ids, dtype=np.int64)
            token_arrays.append(torch.from_numpy(arr_np))
            token_lens.append(torch.tensor(len(ids), dtype=torch.long))

        feature[_GLM4_FIELD] = torch.nn.utils.rnn.pad_sequence(
            token_arrays, batch_first=True, padding_value=KimiAudioProcessor.KIMIA_TEXT_BLANK
        )
        feature[_GLM4_LEN_FIELD] = torch.stack(token_lens)
        return feature

    @staticmethod
    def _extract_raw_audio(mm_data: Mapping[str, object]) -> list[tuple[np.ndarray, int]]:
        audios = mm_data.get("audios") or mm_data.get("audio") or []
        out: list[tuple[np.ndarray, int]] = []
        for aud in audios:
            if isinstance(aud, (tuple, list)) and len(aud) == 2:
                out.append((np.asarray(aud[0]), int(aud[1])))
            elif isinstance(aud, np.ndarray):
                out.append((aud, 16000))
            else:
                out.append((np.asarray(aud), 16000))
        return out

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, Any]:
        return _OMNI_KIMIAUDIO_FIELD_CONFIG

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ) -> Sequence[PromptReplacement]:
        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        glm4_tokens = out_mm_data.get(_GLM4_FIELD)
        glm4_lens = out_mm_data.get(_GLM4_LEN_FIELD)

        if feature_attention_mask is not None:
            audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            ).tolist()
        else:
            audio_output_lens = []

        def get_replacement(item_idx: int):
            num_features = (
                audio_output_lens[item_idx]
                if item_idx < len(audio_output_lens)
                else 376
            )
            if num_features == 0:
                num_features = 376
            if glm4_tokens is not None and item_idx < glm4_tokens.shape[0]:
                length = (
                    int(glm4_lens[item_idx].item())
                    if glm4_lens is not None
                    else glm4_tokens.shape[1]
                )
                ids = glm4_tokens[item_idx, :length].tolist()
                # Whisper feature length is the source of truth for the
                # expansion count — pad/truncate to match so the
                # (audio + whisper) fusion lines up.
                if len(ids) < num_features:
                    ids = ids + [KimiAudioProcessor.KIMIA_TEXT_BLANK] * (num_features - len(ids))
                elif len(ids) > num_features:
                    ids = ids[:num_features]
                return ids
            return [KimiAudioProcessor.KIMIA_TEXT_BLANK] * num_features

        return [
            PromptReplacement(
                modality="audio",
                target=[KimiAudioProcessor.KIMIA_TEXT_BLANK],
                replacement=get_replacement,
            ),
        ]
