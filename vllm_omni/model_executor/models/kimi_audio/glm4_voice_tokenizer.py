"""GLM-4-Voice audio tokenizer wrapper for Kimi-Audio's input audio path.

Upstream Moonshot represents input audio as BOTH discrete codec IDs (from
``THUDM/glm-4-voice-tokenizer``) AND continuous Whisper features. The HF
repo ships only weights + config — the ``WhisperVQEncoder`` modeling code
lives in the GLM-4-Voice GitHub repo, brought in as a Kimi-Audio submodule.
Path is configurable via ``KIMIA_GLM4_SOURCE_DIR``."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch

_DEFAULT_GLM4_SOURCE_PATHS = (
    "/workspace/Kimi-Audio/kimia_infer/models/tokenizer/glm4",
)
_KIMIA_TOKEN_OFFSET = 152064


def _find_glm4_source_dir() -> Path:
    candidates: list[str] = []
    env_path = os.environ.get("KIMIA_GLM4_SOURCE_DIR")
    if env_path:
        candidates.append(env_path)
    candidates.extend(_DEFAULT_GLM4_SOURCE_PATHS)
    for p in candidates:
        path = Path(p)
        if (path / "speech_tokenizer" / "modeling_whisper.py").is_file():
            return path
    raise RuntimeError(
        "Could not locate GLM-4-Voice modeling code. Set "
        "KIMIA_GLM4_SOURCE_DIR to a clone of "
        "https://github.com/THUDM/GLM-4-Voice (the directory containing "
        "speech_tokenizer/modeling_whisper.py)."
    )


@lru_cache(maxsize=1)
def _load_tokenizer():
    src_dir = _find_glm4_source_dir()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    from transformers import WhisperFeatureExtractor

    repo = "THUDM/glm-4-voice-tokenizer"
    model = WhisperVQEncoder.from_pretrained(repo).eval()
    if torch.cuda.is_available():
        model = model.to(torch.cuda.current_device())
    extractor = WhisperFeatureExtractor.from_pretrained(repo)
    return model, extractor


def tokenize_audio(audio_array, sample_rate: int) -> list[int]:
    """Encode a single waveform into Kimi-Audio absolute codec IDs
    (already offset by ``KIMIA_TOKEN_OFFSET``). Inlines upstream's
    ``extract_speech_token`` so we don't depend on the upstream package
    layout."""
    import torchaudio

    model, feature_extractor = _load_tokenizer()
    device = next(model.parameters()).device
    dtype = model.conv1.weight.dtype

    audio_tensor = torch.as_tensor(audio_array).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    if sample_rate != 16000:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=16000
        )
    audio = audio_tensor[0].cpu().numpy()

    pooling_kernel_size = model.config.pooling_kernel_size or 1
    stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
    segments = []
    t = 0
    while t * 16000 < audio.shape[0]:
        segments.append(audio[t * 16000 : (t + 30) * 16000])
        t += 30

    all_tokens: list[int] = []
    with torch.no_grad():
        features = feature_extractor(
            segments,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=stride,
        )
        features["input_features"] = features["input_features"].to(device).to(dtype)
        features["attention_mask"] = features["attention_mask"].to(device)
        outputs = model(**features)
        speech_tokens = outputs.quantized_token_ids
        attn_mask = features["attention_mask"][:, :: model.conv1.stride[0] * model.conv2.stride[0]]
        attn_mask = attn_mask[:, :: pooling_kernel_size]
        for i in range(len(speech_tokens)):
            tokens = speech_tokens[i][attn_mask[i].bool()].tolist()
            all_tokens.extend(tokens)

    return [int(t) + _KIMIA_TOKEN_OFFSET for t in all_tokens]
