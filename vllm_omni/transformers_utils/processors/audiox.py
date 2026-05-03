# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Input transform utilities for the AudioX diffusion pipeline.

Loads and normalizes the raw audio/video conditioning signals (file path / URL /
``data:`` URI / ``np.ndarray`` / ``torch.Tensor``) into the (channels, samples) and
[T, C, H, W] tensors the pipeline needs. The pipeline itself stays focused on model
forward + sampling logic.
"""

from __future__ import annotations

import base64
import os
import tempfile
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import soundfile
import torch
import torch.nn.functional as F
import torchaudio.functional as taF
from einops import rearrange

# AudioX task taxonomy. Tasks beginning with "v" require a video input; tasks containing
# "t" carry a text prompt. tv2*/v2* share the same conditioning pathways.
VIDEO_ONLY_TASKS = frozenset({"v2a", "v2m"})
TEXT_VIDEO_TASKS = frozenset({"tv2a", "tv2m"})
VIDEO_CONDITIONED_TASKS = VIDEO_ONLY_TASKS | TEXT_VIDEO_TASKS

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png"})


def normalize_prompts(prompts: list[Any]) -> list[dict[str, Any]]:
    """Coerce raw prompt entries into ``{"prompt": str, ...}`` dicts (preserves extras)."""
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(prompts):
        if isinstance(raw, str):
            out.append({"prompt": raw.strip()})
        elif isinstance(raw, dict):
            p = dict(raw)
            p["prompt"] = str(p.get("prompt") or "").strip()
            out.append(p)
        else:
            raise TypeError(f"AudioX prompt {i} must be str or dict, got {type(raw)!r}")
    return out


def materialize_media_source(source: str) -> str:
    """Return a local filesystem path for ``source``.

    Accepts a local path, a ``data:<mime>;base64,...`` URI, or an ``http(s)://`` URL.
    Anything non-local is fetched into a NamedTemporaryFile and that path is returned;
    callers don't need to clean the tempfile up (the OS does on exit).
    """
    if source.startswith("data:"):
        _, _, payload = source.partition(",")
        raw = base64.b64decode(payload)
        f = tempfile.NamedTemporaryFile(prefix="audiox_media_", suffix=".bin", delete=False)
        f.write(raw)
        f.close()
        return f.name
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urlopen(source) as resp:
            data = resp.read()
        f = tempfile.NamedTemporaryFile(prefix="audiox_media_", suffix=".bin", delete=False)
        f.write(data)
        f.close()
        return f.name
    return source


def _load_video_path_torchvision(
    path: str,
    *,
    target_fps: int,
    duration: float,
    seek_time: float,
) -> torch.Tensor:
    from torchvision.io import read_video

    path = materialize_media_source(path)
    end_pts = float(seek_time) + float(duration) if duration > 0 else None
    video, _, info = read_video(path, start_pts=float(seek_time), end_pts=end_pts, pts_unit="sec", output_format="TCHW")
    if video.shape[0] == 0:
        raise ValueError(f"No frames in range seek_time={seek_time!r}, duration={duration!r} for {path!r}")
    src_fps = float(info.get("video_fps") or target_fps)
    n_target = max(1, int(round(video.shape[0] * float(target_fps) / src_fps)))
    if n_target >= video.shape[0]:
        return video
    indices = torch.linspace(0, video.shape[0] - 1, n_target).round().long()
    return video[indices]


def load_video_source(
    source: Any,
    *,
    target_fps: int,
    duration: float,
    seek_time: float = 0.0,
) -> torch.Tensor:
    if isinstance(source, str):
        from torchvision.io import read_image

        ext = os.path.splitext(source)[1].lower()
        if ext in _IMAGE_EXTS:
            return read_image(materialize_media_source(source)).unsqueeze(0)
        return _load_video_path_torchvision(
            source,
            target_fps=target_fps,
            duration=duration,
            seek_time=seek_time,
        )

    if isinstance(source, torch.Tensor):
        return source
    if isinstance(source, np.ndarray):
        return torch.from_numpy(source)
    raise TypeError(f"Unsupported video source type: {type(source)!r}")


def normalize_video_tensor(frames: torch.Tensor, size: int = 224) -> torch.Tensor:
    if frames.dim() != 4:
        raise ValueError(f"Expected [T, C, H, W], got {tuple(frames.shape)}")

    frames = frames.float()
    if frames.max() > 1.5:
        frames = frames / 255.0

    if frames.shape[-2:] != (size, size):
        frames = F.interpolate(frames, size=(size, size), mode="bicubic", align_corners=False)
    return frames


def adjust_video_duration(frames: torch.Tensor, duration: float, target_fps: int) -> torch.Tensor:
    target_t = int(duration * target_fps)
    cur_t = frames.shape[0]

    if cur_t > target_t:
        return frames[:target_t]
    if cur_t < target_t:
        last = frames[-1:].repeat(target_t - cur_t, 1, 1, 1)
        return torch.cat([frames, last], dim=0)
    return frames


def prepare_video_reference(
    source: Any,
    *,
    duration: float,
    target_fps: int,
    seek_time: float = 0.0,
) -> torch.Tensor:
    """Decode a video clip (or single image) into the AudioX [T, 3, 224, 224] form."""
    frames = load_video_source(
        source,
        target_fps=target_fps,
        duration=duration,
        seek_time=seek_time,
    )

    if frames.dim() == 4 and frames.shape[-1] == 3:
        frames = rearrange(frames, "t h w c -> t c h w")

    frames = normalize_video_tensor(frames, size=224)
    if duration > 0:
        frames = adjust_video_duration(frames, duration, target_fps)
    return frames


def prepare_audio_reference(
    source: Any,
    *,
    model_sample_rate: int,
    seconds_start: float,
    seconds_total: float,
    device: torch.device,
) -> torch.Tensor:
    """Decode an audio source into a stereo (2, samples) tensor at the model's rate."""
    target_len = int(model_sample_rate * seconds_total)
    start = int(model_sample_rate * seconds_start)
    if isinstance(source, str):
        data, sr = soundfile.read(materialize_media_source(source), dtype="float32", always_2d=True)
        # soundfile returns channels-last (T, C); project convention is (C, T).
        wav = torch.from_numpy(data).transpose(0, 1).contiguous()
        if sr != model_sample_rate:
            wav = taF.resample(wav, sr, model_sample_rate)
    elif isinstance(source, torch.Tensor):
        wav = source.float()
    elif isinstance(source, np.ndarray):
        wav = torch.from_numpy(source).float()
    else:
        raise TypeError(f"Unsupported audio source type: {type(source)!r}")
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    wav = wav[:, start : start + target_len]
    if wav.shape[1] < target_len:
        wav = F.pad(wav, (target_len - wav.shape[1], 0))
    return wav.to(device=device, dtype=torch.float32)


__all__ = [
    "VIDEO_ONLY_TASKS",
    "TEXT_VIDEO_TASKS",
    "VIDEO_CONDITIONED_TASKS",
    "normalize_prompts",
    "materialize_media_source",
    "load_video_source",
    "normalize_video_tensor",
    "adjust_video_duration",
    "prepare_video_reference",
    "prepare_audio_reference",
]
