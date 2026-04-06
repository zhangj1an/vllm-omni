# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load and normalize audio/video paths or tensors for AudioX conditioning.

Audio file paths use :func:`torchaudio.load_with_torchcodec`. Video file paths use
:class:`torchcodec.decoders.VideoDecoder` with :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_in_range`.
Static images (``.jpg`` / ``.jpeg`` / ``.png``) use :func:`torchvision.io.read_image`.

Hugging Face :class:`diffusers.video_processor.VideoProcessor` targets tensors
already decoded for VAE pipelines; it does not read arbitrary files, so decoding
stays here while :func:`normalize_video_tensor` keeps the AudioX [0, 1] float +
224 bicubic contract.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png"})


def _load_audio_path_str(path: str) -> tuple[torch.Tensor, int]:
    """Load [C, T] float32 audio from a file path via TorchCodec."""
    from torchaudio import load_with_torchcodec

    return load_with_torchcodec(path)


def load_audio_source(source: Any, *, target_sample_rate: int | None = None) -> torch.Tensor:
    """Load audio from path/tensor/ndarray into a torch tensor [C, T]."""
    if isinstance(source, str):
        import torchaudio

        wav, sr = _load_audio_path_str(source)
        if target_sample_rate is not None and sr != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        return wav
    if isinstance(source, torch.Tensor):
        return source
    if isinstance(source, np.ndarray):
        return torch.from_numpy(source)
    raise TypeError(f"Unsupported audio source type: {type(source)!r}")


def _load_image_path_torchvision(path: str) -> torch.Tensor:
    from torchvision.io import read_image

    return read_image(path).unsqueeze(0)


def _load_video_path_torchcodec(
    path: str,
    *,
    target_fps: int,
    duration: float,
    seek_time: float,
) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(
        path,
        dimension_order="NCHW",
        device="cpu",
        seek_mode="exact",
    )
    meta_begin = float(decoder.metadata.begin_stream_seconds or 0.0)
    meta_end = float(decoder.metadata.end_stream_seconds or 0.0)
    start_s = max(float(seek_time), meta_begin)
    if duration > 0:
        stop_s = min(float(seek_time) + float(duration), meta_end)
    else:
        stop_s = meta_end
    if start_s >= stop_s:
        raise ValueError(
            f"No frames in range seek_time={seek_time!r}, duration={duration!r} "
            f"(stream [{meta_begin}, {meta_end})) for {path!r}"
        )
    batch = decoder.get_frames_played_in_range(start_s, stop_s, fps=float(target_fps))
    return batch.data


def load_video_source(
    source: Any,
    *,
    target_fps: int,
    duration: float,
    seek_time: float = 0.0,
) -> torch.Tensor:
    """Load video/image/tensor/ndarray into a torch tensor [T, C, H, W] when possible."""
    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
        if ext in _IMAGE_EXTS:
            return _load_image_path_torchvision(source)
        return _load_video_path_torchcodec(
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


def to_2ch_audio(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    elif x.shape[0] > 2:
        x = x[:2]
    return x


def crop_or_pad_1d(x: torch.Tensor, start: int, target_len: int) -> torch.Tensor:
    x = x[:, start : start + target_len]
    cur = x.shape[1]
    if cur < target_len:
        x = F.pad(x, (target_len - cur, 0))
    else:
        x = x[:, :target_len]
    return x


def prepare_audio_reference(
    source: Any,
    *,
    model_sample_rate: int,
    seconds_start: float,
    seconds_total: float,
    device: torch.device,
) -> torch.Tensor:
    target_len = int(model_sample_rate * seconds_total)
    start = int(model_sample_rate * seconds_start)
    wav = load_audio_source(source, target_sample_rate=model_sample_rate)

    wav = to_2ch_audio(wav)
    wav = crop_or_pad_1d(wav, start, target_len)
    return wav.to(device=device, dtype=torch.float32)


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
    frames = load_video_source(
        source,
        target_fps=target_fps,
        duration=duration,
        seek_time=seek_time,
    )

    if frames.dim() == 4 and frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)

    frames = normalize_video_tensor(frames, size=224)
    if duration > 0:
        frames = adjust_video_duration(frames, duration, target_fps)
    return frames
