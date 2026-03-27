from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.media.io import load_video_source


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
