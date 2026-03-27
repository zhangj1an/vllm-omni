from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.media.io import load_audio_source


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
