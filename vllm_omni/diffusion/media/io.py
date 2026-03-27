from __future__ import annotations

import math
import os
from typing import Any

import torch


def load_audio_source(source: Any, *, target_sample_rate: int | None = None) -> torch.Tensor:
    """Load audio from path/tensor/ndarray into a torch tensor [C, T]."""
    if isinstance(source, str):
        import torchaudio

        wav, sr = torchaudio.load(source)
        if target_sample_rate is not None and sr != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        return wav
    if isinstance(source, torch.Tensor):
        return source
    try:
        import numpy as np

        if isinstance(source, np.ndarray):
            return torch.from_numpy(source)
    except ImportError:
        pass
    raise TypeError(f"Unsupported audio source type: {type(source)!r}")


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

        if ext in {".jpg", ".jpeg", ".png"}:
            import numpy as np
            from PIL import Image

            img = Image.open(source).convert("RGB")
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)

        from decord import VideoReader
        from decord import cpu as decord_cpu

        vr = VideoReader(source, ctx=decord_cpu(0))
        fps = float(vr.get_avg_fps())
        step = max(1, int(math.ceil(fps / target_fps)))
        start = int(seek_time * fps)
        if duration > 0:
            target_t = int(duration * target_fps)
            end = min(len(vr), start + target_t * step)
            frame_ids = list(range(start, end, step))
        else:
            frame_ids = list(range(start, len(vr), step))
        return torch.from_numpy(vr.get_batch(frame_ids).asnumpy()).permute(0, 3, 1, 2)

    if isinstance(source, torch.Tensor):
        return source
    try:
        import numpy as np

        if isinstance(source, np.ndarray):
            return torch.from_numpy(source)
    except ImportError:
        pass
    raise TypeError(f"Unsupported video source type: {type(source)!r}")
