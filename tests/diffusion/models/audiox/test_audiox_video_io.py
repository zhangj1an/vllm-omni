# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.audiox.pipeline_audiox import (
    adjust_video_duration,
    coerce_video_source,
    read_audiox_video_from_path,
    tensor_to_audiox_video,
)


def test_adjust_video_duration_pad_and_trim():
    x = torch.zeros(3, 3, 224, 224)
    out = adjust_video_duration(x, duration=2.0, target_fps=2)
    assert out.shape[0] == 4
    out2 = adjust_video_duration(out, duration=1.0, target_fps=2)
    assert out2.shape[0] == 2


def test_adjust_video_duration_pad_uses_integer_repeat_count():
    """Regression: float ``duration`` must still yield an int repeat count for ``Tensor.repeat``."""
    x = torch.zeros(239, 3, 224, 224)
    out = adjust_video_duration(x, duration=10.0, target_fps=24)
    assert out.shape == (240, 3, 224, 224)
    assert out.dtype == x.dtype


def test_torch_repeat_rejects_float_repeat_sizes_documenting_audiox_bug():
    """Upstream AudioX used ``repeat_times`` as float (e.g. 1.0), which breaks ``torch.Tensor.repeat``."""
    last = torch.zeros(1, 3, 224, 224)
    with pytest.raises(TypeError, match="repeat"):
        last.repeat(1.0, 1, 1, 1)


def test_tensor_to_audiox_video_resize_and_fps():
    t = torch.zeros(10, 3, 64, 64)
    out = tensor_to_audiox_video(t, duration=1.0, target_fps=5)
    assert out.shape == (5, 3, 224, 224)


def test_tensor_to_audiox_video_uint8_scaled_before_resize():
    t = torch.zeros(6, 3, 32, 32, dtype=torch.uint8)
    t[..., 0, 0] = 255
    out = tensor_to_audiox_video(t, duration=1.0, target_fps=3)
    assert out.dtype == torch.float32
    assert out.shape == (3, 3, 224, 224)
    assert float(out.max()) <= 1.5


def test_coerce_video_source_numpy_thwc():
    arr = np.zeros((8, 64, 64, 3), dtype=np.float32)
    out = coerce_video_source(arr, seek_time=0.0, duration=1.0, target_fps=4)
    assert out.shape == (4, 3, 224, 224)
    assert out.dtype == torch.float32


def _can_decode_mp4_with_audiox_video_io() -> bool:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    try:
        import decord  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from torchvision.io import read_video  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _can_decode_mp4_with_audiox_video_io(), reason="need ffmpeg+ffprobe and decord or torchvision.io"
)
def test_read_audiox_video_from_path_mp4_decodes_and_pads(tmp_path: Path):
    mp4 = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=24",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    out = read_audiox_video_from_path(str(mp4), seek_time=0.0, duration=10.0, target_fps=2)
    assert out.shape == (20, 3, 224, 224)
    assert out.dtype == torch.float32


def test_read_audiox_video_from_path_png_tiles_to_target_frames(tmp_path: Path):
    from PIL import Image

    png = tmp_path / "one.png"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(png)
    out = read_audiox_video_from_path(str(png), seek_time=0.0, duration=1.5, target_fps=4)
    assert out.shape == (6, 3, 224, 224)
    assert out.dtype == torch.float32


def test_coerce_video_source_str_delegates_to_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from PIL import Image

    png = tmp_path / "c.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(png)
    called: dict[str, object] = {}

    def fake_read(path: str, *, seek_time: float, duration: float, target_fps: int):
        called["path"] = path
        called["seek_time"] = seek_time
        called["duration"] = duration
        called["target_fps"] = target_fps
        return torch.ones(2, 3, 224, 224)

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.audiox.pipeline_audiox.read_audiox_video_from_path",
        fake_read,
    )
    out = coerce_video_source(str(png), seek_time=1.0, duration=2.0, target_fps=8)
    assert out.shape == (2, 3, 224, 224)
    assert called["path"] == str(png)
    assert called["seek_time"] == 1.0
    assert called["duration"] == 2.0
    assert called["target_fps"] == 8
