# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L4 offline inference: Stable Audio with combined FP8 quantization and TeaCache."""

import sys
from pathlib import Path

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from tests.conftest import assert_audio_valid
from tests.e2e.offline_inference.stable_audio_offline_utils import generate_stable_audio_short_clip
from tests.utils import hardware_test
from vllm_omni import Omni

models = ["linyueqian/stable_audio_random"]

_SAMPLE_RATE = 44100
_CLIP_DURATION_S = 2.0


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_stable_audio_quantization_and_teacache(model_name: str) -> None:
    """TeaCache + FP8 quantization in one run (L4 coverage)."""
    m = Omni(
        model=model_name,
        quantization="fp8",
        cache_backend="tea_cache",
        cache_config={"rel_l1_thresh": 0.2},
    )
    try:
        audio = generate_stable_audio_short_clip(m)
        assert_audio_valid(
            audio,
            sample_rate=_SAMPLE_RATE,
            channels=2,
            duration_s=_CLIP_DURATION_S,
        )
    finally:
        m.close()
