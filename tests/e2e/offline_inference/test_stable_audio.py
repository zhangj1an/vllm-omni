# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L2 offline inference: basic Stable Audio deployment and output shape."""

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

# Use random weights model for CI testing (small, no authentication required)
models = ["linyueqian/stable_audio_random"]

_SAMPLE_RATE = 44100
_CLIP_DURATION_S = 2.0


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_stable_audio(model_name: str) -> None:
    m = Omni(model=model_name)
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
