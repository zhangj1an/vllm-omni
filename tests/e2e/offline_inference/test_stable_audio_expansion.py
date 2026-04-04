# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio offline e2e: real weights, FP8 + TeaCache (single job to save GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from tests.conftest import assert_audio_valid
from tests.utils import hardware_test
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

_SAMPLE_RATE = 44100
_CLIP_DURATION_S = 2.0


def generate_stable_audio_short_clip(
    omni: Omni,
    *,
    audio_start_in_s: float = 0.0,
    audio_end_in_s: float = 2.0,
    num_inference_steps: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Run a minimal Stable Audio generation and return audio as (batch, channels, samples)."""
    outputs = omni.generate(
        prompts={
            "prompt": "The sound of a dog barking",
            "negative_prompt": "Low quality.",
        },
        sampling_params_list=OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            num_outputs_per_prompt=1,
            extra_args={
                "audio_start_in_s": audio_start_in_s,
                "audio_end_in_s": audio_end_in_s,
            },
        ),
    )

    assert outputs is not None
    first_output = outputs[0]
    assert first_output.final_output_type == "audio"
    assert hasattr(first_output, "request_output") and first_output.request_output

    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput)
    assert req_out.final_output_type == "audio"
    assert hasattr(req_out, "multimodal_output") and req_out.multimodal_output
    audio = req_out.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    return audio


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "xpu": "B60"})
def test_stable_audio_quantization_and_teacache() -> None:
    """Stable Audio Open on real Hub weights with FP8 + TeaCache (covers former L2 smoke + L4 features).

    CI should provide ``HF_TOKEN`` if the checkpoint is gated.
    """
    m = Omni(
        model="stabilityai/stable-audio-open-1.0",
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
