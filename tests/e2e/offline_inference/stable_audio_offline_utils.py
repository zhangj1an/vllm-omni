# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared generation helpers for Stable Audio offline e2e tests."""

from __future__ import annotations

import numpy as np
import torch

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


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
    assert first_output.final_output_type == "image"
    assert hasattr(first_output, "request_output") and first_output.request_output

    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput)
    assert req_out.final_output_type == "audio"
    assert hasattr(req_out, "multimodal_output") and req_out.multimodal_output
    audio = req_out.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    return audio
