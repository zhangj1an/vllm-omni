# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.models.audiox.pipeline_audiox import prepare_audio_reference


def test_prepare_audio_reference_stereo_pad_trim():
    x = torch.randn(2, 100)
    out = prepare_audio_reference(
        x,
        model_sample_rate=50,
        seconds_start=0.0,
        seconds_total=2.0,
        device=torch.device("cpu"),
    )
    assert out.shape == (2, 100)
