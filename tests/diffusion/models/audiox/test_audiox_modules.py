# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.models.audiox import MAF_Block


def test_maf_block_shapes():
    dim = MAF_Block.DIM
    b = MAF_Block()
    video = torch.randn(2, 5, dim)
    text = torch.randn(2, 3, dim)
    audio = torch.randn(2, 4, dim)
    out = b(video, text, audio)
    assert out["video"].shape == video.shape
    assert out["text"].shape == text.shape
    assert out["audio"].shape == audio.shape
