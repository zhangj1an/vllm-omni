# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for shared Oobleck VAE base wrapper."""

import torch
from torch import nn

from vllm_omni.diffusion.layers.oobleck_vae_base import OobleckVAEBase


class _FakeLatentDist:
    def __init__(self, value: torch.Tensor):
        self._value = value

    def sample(self) -> torch.Tensor:
        return self._value


class _FakeEncodeOutput:
    def __init__(self, value: torch.Tensor):
        self.latent_dist = _FakeLatentDist(value)


class _FakeDecodeOutput:
    def __init__(self, value: torch.Tensor):
        self.sample = value


class _FakeInnerVAE(nn.Module):
    def __init__(self, *, encoded: torch.Tensor, decoded: torch.Tensor):
        super().__init__()
        self.hop_length = 8
        self._encoded = encoded
        self._decoded = decoded
        self.last_decode_input: torch.Tensor | None = None

    def encode(self, _x: torch.Tensor, return_dict: bool = True) -> _FakeEncodeOutput:
        assert return_dict is True
        return _FakeEncodeOutput(self._encoded)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> _FakeDecodeOutput:
        assert return_dict is True
        self.last_decode_input = z
        return _FakeDecodeOutput(self._decoded)


def test_encode_scaled_and_decode_scaled():
    encoded = torch.tensor([[[4.0, -2.0]]], dtype=torch.float32)
    decoded = torch.tensor([[[0.25, 0.5, 0.75]]], dtype=torch.float32)
    inner = _FakeInnerVAE(encoded=encoded, decoded=decoded)
    vae = OobleckVAEBase(
        inner,
        scaling_factor=2.0,
        io_channels=2,
        latent_dim=64,
        sample_rate=44100,
    )

    x = torch.zeros((1, 2, 16), dtype=torch.float32)
    z = torch.tensor([[[1.0, 3.0]]], dtype=torch.float32)

    scaled_latents = vae.encode_scaled(x)
    assert torch.allclose(scaled_latents, torch.tensor([[[2.0, -1.0]]], dtype=torch.float32))

    out_audio = vae.decode_scaled(z)
    assert torch.allclose(out_audio, decoded)
    assert inner.last_decode_input is not None
    assert torch.allclose(inner.last_decode_input, torch.tensor([[[2.0, 6.0]]], dtype=torch.float32))
