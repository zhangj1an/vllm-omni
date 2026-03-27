from __future__ import annotations

import torch
from torch import nn


class OobleckVAEBase(nn.Module):
    """Thin wrapper around Diffusers AutoencoderOobleck with shared logic."""

    def __init__(
        self,
        inner: nn.Module,
        *,
        scaling_factor: float,
        io_channels: int,
        latent_dim: int,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.scaling_factor = float(scaling_factor)
        self.io_channels = int(io_channels)
        self.encoded_channels = int(latent_dim)
        self.downsampling_ratio = int(inner.hop_length)
        self.sample_rate = int(sample_rate)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = super().__getattr__("inner")
            return getattr(inner, name)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.encode(x, return_dict=True).latent_dist.sample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner.decode(z, return_dict=True).sample

    def encode_scaled(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x) / self.scaling_factor

    def decode_scaled(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z * self.scaling_factor)
