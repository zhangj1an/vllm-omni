from __future__ import annotations

import os
import typing as tp

import torch
from diffusers import AutoencoderOobleck
from torch import nn


class AudioXVAE(nn.Module):
    """Thin wrapper around Diffusers ``AutoencoderOobleck`` for AudioX (encode/decode, scaling, metadata)."""

    def __init__(
        self,
        inner: nn.Module,
        *,
        scaling_factor: float,
        io_channels: int,
        latent_dim: int,
        sample_rate: int,
        iterate_batch: bool = False,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.scaling_factor = float(scaling_factor)
        self.io_channels = int(io_channels)
        self.encoded_channels = int(latent_dim)
        self.downsampling_ratio = int(inner.hop_length)
        self.sample_rate = int(sample_rate)
        self.iterate_batch = bool(iterate_batch)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = super().__getattr__("inner")
            return getattr(inner, name)

    def _encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.encode(x, return_dict=True).latent_dist.sample()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sample VAE latents (``OobleckDiagonalGaussianDistribution.sample``)."""
        if self.iterate_batch and x.shape[0] > 1:
            parts = [self._encode_latents(x[i : i + 1]) for i in range(x.shape[0])]
            return torch.cat(parts, dim=0)
        return self._encode_latents(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner.decode(z, return_dict=True).sample

    def encode_scaled(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x) / self.scaling_factor

    def decode_scaled(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z * self.scaling_factor)


def create_pretransform_from_config(
    pretransform_config: dict[str, tp.Any],
    sample_rate: int,
    *,
    model: str,
) -> AudioXVAE:
    """Build ``AudioXVAE`` with Diffusers ``AutoencoderOobleck``, matching Stable Audio:

    ``AutoencoderOobleck.from_pretrained(model, subfolder="vae", ...)`` where ``model`` is
    ``OmniDiffusionConfig.model`` (HF repo id or local directory with a Diffusers ``vae/`` tree).
    """
    allowed_keys = {"type", "config", "scale", "iterate_batch"}
    extra_keys = set(pretransform_config) - allowed_keys
    if extra_keys:
        raise ValueError(f"Unsupported pretransform config keys for AudioX inference: {sorted(extra_keys)}")

    pretransform_type = pretransform_config["type"]

    if pretransform_type != "autoencoder":
        raise NotImplementedError(
            f"AudioX HKUST weights only use pretransform type 'autoencoder'; got {pretransform_type!r}"
        )

    iterate_batch = bool(pretransform_config.get("iterate_batch", False))

    local_files_only = os.path.exists(model)
    inner = AutoencoderOobleck.from_pretrained(
        model,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    icfg = inner.config
    if "scale" in pretransform_config:
        scaling_factor = float(pretransform_config["scale"])
    elif hasattr(icfg, "get") and callable(getattr(icfg, "get", None)):
        scaling_factor = float(icfg.get("scaling_factor", 1.0))
    else:
        scaling_factor = float(getattr(icfg, "scaling_factor", 1.0))
    pretransform = AudioXVAE(
        inner,
        scaling_factor=scaling_factor,
        io_channels=int(getattr(icfg, "audio_channels", 2)),
        latent_dim=int(
            getattr(icfg, "latent_channels", getattr(icfg, "decoder_input_channels", 1)),
        ),
        sample_rate=int(getattr(icfg, "sampling_rate", sample_rate)),
        iterate_batch=iterate_batch,
    )

    pretransform.eval().requires_grad_(False)
    return pretransform
