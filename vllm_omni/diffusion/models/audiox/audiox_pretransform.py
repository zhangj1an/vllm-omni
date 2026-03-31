from __future__ import annotations

import typing as tp
from typing import Any

import torch
from diffusers import AutoencoderOobleck
from torch import nn

from vllm_omni.diffusion.layers.oobleck_vae_base import OobleckVAEBase
from vllm_omni.diffusion.models.audiox.audiox_weights import (
    audiox_oobleck_ae_config_supported,
)


def ae_cfg_to_diffusers_init_kwargs(ae_cfg: dict[str, Any], sample_rate: int) -> dict[str, Any]:
    enc = ae_cfg["encoder"]["config"]
    dec = ae_cfg["decoder"]["config"]
    strides = enc["strides"]
    if list(strides) != list(dec["strides"]):
        raise ValueError("AudioX encoder/decoder strides must match for Diffusers Oobleck.")
    if list(enc["c_mults"]) != list(dec["c_mults"]):
        raise ValueError("AudioX encoder/decoder c_mults must match for Diffusers Oobleck.")
    return {
        "encoder_hidden_size": int(enc["channels"]),
        "downsampling_ratios": [int(s) for s in strides],
        "channel_multiples": [int(c) for c in enc["c_mults"]],
        "decoder_channels": int(dec["channels"]),
        "decoder_input_channels": int(dec["latent_dim"]),
        "audio_channels": int(enc["in_channels"]),
        "sampling_rate": int(sample_rate),
    }


class AudioXVAE(OobleckVAEBase):
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
        super().__init__(
            inner,
            scaling_factor=scaling_factor,
            io_channels=io_channels,
            latent_dim=latent_dim,
            sample_rate=sample_rate,
        )
        self.iterate_batch = bool(iterate_batch)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.iterate_batch and x.shape[0] > 1:
            parts = [super(AudioXVAE, self).encode(x[i : i + 1]) for i in range(x.shape[0])]
            return torch.cat(parts, dim=0)
        return super().encode(x)


def create_pretransform_from_config(pretransform_config: dict[str, tp.Any], sample_rate: int):
    allowed_keys = {"type", "config", "scale", "iterate_batch"}
    extra_keys = set(pretransform_config) - allowed_keys
    if extra_keys:
        raise ValueError(f"Unsupported pretransform config keys for AudioX inference: {sorted(extra_keys)}")

    pretransform_type = pretransform_config["type"]

    if pretransform_type != "autoencoder":
        raise NotImplementedError(
            f"AudioX HKUST weights only use pretransform type 'autoencoder'; got {pretransform_type!r}"
        )

    ae_inner = pretransform_config["config"]
    if not audiox_oobleck_ae_config_supported(ae_inner):
        raise NotImplementedError(
            "AudioX pretransform must match official HKUSTAudio checkpoints (Oobleck encoder/decoder, "
            "VAE bottleneck, use_snake=true, no use_nearest_upsample)."
        )

    scaling_factor = float(pretransform_config.get("scale", 1.0))
    iterate_batch = bool(pretransform_config.get("iterate_batch", False))

    inner = AutoencoderOobleck(**ae_cfg_to_diffusers_init_kwargs(ae_inner, sample_rate))
    pretransform = AudioXVAE(
        inner,
        scaling_factor=scaling_factor,
        io_channels=int(ae_inner["io_channels"]),
        latent_dim=int(ae_inner["latent_dim"]),
        sample_rate=int(sample_rate),
        iterate_batch=iterate_batch,
    )

    pretransform.eval().requires_grad_(False)
    return pretransform
