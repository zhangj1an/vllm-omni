from __future__ import annotations

import typing as tp
from typing import Any

from vllm_omni.diffusion.models.audio.oobleck_vae_base import OobleckVAEBase

# Identity by default for HKUST AudioX; set ``"scale"`` in pretransform config if training used another factor.
DEFAULT_AUDIOX_VAE_SCALING_FACTOR: float = 1.0


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
    """Minimal AudioX adapter around Diffusers AutoencoderOobleck."""


def create_pretransform_from_config(pretransform_config: dict[str, tp.Any], sample_rate: int):
    pretransform_type = pretransform_config.get("type", None)
    assert pretransform_type is not None, "type must be specified in pretransform config"

    if pretransform_type != "autoencoder":
        raise NotImplementedError(
            f"AudioX HKUST weights only use pretransform type 'autoencoder'; got {pretransform_type!r}"
        )

    from vllm_omni.diffusion.models.audiox.audiox_weights import (
        audiox_oobleck_ae_config_supported,
    )

    ae_inner = pretransform_config["config"]
    if not audiox_oobleck_ae_config_supported(ae_inner):
        raise NotImplementedError(
            "AudioX pretransform must match official HKUSTAudio checkpoints (Oobleck encoder/decoder, "
            "VAE bottleneck, use_snake=true, no use_nearest_upsample). Custom autoencoder layouts are "
            "no longer supported."
        )

    scaling_factor = float(pretransform_config.get("scale", DEFAULT_AUDIOX_VAE_SCALING_FACTOR))
    try:
        from diffusers import AutoencoderOobleck

        inner = AutoencoderOobleck(**ae_cfg_to_diffusers_init_kwargs(ae_inner, sample_rate))
        pretransform = AudioXVAE(
            inner,
            scaling_factor=scaling_factor,
            io_channels=int(ae_inner["io_channels"]),
            latent_dim=int(ae_inner["latent_dim"]),
            sample_rate=int(sample_rate),
        )
    except Exception as e:
        raise RuntimeError(
            "AudioX VAE requires Hugging Face diffusers with AutoencoderOobleck "
            "(see requirements/common.txt). Import or construction failed."
        ) from e

    enable_grad = pretransform_config.get("enable_grad", False)
    pretransform.enable_grad = enable_grad
    pretransform.eval().requires_grad_(pretransform.enable_grad)
    return pretransform
