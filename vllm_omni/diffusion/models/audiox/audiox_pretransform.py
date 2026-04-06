from __future__ import annotations

import os
import typing as tp

import torch
from diffusers import AutoencoderOobleck

from vllm_omni.diffusion.models.audiox.audiox_weights import (
    resolve_pretransform_scale,
    validate_audiox_pretransform_config_keys,
)


def create_pretransform_from_config(
    pretransform_config: dict[str, tp.Any],
    *,
    model: str,
) -> AutoencoderOobleck:
    """Load ``AutoencoderOobleck`` (``from_pretrained(model, subfolder="vae")``).

    Sets ``audiox_scaling_factor`` on the module; callers scale latents when encoding/decoding.
    """
    validate_audiox_pretransform_config_keys(pretransform_config)

    pretransform_type = pretransform_config["type"]

    if pretransform_type != "autoencoder":
        raise NotImplementedError(
            f"AudioX HKUST weights only use pretransform type 'autoencoder'; got {pretransform_type!r}"
        )

    local_files_only = os.path.exists(model)
    vae = AutoencoderOobleck.from_pretrained(
        model,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    icfg = vae.config
    scaling_factor = resolve_pretransform_scale(pretransform_config, icfg)
    vae.audiox_scaling_factor = scaling_factor  # type: ignore[attr-defined]

    vae.eval().requires_grad_(False)
    return vae
