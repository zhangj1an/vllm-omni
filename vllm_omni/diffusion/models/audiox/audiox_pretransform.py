from __future__ import annotations

import os
import typing as tp

import torch
from diffusers import AutoencoderOobleck


def create_pretransform_from_config(
    pretransform_config: dict[str, tp.Any],
    *,
    model: str,
) -> AutoencoderOobleck:
    """Load ``AutoencoderOobleck`` (``from_pretrained(model, subfolder="vae")``).

    Sets ``audiox_scaling_factor`` on the module; callers scale latents when encoding/decoding.
    """
    allowed_keys = {"type", "config", "scale"}
    extra_keys = set(pretransform_config) - allowed_keys
    if extra_keys:
        raise ValueError(f"Unsupported pretransform config keys for AudioX inference: {sorted(extra_keys)}")

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
    scaling_factor = float(pretransform_config.get("scale", getattr(icfg, "scaling_factor", 1.0)))
    vae.audiox_scaling_factor = scaling_factor  # type: ignore[attr-defined]

    vae.eval().requires_grad_(False)
    return vae
