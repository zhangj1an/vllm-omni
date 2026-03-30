# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict Diffusers-style component shards for AudioX (vLLM-Omni layout).

Weights under :class:`~vllm_omni.diffusion.models.audiox.pipeline_audiox.AudioXPipeline`
use the submodule prefix ``_model.`` (the conditioned AudioX wrapper). Shard files store
keys **relative to that wrapper** (e.g. ``model.*``, ``conditioner.*``, ``pretransform.*``).

Expected layout (under ``od_config.model``)::

    config.json
    model_index.json   # fixed entries: config/config.json + transformer/conditioners shard names
    transformer/diffusion_pytorch_model.safetensors
    conditioners/diffusion_pytorch_model.safetensors
    vae/diffusion_pytorch_model.safetensors   # optional, only when config has pretransform

Runtime intentionally supports only this one AudioX layout shape.
Use ``convert_to_sharded_layout`` to convert a legacy ``config.json`` + ``model.ckpt`` bundle.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

AUDIOX_WEIGHT_LAYOUT_SHARDED = "vllm_omni_component_sharded"

TRANSFORMER_SAFETENSORS = "transformer/diffusion_pytorch_model.safetensors"
CONDITIONERS_SAFETENSORS = "conditioners/diffusion_pytorch_model.safetensors"
VAE_SAFETENSORS = "vae/diffusion_pytorch_model.safetensors"


def _model_root_has_file(model_root: str, rel: str) -> bool:
    return os.path.isfile(os.path.join(os.path.abspath(model_root), rel))


def load_audiox_model_index(model_root: str) -> dict[str, Any]:
    """Return parsed ``model_index.json`` for AudioX sharded bundles."""
    root = os.path.abspath(model_root)
    index_path = os.path.join(root, "model_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"AudioX sharded layout requires model_index.json under {root}")
    with open(index_path, encoding="utf-8") as f:
        index: dict[str, Any] = json.load(f)
    return index


def model_config_has_pretransform(model_config: dict[str, Any]) -> bool:
    m = model_config.get("model")
    if not isinstance(m, dict):
        return False
    return m.get("pretransform") is not None


def resolve_audiox_bundle_paths(model_root: str) -> tuple[str, dict[str, Any]]:
    """Return ``(config_path, model_index)`` for a **component-sharded** AudioX bundle only.

    Single-file ``model.ckpt`` loading is not supported at runtime; convert first with
    ``convert_to_sharded_layout``.
    """
    root = os.path.abspath(model_root)
    idx = load_audiox_model_index(root)
    if idx.get("weight_layout") != AUDIOX_WEIGHT_LAYOUT_SHARDED:
        raise ValueError(
            f"AudioX model_index.json must declare weight_layout={AUDIOX_WEIGHT_LAYOUT_SHARDED!r}; "
            f"got {idx.get('weight_layout')!r}."
        )
    if idx.get("config") != "config.json":
        raise ValueError("AudioX model_index.json must set 'config' to 'config.json'.")
    if idx.get("transformer_weights") != TRANSFORMER_SAFETENSORS:
        raise ValueError(f"AudioX model_index.json must set transformer_weights={TRANSFORMER_SAFETENSORS!r}.")
    if idx.get("conditioners_weights") != CONDITIONERS_SAFETENSORS:
        raise ValueError(f"AudioX model_index.json must set conditioners_weights={CONDITIONERS_SAFETENSORS!r}.")
    if not _model_root_has_file(root, TRANSFORMER_SAFETENSORS) or not _model_root_has_file(root, CONDITIONERS_SAFETENSORS):
        raise FileNotFoundError(
            f"AudioX sharded layout missing required files under {root}: "
            f"{TRANSFORMER_SAFETENSORS!r}, {CONDITIONERS_SAFETENSORS!r}."
        )

    return os.path.join(root, "config.json"), idx


def load_audiox_bundle_config(model_root: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Load ``config.json`` from the strict component-sharded AudioX bundle."""
    root = os.path.abspath(model_root)
    config_path, idx = resolve_audiox_bundle_paths(root)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"AudioX config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        model_config: dict[str, Any] = json.load(f)
    return config_path, model_config, idx


def build_sharded_component_sources(
    *,
    model_root: str,
    od_config: OmniDiffusionConfig,
    model_config: dict[str, Any],
) -> list[DiffusersPipelineLoader.ComponentSource]:
    """Build ``weights_sources`` for the strict AudioX sharded layout."""
    root = os.path.abspath(model_root)
    rev = getattr(od_config, "revision", None)

    def _src(allow_rel: str) -> DiffusersPipelineLoader.ComponentSource:
        return DiffusersPipelineLoader.ComponentSource(
            model_or_path=root,
            subfolder=None,
            revision=rev,
            prefix="_model.",
            fall_back_to_pt=False,
            allow_patterns_overrides=[allow_rel],
        )

    sources: list[DiffusersPipelineLoader.ComponentSource] = [
        _src(TRANSFORMER_SAFETENSORS),
        _src(CONDITIONERS_SAFETENSORS),
    ]
    if model_config_has_pretransform(model_config):
        if not _model_root_has_file(root, VAE_SAFETENSORS):
            raise FileNotFoundError(
                f"AudioX config includes pretransform but {VAE_SAFETENSORS} was not found under {root}. "
                "Convert the bundle with vllm_omni.diffusion.models.audiox.convert_to_sharded_layout "
                "or add the vae shard."
            )
        sources.append(_src(VAE_SAFETENSORS))
    return sources
