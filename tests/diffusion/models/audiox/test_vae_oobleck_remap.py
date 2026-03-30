# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests avoid ``import vllm_omni`` (heavy optional deps) by loading remap/adapter modules by path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[4]
_REMAP_PATH = _REPO / "vllm_omni/diffusion/models/audiox/vae_oobleck_checkpoint_remap.py"
_ADAPTER_PATH = _REPO / "vllm_omni/diffusion/models/audiox/diffusers_oobleck_adapter.py"


def _load_remap_mod():
    spec = importlib.util.spec_from_file_location("vae_oobleck_checkpoint_remap", _REMAP_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_adapter_mod():
    spec = importlib.util.spec_from_file_location("diffusers_oobleck_adapter", _ADAPTER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


remap_mod = _load_remap_mod()


def test_map_encoder_init_conv():
    r = remap_mod._map_encoder_decoder_suffix("encoder.layers.0.weight_g", num_blocks=5)
    assert r == ("inner.encoder.conv1.weight_g", False)


def test_map_encoder_tail_snake_reshape():
    r = remap_mod._map_encoder_decoder_suffix("encoder.layers.6.alpha", num_blocks=5)
    assert r == ("inner.encoder.snake1.alpha", True)


def test_map_decoder_final_conv():
    r = remap_mod._map_encoder_decoder_suffix("decoder.layers.7.weight_v", num_blocks=5)
    assert r == ("inner.decoder.conv2.weight_v", False)


def test_remap_snake_tensor_shape(monkeypatch):
    monkeypatch.setattr(remap_mod, "should_remap_audiox_vae_to_diffusers", lambda _mc: True)
    mc = {
        "model": {
            "pretransform": {
                "config": {
                    "encoder": {"type": "oobleck", "config": {"strides": [2, 4, 4, 8, 8]}},
                }
            }
        }
    }
    ch = 128
    alpha = torch.randn(ch)
    out = dict(
        remap_mod.remap_audiox_oobleck_weights_for_diffusers(
            [("_model.pretransform.model.encoder.layers.6.alpha", alpha.clone())],
            model_config=mc,
        )
    )
    new_k = "_model.pretransform.model.inner.encoder.snake1.alpha"
    assert new_k in out
    assert out[new_k].shape == (1, ch, 1)


def test_ae_cfg_to_diffusers_init_kwargs_maf():
    """Maps MAF-style JSON to :class:`~diffusers.AutoencoderOobleck` kwargs (no Diffusers import)."""
    adapter_mod = _load_adapter_mod()
    ae_cfg = {
        "encoder": {
            "type": "oobleck",
            "config": {
                "in_channels": 2,
                "channels": 128,
                "c_mults": [1, 2, 4, 8, 16],
                "strides": [2, 4, 4, 8, 8],
                "latent_dim": 128,
                "use_snake": True,
            },
        },
        "decoder": {
            "type": "oobleck",
            "config": {
                "out_channels": 2,
                "channels": 128,
                "c_mults": [1, 2, 4, 8, 16],
                "strides": [2, 4, 4, 8, 8],
                "latent_dim": 64,
                "use_snake": True,
                "final_tanh": False,
            },
        },
        "bottleneck": {"type": "vae"},
        "latent_dim": 64,
        "downsampling_ratio": 2048,
        "io_channels": 2,
    }
    kw = adapter_mod._ae_cfg_to_diffusers_init_kwargs(ae_cfg, 44100)
    assert kw["decoder_input_channels"] == 64
    assert kw["encoder_hidden_size"] == 128
    assert kw["downsampling_ratios"] == [2, 4, 4, 8, 8]
    from math import prod

    assert prod(kw["downsampling_ratios"]) == 2048
