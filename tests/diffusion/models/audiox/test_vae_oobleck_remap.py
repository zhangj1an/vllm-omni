# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for AudioX Oobleck remap helpers."""

import torch

from vllm_omni.diffusion.models.audiox import audiox_weights as remap_mod


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
    new_k = "_model.pretransform.inner.encoder.snake1.alpha"
    assert new_k in out
    assert out[new_k].shape == (1, ch, 1)
