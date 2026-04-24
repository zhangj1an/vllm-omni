# SPDX-License-Identifier: MIT
"""Vendored Moonshot Kimi-Audio BigVGAN vocoder.

Ported from ``Kimi-Audio/kimia_infer/models/detokenizer/vocoder`` so vllm-omni
no longer needs the HF-repurposed ``Qwen2_5OmniToken2WavBigVGANModel``
(which silently dropped the snake + anti-alias activations and produced
fuzzy waveforms from otherwise-correct mel spectrograms)."""
