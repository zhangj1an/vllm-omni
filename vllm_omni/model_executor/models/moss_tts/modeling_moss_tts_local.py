# coding=utf-8
# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Local depth transformer for MossTTSRealtime.

This is a small (4-layer) Qwen3-style decoder that generates the RVQ codebook
codes for one audio frame, autoregressively over codebooks (rvq=16 forward
passes per audio frame). It runs INSIDE the talker's per-step
``make_omni_output`` and is independent from vLLM's main scheduler.

The implementation mirrors upstream
``MossTTSRealtimeLocalTransformerForCausalLM`` but is trimmed to the inference
path (no training/labels/loss). Weight names are kept identical so the
upstream checkpoint loads without remapping.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
    MossTTSLocalTransformerConfig,
)


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x.to(in_dtype)


class _MLP(nn.Module):
    def __init__(self, cfg: MossTTSLocalTransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        # SiLU only — matches upstream's hidden_act="silu".
        self.act = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX style rotation: split last dim into two halves, swap-and-negate.

    Matches upstream's ``rotate_half`` (NOT the GPT-J interleaved style used
    by the audio codec — these are independent files).
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)


class _Attention(nn.Module):
    def __init__(self, cfg: MossTTSLocalTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv = cfg.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)

        # Qwen3 has per-head q/k norm (introduced in Qwen3, missing in Qwen2).
        self.q_norm = _RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, H)
        cos: torch.Tensor,                     # (B, T, head_dim)
        sin: torch.Tensor,
        cache_keys: torch.Tensor | None,       # (B, num_kv, T_cache, head_dim) or None
        cache_values: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.num_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        if cache_keys is not None:
            k = torch.cat([cache_keys, k], dim=2)
            v = torch.cat([cache_values, v], dim=2)
        new_k, new_v = k, v

        k_rep = _repeat_kv(k, self.num_kv_groups)
        v_rep = _repeat_kv(v, self.num_kv_groups)

        out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=(cache_keys is None and T > 1))
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out), new_k, new_v


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: MossTTSLocalTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = _Attention(cfg, layer_idx)
        self.mlp = _MLP(cfg)
        self.input_layernorm = _RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_k: torch.Tensor | None,
        cache_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_layernorm(x)
        h, new_k, new_v = self.self_attn(h, cos, sin, cache_k, cache_v)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_k, new_v


def _build_rope_cache(cfg: MossTTSLocalTransformerConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    half = cfg.head_dim // 2
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(cfg.max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)            # (T, half)
    emb = torch.cat([freqs, freqs], dim=-1)       # (T, head_dim)  (NeoX style)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class MossTTSRealtimeLocalTransformer(nn.Module):
    """Per-frame depth transformer. Mirrors upstream ``...LocalTransformer``.

    State per audio frame:
      - The first input token uses ``backbone_last_hidden_state`` as the
        embedding (codebook 0's "input" is the backbone hidden, not a token).
      - Subsequent tokens (1..rvq-1) embed via ``embed_tokens[idx-1]``.

    Outputs:
      - One logit row per codebook position, projected through
        ``local_lm_heads[codebook_idx]``.
    """

    def __init__(self, cfg: MossTTSLocalTransformerConfig) -> None:
        super().__init__()
        self.config = cfg
        # Upstream: rvq-1 embeddings (codebook 0 uses backbone hidden state).
        self.embed_tokens = nn.ModuleList([
            nn.Embedding(cfg.audio_vocab_size, cfg.hidden_size, padding_idx=cfg.audio_pad_token)
            for _ in range(cfg.rvq - 1)
        ])
        self.layers = nn.ModuleList([_DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = _RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # rope cache lazily realised (we don't know dtype/device at __init__).
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def _rope(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cos is None or self._cos.device != device or self._cos.dtype != dtype:
            self._cos, self._sin = _build_rope_cache(self.config, device, dtype)
        return self._cos, self._sin

    @torch.no_grad()
    def generate_frame(
        self,
        backbone_last_hidden: torch.Tensor,    # (B, H)
        lm_heads: nn.ModuleList,               # ModuleList of rvq Linear(H -> audio_vocab_size)
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate one audio frame (rvq codebook tokens) for batch B.

        Returns a ``(B, rvq)`` LongTensor.
        """
        device = backbone_last_hidden.device
        dtype = backbone_last_hidden.dtype
        B = backbone_last_hidden.shape[0]
        rvq = self.config.rvq

        cos_full, sin_full = self._rope(device, dtype)

        # Per-layer KV cache: each layer holds (B, num_kv, T, head_dim).
        cache_k: list[torch.Tensor | None] = [None] * len(self.layers)
        cache_v: list[torch.Tensor | None] = [None] * len(self.layers)

        codes = backbone_last_hidden.new_zeros((B, rvq), dtype=torch.long)

        for step in range(rvq):
            if step == 0:
                # Codebook 0's "input" is the backbone hidden state itself.
                x = backbone_last_hidden.unsqueeze(1)  # (B, 1, H)
            else:
                emb = self.embed_tokens[step - 1]
                x = emb(codes[:, step - 1].view(B, 1))  # (B, 1, H)

            cos = cos_full[step:step + 1].unsqueeze(0).expand(B, -1, -1)  # (B, 1, head_dim)
            sin = sin_full[step:step + 1].unsqueeze(0).expand(B, -1, -1)

            for li, layer in enumerate(self.layers):
                x, cache_k[li], cache_v[li] = layer(x, cos, sin, cache_k[li], cache_v[li])

            x = self.norm(x)              # (B, 1, H)
            logits = lm_heads[step](x[:, 0, :]).float()  # (B, vocab)
            codes[:, step] = _sample_token(logits, temperature, top_k, top_p, do_sample)

        return codes


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
) -> torch.Tensor:
    """Top-k + top-p sampling (matches upstream's ``sample_token`` for the
    inference branch).
    """
    if not do_sample or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        top_vals, _ = torch.topk(logits, top_k, dim=-1)
        thresh = top_vals[..., -1:].expand_as(logits)
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Drop tail beyond top_p (keep at least one token).
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        # scatter back to original order
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).reshape(probs.shape[:-1])
    return sampled


__all__ = ["MossTTSRealtimeLocalTransformer"]
