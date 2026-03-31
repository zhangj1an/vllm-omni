# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm_omni.diffusion.attention.layer import Attention


class _MAFCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._scale = self._head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)

    def forward(self, experts: torch.Tensor, full_context: torch.Tensor) -> torch.Tensor:
        nh = self._num_heads
        q = rearrange(self.to_q(experts), "b e (nh d) -> b nh e d", nh=nh, d=self._head_dim)
        k, v = self.to_kv(full_context).chunk(2, dim=-1)
        k = rearrange(k, "b l (nh d) -> b nh l d", nh=nh, d=self._head_dim)
        v = rearrange(v, "b l (nh d) -> b nh l d", nh=nh, d=self._head_dim)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self._scale)
        out = rearrange(out, "b nh e d -> b e (nh d)")
        return self.to_out(out)


class _MAFFusionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self._num_heads = num_heads
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.self_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=-1)
        nh = self._num_heads
        q = rearrange(q, "b s (nh d) -> b s nh d", nh=nh)
        k = rearrange(k, "b s (nh d) -> b s nh d", nh=nh)
        v = rearrange(v, "b s (nh d) -> b s nh d", nh=nh)
        q_bsn = q.transpose(1, 2).contiguous()
        k_bsn = k.transpose(1, 2).contiguous()
        v_bsn = v.transpose(1, 2).contiguous()
        out = self.self_attn(q_bsn, k_bsn, v_bsn, attn_metadata=None)
        out = out.transpose(1, 2).contiguous()
        out = rearrange(out, "b s h d -> b s (h d)")
        out = self.out_proj(out)
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x


class MAF_Block(nn.Module):
    DIM = 768
    MLP_RATIO = 4.0

    def __init__(
        self,
        *,
        dim: int = 768,
        num_experts_per_modality: int = 64,
        num_heads: int = 24,
        num_fusion_layers: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        total_experts = num_experts_per_modality * 3

        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
            nn.Sigmoid(),
        )

        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))

        self.cross_block = _MAFCrossAttentionBlock(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.fusion_blocks = nn.ModuleList(
            [_MAFFusionBlock(dim, num_heads, mlp_ratio) for _ in range(num_fusion_layers)]
        )

        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))

    def forward(
        self,
        video_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = video_tokens.shape[0]

        v_global = video_tokens.mean(dim=1)
        t_global = text_tokens.mean(dim=1)
        a_global = audio_tokens.mean(dim=1)

        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1)
        gated_t = text_tokens * w_t.unsqueeze(-1)
        gated_a = audio_tokens * w_a.unsqueeze(-1)

        full_context = torch.cat([gated_v, gated_t, gated_a], dim=1)

        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        cross_out = self.cross_block(experts, full_context)
        updated_experts = self.norm1(experts + cross_out)

        for blk in self.fusion_blocks:
            updated_experts = blk(updated_experts)

        fused_v_experts, fused_t_experts, fused_a_experts = updated_experts.chunk(3, dim=1)

        refinement_v = fused_v_experts.mean(dim=1)
        refinement_t = fused_t_experts.mean(dim=1)
        refinement_a = fused_a_experts.mean(dim=1)

        alpha_v = torch.sigmoid(self.bypass_gate_v)
        alpha_t = torch.sigmoid(self.bypass_gate_t)
        alpha_a = torch.sigmoid(self.bypass_gate_a)

        final_v = video_tokens + alpha_v * self.norm_v2(refinement_v).unsqueeze(1)
        final_t = text_tokens + alpha_t * self.norm_t2(refinement_t).unsqueeze(1)
        final_a = audio_tokens + alpha_a * self.norm_a2(refinement_a).unsqueeze(1)

        return {
            "video": final_v,
            "text": final_t,
            "audio": final_a,
        }
