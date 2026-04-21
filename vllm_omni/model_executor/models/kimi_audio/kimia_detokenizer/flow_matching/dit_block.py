import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func

    HAS_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError):
    flash_attn_varlen_func = None
    flash_attn_varlen_qkvpacked_func = None
    HAS_FLASH_ATTN = False


def _sdpa_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """SDPA fallback for ``flash_attn_varlen_func`` with the same packed
    ``(total_tokens, num_heads, head_dim)`` contract. The per-segment loop
    is fine because the detokenizer runs at batch size 1."""
    q_bounds = cu_seqlens_q.tolist()
    k_bounds = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(q_bounds) - 1):
        qs, qe = q_bounds[i], q_bounds[i + 1]
        ks, ke = k_bounds[i], k_bounds[i + 1]
        q_seg = q[qs:qe].transpose(0, 1).unsqueeze(0)
        k_seg = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_seg = v[ks:ke].transpose(0, 1).unsqueeze(0)
        out_seg = F.scaled_dot_product_attention(
            q_seg, k_seg, v_seg, dropout_p=dropout_p
        )
        outs.append(out_seg.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        flash_attention: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = flash_attention and HAS_FLASH_ATTN

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk_norm = qk_norm
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        seq_len,
        cu_seqlens,
        max_seqlen,
        cu_seqlens_k,
        max_seqlen_k,
        rotary_pos_emb=None,
        incremental_state=None,
        nopadding=True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        dropout_p = self.attn_drop.p if self.training else 0.0

        if nopadding:
            qkv = self.qkv(x)
            qkv = qkv.view(B * N, self.num_heads * 3, self.head_dim)
            q, k, v = qkv.split([self.num_heads] * 3, dim=1)
            q, k = self.q_norm(q), self.k_norm(k)

            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            if rotary_pos_emb is not None:
                q = apply_rotary_emb(q, rotary_pos_emb, use_real=False).type_as(q)
                k = apply_rotary_emb(k, rotary_pos_emb, use_real=False).type_as(k)

            if incremental_state is not None:
                if "prev_k" in incremental_state:
                    prev_k = incremental_state["prev_k"]
                    k = torch.cat([prev_k, k], dim=1)

                if "cur_k" not in incremental_state:
                    incremental_state["cur_k"] = {}
                incremental_state["cur_k"] = k

                if "prev_v" in incremental_state:
                    prev_v = incremental_state["prev_v"]
                    v = torch.cat([prev_v, v], dim=1)

                if "cur_v" not in incremental_state:
                    incremental_state["cur_v"] = {}
                incremental_state["cur_v"] = v

            q = q.view(B * N, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_heads, self.head_dim)
            v = v.view(-1, self.num_heads, self.head_dim)

            if self.fused_attn:
                x = flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=dropout_p,
                )
            else:
                x = _sdpa_varlen_attn(
                    q, k, v, cu_seqlens, cu_seqlens_k, dropout_p=dropout_p
                )
        else:
            if incremental_state is not None:
                raise NotImplementedError(
                    "It is designed for batching inference. AR-chunk is not supported currently."
                )

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            if self.qk_norm:
                q, k, v = qkv.unbind(2)
                q, k = self.q_norm(q), self.k_norm(k)
                # re-bind
                qkv = torch.stack((q, k, v), dim=2)

            # pack qkv with seq_len
            qkv_collect = []
            for i in range(qkv.shape[0]):
                qkv_collect.append(qkv[i, : seq_len[i], :, :, :])

            qkv = torch.cat(qkv_collect, dim=0)

            if self.fused_attn:
                x = flash_attn_varlen_qkvpacked_func(
                    qkv=qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=dropout_p,
                )
            else:
                x = _sdpa_varlen_attn(
                    qkv[:, 0], qkv[:, 1], qkv[:, 2],
                    cu_seqlens, cu_seqlens, dropout_p=dropout_p,
                )

            # unpack and pad 0
            x_collect = []
            for i in range(B):
                x_collect.append(x[cu_seqlens[i] : cu_seqlens[i + 1], :, :])
            x = torch.nn.utils.rnn.pad_sequence(
                x_collect, batch_first=True, padding_value=0
            )

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        ffn_type="conv1d_conv1d",
        ffn_gated_glu=True,
        ffn_act_layer="gelu",
        ffn_conv_kernel_size=5,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if ffn_type == "vanilla_mlp":
            from timm.models.vision_transformer import Mlp

            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        else:
            raise NotImplementedError(f"FFN type {ffn_type} is not implemented")

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x,
        c,
        seq_len,
        cu_seqlens,
        cu_maxlen,
        cu_seqlens_k,
        cu_maxlen_k,
        mask,
        rotary_pos_emb=None,
        incremental_state=None,
        nopadding=True,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=2)
        )

        x_ = modulate(self.norm1(x), shift_msa, scale_msa)

        if incremental_state is not None:
            if "attn_kvcache" not in incremental_state:
                incremental_state["attn_kvcache"] = {}
            inc_attn = incremental_state["attn_kvcache"]
        else:
            inc_attn = None

        x_ = self.attn(
            x_,
            seq_len=seq_len,
            cu_seqlens=cu_seqlens,
            max_seqlen=cu_maxlen,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=cu_maxlen_k,
            rotary_pos_emb=rotary_pos_emb,
            incremental_state=inc_attn,
            nopadding=nopadding,
        )

        if not nopadding:
            x_ = x_ * mask[:, :, None]

        x = x + gate_msa * x_

        x_ = modulate(self.norm2(x), shift_mlp, scale_mlp)

        x_ = self.mlp(x_)

        if not nopadding:
            x_ = x_ * mask[:, :, None]

        x = x + gate_mlp * x_
        return x
