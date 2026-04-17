from __future__ import annotations

import logging
import typing as tp
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.layers.fourier import GaussianFourierProjection
from vllm_omni.diffusion.layers.rope import RotaryEmbedding


class AudioXCrossAttention(nn.Module):
    def __init__(self, dim: int, nheads: int, prefix: str = ""):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        head_dim = dim // nheads
        self.head_dim = head_dim

        # to_kv bundle weights arrive in (head, dim, VK-index) interleaved order; pipeline's
        # load_weights restacks them to [V|K] before MergedColumnParallelLinear consumes them.
        self.to_q = ColumnParallelLinear(dim, dim, bias=False, gather_output=False, prefix=f"{prefix}.to_q")
        self.to_kv = MergedColumnParallelLinear(
            input_size=dim, output_sizes=[dim, dim], bias=False, gather_output=False, prefix=f"{prefix}.to_kv"
        )
        self.q_norm = AudioXRMSNorm(head_dim)
        self.k_norm = AudioXRMSNorm(head_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        tp = get_tensor_model_parallel_world_size()
        local_h = self.nheads // tp
        d = self.head_dim

        q_flat, _ = self.to_q(x)
        q = q_flat.unflatten(-1, (local_h, d)).transpose(1, 2)
        kv_flat, _ = self.to_kv(context)
        v_flat, k_flat = kv_flat.chunk(2, dim=-1)
        # Upstream `CrossAttention.forward` unpacks `pre_attention()` as `q, v, k = ...` (K/V
        # swapped), AND only normalizes the first chunk of to_kv via k_norm. Trained weights
        # depend on this quirk: first chunk normalized -> V; second chunk unnormalized -> K.
        v = v_flat.unflatten(-1, (local_h, d)).transpose(1, 2)
        k = k_flat.unflatten(-1, (local_h, d)).transpose(1, 2)
        q = self.q_norm(q)
        v = self.k_norm(v)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).flatten(-2).contiguous()
        if tp > 1:
            out = tensor_model_parallel_all_gather(out, dim=-1)
        return out


logger = logging.getLogger(__name__)

__all__ = [
    "AudioXMMChannelLastConv1d",
    "AudioXMMConvFeedForward",
    "AudioXMMDiTSelfAttention",
    "AudioXMMDiTBlock",
    "ContinuousMMDiTTransformer",
    "MMDiffusionTransformer",
]


class AudioXMMChannelLastConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


def _slice_weight_loader_out(param: nn.Parameter, loaded: torch.Tensor) -> None:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    out_total = loaded.shape[0]
    out_local = out_total // tp_size
    start = tp_rank * out_local
    param.data.copy_(loaded[start : start + out_local])


def _slice_weight_loader_in(param: nn.Parameter, loaded: torch.Tensor) -> None:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    in_total = loaded.shape[1]
    in_local = in_total // tp_size
    start = tp_rank * in_local
    param.data.copy_(loaded[:, start : start + in_local])


class _ColumnParallelChannelLastConv1d(AudioXMMChannelLastConv1d):
    def __init__(self, in_channels: int, out_channels_total: int, **kwargs: Any):
        tp_size = get_tensor_model_parallel_world_size()
        assert out_channels_total % tp_size == 0, (out_channels_total, tp_size)
        super().__init__(in_channels, out_channels_total // tp_size, **kwargs)
        self.weight.weight_loader = _slice_weight_loader_out


class _RowParallelChannelLastConv1d(AudioXMMChannelLastConv1d):
    def __init__(self, in_channels_total: int, out_channels: int, **kwargs: Any):
        tp_size = get_tensor_model_parallel_world_size()
        assert in_channels_total % tp_size == 0, (in_channels_total, tp_size)
        super().__init__(in_channels_total // tp_size, out_channels, **kwargs)
        self._tp_size = tp_size
        self.weight.weight_loader = _slice_weight_loader_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = tensor_model_parallel_all_reduce(y)
        return y


class AudioXMMConvFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = _ColumnParallelChannelLastConv1d(
            dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding
        )
        self.w2 = _RowParallelChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w3 = _ColumnParallelChannelLastConv1d(
            dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AudioXRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=-1, keepdim=True)
        scale = torch.rsqrt(mean_sq + self.eps)
        return x * scale


class AudioXMMDiTSelfAttention(nn.Module):
    def __init__(self, dim: int, nheads: int, prefix: str = ""):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        head_dim = dim // nheads
        self.head_dim = head_dim
        # Bundle weights arrive in interleaved (h, d, qkv) layout; pipeline's load_weights
        # restacks to [Q|K|V] before QKVParallelLinear's weight_loader consumes them.
        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=nheads,
            bias=True,
            prefix=f"{prefix}.qkv",
        )
        self.q_norm = AudioXRMSNorm(head_dim)
        self.k_norm = AudioXRMSNorm(head_dim)

        self.rope = RotaryEmbedding(is_neox_style=False)

    def apply_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).flatten(-2).contiguous()
        # Downstream linear1/ffn are replicated and expect full hidden dim.
        if get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_gather(out, dim=-1)
        return out

    def pre_attention(self, x: torch.Tensor, rot: tuple[torch.Tensor, torch.Tensor] | None = None):
        qkv, _ = self.qkv(x)
        local_h = self.qkv.num_heads
        d = self.head_dim
        q_size = local_h * d
        q, k, v = qkv.split([q_size, q_size, q_size], dim=-1)
        q = q.unflatten(-1, (local_h, d)).transpose(1, 2)
        k = k.unflatten(-1, (local_h, d)).transpose(1, 2)
        v = v.unflatten(-1, (local_h, d)).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            cos, sin = rot
            q = self.rope(q.transpose(1, 2), cos, sin).transpose(1, 2)
            k = self.rope(k.transpose(1, 2), cos, sin).transpose(1, 2)

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        rot: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x, rot=rot)
        return self.apply_attention(q, k, v)


class AudioXMMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = AudioXMMDiTSelfAttention(dim, nhead, prefix=f"{prefix}.attn")
        self.cross_attn = AudioXCrossAttention(dim, nhead, prefix=f"{prefix}.cross_attn")
        self.linear1 = AudioXMMChannelLastConv1d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = AudioXMMConvFeedForward(dim, int(dim * mlp_ratio), kernel_size=3, padding=1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: tuple[torch.Tensor, torch.Tensor] | None):
        modulation = self.adaLN_modulation(c)
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)
        x = self.norm1(x) * (1 + scale_msa) + shift_msa
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor, ...], context=None):
        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa

        x = x + self.cross_attn(x, context=context)

        r = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + self.ffn(r) * gate_mlp
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        rot: tuple[torch.Tensor, torch.Tensor] | None,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = self.attn.apply_attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions, context=context)
        return x


class MMDiffusionTransformer(nn.Module):
    def __init__(
        self,
        io_channels=32,
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.debug("MMDiffusionTransformer ignoring unused config keys: %s", sorted(kwargs.keys()))

        self.cond_token_dim = cond_token_dim
        if patch_size != 1:
            raise ValueError("AudioX inference-only MMDiT requires patch_size=1.")
        if transformer_type != "continuous_transformer":
            raise ValueError("AudioX inference-only MMDiT requires transformer_type='continuous_transformer'.")
        if global_cond_type != "prepend":
            raise ValueError("AudioX inference-only MMDiT requires global_cond_type='prepend'.")
        if not cond_token_dim > 0:
            raise ValueError("AudioX inference-only MMDiT requires cond_token_dim > 0.")
        if project_cond_tokens:
            raise ValueError(
                "AudioX inference-only MMDiT requires project_cond_tokens=False to match official checkpoints."
            )

        timestep_features_dim = 256
        self.timestep_features = GaussianFourierProjection(
            in_features=1,
            embedding_size=timestep_features_dim // 2,
            scale=1.0,
            trainable=False,
        )
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        if cond_token_dim > 0:
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False),
            )

        if prepend_cond_dim > 0:
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )

        self.input_concat_dim = input_concat_dim
        dim_in = io_channels + self.input_concat_dim
        self.patch_size = patch_size
        self.global_cond_type = global_cond_type

        self.transformer = ContinuousMMDiTTransformer(
            dim=embed_dim,
            depth=depth,
            dim_heads=embed_dim // num_heads,
            dim_in=dim_in * patch_size,
            dim_out=io_channels * patch_size,
        )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                logger.debug("Skipping weight %s - not found in model", name)

        return loaded_params

    def _forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
    ):
        if global_embed is not None:
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None
        prepend_length = 0
        if prepend_cond is not None:
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_inputs = prepend_cond

        if input_concat_cond is not None:
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2],), mode="nearest")
            x = torch.cat([x, input_concat_cond], dim=1)

        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        global_embed = global_embed + timestep_embed.unsqueeze(1) if global_embed is not None else timestep_embed

        if self.global_cond_type == "prepend":
            if prepend_inputs is None:
                prepend_inputs = global_embed.unsqueeze(1)
            else:
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x
        x = x.transpose(1, 2)

        if self.patch_size > 1:
            x = x.unflatten(1, (-1, self.patch_size)).transpose(2, 3).flatten(2)
        output = self.transformer(
            x,
            prepend_embeds=prepend_inputs,
            context=cross_attn_cond,
        )

        output = output.transpose(1, 2)[:, :, prepend_length:]
        if self.patch_size > 1:
            output = output.unflatten(1, (-1, self.patch_size)).transpose(2, 3).flatten(2)
        output = self.postprocess_conv(output) + output

        return output

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        cfg_scale=1.0,
        causal=False,
        scale_phi=0.0,
        **kwargs,
    ):
        assert not causal, "Causal mode is not supported for DiffusionTransformer"

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None):
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)
            batch_global_cond = torch.cat([global_embed, global_embed], dim=0) if global_embed is not None else None
            batch_input_concat_cond = (
                torch.cat([input_concat_cond, input_concat_cond], dim=0) if input_concat_cond is not None else None
            )

            batch_cond = None
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                if negative_cross_attn_cond is not None:
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
                        negative_cross_attn_cond = torch.where(
                            negative_cross_attn_mask, negative_cross_attn_cond, null_embed
                        )
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)
                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

            batch_prepend_cond = None
            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)

            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                input_concat_cond=batch_input_concat_cond,
                global_embed=batch_global_cond,
                prepend_cond=batch_prepend_cond,
            )

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std / out_cfg_std)) + (1 - scale_phi) * cfg_output
            else:
                output = cfg_output

            return output

        return self._forward(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            input_concat_cond=input_concat_cond,
            global_embed=global_embed,
            prepend_cond=prepend_cond,
        )


class ContinuousMMDiTTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in=None,
        dim_out=None,
        dim_heads=64,
        _latent_seq_len=237,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        hidden_dim = dim
        num_heads = dim_heads
        mlp_ratio = 4.0
        self._latent_seq_len = _latent_seq_len

        self.layers = nn.ModuleList(
            [
                AudioXMMDiTBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    prefix=f"layers.{i}",
                )
                for i in range(depth)
            ]
        )
        self.proj_mm_tokens = nn.Linear(768, hidden_dim) if dim != 768 else nn.Identity()
        self.proj_mm_seq_len = nn.Linear(384, self._latent_seq_len) if self._latent_seq_len != 384 else nn.Identity()

        # AudioX RoPE: interleaved (GPT-J) pair layout, theta=10000.
        head_dim = hidden_dim // num_heads
        pos = torch.arange(self._latent_seq_len, dtype=torch.float32, device=self.device)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=self.device) / head_dim))
        ang = torch.outer(pos, inv_freq)
        self.register_buffer("latent_rope_cos", torch.cos(ang), persistent=False)
        self.register_buffer("latent_rope_sin", torch.sin(ang), persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        prepend_embeds=None,
        context=None,
    ):
        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_dim = prepend_embeds.shape[-1]
            assert prepend_dim == x.shape[-1], "prepend dimension must match sequence dimension"
            x = torch.cat((prepend_embeds, x), dim=-2)

        time_cond = prepend_embeds.squeeze(1)
        mm_tokens = context

        mm_tokens = self.proj_mm_tokens(mm_tokens)
        mm_tokens = mm_tokens.transpose(1, 2)
        mm_tokens = self.proj_mm_seq_len(mm_tokens)
        mm_tokens = mm_tokens.transpose(1, 2)

        time_cond = time_cond.unsqueeze(1)
        rot = (
            self.latent_rope_cos.to(device=x.device, dtype=x.dtype),
            self.latent_rope_sin.to(device=x.device, dtype=x.dtype),
        )
        for block in self.layers:
            x = block(x, mm_tokens, rot, context=time_cond)

        x = self.project_out(x)
        return x


def __getattr__(name: str):
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
