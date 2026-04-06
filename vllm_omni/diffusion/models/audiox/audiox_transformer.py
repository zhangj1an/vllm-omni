from __future__ import annotations

import logging
import typing as tp

import torch
import torch.nn as nn
from audiox.models.mm_embeddings import apply_rope, compute_rope_rotations
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

from vllm_omni.diffusion.layers.fourier import GaussianFourierProjection

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

        self.w1 = AudioXMMChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w2 = AudioXMMChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w3 = AudioXMMChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


class AudioXRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=-1, keepdim=True)
        scale = torch.rsqrt(mean_sq + self.eps)
        return x * scale


class AudioXMMDiTSelfAttention(nn.Module):
    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        head_dim = dim // nheads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = AudioXRMSNorm(head_dim)
        self.k_norm = AudioXRMSNorm(head_dim)

        self.split_into_heads = Rearrange("b n (h d j) -> b h n d j", h=nheads, d=dim // nheads, j=3)

    def apply_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = F.scaled_dot_product_attention(q, k, v)
        return rearrange(out, "b h n d -> b n (h d)").contiguous()

    def pre_attention(self, x: torch.Tensor, rot: torch.Tensor | None = None):
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        rot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x, rot=rot)
        return self.apply_attention(q, k, v)


class AudioXMMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = AudioXMMDiTSelfAttention(dim, nhead)
        from audiox.models.mm_transformer_layers import CrossAttention as UpCross

        self.cross_attn = UpCross(dim, nhead)
        self.linear1 = AudioXMMChannelLastConv1d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = AudioXMMConvFeedForward(dim, int(dim * mlp_ratio), kernel_size=3, padding=1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: torch.Tensor | None):
        modulation = self.adaLN_modulation(c)
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor, ...], context=None):
        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa

        x = x + self.cross_attn(x, context=context)

        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        rot: torch.Tensor | None,
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
        x = rearrange(x, "b c t -> b t c")

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)
        output = self.transformer(
            x,
            prepend_embeds=prepend_inputs,
            context=cross_attn_cond,
        )

        output = rearrange(output, "b t c -> b c t")[:, :, prepend_length:]
        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)
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
        """Inference-only forward; unknown keyword arguments are ignored."""
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
                )
                for _ in range(depth)
            ]
        )
        self.proj_mm_tokens = nn.Linear(768, hidden_dim) if dim != 768 else nn.Identity()
        self.proj_mm_seq_len = nn.Linear(384, self._latent_seq_len) if self._latent_seq_len != 384 else nn.Identity()

        base_freq = 1.0
        latent_rot = compute_rope_rotations(
            self._latent_seq_len,
            hidden_dim // num_heads,
            10000,
            freq_scaling=base_freq,
            device=self.device,
        )
        self.register_buffer("latent_rot", latent_rot, persistent=False)

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
        rot = self.latent_rot.to(device=x.device, dtype=self.latent_rot.dtype)
        for block in self.layers:
            x = block(x, mm_tokens, rot, context=time_cond)

        x = self.project_out(x)
        return x


def __getattr__(name: str):
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
