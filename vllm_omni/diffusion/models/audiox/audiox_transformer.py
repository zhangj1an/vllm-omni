"""AudioX multimodal diffusion transformer entry points and MM layers."""

from __future__ import annotations

import typing as tp

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.fourier import GaussianFourierProjection
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

__all__ = [
    "AudioXMMChannelLastConv1d",
    "AudioXMMConvFeedForward",
    "AudioXMMDiTSelfAttention",
    "AudioXMMDiTCrossAttention",
    "AudioXMMDiTBlock",
    "ContinuousMMDiTTransformer",
    "MMDiffusionTransformer",
]


class FourierFeatures(GaussianFourierProjection):
    def __init__(self, in_features, out_features, std=1.0):
        assert out_features % 2 == 0
        super().__init__(
            in_features=in_features,
            embedding_size=out_features // 2,
            scale=std,
            trainable=True,
        )


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


class AudioXMMDiTSelfAttention(nn.Module):
    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        head_dim = dim // nheads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(head_dim, eps=1e-6, has_weight=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6, has_weight=False)
        self._diffusion_attn = Attention(
            num_heads=nheads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
        )
        self.rope = RotaryEmbedding(is_neox_style=False)

        self.split_into_heads = Rearrange("b n (h d j) -> b h n d j", h=nheads, d=dim // nheads, j=3)

    def apply_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = self._diffusion_attn(q, k, v, attn_metadata=None)
        return out.flatten(2, 3).contiguous()

    def pre_attention(self, x: torch.Tensor, rot: tuple[torch.Tensor, torch.Tensor] | None = None):
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            cos, sin = rot
            q = self.rope(q.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2).contiguous()
            k = self.rope(k.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2).contiguous()

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        rot: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x, rot=rot)
        return self.apply_attention(q, k, v)


class AudioXMMDiTCrossAttention(nn.Module):
    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        head_dim = dim // nheads
        self.to_q = ReplicatedLinear(dim, dim, bias=False, disable_tp=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=False, disable_tp=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=False, disable_tp=True)
        self.q_norm = RMSNorm(head_dim, eps=1e-6, has_weight=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6, has_weight=False)
        self._diffusion_attn = Attention(
            num_heads=nheads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
        )

        self.split_q_into_heads = Rearrange("b n (h d) -> b h n d", h=nheads, d=dim // nheads)
        self.split_kv_into_heads = Rearrange("b n (h d j) -> b h n d j", h=nheads, d=dim // nheads, j=2)

    def apply_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = self._diffusion_attn(q, k, v, attn_metadata=None)
        return out.flatten(2, 3).contiguous()

    def pre_attention(self, x: torch.Tensor, context: torch.Tensor | None):
        q, _ = self.to_q(x)
        k_raw, _ = self.to_k(context)
        v_raw, _ = self.to_v(context)
        q = self.split_q_into_heads(q)
        kv = torch.cat([k_raw, v_raw], dim=-1)
        k, v = self.split_kv_into_heads(kv).chunk(2, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k, v

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        q, k, v = self.pre_attention(x, context=context)
        return self.apply_attention(q, k, v)


class AudioXMMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        cross_attend: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = AudioXMMDiTSelfAttention(dim, nhead)
        if cross_attend:
            self.cross_attn = AudioXMMDiTCrossAttention(dim, nhead)
        self.linear1 = AudioXMMChannelLastConv1d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = AudioXMMConvFeedForward(dim, int(dim * mlp_ratio), kernel_size=3, padding=1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: tuple[torch.Tensor, torch.Tensor] | None):
        modulation = self.adaLN_modulation(c)
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor, ...], context=None):
        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa

        if context is not None:
            x = x + self.cross_attn(x, context=context)

        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
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
        fusion_depth=6,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        condition_mask_type: None = None,
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        od_config: OmniDiffusionConfig | None = None,
        **kwargs,
    ):
        super().__init__()

        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config is not None else None
        self.cond_token_dim = cond_token_dim
        if patch_size != 1:
            raise ValueError("AudioX inference-only MMDiT requires patch_size=1.")
        if transformer_type != "continuous_transformer":
            raise ValueError("AudioX inference-only MMDiT requires transformer_type='continuous_transformer'.")
        if condition_mask_type is not None:
            raise ValueError("AudioX inference-only MMDiT does not support condition masks.")
        if global_cond_type != "prepend":
            raise ValueError("AudioX inference-only MMDiT requires global_cond_type='prepend'.")
        if not cond_token_dim > 0:
            raise ValueError("AudioX inference-only MMDiT requires cond_token_dim > 0.")
        if project_cond_tokens:
            raise ValueError(
                "AudioX inference-only MMDiT requires project_cond_tokens=False to match official checkpoints."
            )

        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
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
        self.condition_mask_type = condition_mask_type
        self.global_cond_type = global_cond_type

        self.transformer = ContinuousMMDiTTransformer(
            dim=embed_dim,
            depth=depth,
            fusion_depth=fusion_depth,
            dim_heads=embed_dim // num_heads,
            dim_in=dim_in * patch_size,
            dim_out=io_channels * patch_size,
            cross_attend=True,
            cond_token_dim=cond_embed_dim,
            global_cond_dim=None,
            **kwargs,
        )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        **kwargs,
    ):
        if global_embed is not None:
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2],), mode="nearest")
            x = torch.cat([x, input_concat_cond], dim=1)

        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        global_embed = global_embed + timestep_embed.unsqueeze(1) if global_embed is not None else timestep_embed

        if self.global_cond_type == "prepend":
            if prepend_inputs is None:
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat(
                    [prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1
                )
            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)
        output = self.transformer(
            x,
            prepend_embeds=prepend_inputs,
            context=cross_attn_cond,
            context_mask=cross_attn_cond_mask,
            mask=mask,
            prepend_mask=prepend_mask,
            return_info=return_info,
            **kwargs,
        )

        if return_info:
            output, info = output

        output = rearrange(output, "b t c -> b c t")[:, :, prepend_length:]
        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)
        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info
        return output

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        causal=False,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        **kwargs,
    ):
        del negative_global_embed
        assert not causal, "Causal mode is not supported for DiffusionTransformer"
        if cfg_dropout_prob != 0.0:
            raise ValueError(
                "cfg_dropout_prob must be 0 for vLLM-Omni AudioX (inference-only; training-time CFG dropout was removed)."
            )

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()
            cross_attn_cond_mask = None

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None):
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)
            batch_global_cond = torch.cat([global_embed, global_embed], dim=0) if global_embed is not None else None
            batch_input_concat_cond = (
                torch.cat([input_concat_cond, input_concat_cond], dim=0) if input_concat_cond is not None else None
            )

            batch_cond = None
            batch_cond_masks = None
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
                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)

            batch_prepend_cond = None
            batch_prepend_cond_mask = None
            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)

            batch_masks = torch.cat([mask, mask], dim=0) if mask is not None else None
            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                mask=batch_masks,
                input_concat_cond=batch_input_concat_cond,
                global_embed=batch_global_cond,
                prepend_cond=batch_prepend_cond,
                prepend_cond_mask=batch_prepend_cond_mask,
                return_info=return_info,
                **kwargs,
            )

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std / out_cfg_std)) + (1 - scale_phi) * cfg_output
            else:
                output = cfg_output

            if return_info:
                return output, info
            return output

        return self._forward(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_cond_mask,
            input_concat_cond=input_concat_cond,
            global_embed=global_embed,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            mask=mask,
            return_info=return_info,
            **kwargs,
        )


class ContinuousMMDiTTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        fusion_depth,
        *,
        dim_in=None,
        dim_out=None,
        dim_heads=64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        _latent_seq_len=237,
        **kwargs,
    ):
        del (
            fusion_depth,
            cond_token_dim,
            global_cond_dim,
            causal,
            rotary_pos_emb,
            zero_init_branch_outputs,
            conformer,
            kwargs,
        )
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = False

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        hidden_dim = dim
        num_heads = dim_heads
        mlp_ratio = 4.0
        fused_depth = depth
        cross_attend = True
        self.cross_attend = cross_attend
        self._latent_seq_len = _latent_seq_len

        self.layers = nn.ModuleList(
            [
                AudioXMMDiTBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    cross_attend=cross_attend,
                )
                for _ in range(fused_depth)
            ]
        )
        self.proj_mm_tokens = nn.Linear(768, hidden_dim) if dim != 768 else nn.Identity()
        self.proj_mm_seq_len = nn.Linear(384, self._latent_seq_len) if self._latent_seq_len != 384 else nn.Identity()

        base_freq = 1.0
        rope_dim = hidden_dim // num_heads
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pos = torch.arange(self._latent_seq_len, dtype=torch.float32, device=self.device)
            freqs = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=self.device) / rope_dim))
            freqs *= base_freq
            angles = torch.einsum("n,d->nd", pos, freqs)
            self.register_buffer("latent_cos", torch.cos(angles), persistent=False)
            self.register_buffer("latent_sin", torch.sin(angles), persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        mask=None,
        prepend_embeds=None,
        prepend_mask=None,
        global_cond=None,
        context=None,
        context_mask=None,
        return_info=False,
        **kwargs,
    ):
        del mask, prepend_mask, global_cond, context_mask, kwargs

        info = {"hidden_states": []}
        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], "prepend dimension must match sequence dimension"
            x = torch.cat((prepend_embeds, x), dim=-2)

        time_cond = prepend_embeds.squeeze(1)
        mm_tokens = context

        mm_tokens = self.proj_mm_tokens(mm_tokens)
        mm_tokens = mm_tokens.transpose(1, 2)
        mm_tokens = self.proj_mm_seq_len(mm_tokens)
        mm_tokens = mm_tokens.transpose(1, 2)

        time_cond = time_cond.unsqueeze(1)
        # Keep RoPE tables dtype/device aligned with hidden states for rotary kernels.
        rot = (
            self.latent_cos.to(device=x.device, dtype=x.dtype),
            self.latent_sin.to(device=x.device, dtype=x.dtype),
        )
        for block in self.layers:
            x = block(x, mm_tokens, rot, context=time_cond)

        x = self.project_out(x)
        if return_info:
            return x, info
        return x


def __getattr__(name: str):
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
