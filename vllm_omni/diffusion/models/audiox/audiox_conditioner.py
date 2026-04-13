"""AudioX multi-modal conditioning for diffusion.

Architecture:

- **Upstream encoders** (Transformers / Diffusers): ``T5EncoderModel`` + ``T5TokenizerFast``,
  ``CLIPVisionModelWithProjection``, and the audio VAE loaded via
  :func:`create_pretransform_from_config` (``AutoencoderOobleck``).

- **Thin adapters** below only: normalize / pack inputs, call the upstream module, project to
  ``cond_dim``, and return masks / tensors in the shape vLLM-Omni expects.

- **AudioX-specific glue** (kept because it is weight- and numerics-sensitive): the ``SA_*``
  temporal transformer stack (manual attention — do not swap for SDPA; see module comments),
  sync-frame fusion, empty-video placeholders, and :class:`MultiConditioner` batch assembly.
"""

from __future__ import annotations

import inspect
import typing as tp
from typing import Any

import torch
from diffusers import AutoencoderOobleck
from einops import rearrange
from torch import einsum, nn


def _kwargs_for(cls: type, cfg: dict[str, Any]) -> dict[str, Any]:
    """Filter ``cfg`` to just the keyword arguments accepted by ``cls.__init__``.

    Upstream AudioX configs ship training-only knobs (e.g. ``mask_ratio_*``, ``project_out``)
    alongside inference-relevant keys. Rather than maintain a whitelist, walk the constructor
    signature and drop anything that isn't a named parameter. Constructors that take ``**kwargs``
    are handled by the caller (no filtering needed).
    """
    sig = inspect.signature(cls.__init__)
    accepted = {
        name for name, param in sig.parameters.items() if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    return {k: v for k, v in cfg.items() if k in accepted}


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------


def set_audio_channels(audio: torch.Tensor, target_channels: int) -> torch.Tensor:
    if target_channels == 1:
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


def _stack_pad_audio_list(audio: list[torch.Tensor], device: torch.device | str) -> torch.Tensor:
    bs = len(audio)
    max_len = max(a.shape[-1] for a in audio)
    padded: list[torch.Tensor] = []
    for i in range(bs):
        t = audio[i].to(device)
        pad_len = max_len - t.shape[-1]
        if pad_len > 0:
            t = torch.nn.functional.pad(t, (0, pad_len))
        padded.append(t)
    return torch.cat(padded, dim=0)


# ---------------------------------------------------------------------------
# Video temporal stack (aligned with pip ``audiox.models.temptransformer``)
# Do not replace ``SA_Attention`` with ``vllm_omni.diffusion.attention.layer.Attention`` (SDPA): on the
# same weights it numerically drifts vs upstream AudioX, which degrades conditioning and noticeably
# hurts output quality (e.g. multimodal tasks). See ``debug_audiox_tv2m_layer_compare``.
# ---------------------------------------------------------------------------


class SA_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SA_FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Dropout p=0 preserves upstream ``net.{2,4}`` state-dict keys; inference is identical to no dropout.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


# Manual einsum+softmax only. SDPA/diffusion Attention here degrades conditioning vs upstream and output quality.
class SA_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(0.0),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class SA_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SA_PreNorm(dim, SA_Attention(dim, heads=heads, dim_head=dim_head)),
                        SA_PreNorm(dim, SA_FeedForward(dim, mlp_dim)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ---------------------------------------------------------------------------
# Base + registry
# ---------------------------------------------------------------------------


class Conditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int, project_out: bool = False):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Audio: Diffusers VAE encode + linear projections to conditioning layout
# ---------------------------------------------------------------------------


class AudioVaePromptAdapter(Conditioner):
    """Encode waveform with the loaded ``AutoencoderOobleck`` and project to ``cond_dim``."""

    def __init__(
        self,
        pretransform: tp.Any,
        output_dim: int,
        latent_seq_len: int = 237,
    ):
        icfg = pretransform.config
        enc_ch = int(getattr(icfg, "latent_channels", getattr(icfg, "decoder_input_channels", 1)))
        super().__init__(enc_ch, output_dim)
        self.pretransform = pretransform
        self.latent_seq_len = latent_seq_len
        self.proj_features_128 = nn.Linear(in_features=self.latent_seq_len, out_features=128)

    def forward(
        self, audio: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor], device: torch.device | str
    ) -> list[torch.Tensor]:
        self.to(device)

        if isinstance(audio, (list, tuple)):
            audio = list(audio)
            audio_t = _stack_pad_audio_list(audio, device)
        else:
            audio_t = audio.to(device)

        audio_t = set_audio_channels(audio_t, int(getattr(self.pretransform.config, "audio_channels", 2)))

        vae = self.pretransform
        z = vae.encode(audio_t, return_dict=True).latent_dist.sample()
        latents = z / float(vae.audiox_scaling_factor)
        latents = self.proj_features_128(latents)
        latents = latents.permute(0, 2, 1)
        latents = self.proj_out(latents)
        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


class MultiConditioner(nn.Module):
    def __init__(self, conditioners: dict[str, Conditioner]):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)

    def forward(
        self,
        batch_metadata: list[dict[str, tp.Any]],
        device: torch.device | str,
        *,
        require_single_item_sequence: bool = False,
    ) -> dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            conditioner_inputs = _gather_conditioner_inputs(
                batch_metadata=batch_metadata,
                key=key,
                require_single_item_sequence=require_single_item_sequence,
            )
            output[key] = conditioner(conditioner_inputs, device)
        return output


_AUDIOX_VAE_CONFIG = {
    "audio_channels": 2,
    "channel_multiples": [1, 2, 4, 8, 16],
    "decoder_channels": 128,
    "decoder_input_channels": 64,
    "downsampling_ratios": [2, 4, 4, 8, 8],
    "encoder_hidden_size": 128,
    "sampling_rate": 44100,
}


def create_pretransform_from_config(
    pretransform_config: dict[str, tp.Any],
) -> AutoencoderOobleck:
    """Create ``AutoencoderOobleck`` from config only — weights are loaded later
    through the unified ``load_weights`` path.

    Sets ``audiox_scaling_factor`` on the module; callers scale latents when encoding/decoding.
    """
    pretransform_type = pretransform_config.get("type", "autoencoder")

    if pretransform_type != "autoencoder":
        raise NotImplementedError(
            f"AudioX HKUST weights only use pretransform type 'autoencoder'; got {pretransform_type!r}"
        )

    vae = AutoencoderOobleck(**_AUDIOX_VAE_CONFIG)

    icfg = vae.config
    scaling_factor = float(pretransform_config.get("scale", getattr(icfg, "scaling_factor", 1.0)))
    vae.audiox_scaling_factor = scaling_factor  # type: ignore[attr-defined]

    vae.eval().requires_grad_(False)
    return vae


def _build_pretransform(conditioner_config: dict[str, tp.Any]) -> tp.Any:
    conditioner_config.pop("sample_rate", None)
    pretransform = create_pretransform_from_config(
        conditioner_config.pop("pretransform_config"),
    )
    return pretransform


def _assert_conditioner_key_in_item(item: dict[str, tp.Any], key: str) -> None:
    if key not in item:
        raise ValueError(f"Conditioner key {key} not found in batch metadata")


def _normalize_condition_value(value: tp.Any, *, source_key: str, require_single_item_sequence: bool) -> tp.Any:
    if isinstance(value, (list, tuple)):
        if require_single_item_sequence:
            if len(value) != 1:
                raise ValueError(f"Conditioner input for key {source_key!r} must be scalar or single-item list/tuple.")
            return value[0]
        if len(value) == 1:
            return value[0]
    return value


def _gather_conditioner_inputs(
    *,
    batch_metadata: list[dict[str, tp.Any]],
    key: str,
    require_single_item_sequence: bool,
) -> list[tp.Any]:
    inputs: list[tp.Any] = []
    for item in batch_metadata:
        _assert_conditioner_key_in_item(item, key)
        value = _normalize_condition_value(
            item[key],
            source_key=key,
            require_single_item_sequence=require_single_item_sequence,
        )
        inputs.append(value)
    return inputs


def _with_output_dim(cond_dim: int, cfg: dict[str, tp.Any]) -> dict[str, tp.Any]:
    out = {"output_dim": cond_dim}
    out.update(cfg)
    return out


def create_audiox_fixed_conditioner_from_conditioning_config(
    config: dict[str, tp.Any],
) -> MultiConditioner:
    """Create audio conditioner.  T5 text and CLIP video encoding are handled directly by the pipeline."""
    cond_dim = config["cond_dim"]

    by_id: dict[str, dict[str, tp.Any]] = {}
    for item in config["configs"]:
        if not isinstance(item, dict):
            raise ValueError("Each conditioning config entry must be a dict.")
        cid = item.get("id")
        cconf = item.get("config")
        if not isinstance(cid, str) or not isinstance(cconf, dict):
            raise ValueError("Each conditioning config must include string 'id' and dict 'config'.")
        by_id[cid] = dict(cconf)

    audio_cfg = _with_output_dim(cond_dim, by_id["audio_prompt"])
    pretransform = _build_pretransform(audio_cfg)
    audio_cfg = _kwargs_for(AudioVaePromptAdapter, audio_cfg)

    conditioners: dict[str, Conditioner] = {
        "audio_prompt": AudioVaePromptAdapter(pretransform, **audio_cfg),
    }
    return MultiConditioner(conditioners)


def encode_audiox_conditioning_tensors(
    multi_conditioner: MultiConditioner,
    *,
    batch_metadata: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, tp.Any]:
    return multi_conditioner(
        batch_metadata,
        device,
        require_single_item_sequence=True,
    )
