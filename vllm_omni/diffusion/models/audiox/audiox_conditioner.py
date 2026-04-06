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

import logging
import os
import typing as tp
import warnings
from typing import Any

import einops
import torch
from einops import rearrange
from torch import einsum, nn
from diffusers import AutoencoderOobleck
from torchvision import transforms
from transformers import (
    AutoConfig,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from vllm_omni.diffusion.models.audiox.audiox_weights import (
    filter_audio_prompt_config_after_pretransform_build,
    prepare_audiox_video_text_conditioner_configs,
    resolve_pretransform_scale,
    resolve_vae_audio_channels,
    resolve_vae_latent_channels,
    validate_audiox_pretransform_config_keys,
)


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


def _stack_pad_audio_list(
    audio: list[torch.Tensor], device: torch.device | str
) -> torch.Tensor:
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
# CLIP video: upstream ViT + temporal / sync / empty-feature glue
# ---------------------------------------------------------------------------

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPVideoTemporalSyncConditioner(Conditioner):
    """``CLIPVisionModelWithProjection`` + SA temporal stack, sync fusion, and zero-video masking."""

    DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
    VIDEO_FPS = 5
    DURATION_SECONDS = 10
    TEMPORAL_DIM = 768
    OUT_FEATURES = 128
    CLIP_PATCH_TOKENS = 50

    def __init__(
        self,
        output_dim: int,
        project_out: bool = False,
        *,
        clip_model_name: str | None = None,
    ):
        super().__init__(dim=768, output_dim=output_dim, project_out=project_out)

        name = clip_model_name or self.DEFAULT_CLIP_MODEL
        sa_depth = 4
        num_heads = 16
        dim_head = 64
        hidden_scale = 4
        in_features = self.CLIP_PATCH_TOKENS * self.VIDEO_FPS * self.DURATION_SECONDS

        self.empty_visual_feat = nn.Parameter(
            torch.zeros(1, self.OUT_FEATURES, self.TEMPORAL_DIM), requires_grad=False
        )
        self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(name)
        self.proj = nn.Linear(in_features=in_features, out_features=self.OUT_FEATURES)
        self.proj_sync = nn.Linear(in_features=240, out_features=self.OUT_FEATURES)
        self.sync_weight = nn.Parameter(torch.tensor(0.0))
        self.in_features = in_features
        self.out_features = self.OUT_FEATURES
        self.Temp_transformer = SA_Transformer(
            self.TEMPORAL_DIM, sa_depth, num_heads, dim_head, self.TEMPORAL_DIM * hidden_scale
        )
        self.Temp_pos_embedding = nn.Parameter(
            torch.randn(1, self.DURATION_SECONDS * self.VIDEO_FPS, self.TEMPORAL_DIM)
        )

        self._clip_normalize = transforms.Compose([transforms.Normalize(mean=list(_CLIP_MEAN), std=list(_CLIP_STD))])

    def forward(self, video_list: list[torch.Tensor | dict], device: torch.device | str) -> list[torch.Tensor]:
        self.to(device)
        self.visual_encoder_model.eval()

        if isinstance(video_list[0], dict):
            video_tensors = [item["video_tensors"] for item in video_list]
            video_sync_frames = [item["video_sync_frames"] for item in video_list]
            video_sync_frames = torch.cat(video_sync_frames, dim=0).to(device)
        else:
            video_tensors = video_list
            batch_size = len(video_tensors)
            video_sync_frames = torch.zeros(batch_size, 240, 768, device=device)

        original_videos = torch.cat(video_tensors, dim=0).to(device)
        batch_size, time_length, _, _, _ = original_videos.size()
        is_zero = torch.all(original_videos == 0, dim=(1, 2, 3, 4))

        video_tensors = einops.rearrange(original_videos, "b t c h w -> (b t) c h w")
        # Video frames are expected in [0, 1] float (e.g. from `audiox_reference_media.prepare_video_reference`).
        video_cond_pixel_values = self._clip_normalize(video_tensors).to(device)

        with torch.no_grad():
            outputs = self.visual_encoder_model(pixel_values=video_cond_pixel_values)
        video_hidden = outputs.last_hidden_state
        video_hidden = einops.rearrange(video_hidden, "(b t) q h -> (b q) t h", b=batch_size, t=time_length)
        video_hidden = video_hidden + self.Temp_pos_embedding
        video_hidden = self.Temp_transformer(video_hidden)
        video_hidden = einops.rearrange(video_hidden, "(b q) t h -> b (t q) h", b=batch_size, t=time_length)
        video_hidden = self.proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        video_sync_frames = self.proj_sync(video_sync_frames.view(-1, 240))
        video_sync_frames = video_sync_frames.view(batch_size, self.out_features, -1)
        video_hidden = video_hidden + self.sync_weight * video_sync_frames

        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden)
        return [video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)]


# ---------------------------------------------------------------------------
# T5 text: upstream encoder + projection / mask (no custom T5 blocks)
# ---------------------------------------------------------------------------


def _load_t5_tokenizer_and_encoder(model_name: str) -> tuple[T5TokenizerFast, T5EncoderModel]:
    previous_level = logging.root.manager.disable
    logging.disable(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tokenizer = T5TokenizerFast.from_pretrained(model_name)
            encoder = T5EncoderModel.from_pretrained(model_name).train(False).requires_grad_(False).to(torch.float16)
        finally:
            logging.disable(previous_level)
    return tokenizer, encoder


class T5TextEncoderAdapter(Conditioner):
    """Thin wrapper: tokenize, ``T5EncoderModel`` forward, project to ``cond_dim``, mask padding."""

    def __init__(
        self,
        output_dim: int,
        t5_model_name: str = "t5-base",
        max_length: int = 128,
        project_out: bool = False,
    ):
        hidden_size = AutoConfig.from_pretrained(t5_model_name).hidden_size
        super().__init__(hidden_size, output_dim, project_out=project_out)

        self.max_length = max_length
        self.tokenizer, self.encoder = _load_t5_tokenizer_and_encoder(t5_model_name)

    def forward(self, texts: list[str], device: torch.device | str) -> list[torch.Tensor]:
        self.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.encoder.eval()
        with torch.set_grad_enabled(False):
            embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return [embeddings, attention_mask]


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
        enc_ch = resolve_vae_latent_channels(icfg)
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

        audio_t = set_audio_channels(audio_t, resolve_vae_audio_channels(self.pretransform.config))

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


def create_pretransform_from_config(
    pretransform_config: dict[str, tp.Any],
    *,
    model: str,
) -> AutoencoderOobleck:
    """Load ``AutoencoderOobleck`` (``from_pretrained(model, subfolder="vae")``).

    Sets ``audiox_scaling_factor`` on the module; callers scale latents when encoding/decoding.
    """
    validate_audiox_pretransform_config_keys(pretransform_config)

    pretransform_type = pretransform_config["type"]

    if pretransform_type != "autoencoder":
        raise NotImplementedError(
            f"AudioX HKUST weights only use pretransform type 'autoencoder'; got {pretransform_type!r}"
        )

    local_files_only = os.path.exists(model)
    vae = AutoencoderOobleck.from_pretrained(
        model,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    icfg = vae.config
    scaling_factor = resolve_pretransform_scale(pretransform_config, icfg)
    vae.audiox_scaling_factor = scaling_factor  # type: ignore[attr-defined]

    vae.eval().requires_grad_(False)
    return vae


def _build_pretransform(conditioner_config: dict[str, tp.Any], *, model: str) -> tp.Any:
    sample_rate = conditioner_config.pop("sample_rate", None)
    assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

    pretransform = create_pretransform_from_config(
        conditioner_config.pop("pretransform_config"),
        model=model,
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
    config: dict[str, tp.Any], *, model: str
) -> MultiConditioner:
    cond_dim = config["cond_dim"]
    configs = config["configs"]
    if len(configs) != 3:
        raise ValueError("AudioX fixed conditioner expects exactly 3 configs: video_prompt/text_prompt/audio_prompt.")

    by_id: dict[str, dict[str, tp.Any]] = {}
    for item in configs:
        if not isinstance(item, dict):
            raise ValueError("Each conditioning config entry must be a dict.")
        cid = item.get("id")
        cconf = item.get("config")
        if not isinstance(cid, str) or not isinstance(cconf, dict):
            raise ValueError("Each conditioning config must include string 'id' and dict 'config'.")
        by_id[cid] = dict(cconf)

    expected_ids = {"video_prompt", "text_prompt", "audio_prompt"}
    if set(by_id) != expected_ids:
        raise ValueError(
            "AudioX fixed conditioner ids must be exactly {'video_prompt', 'text_prompt', 'audio_prompt'}."
        )

    audio_cfg: dict[str, tp.Any] = _with_output_dim(cond_dim, by_id["audio_prompt"])
    video_cfg, text_cfg = prepare_audiox_video_text_conditioner_configs(
        cond_dim=cond_dim,
        video_prompt=by_id["video_prompt"],
        text_prompt=by_id["text_prompt"],
    )

    pretransform = _build_pretransform(audio_cfg, model=model)
    audio_cfg = filter_audio_prompt_config_after_pretransform_build(audio_cfg)

    conditioners: dict[str, Conditioner] = {
        "video_prompt": CLIPVideoTemporalSyncConditioner(**video_cfg),
        "text_prompt": T5TextEncoderAdapter(**text_cfg),
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
