from __future__ import annotations

import logging
import typing as tp
import warnings
from typing import Any

import einops
import torch
from einops import rearrange
from torch import einsum, nn
from torchvision import transforms
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
)

from vllm_omni.diffusion.models.audiox.audiox_pretransform import create_pretransform_from_config


def set_audio_channels(audio: torch.Tensor, target_channels: int) -> torch.Tensor:
    if target_channels == 1:
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


# --- Video temporal stack (aligned with pip ``audiox.models.temptransformer``) ---
# Do not replace ``SA_Attention`` with ``vllm_omni.diffusion.attention.layer.Attention`` (SDPA): on the
# same weights it numerically drifts vs upstream AudioX, which degrades conditioning and noticeably
# hurts output quality (e.g. multimodal tasks). See ``debug_audiox_tv2m_layer_compare``.


class SA_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SA_FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Manual einsum+softmax only. SDPA/diffusion Attention here degrades conditioning vs upstream and output quality.
class SA_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SA_PreNorm(dim, SA_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        SA_PreNorm(dim, SA_FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class Conditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int, project_out: bool = False):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class CLIPWithSyncWithEmptyFeatureConditioner(Conditioner):
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    VIDEO_FPS = 5
    DURATION_SECONDS = 10
    TEMPORAL_DIM = 768
    OUT_FEATURES = 128
    CLIP_PATCH_TOKENS = 50

    def __init__(
        self,
        output_dim: int,
        project_out: bool = False,
    ):
        super().__init__(dim=768, output_dim=output_dim, project_out=project_out)

        sa_depth = 4
        num_heads = 16
        dim_head = 64
        hidden_scale = 4
        in_features = self.CLIP_PATCH_TOKENS * self.VIDEO_FPS * self.DURATION_SECONDS

        self.empty_visual_feat = nn.Parameter(torch.zeros(1, self.OUT_FEATURES, self.TEMPORAL_DIM), requires_grad=True)
        nn.init.constant_(self.empty_visual_feat, 0)
        self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(self.CLIP_MODEL_NAME)
        self.proj = nn.Linear(in_features=in_features, out_features=self.OUT_FEATURES)
        self.proj_sync = nn.Linear(in_features=240, out_features=self.OUT_FEATURES)
        self.sync_weight = nn.Parameter(torch.tensor(0.0))
        self.in_features = in_features
        self.out_features = self.OUT_FEATURES
        self.Temp_transformer = SA_Transformer(
            self.TEMPORAL_DIM, sa_depth, num_heads, dim_head, self.TEMPORAL_DIM * hidden_scale, 0.0
        )
        self.Temp_pos_embedding = nn.Parameter(
            torch.randn(1, self.DURATION_SECONDS * self.VIDEO_FPS, self.TEMPORAL_DIM)
        )

        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        self.preprocess_CLIP = transforms.Compose([transforms.Normalize(mean=clip_mean, std=clip_std)])

    def process_video_with_custom_preprocessing(self, video_tensor: torch.Tensor) -> torch.Tensor:
        video_tensor = video_tensor / 255.0
        return self.preprocess_CLIP(video_tensor)

    def forward(self, video_list: list[torch.Tensor | dict], device: torch.device | str) -> list[torch.Tensor]:
        if isinstance(video_list[0], dict):
            video_tensors = [item["video_tensors"] for item in video_list]
            video_sync_frames = [item["video_sync_frames"] for item in video_list]
            video_sync_frames = torch.cat(video_sync_frames, dim=0).to(device)
        else:
            video_tensors = video_list
            batch_size = len(video_tensors)
            video_sync_frames = torch.zeros(batch_size, 240, 768).to(device)

        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)

        original_videos = torch.cat(video_tensors, dim=0).to(device)
        batch_size, time_length, _, _, _ = original_videos.size()
        is_zero = torch.all(original_videos == 0, dim=(1, 2, 3, 4))

        video_tensors = einops.rearrange(original_videos, "b t c h w -> (b t) c h w")
        video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensors).to(device)

        with torch.no_grad():
            outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
        video_hidden = outputs.last_hidden_state
        video_hidden = einops.rearrange(video_hidden, "(b t) q h -> (b q) t h", b=batch_size, t=time_length)
        video_hidden = video_hidden + self.Temp_pos_embedding
        video_hidden = self.Temp_transformer(video_hidden)
        video_hidden = einops.rearrange(video_hidden, "(b q) t h -> b (t q) h", b=batch_size, t=time_length)
        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        video_sync_frames = self.proj_sync(video_sync_frames.view(-1, 240))
        video_sync_frames = video_sync_frames.view(batch_size, self.out_features, -1)
        video_hidden = video_hidden + self.sync_weight * video_sync_frames

        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden)
        return [video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)]


class T5Conditioner(Conditioner):
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

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                self.model = (
                    T5EncoderModel.from_pretrained(t5_model_name).train(False).requires_grad_(False).to(torch.float16)
                )
            finally:
                logging.disable(previous_level)

    def forward(self, texts: list[str], device: torch.device | str) -> list[torch.Tensor]:
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
        dev = torch.device(device)

        self.model.eval()
        if dev.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16), torch.set_grad_enabled(False):
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
        else:
            with torch.set_grad_enabled(False):
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return [embeddings, attention_mask]


class AudioAutoencoderConditionerv2(Conditioner):
    def __init__(
        self,
        pretransform: tp.Any,
        output_dim: int,
        latent_seq_len: int = 237,
        mask_ratio_start: float = 0.0,
        mask_ratio_end: float = 0.0,
    ):
        super().__init__(pretransform.encoded_channels, output_dim)
        self.pretransform = pretransform
        self.latent_seq_len = latent_seq_len
        self.mask_ratio_start = float(mask_ratio_start)
        self.mask_ratio_end = float(mask_ratio_end)
        self.proj_features_128 = nn.Linear(in_features=self.latent_seq_len, out_features=128)

    def mask_audio(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _channels, seq_len = audio.shape
        device = audio.device
        mask_ratios = (
            torch.rand(batch_size, device=device) * (self.mask_ratio_end - self.mask_ratio_start)
            + self.mask_ratio_start
        )
        masked_audio = audio.clone()
        for i in range(batch_size):
            mask_len = int(seq_len * mask_ratios[i])
            start_pos = torch.randint(0, seq_len - mask_len + 1, (1,), device=device)
            masked_audio[i, :, start_pos : start_pos + mask_len] = 0
        return masked_audio, mask_ratios

    def forward(
        self, audio: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor], device: torch.device | str
    ) -> list[torch.Tensor]:
        self.pretransform.to(device)
        self.proj_out.to(device)

        bs = len(audio)
        max_len = max([a.shape[-1] for a in audio])
        for i in range(bs):
            audio[i] = audio[i].to(device)
            pad_len = max_len - audio[i].shape[-1]
            if pad_len > 0:
                audio[i] = torch.nn.functional.pad(audio[i], (0, pad_len))

        audio = torch.cat(audio, dim=0)
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        if self.mask_ratio_start < self.mask_ratio_end:
            audio, _mask_ratios = self.mask_audio(audio)

        latents = self.pretransform.encode_scaled(audio)
        latents = self.proj_features_128(latents)
        latents = latents.permute(0, 2, 1)
        latents = self.proj_out(latents)
        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]


class MultiConditioner(nn.Module):
    def __init__(self, conditioners: dict[str, Conditioner]):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)

    def forward(self, batch_metadata: list[dict[str, tp.Any]], device: torch.device | str) -> dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            conditioner_inputs = _gather_conditioner_inputs(
                batch_metadata=batch_metadata,
                key=key,
                require_single_item_sequence=False,
            )
            output[key] = conditioner(conditioner_inputs, device)
        return output


def _build_pretransform(conditioner_config: dict[str, tp.Any]) -> tp.Any:
    sample_rate = conditioner_config.pop("sample_rate", None)
    assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

    pretransform = create_pretransform_from_config(
        conditioner_config.pop("pretransform_config"), sample_rate=sample_rate
    )
    return pretransform


def _get_source_key(item: dict[str, tp.Any], key: str) -> str:
    source_key = key
    if source_key not in item:
        raise ValueError(f"Conditioner key {key} not found in batch metadata")
    return source_key


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
        source_key = _get_source_key(item, key)
        value = _normalize_condition_value(
            item[source_key],
            source_key=source_key,
            require_single_item_sequence=require_single_item_sequence,
        )
        inputs.append(value)
    return inputs


def _with_output_dim(cond_dim: int, cfg: dict[str, tp.Any]) -> dict[str, tp.Any]:
    out = {"output_dim": cond_dim}
    out.update(cfg)
    return out


def create_audiox_fixed_conditioner_from_conditioning_config(config: dict[str, tp.Any]) -> MultiConditioner:
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

    video_cfg = _with_output_dim(cond_dim, by_id["video_prompt"])
    text_cfg = _with_output_dim(cond_dim, by_id["text_prompt"])
    audio_cfg = _with_output_dim(cond_dim, by_id["audio_prompt"])
    video_allowed = {"output_dim", "project_out", "clip_model_name"}
    video_extra = set(video_cfg) - video_allowed
    if video_extra:
        raise ValueError(f"Unsupported video_prompt config keys for AudioX inference: {sorted(video_extra)}")
    video_cfg = {k: video_cfg[k] for k in video_allowed if k in video_cfg and k != "clip_model_name"}

    text_allowed = {"output_dim", "t5_model_name", "max_length", "project_out"}
    text_extra = set(text_cfg) - text_allowed
    if text_extra:
        raise ValueError(f"Unsupported text_prompt config keys for AudioX inference: {sorted(text_extra)}")
    text_cfg = {k: text_cfg[k] for k in text_allowed if k in text_cfg}

    pretransform = _build_pretransform(audio_cfg)
    audio_allowed = {
        "output_dim",
        "latent_seq_len",
        "pretransform_ckpt_path",
        "mask_ratio_start",
        "mask_ratio_end",
    }
    audio_extra = set(audio_cfg) - audio_allowed
    if audio_extra:
        raise ValueError(f"Unsupported audio_prompt config keys for AudioX inference: {sorted(audio_extra)}")
    audio_cfg = {k: audio_cfg[k] for k in audio_allowed if k in audio_cfg and k != "pretransform_ckpt_path"}

    conditioners: dict[str, Conditioner] = {
        "video_prompt": CLIPWithSyncWithEmptyFeatureConditioner(**video_cfg),
        "text_prompt": T5Conditioner(**text_cfg),
        "audio_prompt": AudioAutoencoderConditionerv2(pretransform, **audio_cfg),
    }
    return MultiConditioner(conditioners)


def encode_audiox_conditioning_tensors(
    multi_conditioner: MultiConditioner,
    *,
    batch_metadata: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, tp.Any]:
    output: dict[str, Any] = {}

    for key, conditioner in multi_conditioner.conditioners.items():
        conditioner_inputs = _gather_conditioner_inputs(
            batch_metadata=batch_metadata,
            key=key,
            require_single_item_sequence=True,
        )
        output[key] = conditioner(conditioner_inputs, device)

    return output
