# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderOobleck
from diffusers.schedulers import EDMDPMSolverMultistepScheduler
from torch import einsum, nn
from torchvision import transforms
from transformers import AutoConfig, CLIPVisionModelWithProjection, T5EncoderModel, T5TokenizerFast
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.audiox.audiox_transformer import MMDiffusionTransformer
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

_VIDEO_ONLY_TASKS = frozenset({"v2a", "v2m"})
_TEXT_VIDEO_TASKS = frozenset({"tv2a", "tv2m"})
_VIDEO_CONDITIONED_TASKS = _VIDEO_ONLY_TASKS | _TEXT_VIDEO_TASKS

# Polyexponential sigma schedule defaults; match upstream AudioX sample scripts (``sigma_min=0.3``, ``sigma_max=500``).
_DEFAULT_UPSTREAM_SIGMA_MIN = 0.3
_DEFAULT_UPSTREAM_SIGMA_MAX = 500.0

logger = init_logger(__name__)


def _load_audiox_bundle_config(model_root: str) -> dict[str, Any]:
    """Load the upstream AudioX bundle config from ``<model_root>/config.json``."""
    with open(os.path.join(os.path.abspath(model_root), "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _audio_conditioning_input_samples(model_config: dict[str, Any]) -> int:
    """``latent_seq_len × downsampling_ratio`` from the ``audio_prompt`` conditioning config."""
    configs = model_config["model"]["conditioning"]["configs"]
    cfg = next(c["config"] for c in configs if c["id"] == "audio_prompt")
    return int(cfg["latent_seq_len"]) * int(cfg["pretransform_config"]["config"]["downsampling_ratio"])


def _mm_path_lookup(
    raw_prompt: Any,
    extra: dict[str, Any],
    batch_index: int,
    *,
    mm_key: str,
    paths_key: str,
    single_key: str,
) -> Any:
    if isinstance(raw_prompt, dict):
        mm = raw_prompt.get("multi_modal_data") or {}
        v = mm.get(mm_key)
        if v is not None:
            return v
    paths = extra.get(paths_key)
    if isinstance(paths, (list, tuple)) and batch_index < len(paths):
        return paths[batch_index]
    return extra.get(single_key)


def _normalize_prompt_item(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        p: dict[str, Any] = {"prompt": raw.strip(), "multi_modal_data": {}}
    elif isinstance(raw, dict):
        p = dict(raw)
        p["prompt"] = str(p.get("prompt") or "").strip()
        mm0 = p.get("multi_modal_data")
        p["multi_modal_data"] = {} if mm0 is None else dict(mm0)
    else:
        raise TypeError(f"AudioX prompt {index} must be str or dict, got {type(raw)!r}")

    ai = p.get("additional_information")
    p["additional_information"] = ai if isinstance(ai, dict) else {}
    return p


def _normalize_prompts(prompts: list[Any]) -> list[dict[str, Any]]:
    return [_normalize_prompt_item(raw, i) for i, raw in enumerate(prompts)]


# ---------------------------------------------------------------------------
# Reference media loading: video/image paths or tensors → torch tensors.
# Decoding is delegated to torchvision.io (already a dep) — both video and image.
# AudioX supports the 6 t2*/v2*/tv2* tasks where audio is the model output, so no
# audio-input loaders are needed here.
# ---------------------------------------------------------------------------

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png"})


def _load_image_path_torchvision(path: str) -> torch.Tensor:
    from torchvision.io import read_image

    return read_image(path).unsqueeze(0)


def _load_video_path_torchvision(
    path: str,
    *,
    target_fps: int,
    duration: float,
    seek_time: float,
) -> torch.Tensor:
    """Decode ``[seek_time, seek_time+duration)`` and uniformly subsample to ``target_fps``."""
    from torchvision.io import read_video

    end_pts = float(seek_time) + float(duration) if duration > 0 else None
    video, _, info = read_video(path, start_pts=float(seek_time), end_pts=end_pts, pts_unit="sec", output_format="TCHW")
    if video.shape[0] == 0:
        raise ValueError(f"No frames in range seek_time={seek_time!r}, duration={duration!r} for {path!r}")
    src_fps = float(info.get("video_fps") or target_fps)
    n_target = max(1, int(round(video.shape[0] * float(target_fps) / src_fps)))
    if n_target >= video.shape[0]:
        return video
    indices = torch.linspace(0, video.shape[0] - 1, n_target).round().long()
    return video[indices]


def load_video_source(
    source: Any,
    *,
    target_fps: int,
    duration: float,
    seek_time: float = 0.0,
) -> torch.Tensor:
    """Load video/image/tensor/ndarray into a torch tensor [T, C, H, W] when possible."""
    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
        if ext in _IMAGE_EXTS:
            return _load_image_path_torchvision(source)
        return _load_video_path_torchvision(
            source,
            target_fps=target_fps,
            duration=duration,
            seek_time=seek_time,
        )

    if isinstance(source, torch.Tensor):
        return source
    if isinstance(source, np.ndarray):
        return torch.from_numpy(source)
    raise TypeError(f"Unsupported video source type: {type(source)!r}")


def normalize_video_tensor(frames: torch.Tensor, size: int = 224) -> torch.Tensor:
    if frames.dim() != 4:
        raise ValueError(f"Expected [T, C, H, W], got {tuple(frames.shape)}")

    frames = frames.float()
    if frames.max() > 1.5:
        frames = frames / 255.0

    if frames.shape[-2:] != (size, size):
        frames = F.interpolate(frames, size=(size, size), mode="bicubic", align_corners=False)
    return frames


def adjust_video_duration(frames: torch.Tensor, duration: float, target_fps: int) -> torch.Tensor:
    target_t = int(duration * target_fps)
    cur_t = frames.shape[0]

    if cur_t > target_t:
        return frames[:target_t]
    if cur_t < target_t:
        last = frames[-1:].repeat(target_t - cur_t, 1, 1, 1)
        return torch.cat([frames, last], dim=0)
    return frames


def prepare_video_reference(
    source: Any,
    *,
    duration: float,
    target_fps: int,
    seek_time: float = 0.0,
) -> torch.Tensor:
    frames = load_video_source(
        source,
        target_fps=target_fps,
        duration=duration,
        seek_time=seek_time,
    )

    if frames.dim() == 4 and frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)

    frames = normalize_video_tensor(frames, size=224)
    if duration > 0:
        frames = adjust_video_duration(frames, duration, target_fps)
    return frames


def _load_audio_source(source: Any, *, target_sample_rate: int | None = None) -> torch.Tensor:
    """Path / tensor / ndarray → ``[channels, samples]`` float32, optionally resampled.

    Audio paths use ``torchaudio.load`` (ffmpeg-backed via torchcodec); only loaded when
    a path is actually passed, so tensor / ndarray callers don't need ffmpeg installed.
    """
    if isinstance(source, str):
        import torchaudio

        wav, sr = torchaudio.load(source)
        if target_sample_rate is not None and sr != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        return wav.float()
    if isinstance(source, torch.Tensor):
        return source.float()
    if isinstance(source, np.ndarray):
        return torch.from_numpy(source).float()
    raise TypeError(f"Unsupported audio source type: {type(source)!r}")


def prepare_audio_reference(
    source: Any,
    *,
    model_sample_rate: int,
    seconds_start: float,
    seconds_total: float,
    device: torch.device,
) -> torch.Tensor:
    """Load a reference audio clip into ``[2, target_len]`` at ``model_sample_rate``."""
    target_len = int(model_sample_rate * seconds_total)
    start = int(model_sample_rate * seconds_start)
    wav = _load_audio_source(source, target_sample_rate=model_sample_rate)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    # Force 2 channels: mono → stereo (repeat); >2 → keep first 2.
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    # Crop / left-pad to target_len.
    wav = wav[:, start : start + target_len]
    if wav.shape[1] < target_len:
        wav = F.pad(wav, (target_len - wav.shape[1], 0))
    return wav.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Multi-modal conditioning (T5 / CLIP encoders are wired directly in the pipeline;
# this section provides the audio VAE adapter, the SA_* temporal stack used for
# CLIP video fusion, and the MultiConditioner batch assembly).
# Manual ``SA_Attention`` (einsum + softmax) is intentional: SDPA here drifts vs upstream.
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


# Manual einsum+softmax only. SDPA/diffusion Attention here degrades conditioning vs upstream.
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
        # [B, N, h*d] -> [B, h, N, d]
        q, k, v = (t.unflatten(-1, (h, -1)).transpose(1, 2) for t in qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        # [B, h, N, d] -> [B, N, h*d]
        out = out.transpose(1, 2).flatten(-2)
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
# Audio VAE adapter: encodes waveform via AutoencoderOobleck → cond_dim tokens.
# AudioX bundle ships a dedicated copy of the VAE weights for this conditioner;
# they are loaded under the legacy ``conditioner.conditioners.audio_prompt.*`` prefix
# (remapped in ``AudioXPipeline.load_weights``).
# ---------------------------------------------------------------------------

_AUDIOX_OOBLECK_CONFIG = {
    "audio_channels": 2,
    "channel_multiples": [1, 2, 4, 8, 16],
    "decoder_channels": 128,
    "decoder_input_channels": 64,
    "downsampling_ratios": [2, 4, 4, 8, 8],
    "encoder_hidden_size": 128,
    "sampling_rate": 44100,
}


def _build_audiox_oobleck(scaling_factor: float = 1.0) -> AutoencoderOobleck:
    vae = AutoencoderOobleck(**_AUDIOX_OOBLECK_CONFIG)
    vae.audiox_scaling_factor = float(scaling_factor)  # type: ignore[attr-defined]
    return vae.eval().requires_grad_(False)


class AudioVaePromptAdapter(nn.Module):
    """Encode the (zero-tensor) audio prompt via Oobleck VAE, project to ``cond_dim`` tokens."""

    def __init__(self, *, cond_dim: int, latent_seq_len: int = 215):
        super().__init__()
        self.pretransform = _build_audiox_oobleck()
        in_ch = int(self.pretransform.config.decoder_input_channels)
        self.proj_features_128 = nn.Linear(latent_seq_len, 128)
        self.proj_out = nn.Linear(in_ch, cond_dim) if in_ch != cond_dim else nn.Identity()

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.pretransform.encode(audio, return_dict=True).latent_dist.sample()
        latents = z / float(self.pretransform.audiox_scaling_factor)
        latents = self.proj_features_128(latents).permute(0, 2, 1)
        latents = self.proj_out(latents)
        ones = torch.ones(latents.shape[0], latents.shape[2], device=latents.device)
        return latents, ones


# ---------------------------------------------------------------------------
# MAF (Multimodal Adaptive Fusion) block
# ---------------------------------------------------------------------------


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
        nh, hd = self._num_heads, self._head_dim
        # [B, *, nh*d] -> [B, nh, *, d]
        q = self.to_q(experts).unflatten(-1, (nh, hd)).transpose(1, 2)
        k, v = (t.unflatten(-1, (nh, hd)).transpose(1, 2) for t in self.to_kv(full_context).chunk(2, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self._scale)
        # [B, nh, e, d] -> [B, e, nh*d]
        return self.to_out(out.transpose(1, 2).flatten(-2))


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
        q = q.unflatten(-1, (nh, -1))
        k = k.unflatten(-1, (nh, -1))
        v = v.unflatten(-1, (nh, -1))
        q_bsn = q.transpose(1, 2).contiguous()
        k_bsn = k.transpose(1, 2).contiguous()
        v_bsn = v.transpose(1, 2).contiguous()
        out = self.self_attn(q_bsn, k_bsn, v_bsn, attn_metadata=None)
        out = out.transpose(1, 2).contiguous()
        out = out.flatten(-2)
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


class AudioXPipeline(nn.Module, SupportAudioOutput, DiffusionPipelineProfilerMixin):
    support_audio_output: ClassVar[bool] = True
    _PROFILER_TARGETS: ClassVar[list[str]] = ["diffuse"]
    _CLIP_SYNC_DURATION_SEC: ClassVar[float] = 10.0
    _VIDEO_SYNC_FRAME_COUNT: ClassVar[int] = 240

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        if od_config.model is None:
            raise ValueError(
                "AudioXPipeline requires od_config.model (directory with unified safetensors; "
                "see https://huggingface.co/zhangj1an/AudioX)."
            )

        self._model_root = os.path.abspath(od_config.model)
        self._model_config = _load_audiox_bundle_config(self._model_root)

        # --- Build sub-modules directly (no wrapper) ---
        model_config = self._model_config["model"]
        diffusion_config = model_config["diffusion"]

        self.model = MMDiffusionTransformer(**dict(diffusion_config["config"]))

        # Audio conditioner: VAE-encode the audio prompt (zeros for the 6 t2*/v2*/tv2* tasks).
        cond_configs = {c["id"]: c.get("config", {}) for c in model_config["conditioning"]["configs"]}
        self.audio_vae_adapter = AudioVaePromptAdapter(
            cond_dim=int(model_config["conditioning"]["cond_dim"]),
            latent_seq_len=int(cond_configs["audio_prompt"]["latent_seq_len"]),
        )

        # T5 text encoder — used directly, no adapter wrapper.
        t5_name = cond_configs.get("text_prompt", {}).get("t5_model_name", "t5-base")
        self._t5_max_length = int(cond_configs.get("text_prompt", {}).get("max_length", 128))
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_name)
        t5_config = AutoConfig.from_pretrained(t5_name)
        self.text_encoder = T5EncoderModel(t5_config).train(False).requires_grad_(False).to(torch.float16)

        # CLIP video encoder + temporal fusion — used directly, no adapter wrapper.
        clip_name = cond_configs.get("video_prompt", {}).get("clip_model_name", "openai/clip-vit-base-patch32")
        clip_config = AutoConfig.from_pretrained(clip_name)
        vision_config = getattr(clip_config, "vision_config", clip_config)
        self.clip_encoder = CLIPVisionModelWithProjection(vision_config)
        _CLIP_PATCH_TOKENS, _VIDEO_FPS, _DURATION_SEC, _DIM = 50, 5, 10, 768
        _in_features = _CLIP_PATCH_TOKENS * _VIDEO_FPS * _DURATION_SEC
        self._clip_in_features = _in_features
        self._clip_out_features = 128
        self.clip_proj = nn.Linear(_in_features, self._clip_out_features)
        self.clip_proj_sync = nn.Linear(240, self._clip_out_features)
        self.clip_sync_weight = nn.Parameter(torch.tensor(0.0))
        self.clip_temp_transformer = SA_Transformer(_DIM, depth=4, heads=16, dim_head=64, mlp_dim=_DIM * 4)
        self.clip_temp_pos_embedding = nn.Parameter(torch.randn(1, _VIDEO_FPS * _DURATION_SEC, _DIM))
        self.clip_empty_visual_feat = nn.Parameter(torch.zeros(1, self._clip_out_features, _DIM), requires_grad=False)
        _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
        self._clip_normalize = transforms.Compose([transforms.Normalize(mean=list(_CLIP_MEAN), std=list(_CLIP_STD))])

        # Final-decode VAE: separate Oobleck instance from the conditioner-side one;
        # bundle ships its own weights under the ``pretransform.*`` prefix.
        self.pretransform = _build_audiox_oobleck(
            scaling_factor=float(model_config["pretransform"].get("scale", 1.0)),
        )

        self.io_channels = model_config["io_channels"]
        self.diffusion_objective = "v"

        gate = bool(diffusion_config.get("gate", False))
        gate_type_config = diffusion_config.get("gate_type_config") or {}
        self.maf_block: MAF_Block | None = None
        if gate and diffusion_config.get("gate_type") == "MAF":
            self.maf_block = MAF_Block(
                dim=768,
                num_experts_per_modality=int(gate_type_config.get("num_experts_per_modality", 64)),
                num_heads=int(gate_type_config.get("num_heads", 24)),
                num_fusion_layers=int(gate_type_config.get("num_fusion_layers", 8)),
                mlp_ratio=float(gate_type_config.get("mlp_ratio", 4.0)),
            )

        logger.debug("AudioX model built from %s", self._model_root)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self._model_root,
                subfolder="transformer",
                revision=getattr(od_config, "revision", None),
                prefix="",
            ),
        ]
        sample_rate = int(self._model_config.get("sample_rate", 48000))
        self._sample_rate = sample_rate
        self._sample_size = int(self._model_config.get("sample_size", sample_rate * 10))
        self._target_fps = int(self._model_config.get("video_fps", 5))
        self._audio_conditioning_samples = _audio_conditioning_input_samples(self._model_config)

        self.setup_diffusion_pipeline_profiler(
            profiler_targets=list(self._PROFILER_TARGETS),
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Bundle stores the audio conditioner under the legacy MultiConditioner prefix.
        # Remap to our flat ``audio_vae_adapter.*`` attribute (which holds the same submodules).
        _legacy_prefix = "conditioner.conditioners.audio_prompt."

        def _remap(items):
            for name, tensor in items:
                if name.startswith(_legacy_prefix):
                    name = "audio_vae_adapter." + name[len(_legacy_prefix) :]
                # T5EncoderModel aliases encoder.embed_tokens.weight ↔ shared.weight (same Parameter).
                # AutoWeightsLoader's name lookup sees only shared.weight; rename so the bundle
                # tensor actually lands on the (shared) embedding instead of being silently dropped.
                if name == "text_encoder.encoder.embed_tokens.weight":
                    name = "text_encoder.shared.weight"
                yield name, tensor

        loaded = AutoWeightsLoader(self).load_weights(_remap(weights))

        self.to(torch.float32)
        self.eval().requires_grad_(False)

        return loaded

    def _conditioning_dtype(self) -> torch.dtype:
        p = next(self.model.parameters())
        return p.dtype if p.dtype.is_floating_point else torch.float32

    @staticmethod
    def _normalize_task(task: str | None) -> str | None:
        if task is None:
            return None
        t = str(task).strip().lower()
        return t or None

    @staticmethod
    def _text_for_task(task_norm: str | None, prompt: str) -> str:
        if task_norm in _VIDEO_ONLY_TASKS:
            return ""
        return prompt

    @staticmethod
    def _ensure_text_video_prompts(task_norm: str | None, prompts: list[str]) -> None:
        if task_norm not in _TEXT_VIDEO_TASKS:
            return
        for i, p in enumerate(prompts):
            if not str(p).strip():
                raise ValueError(
                    f"audiox_task={task_norm!r} requires a non-empty text prompt for item {i}; "
                    "use v2a/v2m for video-only generation."
                )

    def _audio_prompt_tensors(
        self,
        *,
        raw_prompts: list[Any],
        extra: dict[str, Any],
        device: torch.device,
        cond_dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        """Per-prompt audio conditioning tensors.

        Looks up an optional reference clip via ``multi_modal_data['audio']`` /
        ``extra_args['audio_path']`` / ``extra_args['audio_paths'][i]``; falls back to
        silence (zeros) — the architecture's ``audio_prompt`` slot always needs a tensor.
        """
        target_len = self._audio_conditioning_samples
        sample_rate = self._sample_rate
        seconds_start = float(extra.get("seconds_start", 0.0))
        seconds_total = float(target_len) / float(sample_rate)
        out: list[torch.Tensor] = []
        for i, raw in enumerate(raw_prompts):
            src = _mm_path_lookup(raw, extra, i, mm_key="audio", paths_key="audio_paths", single_key="audio_path")
            if src is None:
                out.append(torch.zeros(2, target_len, device=device, dtype=cond_dtype))
                continue
            wav = prepare_audio_reference(
                src,
                model_sample_rate=sample_rate,
                seconds_start=seconds_start,
                seconds_total=seconds_total,
                device=device,
            )
            out.append(wav.to(dtype=cond_dtype))
        return out

    def _video_feature_tensors(
        self,
        *,
        task_norm: str | None,
        raw_prompts: list[Any],
        extra: dict[str, Any],
        seconds_start: float,
        target_fps: int,
        device: torch.device,
        cond_dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        clip_frames = int(round(self._CLIP_SYNC_DURATION_SEC * target_fps))
        if task_norm not in _VIDEO_CONDITIONED_TASKS:
            empty = torch.zeros(clip_frames, 3, 224, 224, device=device, dtype=cond_dtype)
            return [empty for _ in raw_prompts]

        tensors: list[torch.Tensor] = []
        for i, _raw in enumerate(raw_prompts):
            src = _mm_path_lookup(_raw, extra, i, mm_key="video", paths_key="video_paths", single_key="video_path")
            if src is None:
                raise ValueError(
                    f"audiox_task={task_norm!r} requires video input: set extra_args['video_path'], "
                    "extra_args['video_paths'], or multi_modal_data['video'] on the prompt dict."
                )
            vt = prepare_video_reference(
                src,
                duration=float(self._CLIP_SYNC_DURATION_SEC),
                target_fps=target_fps,
                seek_time=seconds_start,
            )
            tensors.append(vt.to(device=device, dtype=cond_dtype))
        return tensors

    def get_conditioning_inputs(self, conditioning_tensors: dict[str, Any], negative: bool = False) -> dict[str, Any]:
        """Extract and fuse cross-attention / global conditioning from encoded tensors."""
        cross_attention_input: list[torch.Tensor] = []
        cross_attention_masks: list[torch.Tensor] = []

        for key in ("video_prompt", "text_prompt", "audio_prompt"):
            cross_attn_in, cross_attn_mask = conditioning_tensors[key]
            if len(cross_attn_in.shape) == 2:
                cross_attn_in = cross_attn_in.unsqueeze(1)
                cross_attn_mask = cross_attn_mask.unsqueeze(1)
            cross_attention_input.append(cross_attn_in)
            cross_attention_masks.append(cross_attn_mask)

        video_feature, text_feature, audio_feature = cross_attention_input
        if self.maf_block is not None:
            refined = self.maf_block(text_feature, video_feature, audio_feature)
            fused = torch.cat(list(refined.values()), dim=1)
        else:
            fused = torch.cat([video_feature, text_feature, audio_feature], dim=1)
        masks = torch.cat(cross_attention_masks, dim=1)

        if negative:
            return {
                "negative_cross_attn_cond": fused,
                "negative_cross_attn_mask": masks,
                "negative_global_embed": None,
            }
        return {
            "cross_attn_cond": fused,
            "cross_attn_cond_mask": masks,
            "global_embed": None,
        }

    def diffuse(
        self,
        *,
        steps: int,
        guidance_scale: float,
        conditioning_tensors: dict[str, Any],
        negative_conditioning_tensors: dict[str, Any] | None,
        batch_size: int,
        sigma_min: float,
        sigma_max: float,
        generator: torch.Generator,
        cfg_rescale: float,
    ) -> torch.Tensor:
        """End-to-end audio sampling: noise → DPM++ sampler → VAE decode."""
        device = self.device
        model_dtype = next(self.model.parameters()).dtype

        # Latent noise.
        latent_len = self._sample_size // int(self.pretransform.hop_length)
        noise = torch.randn(
            [batch_size, self.io_channels, latent_len], device=device, generator=generator, dtype=model_dtype
        )

        # Conditioning (cast to model dtype).
        def _cast(d: dict[str, Any]) -> dict[str, Any]:
            return {k: (v.type(model_dtype) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

        cond = _cast(self.get_conditioning_inputs(conditioning_tensors))
        neg = (
            _cast(self.get_conditioning_inputs(negative_conditioning_tensors, negative=True))
            if negative_conditioning_tensors is not None
            else {}
        )

        # k-diffusion VDenoiser + sample_dpmpp_3m_sde, matching upstream AudioX exactly.
        # Upstream diffusers' EDMDPMSolverMultistepScheduler doesn't implement the same
        # v-prediction preconditioning and stochastic update rule, so the old path produced
        # a fixed ~861 Hz resonance independent of conditioning.
        import k_diffusion as K

        outer = self

        class _ModelFn(nn.Module):
            def forward(self, x: torch.Tensor, t_cond: torch.Tensor, **_: Any) -> torch.Tensor:
                return outer.model(
                    x, t_cond,
                    cross_attn_cond=cond["cross_attn_cond"],
                    cross_attn_cond_mask=cond["cross_attn_cond_mask"],
                    global_embed=None,
                    negative_cross_attn_cond=neg.get("negative_cross_attn_cond"),
                    negative_cross_attn_mask=neg.get("negative_cross_attn_mask"),
                    negative_global_embed=None,
                    cfg_scale=guidance_scale,
                    scale_phi=cfg_rescale,
                )

        denoiser = K.external.VDenoiser(_ModelFn())

        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, 1.0, device=device)
        x = noise * sigmas[0]
        if steps <= 1:
            # k-diffusion's sample_dpmpp_3m_sde has an UnboundLocalError at steps=1
            # (the `h` update is inside an `else` branch). Single-step just returns the denoised.
            s_in = x.new_ones([x.shape[0]])
            sampled = denoiser(x, sigmas[0] * s_in)
        else:
            sampled = K.sampling.sample_dpmpp_3m_sde(
                denoiser, x, sigmas, disable=True, extra_args={},
            )

        # VAE decode.
        vae = self.pretransform.to(device=sampled.device, dtype=torch.float32).eval()
        return vae.decode(sampled.to(torch.float32) * float(vae.audiox_scaling_factor), return_dict=True).sample

    def _encode_text(self, texts: list[str], device: torch.device) -> list[torch.Tensor]:
        """Tokenize and encode text with T5 directly."""
        self.text_encoder.to(device)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self._t5_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.text_encoder.eval()
        with torch.no_grad():
            embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        embeddings = embeddings.float() * attention_mask.unsqueeze(-1).float()
        return [embeddings, attention_mask]

    def _encode_video(self, video_list: list[dict], device: torch.device) -> list[torch.Tensor]:
        """Encode video with CLIP + temporal transformer + sync fusion."""
        self.clip_encoder.to(device).eval()

        video_tensors = [item["video_tensors"] for item in video_list]
        video_sync_frames = torch.cat([item["video_sync_frames"] for item in video_list], dim=0).to(device)

        original_videos = torch.cat(video_tensors, dim=0).to(device)
        batch_size, time_length, _, _, _ = original_videos.size()
        is_zero = torch.all(original_videos == 0, dim=(1, 2, 3, 4))

        frames = original_videos.flatten(0, 1)
        pixel_values = self._clip_normalize(frames).to(device)

        with torch.no_grad():
            outputs = self.clip_encoder(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        # [B*T, q, h] -> [B, T, q, h] -> [B, q, T, h] -> [B*q, T, h]
        hidden = hidden.unflatten(0, (batch_size, time_length)).permute(0, 2, 1, 3).flatten(0, 1)
        hidden = hidden + self.clip_temp_pos_embedding
        hidden = self.clip_temp_transformer(hidden)
        # [B*q, T, h] -> [B, q, T, h] -> [B, T, q, h] -> [B, T*q, h]
        hidden = hidden.unflatten(0, (batch_size, -1)).transpose(1, 2).flatten(1, 2)
        hidden = self.clip_proj(hidden.view(-1, self._clip_in_features))
        hidden = hidden.view(batch_size, self._clip_out_features, -1)

        sync = self.clip_proj_sync(video_sync_frames.view(-1, 240))
        sync = sync.view(batch_size, self._clip_out_features, -1)
        hidden = hidden + self.clip_sync_weight * sync

        empty = self.clip_empty_visual_feat.expand(batch_size, -1, -1)
        hidden = torch.where(is_zero.view(batch_size, 1, 1), empty, hidden)
        return [hidden, torch.ones(batch_size, 1, device=device)]

    def _encode_conditioning_tensors(self, batch_metadata: list[dict[str, Any]]) -> dict[str, Any]:
        device = self.device
        audio = torch.cat([item["audio_prompt"] for item in batch_metadata], dim=0).to(device)
        return {
            "audio_prompt": list(self.audio_vae_adapter(audio)),
            "text_prompt": self._encode_text([item["text_prompt"] for item in batch_metadata], device),
            "video_prompt": self._encode_video([item["video_prompt"] for item in batch_metadata], device),
        }

    def _build_conditioning_batch(
        self,
        *,
        texts: list[str],
        video_tensors_list: list[torch.Tensor],
        audio_prompt_list: list[torch.Tensor],
        sync_features: torch.Tensor,
        seconds_start: float,
        seconds_model: float,
        num_outputs_per_prompt: int,
        task_norm: str | None,
    ) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for i, text in enumerate(texts):
            for _ in range(num_outputs_per_prompt):
                batch.append(
                    {
                        "video_prompt": {
                            "video_tensors": video_tensors_list[i].unsqueeze(0),
                            "video_sync_frames": sync_features,
                        },
                        "text_prompt": self._text_for_task(task_norm, text),
                        "audio_prompt": audio_prompt_list[i].unsqueeze(0),
                        "seconds_start": seconds_start,
                        "seconds_total": seconds_model,
                    }
                )
        return batch

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if req.prompts is None or len(req.prompts) == 0:
            raise ValueError("AudioXPipeline requires at least one prompt.")
        normalized_prompts = _normalize_prompts(list(req.prompts))
        prompts = [p["prompt"] for p in normalized_prompts]

        sampling_params = req.sampling_params
        if sampling_params.num_inference_steps is None:
            raise ValueError("AudioXPipeline requires sampling_params.num_inference_steps.")
        num_inference_steps = int(sampling_params.num_inference_steps)
        extra_args = sampling_params.extra_args or {}
        task_norm = self._normalize_task(extra_args.get("audiox_task"))
        self._ensure_text_video_prompts(task_norm, prompts)

        neg: list[str] | None = None
        if not all(p.get("negative_prompt") is None for p in normalized_prompts):
            neg = [str(p.get("negative_prompt") or "") for p in normalized_prompts]

        guidance_scale = float(sampling_params.guidance_scale)
        if sampling_params.num_outputs_per_prompt <= 0:
            raise ValueError("AudioXPipeline requires sampling_params.num_outputs_per_prompt > 0.")
        num_outputs_per_prompt = int(sampling_params.num_outputs_per_prompt)
        batch_size = len(prompts) * num_outputs_per_prompt

        seconds_start = float(extra_args.get("seconds_start", 0.0))
        seconds_model = self._sample_size / self._sample_rate
        sigma_min = float(extra_args.get("sigma_min", _DEFAULT_UPSTREAM_SIGMA_MIN))
        sigma_max = float(extra_args.get("sigma_max", _DEFAULT_UPSTREAM_SIGMA_MAX))
        cfg_rescale = float(extra_args.get("cfg_rescale", 0.0))
        device = self.device
        generator = sampling_params.generator
        if generator is None:
            raise ValueError("AudioXPipeline requires sampling_params.generator.")
        target_fps = self._target_fps
        sample_rate = self._sample_rate
        cond_dtype = self._conditioning_dtype()

        sync_features = torch.zeros(1, self._VIDEO_SYNC_FRAME_COUNT, 768, device=device, dtype=cond_dtype)

        audio_prompt_list = self._audio_prompt_tensors(
            raw_prompts=normalized_prompts,
            extra=extra_args,
            device=device,
            cond_dtype=cond_dtype,
        )

        video_tensors_list = self._video_feature_tensors(
            task_norm=task_norm,
            raw_prompts=normalized_prompts,
            extra=extra_args,
            seconds_start=seconds_start,
            target_fps=target_fps,
            device=device,
            cond_dtype=cond_dtype,
        )

        conditioning_batch = self._build_conditioning_batch(
            texts=prompts,
            video_tensors_list=video_tensors_list,
            audio_prompt_list=audio_prompt_list,
            sync_features=sync_features,
            seconds_start=seconds_start,
            seconds_model=seconds_model,
            num_outputs_per_prompt=num_outputs_per_prompt,
            task_norm=task_norm,
        )

        negative_conditioning_batch: list[dict[str, Any]] | None = None
        if neg is not None and guidance_scale > 1.0:
            negative_conditioning_batch = self._build_conditioning_batch(
                texts=neg,
                video_tensors_list=video_tensors_list,
                audio_prompt_list=audio_prompt_list,
                sync_features=sync_features,
                seconds_start=seconds_start,
                seconds_model=seconds_model,
                num_outputs_per_prompt=num_outputs_per_prompt,
                task_norm=task_norm,
            )

        conditioning_tensors = self._encode_conditioning_tensors(conditioning_batch)
        negative_conditioning_tensors: dict[str, Any] | None = None
        if negative_conditioning_batch is not None:
            negative_conditioning_tensors = self._encode_conditioning_tensors(negative_conditioning_batch)

        audio = self.diffuse(
            steps=num_inference_steps,
            guidance_scale=guidance_scale,
            conditioning_tensors=conditioning_tensors,
            negative_conditioning_tensors=negative_conditioning_tensors,
            batch_size=batch_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            generator=generator,
            cfg_rescale=cfg_rescale,
        )

        return DiffusionOutput(
            output=audio,
            custom_output={"audiox_task": task_norm},
            stage_durations=self.stage_durations
            if getattr(self, "enable_diffusion_pipeline_profiler", False)
            else None,
        )
