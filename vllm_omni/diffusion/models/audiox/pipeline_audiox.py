# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect
import json
import math
import os
import typing as tp
from collections.abc import Iterable
from typing import Any, ClassVar

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderOobleck
from diffusers.schedulers import EDMDPMSolverMultistepScheduler
from einops import rearrange
from scipy.signal import resample_poly
from torch import einsum, nn
from torchvision import transforms
from transformers import AutoConfig, CLIPVisionModelWithProjection, T5EncoderModel, T5TokenizerFast
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.audiox.audiox_transformer import MMDiffusionTransformer
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

_VIDEO_ONLY_TASKS = frozenset({"v2a", "v2m"})
_TEXT_VIDEO_TASKS = frozenset({"tv2a", "tv2m"})
_VIDEO_CONDITIONED_TASKS = _VIDEO_ONLY_TASKS | _TEXT_VIDEO_TASKS

# Polyexponential sigma schedule defaults; match upstream AudioX sample scripts (``sigma_min=0.3``, ``sigma_max=500``).
_DEFAULT_UPSTREAM_SIGMA_MIN = 0.3
_DEFAULT_UPSTREAM_SIGMA_MAX = 500.0

logger = init_logger(__name__)


def _default_audiox_device() -> torch.device:
    """Single-process device; placement for multi-GPU runs is handled outside this module."""
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _load_audiox_bundle_config(model_root: str) -> dict[str, Any]:
    """Load the upstream AudioX bundle config from ``<model_root>/config.json``."""
    with open(os.path.join(os.path.abspath(model_root), "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _audio_conditioning_input_samples(model_config: dict[str, Any]) -> int | None:
    """``latent_seq_len × downsampling_ratio`` from the nested ``audio_prompt`` conditioning config."""
    m = model_config.get("model")
    if not isinstance(m, dict):
        return None
    cond = m.get("conditioning")
    if not isinstance(cond, dict):
        return None
    for item in cond.get("configs", []):
        if not isinstance(item, dict) or item.get("id") != "audio_prompt":
            continue
        c = item.get("config")
        if not isinstance(c, dict):
            continue
        ls = c.get("latent_seq_len")
        pt = c.get("pretransform_config")
        ds = None
        if isinstance(pt, dict):
            ptc = pt.get("config")
            if isinstance(ptc, dict):
                ds = ptc.get("downsampling_ratio")
        if isinstance(ls, (int, float)) and isinstance(ds, (int, float)):
            return int(ls) * int(ds)
    return None


def resample_audiox_waveform_poly(audio_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio_data

    return resample_poly(audio_data.astype(np.float32), up=int(dst_rate), down=int(src_rate), axis=0)


def get_audiox_post_process_func(_od_config: OmniDiffusionConfig):
    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type in ("latent", "pt"):
            return audio
        return audio.detach().cpu().float().numpy()

    return post_process_func


def _resolve_audio_source(
    raw_prompt: Any,
    extra: dict[str, Any],
    batch_index: int,
) -> Any:
    return _mm_path_lookup(
        raw_prompt, extra, batch_index, mm_key="audio", paths_key="audio_paths", single_key="audio_path"
    )


def get_audiox_pre_process_func(od_config: OmniDiffusionConfig):
    if od_config.model is None:
        raise ValueError("AudioX pre-process requires od_config.model.")

    model_root = os.path.abspath(od_config.model)
    model_cfg = _load_audiox_bundle_config(model_root)

    sample_rate = int(model_cfg.get("sample_rate", 48000))
    sample_size = int(model_cfg.get("sample_size", sample_rate * 10))
    video_fps = int(model_cfg.get("video_fps", 5))
    ac_samples = _audio_conditioning_input_samples(model_cfg)
    audio_conditioning_samples = ac_samples if ac_samples is not None else sample_size
    seconds_model = float(sample_size) / float(sample_rate)
    clip_duration = 10.0

    cpu = torch.device("cpu")

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        sp = request.sampling_params
        extra = sp.extra_args or {}
        seconds_start = float(extra.get("seconds_start", 0.0))
        user_seconds_total = float(extra.get("seconds_total", seconds_model))
        cond_seconds = float(audio_conditioning_samples) / float(sample_rate)

        task_norm = AudioXPipeline._normalize_task(extra.get("audiox_task"))

        normalized = _normalize_prompts(list(request.prompts))
        new_prompts: list[Any] = []
        for i, p in enumerate(normalized):
            mm = p["multi_modal_data"]
            ai = p["additional_information"]

            if task_norm in _VIDEO_CONDITIONED_TASKS:
                vsrc = _mm_path_lookup(p, extra, i, mm_key="video", paths_key="video_paths", single_key="video_path")
                if vsrc is not None:
                    mm["video"] = prepare_video_reference(
                        vsrc,
                        duration=float(clip_duration),
                        target_fps=video_fps,
                        seek_time=seconds_start,
                    ).to(device=cpu, dtype=torch.float32)

            asrc = _resolve_audio_source(p, extra, i)
            if asrc is not None:
                mm["audio"] = prepare_audio_reference(
                    asrc,
                    model_sample_rate=sample_rate,
                    seconds_start=seconds_start,
                    seconds_total=cond_seconds,
                    device=cpu,
                )

            ai["audiox_preprocess"] = {
                "seconds_model": seconds_model,
                "user_seconds_total": user_seconds_total,
                "sample_rate": sample_rate,
                "video_fps": video_fps,
            }
            new_prompts.append(p)

        request.prompts = new_prompts
        return request

    return pre_process_func


def _conditioning_item(
    *,
    text: str,
    video_tensor: torch.Tensor,
    audio_tensor: torch.Tensor,
    sync_features: torch.Tensor,
    seconds_start: float,
    seconds_model: float,
) -> dict[str, Any]:
    return {
        "video_prompt": {
            "video_tensors": video_tensor.unsqueeze(0),
            "video_sync_frames": sync_features,
        },
        "text_prompt": text,
        "audio_prompt": audio_tensor.unsqueeze(0),
        "seconds_start": seconds_start,
        "seconds_total": seconds_model,
    }


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
# Reference media loading: audio/video paths or tensors → torch tensors
# Audio file paths use ``torchaudio.load_with_torchcodec``. Video file paths use
# ``torchcodec.decoders.VideoDecoder``. Static images use ``torchvision.io.read_image``.
# ---------------------------------------------------------------------------

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png"})


def _load_audio_path_str(path: str) -> tuple[torch.Tensor, int]:
    """Load [C, T] float32 audio from a file path via TorchCodec."""
    from torchaudio import load_with_torchcodec

    return load_with_torchcodec(path)


def load_audio_source(source: Any, *, target_sample_rate: int | None = None) -> torch.Tensor:
    """Load audio from path/tensor/ndarray into a torch tensor [C, T]."""
    if isinstance(source, str):
        import torchaudio

        wav, sr = _load_audio_path_str(source)
        if target_sample_rate is not None and sr != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        return wav
    if isinstance(source, torch.Tensor):
        return source
    if isinstance(source, np.ndarray):
        return torch.from_numpy(source)
    raise TypeError(f"Unsupported audio source type: {type(source)!r}")


def _load_image_path_torchvision(path: str) -> torch.Tensor:
    from torchvision.io import read_image

    return read_image(path).unsqueeze(0)


def _load_video_path_torchcodec(
    path: str,
    *,
    target_fps: int,
    duration: float,
    seek_time: float,
) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(
        path,
        dimension_order="NCHW",
        device="cpu",
        seek_mode="exact",
    )
    meta_begin = float(decoder.metadata.begin_stream_seconds or 0.0)
    meta_end = float(decoder.metadata.end_stream_seconds or 0.0)
    start_s = max(float(seek_time), meta_begin)
    if duration > 0:
        stop_s = min(float(seek_time) + float(duration), meta_end)
    else:
        stop_s = meta_end
    if start_s >= stop_s:
        raise ValueError(
            f"No frames in range seek_time={seek_time!r}, duration={duration!r} "
            f"(stream [{meta_begin}, {meta_end})) for {path!r}"
        )
    batch = decoder.get_frames_played_in_range(start_s, stop_s, fps=float(target_fps))
    return batch.data


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
        return _load_video_path_torchcodec(
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


def to_2ch_audio(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    elif x.shape[0] > 2:
        x = x[:2]
    return x


def crop_or_pad_1d(x: torch.Tensor, start: int, target_len: int) -> torch.Tensor:
    x = x[:, start : start + target_len]
    cur = x.shape[1]
    if cur < target_len:
        x = F.pad(x, (target_len - cur, 0))
    else:
        x = x[:, :target_len]
    return x


def prepare_audio_reference(
    source: Any,
    *,
    model_sample_rate: int,
    seconds_start: float,
    seconds_total: float,
    device: torch.device,
) -> torch.Tensor:
    target_len = int(model_sample_rate * seconds_total)
    start = int(model_sample_rate * seconds_start)
    wav = load_audio_source(source, target_sample_rate=model_sample_rate)

    wav = to_2ch_audio(wav)
    wav = crop_or_pad_1d(wav, start, target_len)
    return wav.to(device=device, dtype=torch.float32)


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


# ---------------------------------------------------------------------------
# Multi-modal conditioning (T5 / CLIP encoders are wired directly in the pipeline;
# this section provides the audio VAE adapter, the SA_* temporal stack used for
# CLIP video fusion, and the MultiConditioner batch assembly).
# Manual ``SA_Attention`` (einsum + softmax) is intentional: SDPA here drifts vs upstream.
# ---------------------------------------------------------------------------


def _kwargs_for(cls: type, cfg: dict[str, Any]) -> dict[str, Any]:
    """Filter ``cfg`` to just the keyword arguments accepted by ``cls.__init__``.

    Upstream AudioX configs ship training-only knobs alongside inference-relevant keys.
    """
    sig = inspect.signature(cls.__init__)
    accepted = {
        name for name, param in sig.parameters.items() if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    return {k: v for k, v in cfg.items() if k in accepted}


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


class Conditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int, project_out: bool = False):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


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


# ---------------------------------------------------------------------------
# Diffusion sampling runtime: polyexponential sigma schedule + EDM-DPM++ scheduler
# ---------------------------------------------------------------------------

_PROGRESS = ProgressBarMixin()


def _append_zero(sigmas: torch.Tensor) -> torch.Tensor:
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _sigmas_polyexponential(
    n: int, sigma_min: float, sigma_max: float, rho: float, device: torch.device
) -> torch.Tensor:
    """Polynomial-in-log-sigma noise schedule (upstream AudioX / ``get_sigmas_polyexponential``)."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return _append_zero(sigmas)


def _sigma_to_t_vdiffusion(sigma: torch.Tensor) -> torch.Tensor:
    """Timestep embedding input ``atan(sigma) * 2 / pi`` (AudioX DiT conditioning)."""
    return sigma.atan() / math.pi * 2


def sample_k(
    model_fn,
    noise,
    steps: int = 100,
    sigma_min: float = 0.3,
    sigma_max: float = 500.0,
    rho: float = 1.0,
    device: str | torch.device = "cuda",
    callback=None,
    generator: torch.Generator | None = None,
    **extra_args,
):
    """Sample with :class:`~diffusers.schedulers.EDMDPMSolverMultistepScheduler` (``sde-dpmsolver++``, ``sigma_data=1``)."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    sigmas_full = _sigmas_polyexponential(steps, sigma_min, sigma_max, rho, dev)

    scheduler = EDMDPMSolverMultistepScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=1.0,
        sigma_schedule="exponential",
        prediction_type="v_prediction",
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
        solver_type="midpoint",
        final_sigmas_type="zero",
    )
    scheduler.set_timesteps(steps, device=device)
    scheduler.sigmas = sigmas_full.detach().cpu().to(torch.float32)
    scheduler.timesteps = scheduler.precondition_noise(sigmas_full[:-1]).detach().cpu().to(torch.float32)
    scheduler.num_inference_steps = steps

    latents = noise * sigmas_full[0].to(device=noise.device, dtype=noise.dtype)

    _PROGRESS.set_progress_bar_config(disable=False)
    with _PROGRESS.progress_bar(total=len(scheduler.timesteps)) as pbar:
        for t in scheduler.timesteps:
            latent_in = scheduler.scale_model_input(latents, t)
            sigma = scheduler.sigmas[scheduler.step_index].to(device=latent_in.device, dtype=latent_in.dtype)
            s_in = latent_in.new_ones([latent_in.shape[0]])
            t_cond = _sigma_to_t_vdiffusion(sigma * s_in)
            model_output = model_fn(latent_in, t_cond, **extra_args)
            if callback is not None:
                callback(
                    {
                        "x": latents,
                        "i": int(scheduler.step_index),
                        "sigma": sigma,
                        "sigma_hat": sigma,
                        "denoised": None,
                    }
                )
            latents = scheduler.step(model_output, t, latents, generator=generator).prev_sample
            pbar.update()

    return latents


def generate_diffusion_cond(
    pipeline,
    steps: int = 250,
    cfg_scale=6,
    conditioning_tensors: dict | None = None,
    negative_conditioning_tensors: dict | None = None,
    batch_size: int = 1,
    sample_size: int = 2097152,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
    **sample_kwargs,
) -> torch.Tensor:
    """Run diffusion sampling using the AudioXPipeline.

    ``pipeline`` must expose ``.pretransform``, ``.io_channels``, ``.model``,
    ``.diffusion_objective``, and ``.get_conditioning_inputs()``.
    """
    pt = pipeline.pretransform
    if pt is None:
        raise RuntimeError("AudioX inference-only path requires a pretransform.")
    if not isinstance(pt, AutoencoderOobleck):
        raise RuntimeError(f"Expected AutoencoderOobleck pretransform, got {type(pt)!r}.")

    sample_size = sample_size // int(pt.hop_length)

    if generator is None:
        raise ValueError("AudioX generation requires a torch.Generator.")
    noise = torch.randn([batch_size, pipeline.io_channels, sample_size], device=device, generator=generator)

    conditioning_inputs = pipeline.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors is not None:
        negative_conditioning_tensors = pipeline.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    model_dtype = next(pipeline.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    sampled = sample_k(
        pipeline.model,
        noise,
        steps,
        **sample_kwargs,
        **conditioning_inputs,
        **negative_conditioning_tensors,
        cfg_scale=cfg_scale,
        device=device,
        generator=generator,
    )

    dev = sampled.device
    vae = pt.to(device=dev, dtype=torch.float32).eval()
    pipeline.pretransform = vae
    z = sampled.to(dtype=torch.float32)
    sampled = vae.decode(z * float(vae.audiox_scaling_factor), return_dict=True).sample

    return sampled


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
        self.device = _default_audiox_device()
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
        self.conditioner = create_audiox_fixed_conditioner_from_conditioning_config(
            model_config["conditioning"],
        )

        # T5 text encoder — used directly, no adapter wrapper.
        cond_configs = {c["id"]: c.get("config", {}) for c in model_config["conditioning"]["configs"]}
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

        pretransform_cfg = model_config["pretransform"]
        self.pretransform = create_pretransform_from_config(pretransform_cfg)

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
        sample_size = int(self._model_config.get("sample_size", sample_rate * 10))
        ac_samples = _audio_conditioning_input_samples(self._model_config)
        audio_conditioning_samples = ac_samples if ac_samples is not None else sample_size
        self._sample_rate = sample_rate
        self._sample_size = sample_size
        self._target_fps = int(self._model_config.get("video_fps", 5))
        self._audio_conditioning_samples = audio_conditioning_samples

        self.setup_diffusion_pipeline_profiler(
            profiler_targets=list(self._PROFILER_TARGETS),
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights)

        # T5EncoderModel ties shared.weight ↔ encoder.embed_tokens.weight.
        # The unified safetensors omits shared.weight; reconstruct the tie.
        if hasattr(self.text_encoder, "shared"):
            self.text_encoder.shared.weight = self.text_encoder.encoder.embed_tokens.weight
            loaded.add("text_encoder.shared.weight")

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
        seconds_start: float,
        sample_rate: int,
        device: torch.device,
        cond_dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        target_len = self._audio_conditioning_samples
        cond_seconds = float(target_len) / float(sample_rate)
        for i, _raw in enumerate(raw_prompts):
            src = _resolve_audio_source(_raw, extra, i)
            if src is None:
                out.append(torch.zeros(2, target_len, device=device, dtype=cond_dtype))
                continue
            out.append(
                prepare_audio_reference(
                    src,
                    model_sample_rate=sample_rate,
                    seconds_start=seconds_start,
                    seconds_total=cond_seconds,
                    device=device,
                ).to(dtype=cond_dtype)
            )
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
        return generate_diffusion_cond(
            self,
            steps=steps,
            cfg_scale=guidance_scale,
            conditioning_tensors=conditioning_tensors,
            negative_conditioning_tensors=negative_conditioning_tensors,
            batch_size=batch_size,
            sample_size=self._sample_size,
            device=self.device,
            generator=generator,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            scale_phi=cfg_rescale,
        )

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

        frames = einops.rearrange(original_videos, "b t c h w -> (b t) c h w")
        pixel_values = self._clip_normalize(frames).to(device)

        with torch.no_grad():
            outputs = self.clip_encoder(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        hidden = einops.rearrange(hidden, "(b t) q h -> (b q) t h", b=batch_size, t=time_length)
        hidden = hidden + self.clip_temp_pos_embedding
        hidden = self.clip_temp_transformer(hidden)
        hidden = einops.rearrange(hidden, "(b q) t h -> b (t q) h", b=batch_size, t=time_length)
        hidden = self.clip_proj(hidden.view(-1, self._clip_in_features))
        hidden = hidden.view(batch_size, self._clip_out_features, -1)

        sync = self.clip_proj_sync(video_sync_frames.view(-1, 240))
        sync = sync.view(batch_size, self._clip_out_features, -1)
        hidden = hidden + self.clip_sync_weight * sync

        empty = self.clip_empty_visual_feat.expand(batch_size, -1, -1)
        hidden = torch.where(is_zero.view(batch_size, 1, 1), empty, hidden)
        return [hidden, torch.ones(batch_size, 1, device=device)]

    def _encode_conditioning_tensors(self, batch_metadata: list[dict[str, Any]]) -> dict[str, Any]:
        # Encode audio through the MultiConditioner.
        output = encode_audiox_conditioning_tensors(
            self.conditioner,
            batch_metadata=batch_metadata,
            device=self.device,
        )
        # Encode text directly with T5.
        texts = [item["text_prompt"] for item in batch_metadata]
        output["text_prompt"] = self._encode_text(texts, self.device)
        # Encode video directly with CLIP.
        video_inputs = [item["video_prompt"] for item in batch_metadata]
        output["video_prompt"] = self._encode_video(video_inputs, self.device)
        return output

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
                    _conditioning_item(
                        text=self._text_for_task(task_norm, text),
                        video_tensor=video_tensors_list[i],
                        audio_tensor=audio_prompt_list[i],
                        sync_features=sync_features,
                        seconds_start=seconds_start,
                        seconds_model=seconds_model,
                    )
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
            seconds_start=seconds_start,
            sample_rate=sample_rate,
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
