# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

import einops
from torchvision import transforms
from transformers import AutoConfig, CLIPVisionModelWithProjection, T5EncoderModel, T5TokenizerFast
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.audiox.audiox_conditioner import (
    SA_Transformer,
    create_audiox_fixed_conditioner_from_conditioning_config,
    create_pretransform_from_config,
    encode_audiox_conditioning_tensors,
)
from vllm_omni.diffusion.models.audiox.audiox_maf import MAF_Block
from vllm_omni.diffusion.models.audiox.audiox_reference_media import (
    prepare_audio_reference,
    prepare_video_reference,
)
from vllm_omni.diffusion.models.audiox.audiox_runtime import generate_diffusion_cond
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
        self.clip_encoder = CLIPVisionModelWithProjection(clip_config)
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
