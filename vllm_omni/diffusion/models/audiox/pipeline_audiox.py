# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream: https://github.com/ZeyueT/AudioX/tree/main/audiox/models

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.audiox.audiox_conditioner import (
    encode_audiox_conditioning_tensors,
)
from vllm_omni.diffusion.models.audiox.audiox_runtime import create_model_from_config
from vllm_omni.diffusion.models.audiox.audiox_weights import (
    build_sharded_component_sources,
    load_audiox_bundle_config,
    load_audiox_weights,
)
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.postprocess.audio import build_audio_post_process_func
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

_VIDEO_ONLY_TASKS = frozenset({"v2a", "v2m"})
_TEXT_VIDEO_TASKS = frozenset({"tv2a", "tv2m"})
_VIDEO_CONDITIONED_TASKS = _VIDEO_ONLY_TASKS | _TEXT_VIDEO_TASKS

logger = init_logger(__name__)


def _to_2ch_audio(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    elif x.shape[0] > 2:
        x = x[:2]
    return x


def _crop_or_pad_1d(x: torch.Tensor, start: int, target_len: int) -> torch.Tensor:
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

    if isinstance(source, str):
        import torchaudio

        wav, sr = torchaudio.load(source)
        if sr != model_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, model_sample_rate)
    elif isinstance(source, torch.Tensor):
        wav = source
    else:
        try:
            import numpy as np

            if isinstance(source, np.ndarray):
                wav = torch.from_numpy(source)
            else:
                raise TypeError(f"Unsupported audio source type: {type(source)!r}")
        except ImportError:
            raise TypeError(f"Unsupported audio source type: {type(source)!r}") from None

    wav = _to_2ch_audio(wav)
    wav = _crop_or_pad_1d(wav, start, target_len)
    return wav.to(device=device, dtype=torch.float32)


def _normalize_video_tensor(frames: torch.Tensor, size: int = 224) -> torch.Tensor:
    if frames.dim() != 4:
        raise ValueError(f"Expected [T, C, H, W], got {tuple(frames.shape)}")

    frames = frames.float()
    if frames.max() > 1.5:
        frames = frames / 255.0

    if frames.shape[-2:] != (size, size):
        frames = F.interpolate(frames, size=(size, size), mode="bicubic", align_corners=False)
    return frames


def _adjust_video_duration(frames: torch.Tensor, duration: float, target_fps: int) -> torch.Tensor:
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
    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()

        if ext in {".jpg", ".jpeg", ".png"}:
            import numpy as np
            from PIL import Image

            img = Image.open(source).convert("RGB")
            frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
            frames = frame
        else:
            from decord import VideoReader
            from decord import cpu as decord_cpu

            vr = VideoReader(source, ctx=decord_cpu(0))
            fps = float(vr.get_avg_fps())
            step = max(1, int(math.ceil(fps / target_fps)))
            start = int(seek_time * fps)
            if duration > 0:
                target_t = int(duration * target_fps)
                end = min(len(vr), start + target_t * step)
                frame_ids = list(range(start, end, step))
            else:
                frame_ids = list(range(start, len(vr), step))
            frames = torch.from_numpy(vr.get_batch(frame_ids).asnumpy()).permute(0, 3, 1, 2)
    elif isinstance(source, torch.Tensor):
        frames = source
    else:
        try:
            import numpy as np

            if isinstance(source, np.ndarray):
                frames = torch.from_numpy(source)
            else:
                raise TypeError(f"Unsupported video source type: {type(source)!r}")
        except ImportError:
            raise TypeError(f"Unsupported video source type: {type(source)!r}") from None

    if frames.dim() == 4 and frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)

    frames = _normalize_video_tensor(frames, size=224)
    if duration > 0:
        frames = _adjust_video_duration(frames, duration, target_fps)
    return frames


def get_audiox_post_process_func(_od_config: OmniDiffusionConfig):
    """Return a post-processor for AudioX outputs (contributing guide section 2.4).

    Separates final tensor handling from ``AudioXPipeline.forward``: converts denoised
    waveforms to CPU float numpy for saving, unless ``output_type`` requests tensors.
    ``_od_config`` is reserved for future use (e.g. bundle-specific output normalization).
    """

    return build_audio_post_process_func()


def _resolve_audio_ref_for_preprocess(
    raw_prompt: Any,
    extra: dict[str, Any],
    sp: Any,
    batch_index: int,
    od_config: OmniDiffusionConfig,
) -> Any:
    src = _mm_path_lookup(
        raw_prompt, extra, batch_index, mm_key="audio", paths_key="audio_paths", single_key="audio_path"
    )
    if src is not None:
        return src
    ap = getattr(sp, "audiox_audio_path", None)
    if ap:
        return ap
    return od_config.audiox_reference_audio_path


def get_audiox_pre_process_func(od_config: OmniDiffusionConfig):
    """Return a request pre-processor for text / video / audio (contributing guide section 2.4).

    Normalizes each entry in ``OmniDiffusionRequest.prompts`` to a dict with
    ``multi_modal_data`` and optional ``additional_information``. Strips text prompts.
    When possible, resolves **video** and **reference audio** paths to CPU tensors early
    so ``AudioXPipeline.forward`` only moves data to the inference device.

    Video loading is skipped for text-only tasks (``t2a`` / ``t2m``). Reference audio is
    loaded when a source path or tensor is present (including ``OmniDiffusionConfig`` /
    sampling-params fallbacks used by the pipeline).
    """

    def _noop(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        return request

    if od_config.model is None:
        return _noop

    model_root = os.path.abspath(od_config.model)
    try:
        _, model_cfg, _ = load_audiox_bundle_config(model_root)
    except FileNotFoundError as e:
        logger.warning("AudioX pre-process: %s; skipping media preload.", e)
        return _noop

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
        user_seconds_total = extra.get("seconds_total")
        if user_seconds_total is None:
            user_seconds_total = sample_size / sample_rate
        user_seconds_total = float(user_seconds_total)
        cond_seconds = float(audio_conditioning_samples) / float(sample_rate)

        task_norm = AudioXPipeline._normalize_task(sp.audiox_task)

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

            asrc = _resolve_audio_ref_for_preprocess(p, extra, sp, i, od_config)
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


def _audio_conditioning_input_samples(model_config: dict[str, Any]) -> int | None:
    """Waveform length (per channel) so ``AudioAutoencoderConditionerv2`` encode matches ``latent_seq_len``.

    Upstream ties ``proj_features_128`` to ``latent_seq_len``; input length must match
    ``latent_seq_len * pretransform.downsampling_ratio`` (see ``audiox/models/conditioners.py``).
    """
    try:
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
            if ls is not None and ds is not None:
                return int(ls) * int(ds)
    except (TypeError, ValueError):
        return None
    return None


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
    """AudioX conditional diffusion (registry pipeline; weights via ``weights_sources`` + ``load_weights``).

    Tasks (``audiox_task`` on sampling params):

    - ``t2a`` / ``t2m``: text — zero video and zero reference audio.
    - ``v2a`` / ``v2m``: video — empty text.
    - ``tv2a`` / ``tv2m``: text + video — non-empty text per item.

    Video: ``extra_args["video_path"]`` / ``["video_paths"]`` or ``multi_modal_data["video"]``.

    Reference audio: ``extra_args["audio_path"]`` / ``["audio_paths"]``, ``audiox_audio_path`` on sampling
    params, or ``OmniDiffusionConfig.audiox_reference_audio_path`` (CLI ``--audiox-reference-audio-path``).

    Requires ``od_config.model`` to be a **vLLM-Omni sharded** tree: ``config.json`` plus
    ``transformer/diffusion_pytorch_model.safetensors`` and
    ``conditioners/diffusion_pytorch_model.safetensors`` (and ``vae/`` when the config includes a
    pretransform). Produce this layout from a Hugging Face ``config.json`` + ``model.ckpt`` bundle via
    ``python -m vllm_omni.diffusion.models.audiox.audiox_weights``. Also requires
    third-party Python deps used by the inlined AudioX modules under this package
    (``inference/``, ``models/``, ``data/``). VAE weights are remapped onto Hugging Face
    :class:`~diffusers.AutoencoderOobleck` (see :mod:`vllm_omni.diffusion.models.audiox.audiox_weights`).
    """

    support_audio_output: ClassVar[bool] = True
    _PROFILER_TARGETS: ClassVar[list[str]] = ["diffuse"]
    # ``CLIPWithSyncWithEmptyFeatureConditioner`` hard-codes ``duration = 10`` (seconds) and a positional
    # embedding of length ``duration * video_fps`` (see ``audiox/models/conditioners.py``). Video tensors
    # must match that length or forward fails (e.g. ``video_hidden += Temp_pos_embedding``).
    # If future checkpoints use a different duration, read from model config.
    _CLIP_SYNC_DURATION_SEC: ClassVar[float] = 10.0
    # Sync placeholder shape is fixed in AudioX: ``proj_sync = Linear(240, out_features)`` and
    # ``video_sync_frames.view(-1, 240)`` (see ``conditioners.py``). The middle dim must be 240, not
    # ``duration * video_fps``; otherwise the view yields the wrong last dim (e.g. 160 vs 768) when added
    # to ``video_hidden``. Matches upstream Gradio ``torch.zeros(1, 240, 768)``.
    _VIDEO_SYNC_FRAME_COUNT: ClassVar[int] = 240

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()
        if od_config.model is None:
            raise ValueError(
                "AudioXPipeline requires od_config.model (directory with component-sharded safetensors; "
                "see audiox_weights)."
            )

        self._model_root = os.path.abspath(od_config.model)
        config_path, self._model_config, _ = load_audiox_bundle_config(self._model_root)

        try:
            from vllm_omni.diffusion.models.audiox.audiox_runtime import generate_diffusion_cond
        except ImportError as e:
            raise RuntimeError(
                "Failed to import inlined AudioX modules under vllm_omni.diffusion.models.audiox "
                "(missing files or unsatisfied optional dependencies?)."
            ) from e

        self._generate_diffusion_cond = generate_diffusion_cond
        self._model = create_model_from_config(self._model_config, od_config=od_config)
        logger.debug("AudioX conditioned model built from %s", config_path)

        self.weights_sources = build_sharded_component_sources(
            model_root=self._model_root,
            od_config=od_config,
            model_config=self._model_config,
        )
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
        """Load AudioX checkpoint shards from ``weights`` (from ``DiffusersPipelineLoader.get_all_weights``)."""
        load_audiox_weights(self, weights, model_config=self._model_config)

        # Upstream AudioX inference is effectively float32 (e.g. T5 ``proj_out(embeddings.float())``).
        self._model.to(torch.float32)
        self._model.eval().requires_grad_(False)

        # ``AutoWeightsLoader`` only reports tensors present in the iterator; ``DiffusersPipelineLoader``
        # strict check expects every parameter/buffer under ``weights_sources`` prefixes. Treat init
        # defaults as satisfying coverage.
        names = {n for n, _ in self.named_parameters()}
        names.update(n for n, _ in self.named_buffers())
        return names

    def _conditioning_dtype(self) -> torch.dtype:
        """Dtype of the inner model's floating-point parameters (forced to float32 after ``load_weights``)."""
        model = getattr(self, "_model", None)
        if model is None:
            return torch.float32
        try:
            p = next(model.parameters())
        except StopIteration:
            return torch.float32
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

    def _resolve_audio_source(self, raw_prompt: Any, extra: dict[str, Any], sp: Any, batch_index: int) -> Any:
        src = _mm_path_lookup(
            raw_prompt, extra, batch_index, mm_key="audio", paths_key="audio_paths", single_key="audio_path"
        )
        if src is not None:
            return src
        ap = getattr(sp, "audiox_audio_path", None)
        if ap:
            return ap
        return self.od_config.audiox_reference_audio_path

    def _audio_prompt_tensors(
        self,
        *,
        raw_prompts: list[Any],
        extra: dict[str, Any],
        sp: Any,
        seconds_start: float,
        sample_rate: int,
        device: torch.device,
        cond_dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        target_len = self._audio_conditioning_samples
        cond_seconds = float(target_len) / float(sample_rate)
        for i, _raw in enumerate(raw_prompts):
            src = self._resolve_audio_source(_raw, extra, sp, i)
            if src is None:
                out.append(torch.zeros(2, target_len, device=device, dtype=cond_dtype))
                continue
            try:
                out.append(
                    prepare_audio_reference(
                        src,
                        model_sample_rate=sample_rate,
                        seconds_start=seconds_start,
                        seconds_total=cond_seconds,
                        device=device,
                    ).to(dtype=cond_dtype)
                )
            except (OSError, TypeError, ValueError) as e:
                raise ValueError(f"Failed to load AudioX reference audio for batch item {i}: {e}") from e
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

    def diffuse(
        self,
        *,
        steps: int,
        guidance_scale: float,
        conditioning_tensors: dict[str, Any],
        negative_conditioning_tensors: dict[str, Any] | None,
        batch_size: int,
        sampler_type: str,
        sigma_min: float,
        sigma_max: float,
        generator: torch.Generator,
        cfg_rescale: float,
    ) -> torch.Tensor:
        """Run k-diffusion sampling (profiled when ``enable_diffusion_pipeline_profiler`` is set)."""
        return self._generate_diffusion_cond(
            self._model,
            steps=steps,
            cfg_scale=guidance_scale,
            conditioning_tensors=conditioning_tensors,
            negative_conditioning_tensors=negative_conditioning_tensors,
            batch_size=batch_size,
            sample_size=self._sample_size,
            device=self.device,
            generator=generator,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            scale_phi=cfg_rescale,
        )

    def _encode_conditioning_tensors(self, batch_metadata: list[dict[str, Any]]) -> dict[str, Any]:
        return encode_audiox_conditioning_tensors(
            getattr(self._model, "conditioner", None),
            batch_metadata=batch_metadata,
            device=self.device,
        )

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Execute one generation request (vLLM-Omni pipeline contract; see contributing guide section 2.3).

        Reads :class:`~vllm_omni.diffusion.request.OmniDiffusionRequest`: ``prompts`` (strings or dicts)
        and :class:`~vllm_omni.inputs.data.OmniDiffusionSamplingParams` for steps, guidance, seed, and
        ``extra_args`` (sampler, sigmas, timing, paths). Dict prompts may include ``multi_modal_data``
        (e.g. ``video``, ``audio``) per item.

        Returns :class:`~vllm_omni.diffusion.data.DiffusionOutput` with generated audio as ``output``.
        """
        # --- Prompts (and optional negative prompts for CFG) ---
        if req.prompts is None or len(req.prompts) == 0:
            raise ValueError("AudioXPipeline requires at least one prompt.")
        normalized_prompts = _normalize_prompts(list(req.prompts))
        prompts = [p["prompt"] for p in normalized_prompts]

        sampling_params = req.sampling_params
        task_norm = self._normalize_task(sampling_params.audiox_task)
        self._ensure_text_video_prompts(task_norm, prompts)

        neg: list[str] | None = None
        if not all(p.get("negative_prompt") is None for p in normalized_prompts):
            neg = [str(p.get("negative_prompt") or "") for p in normalized_prompts]

        # --- Sampling parameters (OmniDiffusionSamplingParams + extra_args) ---
        num_inference_steps = (
            sampling_params.num_inference_steps if sampling_params.num_inference_steps is not None else 100
        )
        extra_args = sampling_params.extra_args or {}
        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale
        else:
            guidance_scale = float(extra_args.get("cfg_scale", 6.0))
        num_outputs_per_prompt = (
            int(sampling_params.num_outputs_per_prompt) if sampling_params.num_outputs_per_prompt > 0 else 1
        )
        batch_size = len(prompts) * num_outputs_per_prompt

        seconds_start = float(extra_args.get("seconds_start", 0.0))
        seconds_model = self._sample_size / self._sample_rate
        sampler_type = extra_args.get("sampler_type", "dpmpp-3m-sde")
        sigma_min = float(extra_args.get("sigma_min", 0.03))
        sigma_max = float(extra_args.get("sigma_max", 1000.0))
        cfg_rescale = float(extra_args.get("cfg_rescale", 0.0))
        device = self.device
        generator = sampling_params.generator
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(0)
        target_fps = self._target_fps
        sample_rate = self._sample_rate
        cond_dtype = self._conditioning_dtype()

        sync_features = torch.zeros(1, self._VIDEO_SYNC_FRAME_COUNT, 768, device=device, dtype=cond_dtype)

        audio_prompt_list = self._audio_prompt_tensors(
            raw_prompts=normalized_prompts,
            extra=extra_args,
            sp=sampling_params,
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

        conditioning_batch: list[dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            for _ in range(num_outputs_per_prompt):
                conditioning_batch.append(
                    _conditioning_item(
                        text=self._text_for_task(task_norm, prompt),
                        video_tensor=video_tensors_list[i],
                        audio_tensor=audio_prompt_list[i],
                        sync_features=sync_features,
                        seconds_start=seconds_start,
                        seconds_model=seconds_model,
                    )
                )

        negative_conditioning_batch: list[dict[str, Any]] | None = None
        if neg is not None and guidance_scale > 1.0:
            negative_conditioning_batch = []
            for i, nprompt in enumerate(neg):
                for _ in range(num_outputs_per_prompt):
                    negative_conditioning_batch.append(
                        _conditioning_item(
                            text=self._text_for_task(task_norm, nprompt),
                            video_tensor=video_tensors_list[i],
                            audio_tensor=audio_prompt_list[i],
                            sync_features=sync_features,
                            seconds_start=seconds_start,
                            seconds_model=seconds_model,
                        )
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
            sampler_type=sampler_type,
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
