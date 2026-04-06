from __future__ import annotations

import contextlib
import logging
import math
import os
import typing as tp

import torch
from diffusers import AutoencoderOobleck
from diffusers.schedulers import EDMDPMSolverMultistepScheduler
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.audiox.audiox_conditioner import (
    MultiConditioner,
    create_audiox_fixed_conditioner_from_conditioning_config,
)
from vllm_omni.diffusion.models.audiox.audiox_maf import MAF_Block
from vllm_omni.diffusion.models.audiox.audiox_pretransform import create_pretransform_from_config
from vllm_omni.diffusion.models.audiox.audiox_transformer import MMDiffusionTransformer
from vllm_omni.diffusion.models.audiox.audiox_weights import strip_diffusion_model_config_for_audiox_dit
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

_PROGRESS = ProgressBarMixin()


def _use_upstream_pip_audiox_dit() -> bool:
    v = os.environ.get("VLLM_OMNI_AUDIOX_USE_UPSTREAM_DIT", "")
    return str(v).strip().lower() in ("1", "true", "yes")


def create_model_from_config(model_config, od_config: OmniDiffusionConfig | None = None):
    return create_diffusion_cond_from_config(model_config, od_config=od_config)


class ConditionedDiffusionModel(nn.Module):
    def __init__(
        self,
        *args,
        supports_cross_attention: bool = False,
        supports_input_concat: bool = False,
        supports_global_cond: bool = False,
        supports_prepend_cond: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
        input_concat_cond: torch.Tensor = None,
        global_embed: torch.Tensor = None,
        prepend_cond: torch.Tensor = None,
        prepend_cond_mask: torch.Tensor = None,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        batch_cfg: bool = False,
        rescale_cfg: bool = False,
        **kwargs,
    ):
        raise NotImplementedError()


class ConditionedDiffusionModelWrapper(nn.Module):
    def __init__(
        self,
        model: ConditionedDiffusionModel,
        conditioner: MultiConditioner,
        io_channels,
        sample_rate,
        min_input_length: int,
        diffusion_objective: tp.Literal["v"] = "v",
        pretransform: AutoencoderOobleck | None = None,
        cross_attn_cond_ids: list[str] | None = None,
        global_cond_ids: list[str] | None = None,
        od_config: OmniDiffusionConfig | None = None,
        *,
        gate: bool = False,
        gate_type: str | None = None,
        gate_type_config: dict[str, tp.Any] | None = None,
    ):
        super().__init__()
        self.model = model
        self.conditioner = conditioner
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config is not None else None
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids or []
        self.global_cond_ids = global_cond_ids or []
        self.min_input_length = min_input_length

        self.maf_block: MAF_Block | None = None
        if gate and gate_type == "MAF":
            gtc = gate_type_config or {}
            self.maf_block = MAF_Block(
                dim=768,
                num_experts_per_modality=int(gtc.get("num_experts_per_modality", 64)),
                num_heads=int(gtc.get("num_heads", 24)),
                num_fusion_layers=int(gtc.get("num_fusion_layers", 8)),
                mlp_ratio=float(gtc.get("mlp_ratio", 4.0)),
            )

    def get_conditioning_inputs(self, conditioning_tensors: dict[torch.Tensor, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)
                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            video_feature, text_feature, audio_feature = cross_attention_input
            if self.maf_block is not None:
                refined_branches = self.maf_block(text_feature, video_feature, audio_feature)
                cross_attention_input = torch.cat(list(refined_branches.values()), dim=1)
            else:
                cross_attention_input = torch.cat([video_feature, text_feature, audio_feature], dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]
                global_conds.append(global_cond_input)
            global_cond = torch.cat(global_conds, dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_embed": global_cond,
            }
        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_cond_mask": cross_attention_masks,
            "global_embed": global_cond,
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)


def create_diffusion_cond_from_config(config: dict[str, tp.Any], od_config: OmniDiffusionConfig | None = None):
    model_config = config["model"]
    diffusion_config = model_config["diffusion"]
    diffusion_model_config = diffusion_config["config"]
    diffusion_model_config = strip_diffusion_model_config_for_audiox_dit(dict(diffusion_model_config))

    diffusion_build_kwargs = dict(diffusion_model_config)
    if od_config is not None:
        diffusion_build_kwargs["od_config"] = od_config

    if _use_upstream_pip_audiox_dit():
        try:
            from audiox.models.dit import MMDiffusionTransformer as UpstreamMMDiffusionTransformer
        except ImportError as e:
            raise ImportError(
                "VLLM_OMNI_AUDIOX_USE_UPSTREAM_DIT is set but the ``audiox`` package is not importable. "
                "Install upstream AudioX in the same environment (e.g. ``pip install audiox``)."
            ) from e
        kw = dict(diffusion_build_kwargs)
        kw.pop("od_config", None)
        diffusion_model = UpstreamMMDiffusionTransformer(**kw)
        logger.info(
            "Using upstream pip ``audiox.models.dit.MMDiffusionTransformer`` (VLLM_OMNI_AUDIOX_USE_UPSTREAM_DIT)."
        )
    else:
        diffusion_model = MMDiffusionTransformer(**diffusion_build_kwargs)

    io_channels = model_config["io_channels"]
    sample_rate = config["sample_rate"]

    diffusion_objective: tp.Literal["v"] = "v"

    if od_config is None or not getattr(od_config, "model", None):
        raise ValueError(
            "AudioX requires OmniDiffusionConfig.model (Hugging Face repo id or local path with Diffusers layout "
            "including vae/) to load AutoencoderOobleck — same contract as Stable Audio Open."
        )
    model_path = od_config.model

    conditioning_config = model_config["conditioning"]
    conditioner = create_audiox_fixed_conditioner_from_conditioning_config(conditioning_config, model=model_path)

    cross_attention_ids = ["video_prompt", "text_prompt", "audio_prompt"]
    global_cond_ids: list[str] = []

    diffusion_full = model_config.get("diffusion") or {}
    gate = bool(diffusion_full.get("gate", False))
    gate_type = diffusion_full.get("gate_type")
    gate_type_config = diffusion_full.get("gate_type_config")

    pretransform_cfg = model_config["pretransform"]
    pretransform = create_pretransform_from_config(pretransform_cfg, model=model_path)
    min_input_length = int(pretransform.hop_length)

    min_input_length *= diffusion_model.patch_size

    return ConditionedDiffusionModelWrapper(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        diffusion_objective=diffusion_objective,
        od_config=od_config,
        gate=gate,
        gate_type=gate_type,
        gate_type_config=gate_type_config,
    )


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


def set_audio_channels(audio: torch.Tensor, target_channels: int) -> torch.Tensor:
    if target_channels == 1:
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


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

    amp = (
        torch.autocast(device_type="cuda", enabled=True)
        if dev.type == "cuda"
        else contextlib.nullcontext()
    )

    _PROGRESS.set_progress_bar_config(disable=False)
    with _PROGRESS.progress_bar(total=len(scheduler.timesteps)) as pbar, amp:
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
    model,
    steps: int = 250,
    cfg_scale=6,
    conditioning_tensors: dict | None = None,
    negative_conditioning_tensors: dict | None = None,
    batch_size: int = 1,
    sample_size: int = 2097152,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
    return_latents: bool = False,
    **sample_kwargs,
) -> torch.Tensor:
    pt = model.pretransform
    if pt is None:
        raise RuntimeError("AudioX inference-only path requires an AudioX pretransform.")
    if not isinstance(pt, AutoencoderOobleck):
        raise RuntimeError(f"Expected AutoencoderOobleck pretransform, got {type(pt)!r}.")

    sample_size = sample_size // int(pt.hop_length)

    if generator is None:
        raise ValueError("AudioX generation requires a torch.Generator.")
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device, generator=generator)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors is not None:
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    if model.diffusion_objective != "v":
        raise ValueError(
            f"Unsupported diffusion objective for inference-only AudioX path: {model.diffusion_objective!r}. "
            "Expected 'v'."
        )
    sampled = sample_k(
        model.model,
        noise,
        steps,
        **sample_kwargs,
        **conditioning_inputs,
        **negative_conditioning_tensors,
        cfg_scale=cfg_scale,
        batch_cfg=True,
        rescale_cfg=True,
        device=device,
        generator=generator,
    )
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()

    if not return_latents:
        sampled = sampled.to(next(pt.parameters()).dtype)
        model.pretransform = pt.to(dtype=torch.float32).eval()
        with torch.cuda.amp.autocast(enabled=False):
            vae = model.pretransform
            z = sampled.to(dtype=torch.float32)
            sampled = vae.decode(z * float(vae.audiox_scaling_factor), return_dict=True).sample

    return sampled


__all__ = [
    "ConditionedDiffusionModel",
    "ConditionedDiffusionModelWrapper",
    "create_diffusion_cond_from_config",
    "create_model_from_config",
    "generate_diffusion_cond",
    "sample_k",
    "set_audio_channels",
]
