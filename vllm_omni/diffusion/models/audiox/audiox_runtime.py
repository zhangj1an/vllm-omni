from __future__ import annotations

import logging
import os
import typing as tp

import k_diffusion as K
import k_diffusion.sampling as ks
import torch
from k_diffusion.sampling import BrownianTreeNoiseSampler
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.audiox.audiox_conditioner import (
    MultiConditioner,
    create_audiox_fixed_conditioner_from_conditioning_config,
)
from vllm_omni.diffusion.models.audiox.audiox_maf import MAF_Block
from vllm_omni.diffusion.models.audiox.audiox_pretransform import (
    AudioXVAE,
    create_pretransform_from_config,
)
from vllm_omni.diffusion.models.audiox.audiox_transformer import MMDiffusionTransformer
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

_PATCH_ATTR = "_vllm_omni_dpmpp_3m_sde_fixed"
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
        pretransform: AudioXVAE | None = None,
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
    diffusion_model_config = dict(diffusion_model_config)
    if diffusion_model_config.get("video_fps", None) is not None:
        diffusion_model_config.pop("video_fps")

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

    conditioning_config = model_config["conditioning"]
    conditioner = create_audiox_fixed_conditioner_from_conditioning_config(conditioning_config)

    cross_attention_ids = ["video_prompt", "text_prompt", "audio_prompt"]
    global_cond_ids: list[str] = []

    diffusion_full = model_config.get("diffusion") or {}
    gate = bool(diffusion_full.get("gate", False))
    gate_type = diffusion_full.get("gate_type")
    gate_type_config = diffusion_full.get("gate_type_config")

    pretransform_cfg = model_config["pretransform"]
    pretransform = create_pretransform_from_config(pretransform_cfg, sample_rate)
    min_input_length = pretransform.downsampling_ratio

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


@torch.no_grad()
def _sample_dpmpp_3m_sde_fixed(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    _PROGRESS.set_progress_bar_config(disable=disable)
    with _PROGRESS.progress_bar(total=len(sigmas) - 1) as pbar:
        for i in range(len(sigmas) - 1):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
            if sigmas[i + 1] == 0:
                x = denoised
            else:
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                h_eta = h * (eta + 1)

                x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

                if h_2 is not None:
                    r0 = h_1 / h
                    r1 = h_2 / h
                    d1_0 = (denoised - denoised_1) / r0
                    d1_1 = (denoised_1 - denoised_2) / r1
                    d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                    d2 = (d1_0 - d1_1) / (r0 + r1)
                    phi_2 = h_eta.neg().expm1() / h_eta + 1
                    phi_3 = phi_2 / h_eta - 0.5
                    x = x + phi_2 * d1 - phi_3 * d2
                elif h_1 is not None:
                    r = h_1 / h
                    d = (denoised - denoised_1) / r
                    phi_2 = h_eta.neg().expm1() / h_eta + 1
                    x = x + phi_2 * d

                if eta:
                    x = (
                        x
                        + noise_sampler(sigmas[i], sigmas[i + 1])
                        * sigmas[i + 1]
                        * (-2 * h * eta).expm1().neg().sqrt()
                        * s_noise
                    )

                denoised_1, denoised_2 = denoised, denoised_1
                h_1, h_2 = h, h_1
            pbar.update()
    return x


def patch_k_diffusion_sample_dpmpp_3m_sde() -> None:
    if getattr(ks, _PATCH_ATTR, False):
        return
    ks.sample_dpmpp_3m_sde = _sample_dpmpp_3m_sde_fixed
    setattr(ks, _PATCH_ATTR, True)


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
    sigma_min: float = 0.5,
    sigma_max: float = 50,
    rho: float = 1.0,
    device: str = "cuda",
    callback=None,
    **extra_args,
):
    patch_k_diffusion_sample_dpmpp_3m_sde()
    denoiser = K.external.VDenoiser(model_fn)

    sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
    noise = noise * sigmas[0]
    x = noise

    with torch.cuda.amp.autocast():
        return K.sampling.sample_dpmpp_3m_sde(
            denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args
        )


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
    **sampler_kwargs,
) -> torch.Tensor:
    pt = model.pretransform
    if pt is None:
        raise RuntimeError("AudioX inference-only path requires an AudioX pretransform.")
    if not isinstance(pt, AudioXVAE):
        raise RuntimeError(f"Expected AudioXVAE pretransform, got {type(pt)!r}.")

    sample_size = sample_size // pt.downsampling_ratio

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
        **sampler_kwargs,
        **conditioning_inputs,
        **negative_conditioning_tensors,
        cfg_scale=cfg_scale,
        batch_cfg=True,
        rescale_cfg=True,
        device=device,
    )
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()

    if not return_latents:
        sampled = sampled.to(next(pt.parameters()).dtype)
        model.pretransform = pt.to(dtype=torch.float32).eval()
        with torch.cuda.amp.autocast(enabled=False):
            sampled = model.pretransform.decode_scaled(sampled.to(dtype=torch.float32))

    return sampled


__all__ = [
    "ConditionedDiffusionModel",
    "ConditionedDiffusionModelWrapper",
    "create_diffusion_cond_from_config",
    "create_model_from_config",
    "generate_diffusion_cond",
    "patch_k_diffusion_sample_dpmpp_3m_sde",
    "sample_k",
    "set_audio_channels",
]
