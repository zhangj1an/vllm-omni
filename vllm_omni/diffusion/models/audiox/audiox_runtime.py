from __future__ import annotations

import math

import torch
from diffusers import AutoencoderOobleck
from diffusers.schedulers import EDMDPMSolverMultistepScheduler

from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

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


__all__ = [
    "generate_diffusion_cond",
    "sample_k",
    "set_audio_channels",
]
