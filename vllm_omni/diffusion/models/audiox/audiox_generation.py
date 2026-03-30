from __future__ import annotations

import k_diffusion as K
import torch
from tqdm.auto import trange

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")  # FP32

_PATCH_ATTR = "_vllm_omni_dpmpp_3m_sde_fixed"


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
    """DPM-Solver++(3M) SDE with terminal-sigma fix for upstream k-diffusion bug."""
    from k_diffusion.sampling import BrownianTreeNoiseSampler

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
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
    return x


def patch_k_diffusion_sample_dpmpp_3m_sde() -> None:
    """Idempotently replace ``k_diffusion.sampling.sample_dpmpp_3m_sde`` with fixed version."""
    import k_diffusion.sampling as ks

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
    sampler_type: str = "dpmpp-3m-sde",
    sigma_min: float = 0.5,
    sigma_max: float = 50,
    rho: float = 1.0,
    device: str = "cuda",
    callback=None,
    **extra_args,
):
    """Inference-only AudioX sampler (k-diffusion)."""
    patch_k_diffusion_sample_dpmpp_3m_sde()
    denoiser = K.external.VDenoiser(model_fn)

    sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
    noise = noise * sigmas[0]
    x = noise

    with torch.cuda.amp.autocast():
        if sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        if sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
    raise ValueError(
        f"Unsupported sampler_type={sampler_type!r} for inference-only AudioX path. "
        "Supported: 'dpmpp-2m-sde', 'dpmpp-3m-sde'."
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
    """Generate audio in inference-only mode (no variation/inpainting)."""
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio

    if generator is None:
        raise ValueError("AudioX generation requires a torch.Generator; seed/random fallback is disabled.")
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

    diff_objective = model.diffusion_objective
    if diff_objective == "v":
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
    else:
        raise ValueError(
            f"Unsupported diffusion objective for inference-only AudioX path: {diff_objective!r}. "
            "Expected 'v'."
        )
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()

    if model.pretransform is not None and not return_latents:
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)

        model.pretransform = model.pretransform.to(dtype=torch.float32).eval()
        with torch.cuda.amp.autocast(enabled=False):
            pt = model.pretransform
            scale = getattr(getattr(pt, "config", None), "scaling_factor", None)
            if scale is None:
                scale = getattr(pt, "scaling_factor", 1.0)
            sampled = sampled * float(scale)
            sampled = model.pretransform.decode(sampled.to(dtype=torch.float32))

    return sampled


__all__ = ["generate_diffusion_cond", "patch_k_diffusion_sample_dpmpp_3m_sde", "sample_k", "set_audio_channels"]

