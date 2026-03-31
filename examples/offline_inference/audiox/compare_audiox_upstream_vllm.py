#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare upstream AudioX vs vLLM-Omni AudioX on a fixed dummy t2a conditioning + one denoiser step.

1) **Conditioning parity** — identical batch metadata (zeros video/sync/audio, fixed text, model-native
   ``seconds_total`` / audio conditioning length). Runs each tree's conditioner + ``get_conditioning_inputs``,
   then reports ``cross_attn_cond`` / mask stats and max-abs diff.

2) **Single denoiser step** — optional shared ``noise.pt`` (or generated with a fixed seed), same ``sigma``
    tensor, ``forward(x, t, conditioning_tensors)`` on both wrappers; compares ``denoised`` max-abs diff.

3) **``--dit-dummy``** — bypasses wrapper conditioning: same ``x``, ``t``, and **identical** ``cross_attn_cond`` /
   mask tensors fed to **both** inner diffusion modules (upstream ``MMDiTWrapper`` vs vLLM ``MMDiffusionTransformer``).
   Use ``--dit-dummy-source random`` (default) for fixed-seed synthetic tensors, or ``upstream`` to reuse
   upstream-encoded conditioning on **both** sides (isolates DiT vs encoders).

4) **``--inject-upstream-cross-attn``** — run vLLM's inner DiT with **upstream's** fused ``cross_attn_cond`` /
   ``cross_attn_mask`` from ``upstream.get_conditioning_inputs(ct_u)``. If **denoised** matches **upstream(x,t,ct_u)**
   (~0) but differs from **vllm(x,t,ct_v)**, the full-path gap is from vLLM's encoder/MAF path (``ct_v``), not the DiT.

5) **``--vllm-forward-upstream-raw-ct``** — **vllm(x,t, ct_u)** so upstream per-modality tensors go through **vLLM's**
   ``get_conditioning_inputs`` (vLLM MAF). Compares MAF/encoder split vs fused inject.

Usage (repo root on ``PYTHONPATH``)::

    export PYTHONPATH=/path/to/vllm-omni
    /path/to/.venv/bin/python examples/offline_inference/audiox/compare_audiox_upstream_vllm.py

Env:

- ``DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`` — recommended for stable runs on CUDA.
- ``VLLM_OMNI_AUDIOX_USE_UPSTREAM_DIT=1`` — build the vLLM pipeline’s inner DiT from pip ``audiox.models.dit.MMDiffusionTransformer`` (see ``audiox_runtime.create_diffusion_cond_from_config``). Weight load merges split ``cross_attn`` tensors back to ``to_kv`` for upstream ``CrossAttention``. Use with ``--dit-dummy --dit-dummy-source upstream`` to confirm **bit-identical** one-step ``denoised`` on shared conditioning; any remaining full-path gap is then **not** the forked DiT.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

# --- vLLM config (CustomOp / RMSNorm) ---
def _vllm_config_cm():
    try:
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
        device = "cuda" if has_gpu else "cpu"
        return set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device)))
    except Exception:  # pragma: no cover
        import contextlib

        return contextlib.nullcontext()


def _fix_vae_ckpt_paths(obj: object, weights_dir: Path) -> None:
    vae = weights_dir / "VAE.ckpt"
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "pretransform_ckpt_path" and isinstance(v, str) and "VAE" in v:
                obj[k] = str(vae)
            else:
                _fix_vae_ckpt_paths(v, weights_dir)
    elif isinstance(obj, list):
        for item in obj:
            _fix_vae_ckpt_paths(item, weights_dir)


def _audio_conditioning_samples(model_config: dict[str, Any]) -> int | None:
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


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float().cpu()
    return {
        "shape": tuple(x.shape),
        "mean": float(xf.mean()),
        "std": float(xf.std()),
        "min": float(xf.min()),
        "max": float(xf.max()),
    }


def _diff_report(a: torch.Tensor, b: torch.Tensor, name: str) -> dict[str, Any]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {"name": name, "error": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    d = (a - b).abs()
    return {
        "name": name,
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rmse": float(torch.sqrt((d**2).mean())),
        "stats_a": _tensor_stats(a),
        "stats_b": _tensor_stats(b),
    }


def _build_dummy_batch(
    *,
    device: torch.device,
    fixed_text: str,
    sample_rate: int,
    sample_size: int,
    video_fps: int,
    clip_duration_sec: float,
    audio_cond_samples: int,
) -> list[dict[str, Any]]:
    seconds_model = float(sample_size) / float(sample_rate)
    clip_frames = int(round(clip_duration_sec * video_fps))
    video = torch.zeros(clip_frames, 3, 224, 224, device=device, dtype=torch.float32)
    audio = torch.zeros(2, audio_cond_samples, device=device, dtype=torch.float32)
    sync = torch.zeros(1, 240, 768, device=device, dtype=torch.float32)
    return [
        {
            "video_prompt": {"video_tensors": video.unsqueeze(0), "video_sync_frames": sync},
            "text_prompt": fixed_text,
            "audio_prompt": audio.unsqueeze(0),
            "seconds_start": 0.0,
            "seconds_total": seconds_model,
        }
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Upstream vs vLLM-Omni AudioX conditioning / one-step compare")
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory with config.json + model.ckpt (default: ./audiox_weights next to this script)",
    )
    p.add_argument("--text", type=str, default="Fireworks burst twice, then silence, then a clock ticks.")
    p.add_argument("--sigma", type=float, default=10.0, help="Scalar noise level t passed to the DiT (same both sides).")
    p.add_argument("--noise-seed", type=int, default=42)
    p.add_argument(
        "--conditioner-seed",
        type=int,
        default=42,
        help=(
            "Reset torch.manual_seed before each full conditioner forward (upstream vs vLLM) so VAE "
            "sampling matches when comparing; video/text are deterministic, audio uses the same RNG draw."
        ),
    )
    p.add_argument("--noise-pt", type=Path, default=None, help="Load latent noise from this .pt (shape [B,C,T]).")
    p.add_argument("--save-noise-pt", type=Path, default=None, help="Save generated noise to this path.")
    p.add_argument("--skip-denoiser", action="store_true")
    p.add_argument(
        "--dit-dummy",
        action="store_true",
        help="Compare inner DiT only: identical cross_attn tensors on both sides (see --dit-dummy-source).",
    )
    p.add_argument(
        "--dit-dummy-source",
        type=str,
        choices=("random", "upstream"),
        default="random",
        help="random: same-shape randn/ones; upstream: clone upstream get_conditioning_inputs tensors on both sides.",
    )
    p.add_argument("--dit-dummy-seed", type=int, default=12345, help="RNG for --dit-dummy-source random.")
    p.add_argument(
        "--inject-upstream-cross-attn",
        action="store_true",
        help=(
            "After the main denoiser compare: run vLLM inner DiT with upstream's fused "
            "cross_attn_cond / mask from upstream.get_conditioning_inputs(ct_u). "
            "Should match den_u (~0 vs upstream) and isolate the ~0.2 full-path gap as vLLM ct_v vs ct_u."
        ),
    )
    p.add_argument(
        "--vllm-forward-upstream-raw-ct",
        action="store_true",
        help=(
            "Also run vllm(x,t, ct_u): upstream per-modality conditioner outputs through vLLM's "
            "get_conditioning_inputs (vLLM MAF + vLLM DiT). Compare to vllm(x,t, ct_v)."
        ),
    )
    p.add_argument(
        "--per-modality-cond-diff",
        action="store_true",
        help=(
            "After encoding: print max_abs/mean_abs for each branch tensor (and mask) of ct_u vs ct_v "
            "for video_prompt, text_prompt, audio_prompt — see which modality drives ct_v vs ct_u."
        ),
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    weights_dir = args.weights_dir or (here / "audiox_weights")
    cfg_path = weights_dir / "config.json"
    ckpt_path = weights_dir / "model.ckpt"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        raise SystemExit(f"Need {cfg_path} and {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    _fix_vae_ckpt_paths(cfg, weights_dir)

    sample_rate = int(cfg["sample_rate"])
    sample_size = int(cfg["sample_size"])
    video_fps = int(cfg.get("video_fps", 5))
    ac = _audio_conditioning_samples(cfg)
    audio_cond_samples = ac if ac is not None else sample_size

    batch = _build_dummy_batch(
        device=device,
        fixed_text=args.text,
        sample_rate=sample_rate,
        sample_size=sample_size,
        video_fps=video_fps,
        clip_duration_sec=10.0,
        audio_cond_samples=audio_cond_samples,
    )

    # ---- Upstream (pip audiox) ----
    from audiox.models.diffusion import create_diffusion_cond_from_config as ax_create
    from audiox.models.utils import load_ckpt_state_dict as ax_load_ckpt
    from audiox.training.utils import copy_state_dict as ax_copy

    upstream = ax_create(cfg).to(device).eval()
    ax_copy(upstream, ax_load_ckpt(str(ckpt_path)))
    torch.manual_seed(int(args.conditioner_seed))
    ct_u = upstream.conditioner(batch, device)
    gi_u = upstream.get_conditioning_inputs(ct_u)
    cac_u = gi_u["cross_attn_cond"]
    cam_u = gi_u.get("cross_attn_mask")

    # ---- vLLM-Omni ----
    repo_root = here.parents[2]
    if str(repo_root) not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = f"{repo_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    from vllm_omni.diffusion.models.audiox.audiox_conditioner import encode_audiox_conditioning_tensors
    from vllm_omni.diffusion.models.audiox.audiox_runtime import create_model_from_config
    from vllm_omni.diffusion.models.audiox.audiox_weights import (
        CONDITIONERS_SAFETENSORS,
        TRANSFORMER_SAFETENSORS,
        VAE_SAFETENSORS,
        _load_checkpoint_state_dict,
        load_audiox_weights,
    )

    # ReplicatedLinear / ModelWeightParameter expect TP groups; standalone script uses TP=1.
    import vllm.model_executor.parameter as vllm_param  # noqa: PLC0415

    class _Harness(nn.Module):
        """Same layout as ``AudioXPipeline`` (``_model`` child) so ``load_audiox_weights`` + ``AutoWeightsLoader`` match production."""

        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self._model = inner

    def _load_vllm_inner_weights(inner: nn.Module, wdir: Path) -> None:
        """Load sharded / ckpt weights the same way as ``AudioXPipeline`` (remap + ``AutoWeightsLoader``)."""
        from safetensors.torch import load_file  # noqa: PLC0415

        chunks: list[tuple[str, torch.Tensor]] = []
        for rel in (TRANSFORMER_SAFETENSORS, CONDITIONERS_SAFETENSORS, VAE_SAFETENSORS):
            p = wdir / rel
            if p.is_file():
                chunks.extend(list(load_file(p).items()))
        if not chunks:
            chunks = list(_load_checkpoint_state_dict(str(wdir / "model.ckpt")).items())
            print("[vLLM weights] using model.ckpt (no sharded safetensors found)")
        else:
            print(f"[vLLM weights] merged {len(chunks)} tensors from sharded safetensors")

        load_audiox_weights(_Harness(inner), chunks, model_config=cfg)
        inner.to(torch.float32).eval().requires_grad_(False)

    with _vllm_config_cm():
        with (
            patch.object(vllm_param, "get_tensor_model_parallel_rank", return_value=0),
            patch.object(vllm_param, "get_tensor_model_parallel_world_size", return_value=1),
        ):
            vllm = create_model_from_config(cfg, od_config=None).to(device).eval()
            _load_vllm_inner_weights(vllm, weights_dir)
    torch.manual_seed(int(args.conditioner_seed))
    ct_v = encode_audiox_conditioning_tensors(vllm.conditioner, batch_metadata=batch, device=device)
    gi_v = vllm.get_conditioning_inputs(ct_v)
    cac_v = gi_v["cross_attn_cond"]
    cam_v = gi_v.get("cross_attn_cond_mask")

    print("=== Conditioning: cross_attn_cond ===")
    print(_diff_report(cac_u, cac_v, "cross_attn_cond"))
    print("=== Upstream cross_attn_cond stats ===")
    print(_tensor_stats(cac_u))
    print("=== vLLM cross_attn_cond stats ===")
    print(_tensor_stats(cac_v))

    if cam_u is not None and cam_v is not None:
        print("=== cross_attn mask (upstream: cross_attn_mask, vLLM: cross_attn_cond_mask) ===")
        print(_diff_report(cam_u, cam_v, "mask"))
    else:
        print(f"=== Masks: upstream={type(cam_u)} vLLM={type(cam_v)} ===")

    if args.per_modality_cond_diff:
        print("=== Per-modality ct_u vs ct_v (before MAF / concat in get_conditioning_inputs) ===")
        for key in ("video_prompt", "text_prompt", "audio_prompt"):
            if key not in ct_u or key not in ct_v:
                print(f"  {key}: missing on one side, skip")
                continue
            u_pair, v_pair = ct_u[key], ct_v[key]
            if not isinstance(u_pair, (list, tuple)) or not isinstance(v_pair, (list, tuple)):
                print(f"  {key}: unexpected type, skip")
                continue
            tu, tv = u_pair[0], v_pair[0]
            print(_diff_report(tu, tv, f"{key}.tensor"))
            if len(u_pair) > 1 and len(v_pair) > 1:
                mu, mv = u_pair[1], v_pair[1]
                print(_diff_report(mu, mv, f"{key}.mask"))

    if args.skip_denoiser:
        return

    # Latent shape for diffusion (matches generate_diffusion_cond)
    pt = vllm.pretransform
    if pt is None:
        raise SystemExit("vLLM model missing pretransform")
    down = int(pt.downsampling_ratio)
    latent_t = sample_size // down
    io_ch = int(vllm.io_channels)

    if args.noise_pt is not None:
        x = torch.load(args.noise_pt, map_location=device, weights_only=True)
        if not isinstance(x, torch.Tensor):
            raise SystemExit("--noise-pt must contain a tensor")
        x = x.to(device=device, dtype=torch.float32)
    else:
        g = torch.Generator(device=device)
        g.manual_seed(int(args.noise_seed))
        x = torch.randn(1, io_ch, latent_t, device=device, generator=g, dtype=torch.float32)
        if args.save_noise_pt:
            args.save_noise_pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(x.cpu(), args.save_noise_pt)
            print(f"Saved noise to {args.save_noise_pt}")

    t = torch.full((1,), float(args.sigma), device=device, dtype=torch.float32)

    with torch.no_grad():
        den_u = upstream(x, t, ct_u)
        den_v = vllm(x, t, ct_v)

    print("=== Single denoiser step (same x, t, respective conditioning_tensors) ===")
    print(_diff_report(den_u, den_v, "denoised"))
    print("=== denoised upstream stats ===")
    print(_tensor_stats(den_u))
    print("=== denoised vLLM stats ===")
    print(_tensor_stats(den_v))

    if args.vllm_forward_upstream_raw_ct:
        with torch.no_grad():
            den_v_ctu = vllm(x, t, ct_u)
        print("=== vLLM forward with upstream conditioner dict ct_u (vLLM MAF + DiT) ===")
        print(_diff_report(den_u, den_v_ctu, "denoised_upstream_vs_vllm_ct_u"))
        print(_diff_report(den_v, den_v_ctu, "denoised_vllm_ct_v_vs_vllm_ct_u"))
        print("=== denoised vllm(ct_u) stats ===")
        print(_tensor_stats(den_v_ctu))

    if args.inject_upstream_cross_attn:
        gi_u = upstream.get_conditioning_inputs(ct_u)
        gac = gi_u.get("cross_attn_cond")
        gam = gi_u.get("cross_attn_mask")
        if gac is None or gam is None:
            raise SystemExit("inject-upstream-cross-attn: upstream get_conditioning_inputs missing cross tensors")
        extra: dict[str, Any] = {}
        gglob = gi_u.get("global_cond")
        if gglob is not None:
            extra["global_embed"] = gglob
        with torch.no_grad():
            den_v_inj = vllm.model(
                x,
                t,
                cross_attn_cond=gac,
                cross_attn_cond_mask=gam,
                cfg_scale=1.0,
                **extra,
            )
        print("=== vLLM inner DiT with upstream fused cross_attn (same tensors as upstream forward) ===")
        print(_diff_report(den_u, den_v_inj, "denoised_upstream_vs_vllm_upstream_cross_injected"))
        print(_diff_report(den_v, den_v_inj, "denoised_vllm_full_vs_vllm_upstream_cross_injected"))
        print("=== denoised vLLM (upstream cross injected) stats ===")
        print(_tensor_stats(den_v_inj))

    if args.dit_dummy:
        # Inner modules only: upstream wraps MMDiffusionTransformer in MMDiTWrapper (cross_attn_mask kw).
        if args.dit_dummy_source == "upstream":
            cac_d = cac_u.detach().clone()
            cam_d = cam_u.detach().clone() if cam_u is not None else None
            src_note = "cloned upstream cross_attn_cond / mask on BOTH sides"
        else:
            g = torch.Generator(device=device)
            g.manual_seed(int(args.dit_dummy_seed))
            cac_d = torch.randn(cac_u.shape, device=device, dtype=torch.float32, generator=g)
            cam_d = (
                torch.ones(cam_u.shape, device=device, dtype=cam_u.dtype) if cam_u is not None else None
            )
            src_note = f"randn / ones (--dit-dummy-seed={args.dit_dummy_seed})"

        with torch.no_grad():
            den_ud = upstream.model(
                x,
                t,
                cross_attn_cond=cac_d,
                cross_attn_mask=cam_d,
                cfg_scale=1.0,
            )
            den_vd = vllm.model(
                x,
                t,
                cross_attn_cond=cac_d,
                cross_attn_cond_mask=cam_d,
                cfg_scale=1.0,
            )

        print("=== DiT-only forward (identical conditioning tensors on both sides) ===")
        print(f"    source: {src_note}")
        print(_diff_report(den_ud, den_vd, "denoised_dit_only"))
        print("=== denoised_dit_only upstream stats ===")
        print(_tensor_stats(den_ud))
        print("=== denoised_dit_only vLLM stats ===")
        print(_tensor_stats(den_vd))


if __name__ == "__main__":
    main()
