#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run upstream vs vLLM-Omni AudioX conditioners on the same dummy batch (video / text / audio).

Loads ``config.json`` + weights from ``--weights-dir``, builds one batch with fixed shapes (same as
``compare_audiox_upstream_vllm.py``), runs ``upstream.conditioner`` and
``encode_audiox_conditioning_tensors(vllm.conditioner, ...)``, then prints per-modality tensor/mask
diffs and fused ``cross_attn_cond`` / mask diffs.

Usage::

    export PYTHONPATH=/path/to/vllm-omni
    export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
    python examples/offline_inference/audiox/dummy_conditioners_compare_upstream_vllm.py

Use ``--dummy-seed`` to fill video / audio / sync tensors with reproducible ``randn`` instead of zeros.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import nn


def _load_compare_helpers():
    """Load private helpers from ``compare_audiox_upstream_vllm.py`` (same directory)."""
    here = Path(__file__).resolve().parent
    path = here / "compare_audiox_upstream_vllm.py"
    if not path.is_file():
        raise SystemExit(f"Missing sibling script: {path}")
    spec = importlib.util.spec_from_file_location("audiox_upstream_vllm_compare", path)
    if spec is None or spec.loader is None:
        raise SystemExit("importlib failed to load compare_audiox_upstream_vllm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_dummy_batch_seeded(
    *,
    device: torch.device,
    fixed_text: str,
    sample_rate: int,
    sample_size: int,
    video_fps: int,
    clip_duration_sec: float,
    audio_cond_samples: int,
    dummy_seed: int | None,
) -> list[dict[str, Any]]:
    """Same layout as ``compare_audiox_upstream_vllm._build_dummy_batch``; optional randn fill."""
    seconds_model = float(sample_size) / float(sample_rate)
    clip_frames = int(round(clip_duration_sec * video_fps))
    if dummy_seed is None:
        video = torch.zeros(clip_frames, 3, 224, 224, device=device, dtype=torch.float32)
        audio = torch.zeros(2, audio_cond_samples, device=device, dtype=torch.float32)
        sync = torch.zeros(1, 240, 768, device=device, dtype=torch.float32)
    else:
        g = torch.Generator(device=device)
        g.manual_seed(int(dummy_seed))
        video = torch.randn(clip_frames, 3, 224, 224, device=device, dtype=torch.float32, generator=g)
        audio = torch.randn(2, audio_cond_samples, device=device, dtype=torch.float32, generator=g)
        sync = torch.randn(1, 240, 768, device=device, dtype=torch.float32, generator=g)
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
    p = argparse.ArgumentParser(
        description="Dummy batch → upstream vs vLLM-Omni conditioner outputs (per modality + fused cross_attn)."
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory with config.json + model.ckpt (default: ./audiox_weights next to this script)",
    )
    p.add_argument("--text", type=str, default="dummy caption for t2a conditioner compare.")
    p.add_argument(
        "--dummy-seed",
        type=int,
        default=None,
        help="If set, video / audio / sync tensors use randn with this seed; else zeros (deterministic).",
    )
    p.add_argument(
        "--clip-duration-sec",
        type=float,
        default=10.0,
        help="Clip length used to compute video frame count (same as compare script default).",
    )
    args = p.parse_args()

    cmp = _load_compare_helpers()
    _fix_vae_ckpt_paths = cmp._fix_vae_ckpt_paths
    _audio_conditioning_samples = cmp._audio_conditioning_samples
    _diff_report = cmp._diff_report
    _tensor_stats = cmp._tensor_stats
    _vllm_config_cm = cmp._vllm_config_cm

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

    batch = _build_dummy_batch_seeded(
        device=device,
        fixed_text=args.text,
        sample_rate=sample_rate,
        sample_size=sample_size,
        video_fps=video_fps,
        clip_duration_sec=float(args.clip_duration_sec),
        audio_cond_samples=audio_cond_samples,
        dummy_seed=args.dummy_seed,
    )

    repo_root = here.parents[2]
    if str(repo_root) not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = f"{repo_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    # ---- Upstream (pip audiox) ----
    from audiox.models.diffusion import create_diffusion_cond_from_config as ax_create
    from audiox.models.utils import load_ckpt_state_dict as ax_load_ckpt
    from audiox.training.utils import copy_state_dict as ax_copy

    upstream = ax_create(cfg).to(device).eval()
    ax_copy(upstream, ax_load_ckpt(str(ckpt_path)))
    # Same global RNG before each full conditioner forward so VAE sampling order matches (video/text are deterministic).
    torch.manual_seed(42)
    with torch.no_grad():
        ct_u = upstream.conditioner(batch, device)
    gi_u = upstream.get_conditioning_inputs(ct_u)
    cac_u = gi_u["cross_attn_cond"]
    cam_u = gi_u.get("cross_attn_mask")

    # ---- vLLM-Omni ----
    from vllm_omni.diffusion.models.audiox.audiox_conditioner import encode_audiox_conditioning_tensors
    from vllm_omni.diffusion.models.audiox.audiox_runtime import create_model_from_config
    from vllm_omni.diffusion.models.audiox.audiox_weights import (
        CONDITIONERS_SAFETENSORS,
        TRANSFORMER_SAFETENSORS,
        VAE_SAFETENSORS,
        _load_checkpoint_state_dict,
        load_audiox_weights,
    )

    import vllm.model_executor.parameter as vllm_param  # noqa: PLC0415

    class _Harness(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self._model = inner

    def _load_vllm_inner_weights(inner: nn.Module, wdir: Path) -> None:
        from safetensors.torch import load_file  # noqa: PLC0415

        chunks: list[tuple[str, torch.Tensor]] = []
        for rel in (TRANSFORMER_SAFETENSORS, CONDITIONERS_SAFETENSORS, VAE_SAFETENSORS):
            fp = wdir / rel
            if fp.is_file():
                chunks.extend(list(load_file(fp).items()))
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

    torch.manual_seed(42)
    with torch.no_grad():
        ct_v = encode_audiox_conditioning_tensors(vllm.conditioner, batch_metadata=batch, device=device)
    gi_v = vllm.get_conditioning_inputs(ct_v)
    cac_v = gi_v["cross_attn_cond"]
    cam_v = gi_v.get("cross_attn_cond_mask")

    seed_note = f"dummy-seed={args.dummy_seed}" if args.dummy_seed is not None else "zeros (no seed)"
    print("=== Dummy batch ===")
    print(f"    text: {args.text!r}")
    print(f"    tensors: {seed_note}")
    print("=== Fused cross_attn_cond (upstream get_conditioning_inputs vs vLLM) ===")
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

    print("=== Per-modality conditioner outputs ct_u vs ct_v (video_prompt / text_prompt / audio_prompt) ===")
    for key in ("video_prompt", "text_prompt", "audio_prompt"):
        if key not in ct_u or key not in ct_v:
            print(f"  {key}: missing on one side, skip")
            continue
        u_pair, v_pair = ct_u[key], ct_v[key]
        if not isinstance(u_pair, (list, tuple)) or not isinstance(v_pair, (list, tuple)):
            print(f"  {key}: unexpected type {type(u_pair)}/{type(v_pair)}, skip")
            continue
        tu, tv = u_pair[0], v_pair[0]
        print(_diff_report(tu, tv, f"{key}.tensor"))
        if len(u_pair) > 1 and len(v_pair) > 1:
            mu, mv = u_pair[1], v_pair[1]
            print(_diff_report(mu, mv, f"{key}.mask"))

    print("Done.")


if __name__ == "__main__":
    main()
