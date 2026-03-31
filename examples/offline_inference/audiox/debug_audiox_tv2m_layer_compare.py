#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Layer-by-layer conditioning compare for **tv2m** (upstream pip AudioX vs vLLM-Omni).

Builds the same batch as ``run_audiox_bench.py`` (``videos/tv2m.mp4``, ``PR_DRAFT_PROMPTS``),
runs both conditioners with ``--conditioner-seed`` (default 42) before each full encode so VAE
RNG matches, then reports **max_abs / mean_abs / rmse** at:

1. **Per-modality outputs** — ``video_prompt``, ``text_prompt``, ``audio_prompt`` (tensor + mask).
2. **Fused cross-attention** — ``get_conditioning_inputs`` output (after MAF).
3. **Optional submodules** — CLIP ``last_hidden_state`` after ``visual_encoder_model``, output of
   ``Temp_transformer`` (video path only), via forward hooks.

On a real **tv2m** batch (same pixels as the bench), measured **max_abs** ordering is typically:
**``video.temporal_transformer_out``** > **``video.clip_visual_encoder_last_hidden``** >
**``video_prompt.tensor``** ≈ **``fused.cross_attn_cond``**, with **``text_prompt``** at ~**1e-3** and
silent **``audio_prompt``** matching. Video pixels are shared via ``prepare_video_reference``; the
divergence is **inside** CLIP + temporal self-attention, not the MP4 decode.

Usage (repo root on ``PYTHONPATH``)::

    export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
    python examples/offline_inference/audiox/debug_audiox_tv2m_layer_compare.py \\
        --weights-dir examples/offline_inference/audiox/audiox_weights \\
        --out-json examples/offline_inference/audiox/output/debug_tv2m/layer_report.json

Requires pip **audiox** and **vLLM-Omni** in the same env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Repo root + this dir
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
for _p in (_REPO_ROOT, _HERE):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Reuse bench prompts / video layout
import run_audiox_bench  # noqa: E402


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


def _diff_stats(a: torch.Tensor, b: torch.Tensor, name: str) -> dict[str, Any]:
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
    }


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().float().cpu()
    std = float(x.std(unbiased=False)) if x.numel() > 1 else 0.0
    return {
        "shape": list(x.shape),
        "mean": float(x.mean()),
        "std": std,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _build_tv2m_batch(base: Path, device: torch.device) -> list[dict[str, Any]]:
    from vllm_omni.diffusion.media.video import prepare_video_reference

    cfg_path = base / "audiox_weights" / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    sample_rate = int(cfg["sample_rate"])
    sample_size = int(cfg["sample_size"])
    target_fps = int(cfg.get("video_fps", 5))
    seconds_input = sample_size / float(sample_rate)
    clip_sec = 10.0

    video_path = str(base / "videos" / "tv2m.mp4")
    if not Path(video_path).is_file():
        raise SystemExit(f"Missing {video_path}")

    vt = prepare_video_reference(
        video_path,
        duration=float(clip_sec),
        target_fps=target_fps,
        seek_time=0.0,
    ).to(device=device, dtype=torch.float32)
    sync = torch.zeros(1, 240, 768, device=device, dtype=torch.float32)
    audio_tensor = torch.zeros(2, int(sample_rate * clip_sec), device=device, dtype=torch.float32)
    text = run_audiox_bench.PR_DRAFT_PROMPTS["tv2m"]

    return [
        {
            "video_prompt": {"video_tensors": vt.unsqueeze(0), "video_sync_frames": sync},
            "text_prompt": text,
            "audio_prompt": audio_tensor.unsqueeze(0),
            "seconds_start": 0.0,
            "seconds_total": seconds_input,
        }
    ]


def _register_clip_hooks(
    video_module: nn.Module,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Hooks on CLIP encoder and temporal transformer. Returns (captures list, handles)."""
    captures: list[dict[str, Any]] = []
    handles: list[Any] = []

    def _hook_clip(_m, _inp, out):
        if hasattr(out, "last_hidden_state"):
            captures.append({"stage": "clip_visual_encoder_last_hidden", "tensor": out.last_hidden_state.detach()})
        else:
            captures.append({"stage": "clip_visual_encoder_out", "tensor": out})

    def _hook_temp(_m, _inp, out):
        captures.append({"stage": "temporal_transformer_out", "tensor": out.detach()})

    if hasattr(video_module, "visual_encoder_model"):
        handles.append(video_module.visual_encoder_model.register_forward_hook(_hook_clip))
    if hasattr(video_module, "Temp_transformer"):
        handles.append(video_module.Temp_transformer.register_forward_hook(_hook_temp))
    return captures, handles


def main() -> None:
    if not os.environ.get("DIFFUSION_ATTENTION_BACKEND", "").strip():
        os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

    p = argparse.ArgumentParser(description="tv2m layer-by-layer upstream vs vLLM conditioning compare")
    p.add_argument("--weights-dir", type=Path, default=None)
    p.add_argument("--conditioner-seed", type=int, default=42)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    base = _HERE
    weights_dir = args.weights_dir or (base / "audiox_weights")
    cfg_path = weights_dir / "config.json"
    ckpt_path = weights_dir / "model.ckpt"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        raise SystemExit(f"Need {cfg_path} and {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    _fix_vae_ckpt_paths(cfg, weights_dir)

    batch = _build_tv2m_batch(base, device)

    report: dict[str, Any] = {
        "task": "tv2m",
        "conditioner_seed": int(args.conditioner_seed),
        "device": str(device),
        "stages": [],
    }

    # ---- Upstream ----
    from audiox.models.diffusion import create_diffusion_cond_from_config as ax_create
    from audiox.models.utils import load_ckpt_state_dict as ax_load_ckpt
    from audiox.training.utils import copy_state_dict as ax_copy

    upstream = ax_create(cfg).to(device).eval()
    ax_copy(upstream, ax_load_ckpt(str(ckpt_path)))

    # Hooks must run on the *same* full conditioner forward that produces ct_u / ct_v
    # (a separate video-only forward can disagree with ct_* on some stacks).
    up_video = upstream.conditioner.conditioners["video_prompt"]
    up_captures, up_hook_handles = _register_clip_hooks(up_video)
    try:
        torch.manual_seed(int(args.conditioner_seed))
        ct_u = upstream.conditioner(batch, device)
    finally:
        for h in up_hook_handles:
            h.remove()
    up_stage = {c["stage"]: c["tensor"] for c in up_captures}

    gi_u = upstream.get_conditioning_inputs(ct_u)
    cac_u = gi_u["cross_attn_cond"]
    cam_u = gi_u.get("cross_attn_mask")

    # ---- vLLM-Omni ----
    from unittest.mock import patch

    import vllm.model_executor.parameter as vllm_param  # noqa: PLC0415

    from vllm_omni.diffusion.models.audiox.audiox_conditioner import encode_audiox_conditioning_tensors
    from vllm_omni.diffusion.models.audiox.audiox_runtime import create_model_from_config
    from vllm_omni.diffusion.models.audiox.audiox_weights import (
        CONDITIONERS_SAFETENSORS,
        TRANSFORMER_SAFETENSORS,
        VAE_SAFETENSORS,
        _load_checkpoint_state_dict,
        load_audiox_weights,
    )

    class _Harness(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self._model = inner

    def _load_vllm_inner_weights(inner: nn.Module, wdir: Path) -> None:
        from safetensors.torch import load_file  # noqa: PLC0415

        chunks: list[tuple[str, torch.Tensor]] = []
        for rel in (TRANSFORMER_SAFETENSORS, CONDITIONERS_SAFETENSORS, VAE_SAFETENSORS):
            p = wdir / rel
            if p.is_file():
                chunks.extend(list(load_file(p).items()))
        if not chunks:
            chunks = list(_load_checkpoint_state_dict(str(wdir / "model.ckpt")).items())
        load_audiox_weights(_Harness(inner), chunks, model_config=cfg)
        inner.to(torch.float32).eval().requires_grad_(False)

    with _vllm_config_cm():
        with (
            patch.object(vllm_param, "get_tensor_model_parallel_rank", return_value=0),
            patch.object(vllm_param, "get_tensor_model_parallel_world_size", return_value=1),
        ):
            vllm = create_model_from_config(cfg, od_config=None).to(device).eval()
            _load_vllm_inner_weights(vllm, weights_dir)

    vl_video = vllm.conditioner.conditioners["video_prompt"]
    vl_captures, vl_hook_handles = _register_clip_hooks(vl_video)
    try:
        torch.manual_seed(int(args.conditioner_seed))
        ct_v = encode_audiox_conditioning_tensors(vllm.conditioner, batch_metadata=batch, device=device)
    finally:
        for h in vl_hook_handles:
            h.remove()
    vl_stage = {c["stage"]: c["tensor"] for c in vl_captures}

    gi_v = vllm.get_conditioning_inputs(ct_v)
    cac_v = gi_v["cross_attn_cond"]
    cam_v = gi_v.get("cross_attn_cond_mask")

    # ---- Submodule diffs (video) ----
    for stage_name in ("clip_visual_encoder_last_hidden", "temporal_transformer_out"):
        if stage_name in up_stage and stage_name in vl_stage:
            a, b = up_stage[stage_name], vl_stage[stage_name]
            st = _diff_stats(a, b, f"video.{stage_name}")
            st["stats_a"] = _tensor_stats(a)
            st["stats_b"] = _tensor_stats(b)
            report["stages"].append(st)

    # ---- Per-modality ----
    for key in ("video_prompt", "text_prompt", "audio_prompt"):
        tu, tv = ct_u[key], ct_v[key]
        if not isinstance(tu, (list, tuple)) or not isinstance(tv, (list, tuple)):
            report["stages"].append({"name": key, "error": "unexpected structure"})
            continue
        st_t = _diff_stats(tu[0], tv[0], f"{key}.tensor")
        st_t["stats_a"] = _tensor_stats(tu[0])
        st_t["stats_b"] = _tensor_stats(tv[0])
        report["stages"].append(st_t)
        if len(tu) > 1 and len(tv) > 1:
            st_m = _diff_stats(tu[1], tv[1], f"{key}.mask")
            st_m["stats_a"] = _tensor_stats(tu[1])
            st_m["stats_b"] = _tensor_stats(tv[1])
            report["stages"].append(st_m)

    # ---- Fused ----
    st_f = _diff_stats(cac_u, cac_v, "fused.cross_attn_cond")
    st_f["stats_a"] = _tensor_stats(cac_u)
    st_f["stats_b"] = _tensor_stats(cac_v)
    report["stages"].append(st_f)
    if cam_u is not None and cam_v is not None:
        st_m = _diff_stats(cam_u, cam_v, "fused.cross_attn_mask")
        report["stages"].append(st_m)

    # ---- Rank by max_abs ----
    ranked = []
    for s in report["stages"]:
        if "max_abs" in s:
            ranked.append((s["max_abs"], s["name"]))
    ranked.sort(reverse=True)
    report["ranked_by_max_abs"] = [{"name": n, "max_abs": float(m)} for m, n in ranked]
    report["largest_gap"] = report["ranked_by_max_abs"][0] if report["ranked_by_max_abs"] else None

    out_path = args.out_json or (base / "output" / "debug_tv2m" / "layer_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["ranked_by_max_abs"][:8], indent=2))
    print(f"Wrote {out_path}")
    if report.get("largest_gap"):
        print(f"Largest gap: {report['largest_gap']}")


if __name__ == "__main__":
    main()
