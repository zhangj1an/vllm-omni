# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark pip **AudioX** (upstream) vs **vLLM-Omni** (`AudioXPipeline` via Omni).

- **vLLM-Omni:** ``end2end.run_single_inference`` → ``output/vllm_omni/{task}.wav``
- **Upstream:** pip ``audiox`` ``generate_diffusion_cond`` → ``output/upstream_audiox/{task}.wav``

Shared defaults: seed **42**, **250** steps, **cfg_scale=7**, **sigma_min=0.3**, **sigma_max=500**,
**seconds_total=10** (conditioning / video clip), model-native **44100 Hz** WAV export for vLLM-Omni.

Run::

    cd /path/to/vllm-omni
    /path/to/.venv/bin/python examples/offline_inference/audiox/run_audiox_bench.py --backend both

Env (optional): ``AUDIOX_BENCH_BACKEND``, ``AUDIOX_BENCH_TASKS``,
``AUDIOX_VIDEO`` (one clip for all video tasks), or ``AUDIOX_VIDEO_TV2A`` / ``TV2M`` / ``V2A`` / ``V2M`` for per-task paths.
Prompts match ``pr_draft.md``; video-conditioned tasks default to ``videos/{task}.mp4`` (``--video-dir`` to override).

Requires: ``pip install -e '.[audiox]'`` and ``pip install 'git+https://github.com/ZeyueT/AudioX.git'``
for the upstream path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Literal

# Repo root (``vllm_omni``) + this dir (sibling ``end2end``).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
for _p in (_REPO_ROOT, _HERE):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import end2end  # noqa: E402

Backend = Literal["vllm", "upstream", "both"]

# Prompts aligned with ``pr_draft.md`` (Input Prompts table). ``v2a`` / ``v2m`` are video-only (empty text).
PR_DRAFT_PROMPTS: dict[str, str] = {
    "t2a": "Fireworks burst twice, followed by a period of silence before a clock begins ticking.",
    "t2m": "Uplifting ukulele tune for a travel vlog",
    "v2a": "",
    "v2m": "",
    "tv2a": "drum beating sound and human talking",
    "tv2m": "uplifting music matching the scene",
}

# Per-task clip under ``videos/`` (see ``pr_draft.md`` Sample Video Input). v2* and tv2* use distinct files.
_DEFAULT_VIDEOS_DIRNAME = "videos"


def _default_video_path_for_task(base: Path, task: str) -> str:
    return str(base / _DEFAULT_VIDEOS_DIRNAME / f"{task}.mp4")


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


def _ensure_sample_video(video_path: str) -> None:
    p = Path(video_path)
    if p.is_file():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = end2end.ROOT / "assets"
    end2end.download_sample_assets(assets_dir, trim_seconds=int(end2end.DEFAULT_RUN_CONFIG["assets"]["trim_seconds"]))
    if not p.is_file():
        raise SystemExit(
            f"Video required for bench but missing after download: {video_path}. "
            "Set AUDIOX_VIDEO to an existing MP4 or fix network/ffmpeg."
        )


def _build_cases(
    base: Path,
    task_filter: list[str] | None,
    *,
    video_overrides: dict[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """Each task gets ``pr_draft`` prompts and a per-task video path when applicable."""
    overrides = video_overrides or {}
    cases: list[tuple[str, str, str]] = []
    for task in end2end.ALL_TASKS_ORDERED:
        prompt = PR_DRAFT_PROMPTS.get(task, "")
        if task in end2end.VIDEO_TASKS:
            vid = overrides.get(task) or _default_video_path_for_task(base, task)
        else:
            vid = ""
        cases.append((task, prompt, vid))
    if task_filter:
        allow = {t.strip().lower() for t in task_filter if t.strip()}
        cases = [c for c in cases if c[0] in allow]
    return cases


def _metrics_row(
    *,
    task: str,
    latency: float,
    out_wav: Path,
) -> dict[str, Any]:
    with wave.open(str(out_wav), "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate())
    rtf = latency / duration if duration > 0 else None
    throughput = (1.0 / latency) if latency > 0 else None
    return {
        "task": task,
        "latency_s": latency,
        "duration_s": duration,
        "rtf": rtf,
        "ttft_s": latency,
        "tpot_s_per_token": latency,
        "itl_s_per_token": None,
        "output_throughput_tokens_per_s": throughput,
        "total_throughput_tokens_per_s": throughput,
        "efficiency_proxy": throughput,
        "output_wav": str(out_wav),
    }


def _summary(results: list[dict[str, Any]], n_cases: int) -> dict[str, Any]:
    n = len(results)
    return {
        "success_rate": f"100% ({n}/{n_cases} tasks)",
        "latency_s_avg": sum(r["latency_s"] for r in results) / n,
        "duration_s_avg": sum(r["duration_s"] for r in results) / n,
        "rtf_avg": sum(r["rtf"] for r in results) / n,
        "ttft_s_avg": sum(r["ttft_s"] for r in results) / n,
        "tpot_s_per_token_avg": sum(r["tpot_s_per_token"] for r in results) / n,
        "itl_s_per_token_avg": None,
        "output_throughput_avg": sum(r["output_throughput_tokens_per_s"] for r in results) / n,
        "total_throughput_avg": sum(r["total_throughput_tokens_per_s"] for r in results) / n,
        "efficiency_proxy_avg": sum(r["efficiency_proxy"] for r in results) / n,
    }


def run_vllm_omni(
    cases: list[tuple[str, str, str]],
    *,
    base: Path,
    out_dir: Path,
    weights: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    end2end._default_diffusion_attention_backend()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = Path(weights)
    model_sr = end2end._get_audiox_model_sample_rate(weights_path)
    results: list[dict[str, Any]] = []
    for task, prompt, video in cases:
        print(f"[vLLM-Omni] === RUN {task} ===", flush=True)
        out = out_dir / f"{task}.wav"
        latency = end2end.run_single_inference(
            model_dir=weights,
            task=task,
            prompt=prompt,
            video_path=video,
            reference_audio_path="",
            output=str(out),
            sample_rate=model_sr,
            seed=42,
            num_inference_steps=250,
            guidance_scale=7.0,
            seconds_start=0.0,
            seconds_total=10.0,
            negative_prompt="",
            enable_profiler=False,
            enable_cpu_offload=False,
        )
        results.append(_metrics_row(task=task, latency=latency, out_wav=out))
    summary = _summary(results, len(cases))
    metrics_path = out_dir / "metrics_vllm_omni.json"
    metrics_path.write_text(json.dumps({"results": results, "summary": summary}, indent=2), encoding="utf-8")
    print(f"[vLLM-Omni] Wrote {metrics_path}")
    return results, summary


def run_upstream(
    cases: list[tuple[str, str, str]],
    *,
    base: Path,
    out_dir: Path,
    weights_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from einops import rearrange

    from audiox.inference.generation import generate_diffusion_cond
    from audiox.models.factory import create_model_from_config
    from audiox.models.utils import load_ckpt_state_dict
    from audiox.training.utils import copy_state_dict

    from vllm_omni.diffusion.media.video import prepare_video_reference

    cfg_path = weights_dir / "config.json"
    ckpt_path = weights_dir / "model.ckpt"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        raise SystemExit(f"Missing {cfg_path} or {ckpt_path}.")

    with open(cfg_path, encoding="utf-8") as f:
        model_config = json.load(f)
    _fix_vae_ckpt_paths(model_config, weights_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(str(ckpt_path)))
    model.to(device).eval().requires_grad_(False)

    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])
    target_fps = int(model_config.get("video_fps", 5))
    seconds_input = sample_size / float(sample_rate)
    clip_sec = 10.0

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for task, prompt, video in cases:
        print(f"[upstream AudioX] === RUN {task} ===", flush=True)
        if task in end2end.VIDEO_TASKS and not str(video).strip():
            raise SystemExit(f"task {task} requires a video path.")

        if task in end2end.VIDEO_TASKS and video:
            vt = prepare_video_reference(
                video,
                duration=float(clip_sec),
                target_fps=target_fps,
                seek_time=0.0,
            ).to(device=device, dtype=torch.float32)
        else:
            vt = torch.zeros(int(target_fps * clip_sec), 3, 224, 224, device=device, dtype=torch.float32)

        audio_tensor = torch.zeros(2, int(sample_rate * clip_sec), device=device, dtype=torch.float32)
        sync = torch.zeros(1, 240, 768, device=device, dtype=torch.float32)
        text = prompt if task in end2end.TEXT_TASKS else ""

        conditioning = [
            {
                "video_prompt": {"video_tensors": vt.unsqueeze(0), "video_sync_frames": sync},
                "text_prompt": text,
                "audio_prompt": audio_tensor.unsqueeze(0),
                "seconds_start": 0.0,
                "seconds_total": seconds_input,
            }
        ]

        t0 = time.perf_counter()
        output = generate_diffusion_cond(
            model,
            steps=250,
            cfg_scale=7.0,
            conditioning=conditioning,
            sample_size=sample_size,
            seed=42,
            device=device,
            sampler_type="dpmpp-3m-sde",
            sigma_min=0.3,
            sigma_max=500,
        )
        latency = time.perf_counter() - t0

        output = rearrange(output, "b d n -> d (b n)")
        # Peak-normalize like upstream samples; avoid div-by-zero if the waveform is (near) silent.
        peak = torch.max(torch.abs(output.to(torch.float32))).clamp_min(1e-8)
        output = (
            (output.to(torch.float32) / peak)
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        pcm = output.numpy()
        if pcm.ndim != 2:
            raise RuntimeError(f"Expected 2D audio, got {pcm.shape}")
        nch, nframes = pcm.shape
        out_wav = out_dir / f"{task}.wav"
        interleaved = pcm.T.reshape(-1)
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(nch)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(interleaved.tobytes())
        print(f"[upstream AudioX] Saved {out_wav} ({latency:.2f}s)", flush=True)
        results.append(_metrics_row(task=task, latency=latency, out_wav=out_wav))

    summary = _summary(results, len(cases))
    metrics_path = out_dir / "metrics_upstream_audiox.json"
    metrics_path.write_text(json.dumps({"results": results, "summary": summary}, indent=2), encoding="utf-8")
    print(f"[upstream AudioX] Wrote {metrics_path}")
    return results, summary


def main() -> None:
    p = argparse.ArgumentParser(description="AudioX benchmark: upstream pip vs vLLM-Omni.")
    p.add_argument(
        "--backend",
        choices=("vllm", "upstream", "both"),
        default=os.environ.get("AUDIOX_BENCH_BACKEND", "both"),
        help="Which stack to run (default: env AUDIOX_BENCH_BACKEND or both).",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma/space-separated task ids (default: all six, or AUDIOX_BENCH_TASKS).",
    )
    p.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing {task}.mp4 per video task (default: ./videos next to this script).",
    )
    args = p.parse_args()

    backend: Backend = args.backend.lower()  # type: ignore[assignment]
    base = _HERE
    weights = str(base / "audiox_weights")
    weights_dir = Path(weights)

    task_tokens: list[str] = []
    if args.tasks:
        task_tokens = [t for t in args.tasks.replace(",", " ").split() if t.strip()]
    elif os.environ.get("AUDIOX_BENCH_TASKS", "").strip():
        task_tokens = [t for t in os.environ["AUDIOX_BENCH_TASKS"].replace(",", " ").split() if t.strip()]

    video_dir = Path(args.video_dir).resolve() if args.video_dir else (base / _DEFAULT_VIDEOS_DIRNAME)
    video_overrides: dict[str, str] = {}
    for env_key, task in (
        ("AUDIOX_VIDEO_V2A", "v2a"),
        ("AUDIOX_VIDEO_V2M", "v2m"),
        ("AUDIOX_VIDEO_TV2A", "tv2a"),
        ("AUDIOX_VIDEO_TV2M", "tv2m"),
    ):
        p = os.environ.get(env_key, "").strip()
        if p:
            video_overrides[task] = str(Path(p).expanduser().resolve())

    legacy = os.environ.get("AUDIOX_VIDEO", "").strip()
    legacy_resolved = str(Path(legacy).expanduser().resolve()) if legacy else ""
    if legacy_resolved:
        for t in end2end.VIDEO_TASKS:
            if t not in video_overrides:
                video_overrides[t] = legacy_resolved

    for t in end2end.VIDEO_TASKS:
        if t not in video_overrides:
            video_overrides[t] = str((video_dir / f"{t}.mp4").resolve())

    cases = _build_cases(base, task_tokens if task_tokens else None, video_overrides=video_overrides)
    if not cases:
        raise SystemExit("No tasks selected (empty filter).")

    needs_video = any(t in end2end.VIDEO_TASKS for t, _, _ in cases)
    if needs_video:
        for t, _, vp in cases:
            if t not in end2end.VIDEO_TASKS:
                continue
            vp_path = Path(vp)
            if vp_path.is_file():
                continue
            # Optional download only for the default animal asset path used by ``end2end``.
            if vp_path.name == "sample_animal.mp4" and "assets" in vp_path.parts:
                _ensure_sample_video(str(vp_path))
            else:
                raise SystemExit(
                    f"Missing video file for task {t}: {vp}\n"
                    f"Place {t}.mp4 under {video_dir} or set AUDIOX_VIDEO or AUDIOX_VIDEO_{t.upper()}."
                )

    out_upstream = base / "output" / "upstream_audiox"
    out_vllm = base / "output" / "vllm_omni"

    combined: dict[str, Any] = {
        "tasks": [c[0] for c in cases],
        "num_inference_steps": 250,
        "guidance_scale": 7.0,
        "seed": 42,
        "sigma_min": 0.3,
        "sigma_max": 500.0,
    }

    if backend in ("upstream", "both"):
        _, su = run_upstream(cases, base=base, out_dir=out_upstream, weights_dir=weights_dir)
        combined["upstream_audiox"] = {
            "metrics_path": str(out_upstream / "metrics_upstream_audiox.json"),
            "summary": su,
        }
    else:
        combined["upstream_audiox"] = {"skipped": True}

    if backend in ("vllm", "both"):
        try:
            _ensure_audiox_sharded_weights = end2end._ensure_audiox_sharded_weights
            _audiox_bundle_is_sharded = end2end._audiox_bundle_is_sharded
            wd = Path(weights).resolve()
            if not _audiox_bundle_is_sharded(wd):
                ckpt = wd / "model.ckpt"
                if ckpt.is_file():
                    _ensure_audiox_sharded_weights(wd)
        except Exception as e:
            print(f"[vLLM-Omni] weight prep: {e}", flush=True)

        _, sv = run_vllm_omni(cases, base=base, out_dir=out_vllm, weights=weights)
        combined["vllm_omni"] = {
            "metrics_path": str(out_vllm / "metrics_vllm_omni.json"),
            "summary": sv,
        }
    else:
        combined["vllm_omni"] = {"skipped": True}

    agg_path = base / "output" / "metrics_audiox_bench.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Wrote aggregate metrics to {agg_path}")


if __name__ == "__main__":
    main()
