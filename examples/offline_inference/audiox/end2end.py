# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end AudioX offline example: Hugging Face weight download, sample video assets, and Omni inference.

Weights are fetched from **Hugging Face** (same repos as ``HKUSTAudio/AudioX*``). Inference code is
inlined under ``vllm_omni.diffusion.models.audiox`` (no separate AudioX clone).

Install extra Python deps from the repo root: ``pip install -e ".[audiox]"``
(see ``README.md``).
Unless ``DIFFUSION_ATTENTION_BACKEND`` is set, this script defaults it to ``TORCH_SDPA`` so
``infer`` / ``run`` work when Flash / fa3-fwd rejects FP16 on the current GPU.

Typical flow::

    cd examples/offline_inference/audiox
    python end2end.py run

Or use ``./run_audiox_sample_task.sh`` (sets ``PYTHONPATH`` and runs the same command).

Single-task debugging::

    python end2end.py infer --model ./audiox_weights --task t2a --prompt "…"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

# --- Hugging Face model id (weights only) ---
DEFAULT_HF_MODEL_KEY = "maf-mmdit"
REPO_BY_MODEL: dict[str, str] = {
    DEFAULT_HF_MODEL_KEY: "HKUSTAudio/AudioX-MAF-MMDiT",
}

# --- Sample videos (Pexels); see https://www.pexels.com/license/ ---
PEXELS_SAMPLE_ANIMAL_URL = "https://www.pexels.com/download/video/5871756/"
_FFMPEG_PEXELS_HEADERS = (
    "Referrer: https://www.pexels.com/\r\n"
    "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36\r\n"
)

# --- Prompts for bundled animal clip runs ---
SAMPLE_PROMPTS: dict[str, str] = {
    "t2a": "Soft indoor room tone, faint fabric rustle, and gentle cat breathing or purr in a quiet home",
    "t2m": "Playful light acoustic or piano motif for an intimate close-up of a curious tabby cat indoors",
    "v2a": "",
    "v2m": "",
    "tv2a": "Quiet domestic ambience—upholstery, subtle shifts, and close-up foley matching a tabby cat on soft grey fabric",
    "tv2m": "Warm whimsical instrumental for a cute tabby cat close-up with playful upside-down framing",
}

# --- Inlined default run config (previously configs/animal.json) ---
DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "weights": {
        "hf_model": DEFAULT_HF_MODEL_KEY,
        "local_dir": "audiox_weights",
        "full": True,
        "download_if_missing": True,
    },
    "assets": {
        "local_dir": "assets",
        "trim_seconds": 5,
        "download_if_missing": True,
    },
    "run": {
        "tasks": None,
        "output_root": "audiox_task_outputs",
        "model_slug": None,
        "num_inference_steps": 50,
        "seconds_total": 2.0,
        "guidance_scale": 6.0,
        "seed": 42,
        "sample_rate": 48000,
        "enable_cpu_offload": False,
        "enable_diffusion_pipeline_profiler": False,
    },
}

VIDEO_TASKS = frozenset({"v2a", "v2m", "tv2a", "tv2m"})
TEXT_VIDEO_TASKS = frozenset({"tv2a", "tv2m"})
TEXT_TASKS = frozenset({"t2a", "t2m", "tv2a", "tv2m"})
ALL_TASKS_ORDERED = ("t2a", "t2m", "v2a", "v2m", "tv2a", "tv2m")


def _timed_hf_download(repo_id: str, local_dir: Path, *, allow_patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    if local_dir.is_dir() and any(local_dir.iterdir()):
        print(f"Directory {local_dir} is non-empty. HF may skip existing files.")
    print(f"Downloading {repo_id} -> {local_dir}")
    t0 = time.perf_counter()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print(f"Finished in {time.perf_counter() - t0:.1f}s")


def _ensure_transformer_stub_config(output_dir: Path) -> None:
    """HF README / tooling sometimes expect ``transformer/config.json`` under the bundle."""
    tf_dir = output_dir / "transformer"
    tf_dir.mkdir(parents=True, exist_ok=True)
    stub = tf_dir / "config.json"
    if not stub.is_file():
        stub.write_text("{}\n", encoding="utf-8")
        print(f"Wrote {stub}")


def _audiox_bundle_is_sharded(weights_dir: Path) -> bool:
    from vllm_omni.diffusion.models.audiox.audiox_weights import resolve_audiox_bundle_paths

    try:
        resolve_audiox_bundle_paths(str(weights_dir.resolve()))
        return True
    except (OSError, ValueError):
        return False


def _get_audiox_model_sample_rate(weights_dir: Path) -> int:
    from vllm_omni.diffusion.models.audiox.audiox_weights import load_audiox_bundle_config

    _, model_cfg, _ = load_audiox_bundle_config(str(weights_dir.resolve()))
    return int(model_cfg.get("sample_rate", 48000))


def _ensure_audiox_sharded_weights(weights_dir: Path) -> None:
    """vLLM-Omni loads only component safetensors; convert ``model.ckpt`` in place when needed."""
    from vllm_omni.diffusion.models.audiox.audiox_weights import convert_audiox_bundle

    if _audiox_bundle_is_sharded(weights_dir):
        return
    ckpt = weights_dir / "model.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"No sharded weights and no model.ckpt under {weights_dir} "
            "(download HF bundle or run audiox_weights conversion on a checkpoint directory)."
        )
    print(f"Converting {ckpt.name} to vLLM-Omni sharded safetensors under {weights_dir} …")
    convert_audiox_bundle(str(weights_dir.resolve()), str(weights_dir.resolve()), copy_config=True)
    if not _audiox_bundle_is_sharded(weights_dir):
        raise RuntimeError(f"Sharded layout still incomplete after conversion under {weights_dir}")


def download_weights(
    *,
    hf_model: str,
    output_dir: Path,
    full: bool,
    repo_id: str | None = None,
) -> None:
    """Download checkpoint bundle from Hugging Face and write vLLM-Omni index files."""
    if hf_model != DEFAULT_HF_MODEL_KEY:
        raise SystemExit(f"Unsupported hf_model={hf_model!r}. This example only supports {DEFAULT_HF_MODEL_KEY!r}.")
    repo_id = repo_id or REPO_BY_MODEL.get(hf_model, hf_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    allow = ["config.json", "model.ckpt", "README.md", ".gitattributes"]
    if full:
        allow.extend(["VAE.ckpt"])
    _timed_hf_download(repo_id, output_dir, allow_patterns=allow)
    _ensure_transformer_stub_config(output_dir)
    _ensure_audiox_sharded_weights(output_dir)


def download_sample_assets(output_dir: Path, *, trim_seconds: int) -> None:
    """Stream Pexels clips via ffmpeg and trim to ``trim_seconds``."""
    pairs = [
        (PEXELS_SAMPLE_ANIMAL_URL, output_dir / "sample_animal.mp4"),
    ]
    for url, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Streaming and trimming {url} -> {dest} ({trim_seconds} s)")
        cmd = [
            "ffmpeg",
            "-y",
            "-headers",
            _FFMPEG_PEXELS_HEADERS,
            "-i",
            url,
            "-t",
            str(trim_seconds),
            "-c",
            "copy",
            str(dest),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr or proc.stdout, end="")
            proc.check_returncode()
        print(f"Saved {dest} ({dest.stat().st_size / (1024 * 1024):.2f} MB)")


def _save_audio(audio_data: np.ndarray, output_path: str, sample_rate: int) -> None:
    try:
        import soundfile as sf

        sf.write(output_path, audio_data, sample_rate)
    except ImportError:
        import scipy.io.wavfile as wav

        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        wav.write(output_path, sample_rate, audio_data)


def run_single_inference(
    *,
    model_dir: str,
    task: str,
    prompt: str,
    video_path: str,
    reference_audio_path: str,
    output: str,
    sample_rate: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    seconds_start: float,
    seconds_total: float | None,
    negative_prompt: str,
    enable_profiler: bool,
    enable_cpu_offload: bool,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
) -> float:
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    task = task.lower()
    if task in TEXT_TASKS and not str(prompt).strip():
        raise SystemExit(f"task {task} requires a non-empty prompt.")
    if task in VIDEO_TASKS and not str(video_path).strip():
        raise SystemExit(f"task {task} requires a video path.")

    wd = Path(os.path.abspath(os.path.expanduser(model_dir)))
    try:
        _ensure_audiox_sharded_weights(wd)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        raise SystemExit(str(e)) from e

    prompt_text = prompt if task in TEXT_TASKS else ""
    model_sample_rate = _get_audiox_model_sample_rate(wd)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    omni = Omni(
        model=os.path.abspath(os.path.expanduser(model_dir)),
        model_class_name="AudioXPipeline",
        enable_diffusion_pipeline_profiler=enable_profiler,
        enable_cpu_offload=enable_cpu_offload,
    )
    extra: dict = {"seconds_start": seconds_start}
    if seconds_total is not None:
        extra["seconds_total"] = float(seconds_total)
    if video_path.strip():
        extra["video_path"] = os.path.abspath(os.path.expanduser(video_path))
    extra["audiox_task"] = task
    if reference_audio_path.strip():
        extra["audio_path"] = os.path.abspath(os.path.expanduser(reference_audio_path))
    if sigma_min is not None:
        extra["sigma_min"] = float(sigma_min)
    if sigma_max is not None:
        extra["sigma_max"] = float(sigma_max)

    sampling = OmniDiffusionSamplingParams(
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        extra_args=extra,
    )
    user_prompt: str | dict
    if negative_prompt.strip():
        user_prompt = {"prompt": prompt_text, "negative_prompt": negative_prompt}
    else:
        user_prompt = prompt_text

    t0 = time.perf_counter()
    outputs = omni.generate(user_prompt, sampling)
    elapsed = time.perf_counter() - t0
    omni.close()

    if not outputs:
        raise RuntimeError("No output from omni.generate().")
    out = outputs[0]
    ro = out.request_output
    mm = getattr(ro, "multimodal_output", None) if ro is not None else None
    if mm is None:
        mm = out.multimodal_output
    audio = mm.get("audio") if isinstance(mm, dict) else None
    if audio is None:
        raise RuntimeError("No audio in multimodal_output.")
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.T
    elif audio.ndim != 2:
        raise RuntimeError(f"Unexpected audio shape {audio.shape}")

    # Preserve duration when output sample rate differs from model-native rate.
    if model_sample_rate != sample_rate:
        from vllm_omni.diffusion.models.audiox.pipeline_audiox import resample_audiox_waveform_poly

        audio = resample_audiox_waveform_poly(audio, src_rate=model_sample_rate, dst_rate=sample_rate)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_audio(audio, str(out_path), sample_rate)
    print(f"Saved {out_path} (generation {elapsed:.2f}s)")
    return elapsed


def _task_is_allowed(q: str) -> bool:
    return q in ALL_TASKS_ORDERED


def _parse_requested_tasks_from_env() -> list[str] | None:
    """Return explicit task list from AUDIOX_TASKS / AUDIOX_TASKS_FILE, or None to use config."""
    requested: list[str] = []

    def append_token(raw: str) -> None:
        t = raw.strip().lower()
        if not t:
            return
        if not _task_is_allowed(t):
            raise SystemExit(f"Unknown task: {raw} (allowed: {ALL_TASKS_ORDERED})")
        if t not in requested:
            requested.append(t)

    tfile = os.environ.get("AUDIOX_TASKS_FILE", "").strip()
    if tfile:
        p = Path(tfile)
        if not p.is_file():
            raise SystemExit(f"AUDIOX_TASKS_FILE not found: {tfile}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0]
            for tok in line.replace(",", " ").split():
                append_token(tok)

    ts = os.environ.get("AUDIOX_TASKS", "").strip()
    if ts:
        for tok in ts.replace(",", " ").split():
            append_token(tok)

    if tfile or ts:
        if not requested:
            raise SystemExit("AUDIOX_TASKS(_FILE) set but no tasks parsed.")
        run_order = [t for t in ALL_TASKS_ORDERED if t in requested]
        if not run_order:
            raise SystemExit("No valid tasks after env parse.")
        return run_order
    return None


def cmd_run(args: argparse.Namespace) -> None:
    cfg = DEFAULT_RUN_CONFIG

    w = cfg.get("weights") or {}
    a = cfg.get("assets") or {}
    r = cfg.get("run") or {}

    hf_model = w.get("hf_model", DEFAULT_HF_MODEL_KEY)
    weights_dir = Path(w.get("local_dir", "audiox_weights"))
    if not weights_dir.is_absolute():
        weights_dir = ROOT / weights_dir
    full = bool(w.get("full", True))
    dl_w = bool(w.get("download_if_missing", True)) and not args.skip_download_weights
    repo_override = (w.get("repo_id") or "").strip() or None

    model_override = os.environ.get("AUDIOX_MODEL", "").strip()
    variant = os.environ.get("AUDIOX_MODEL_VARIANT", "").strip().lower()
    if variant and not model_override:
        if variant != DEFAULT_HF_MODEL_KEY:
            raise SystemExit(f"AUDIOX_MODEL_VARIANT={variant!r} is not supported. Use {DEFAULT_HF_MODEL_KEY!r}.")
        hf_model = DEFAULT_HF_MODEL_KEY
        weights_dir = ROOT / "audiox_weights"

    if hf_model != DEFAULT_HF_MODEL_KEY:
        raise SystemExit(f"weights.hf_model={hf_model!r} is not supported. Use {DEFAULT_HF_MODEL_KEY!r}.")

    assets_dir = Path(a.get("local_dir", "assets"))
    if not assets_dir.is_absolute():
        assets_dir = ROOT / assets_dir
    trim_sec = int(a.get("trim_seconds", 5))
    dl_a = bool(a.get("download_if_missing", True)) and not args.skip_download_assets

    clip = "animal"
    video_path = os.environ.get("AUDIOX_VIDEO", "").strip()
    if not video_path:
        video_path = str(assets_dir / f"sample_{clip}.mp4")

    if model_override:
        weights_dir = Path(os.path.abspath(os.path.expanduser(model_override)))
        dl_w = False

    ckpt = weights_dir / "model.ckpt"
    if not _audiox_bundle_is_sharded(weights_dir):
        if dl_w and not ckpt.is_file():
            download_weights(
                hf_model=hf_model,
                output_dir=weights_dir,
                full=full,
                repo_id=repo_override,
            )
        elif ckpt.is_file():
            try:
                _ensure_audiox_sharded_weights(weights_dir)
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                raise SystemExit(str(e)) from e
        else:
            raise SystemExit(
                f"Missing AudioX weights under {weights_dir}. Expected either component-sharded "
                f"safetensors (transformer/ + conditioners/) or model.ckpt to convert. "
                f"Run with download_if_missing true or set AUDIOX_MODEL.\n"
                f"HF model key in config: {hf_model!r}"
            )

    v2_needed = False
    tasks_cfg = r.get("tasks")
    env_tasks = _parse_requested_tasks_from_env()
    if env_tasks is not None:
        run_tasks = env_tasks
    elif tasks_cfg is None:
        run_tasks = list(ALL_TASKS_ORDERED)
    else:
        run_tasks = [str(t).lower() for t in tasks_cfg]
        for t in run_tasks:
            if not _task_is_allowed(t):
                raise SystemExit(f"Unknown task in config run.tasks: {t!r} (allowed: {ALL_TASKS_ORDERED})")
    for t in run_tasks:
        if t in VIDEO_TASKS:
            v2_needed = True

    if v2_needed:
        vp = Path(video_path)
        if not vp.is_file():
            if dl_a:
                download_sample_assets(assets_dir, trim_seconds=trim_sec)
            if not vp.is_file():
                raise SystemExit(f"Video required but missing: {video_path}")

    model_slug = os.environ.get("AUDIOX_PR_MODEL_SLUG", "").strip() or r.get("model_slug")
    if not model_slug:
        model_slug = hf_model

    out_root = Path(r.get("output_root", "audiox_task_outputs"))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_dir = os.environ.get("AUDIOX_OUT_DIR", "").strip()
    if out_dir:
        final_out = Path(os.path.abspath(os.path.expanduser(out_dir)))
    else:
        task_root = os.environ.get("AUDIOX_TASK_OUTPUT_ROOT", "").strip()
        if task_root:
            out_root = ROOT / task_root
        final_out = out_root / model_slug / clip
    final_out.mkdir(parents=True, exist_ok=True)

    steps = int(os.environ.get("AUDIOX_STEPS", r.get("num_inference_steps", 50)))
    sec_total = float(os.environ.get("AUDIOX_SECONDS", r.get("seconds_total", 2.0)))
    gscale = float(r.get("guidance_scale", 6.0))
    seed = int(r.get("seed", 42))
    sr = int(r.get("sample_rate", 48000))
    prof = bool(r.get("enable_diffusion_pipeline_profiler", False))
    offload = bool(r.get("enable_cpu_offload", False))

    print(
        f"model_slug={model_slug} clip={clip} tasks={' '.join(run_tasks)} "
        f"weights={weights_dir} video={video_path} out={final_out}"
    )

    for task in run_tasks:
        print(f"=== {task} ===", flush=True)
        prompt = SAMPLE_PROMPTS.get(task, "")
        vid_arg = video_path if task in VIDEO_TASKS else ""
        run_single_inference(
            model_dir=str(weights_dir),
            task=task,
            prompt=prompt,
            video_path=vid_arg,
            reference_audio_path="",
            output=str(final_out / f"{task}.wav"),
            sample_rate=sr,
            seed=seed,
            num_inference_steps=steps,
            guidance_scale=gscale,
            seconds_start=0.0,
            seconds_total=sec_total,
            negative_prompt="",
            enable_profiler=prof,
            enable_cpu_offload=offload,
        )
    print(f"Wrote WAVs under {final_out}")


def cmd_infer(args: argparse.Namespace) -> None:
    run_single_inference(
        model_dir=args.model,
        task=args.task,
        prompt=args.prompt,
        video_path=args.video_path,
        reference_audio_path=args.reference_audio_path,
        output=args.output,
        sample_rate=args.sample_rate,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seconds_start=args.seconds_start,
        seconds_total=args.seconds_total,
        negative_prompt=args.negative_prompt,
        enable_profiler=args.enable_diffusion_pipeline_profiler,
        enable_cpu_offload=args.enable_cpu_offload,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AudioX offline end-to-end (HF weights + Omni).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run inlined animal sample config; download assets/weights if needed.")
    pr.add_argument("--skip-download-weights", action="store_true")
    pr.add_argument("--skip-download-assets", action="store_true")
    pr.set_defaults(func=cmd_run)

    pi = sub.add_parser("infer", help="Single-task inference (CLI flags).")
    pi.add_argument("--model", type=str, required=True)
    pi.add_argument("--task", type=str, required=True, choices=sorted(VIDEO_TASKS | TEXT_TASKS))
    pi.add_argument("--prompt", type=str, default="")
    pi.add_argument("--negative-prompt", type=str, default="")
    pi.add_argument("--video-path", type=str, default="")
    pi.add_argument("--reference-audio-path", type=str, default="")
    pi.add_argument("--output", type=str, default="audiox_out.wav")
    pi.add_argument("--sample-rate", type=int, default=48000)
    pi.add_argument("--seed", type=int, default=42)
    pi.add_argument("--num-inference-steps", type=int, default=100)
    pi.add_argument("--guidance-scale", type=float, default=6.0)
    pi.add_argument("--seconds-start", type=float, default=0.0)
    pi.add_argument("--seconds-total", type=float, default=None)
    pi.add_argument("--enable-diffusion-pipeline-profiler", action="store_true")
    pi.add_argument("--enable-cpu-offload", action="store_true")
    pi.set_defaults(func=cmd_infer)

    return p


def _default_diffusion_attention_backend() -> None:
    """Avoid AudioX dummy-run failures when default FLASH_ATTN uses fa3-fwd without FP16 on this GPU."""
    if not os.environ.get("DIFFUSION_ATTENTION_BACKEND", "").strip():
        os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"


def main() -> None:
    _default_diffusion_attention_backend()
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
