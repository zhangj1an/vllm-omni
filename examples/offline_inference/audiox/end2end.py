# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end AudioX offline example covering the 6 t2*/v2*/tv2* tasks.

Provide a directory with the **vLLM-Omni AudioX safetensors bundle** (e.g. from
``zhangj1an/AudioX`` on Hugging Face). Inference code lives under
``vllm_omni.diffusion.models.audiox`` (no separate AudioX clone).

Typical use::

    huggingface-cli download zhangj1an/AudioX --local-dir ./audiox_weights
    python end2end.py --model ./audiox_weights
    python end2end.py --model ./audiox_weights --tasks t2a tv2a
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as TF

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

ROOT = Path(__file__).resolve().parent

SAMPLE_PROMPTS: dict[str, str] = {
    "t2a": "Fireworks burst twice, followed by a period of silence before a clock begins ticking.",
    "t2m": "Uplifting ukulele tune for a travel vlog",
    "v2a": "",
    "v2m": "",
    "tv2a": "drum beating sound and human talking",
    "tv2m": "uplifting music matching the scene",
}

# AudioX project-page sample clips. v2a/tv2a share one video; v2m/tv2m share another.
_V2A_URL = "https://zeyuet.github.io/AudioX/static/samples/samples-v2/V2A/JIemsK_0lXc_000364_compressed.mp4"
_V2M_URL = "https://zeyuet.github.io/AudioX/static/samples/V2M/1XeBotOFqHA.mp4"
SAMPLE_VIDEO_URLS: dict[str, str] = {
    "v2a": _V2A_URL,
    "tv2a": _V2A_URL,
    "v2m": _V2M_URL,
    "tv2m": _V2M_URL,
}
ALL_TASKS = ("t2a", "t2m", "v2a", "v2m", "tv2a", "tv2m")
VIDEO_TASKS = frozenset({"v2a", "v2m", "tv2a", "tv2m"})
TEXT_TASKS = frozenset({"t2a", "t2m", "tv2a", "tv2m"})

_REQUIRED_BUNDLE_FILES = ("config.json", "transformer/diffusion_pytorch_model.safetensors")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AudioX offline end-to-end (6 t2*/v2*/tv2* tasks).")
    p.add_argument("--model", type=str, default=str(ROOT / "audiox_weights"), help="Path to AudioX weight bundle.")
    p.add_argument("--tasks", nargs="+", default=list(ALL_TASKS), choices=ALL_TASKS, help="Subset of tasks to run.")
    p.add_argument(
        "--video-dir",
        type=str,
        default=str(ROOT / "assets"),
        help="Where to cache downloaded sample videos for v2*/tv2*.",
    )
    p.add_argument(
        "--reference-audio", type=str, default="", help="Optional audio clip for audio-conditioned generation."
    )
    p.add_argument("--output-dir", type=str, default=str(ROOT / "audiox_task_outputs"))
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--seconds-total", type=float, default=2.0)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rate", type=int, default=48000, help="Output WAV rate (resampled if != model rate).")
    p.add_argument("--enable-cpu-offload", action="store_true")
    p.add_argument("--enable-profiler", action="store_true")
    return p.parse_args()


def require_bundle(weights_dir: Path) -> None:
    missing = [f for f in _REQUIRED_BUNDLE_FILES if not (weights_dir / f).is_file()]
    if missing:
        raise SystemExit(
            f"Missing AudioX bundle files under {weights_dir}: {missing}\n"
            f"  huggingface-cli download zhangj1an/AudioX --local-dir {weights_dir}"
        )


def download_sample_video(url: str, dest: Path) -> None:
    """Fetch a sample mp4 to ``dest``. No-op if ``dest`` exists."""
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample video {url} → {dest}")
    urllib.request.urlretrieve(url, dest)


def video_path_for(task: str, video_dir: Path) -> Path:
    return video_dir / Path(SAMPLE_VIDEO_URLS[task]).name


def save_wav(audio: torch.Tensor, path: Path, sample_rate: int) -> None:
    """Write 16-bit PCM WAV. ``audio`` is ``[channels, samples]`` float in [-1, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.clamp(-1.0, 1.0).cpu(), sample_rate, encoding="PCM_S", bits_per_sample=16)


def generate_audio(omni: Omni, task: str, video_dir: Path, args: argparse.Namespace) -> torch.Tensor:
    extra: dict = {"audiox_task": task, "seconds_start": 0.0, "seconds_total": float(args.seconds_total)}
    if task in VIDEO_TASKS:
        extra["video_path"] = str(video_path_for(task, video_dir))
    if args.reference_audio:
        extra["audio_path"] = args.reference_audio
    prompt = SAMPLE_PROMPTS[task] if task in TEXT_TASKS else ""
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    outputs = omni.generate(
        prompt,
        OmniDiffusionSamplingParams(
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            extra_args=extra,
        ),
    )
    audio = outputs[0].request_output.multimodal_output.get("audio")
    if audio is None:
        raise RuntimeError(f"No audio produced for task {task!r}")
    audio = torch.as_tensor(audio).detach().cpu().float()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim != 2:
        raise RuntimeError(f"Unexpected audio shape {tuple(audio.shape)}")
    return audio  # [channels, samples]


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    require_bundle(weights_dir)

    for task in args.tasks:
        if task in VIDEO_TASKS:
            download_sample_video(SAMPLE_VIDEO_URLS[task], video_path_for(task, video_dir))

    model_sr = int(json.loads((weights_dir / "config.json").read_text()).get("sample_rate", 48000))

    omni = Omni(
        model=str(weights_dir),
        model_class_name="AudioXPipeline",
        enable_diffusion_pipeline_profiler=args.enable_profiler,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    for task in args.tasks:
        print(f"=== {task} ===", flush=True)
        t0 = time.perf_counter()
        audio = generate_audio(omni, task, video_dir, args)
        if model_sr != args.sample_rate:
            audio = TF.resample(audio, model_sr, args.sample_rate)
        out_path = output_dir / f"{task}.wav"
        save_wav(audio, out_path, args.sample_rate)
        print(f"Saved {out_path} ({time.perf_counter() - t0:.2f}s)")

    omni.close()


if __name__ == "__main__":
    main()
