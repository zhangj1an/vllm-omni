#!/usr/bin/env python3
"""AudioX OpenAI-compatible chat client.

AudioX supports 6 tasks (t2a, t2m, v2a, v2m, tv2a, tv2m). Text-only tasks send the prompt as the
chat message; video-conditioned tasks additionally attach the video via a ``video_url`` content
item (data URI for local files). Task + generation knobs (steps, cfg, sigma range, seconds, seed)
are sent via the OpenAI SDK's ``extra_body`` as ``extra_args`` — the same pipeline-agnostic escape
hatch used by the /v1/videos endpoint's ``extra_params`` field.

Usage:
  python openai_chat_client.py --task t2a --prompt "Fireworks burst twice..." --output t2a.wav
  python openai_chat_client.py --task tv2a --prompt "drum beating" --video clip.mp4 -o tv2a.wav
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import sys
import wave
from pathlib import Path

import numpy as np
import requests
import torch

VIDEO_TASKS = frozenset({"v2a", "v2m", "tv2a", "tv2m"})
TEXT_TASKS = frozenset({"t2a", "t2m", "tv2a", "tv2m"})


def _to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "video/mp4"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _save_wav(audio: torch.Tensor, path: Path, sample_rate: int) -> None:
    audio = audio.to(torch.float32)
    peak = audio.abs().max().clamp(min=1e-8)
    audio = audio / peak
    pcm = (audio.clamp(-1.0, 1.0) * 32767).to(torch.int16).T.contiguous().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _decode_audio_from_response(body: dict) -> tuple[torch.Tensor, int]:
    import io

    for choice in body.get("choices", []):
        audio_obj = choice.get("message", {}).get("audio")
        if not (isinstance(audio_obj, dict) and audio_obj.get("data")):
            continue
        raw = base64.b64decode(audio_obj["data"])
        with wave.open(io.BytesIO(raw), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        arr = np.frombuffer(frames, dtype=np.int16).reshape(-1, nch).astype(np.float32) / 32768.0
        return torch.from_numpy(arr).transpose(0, 1), sr
    brief = {k: v for k, v in body.items() if k != "choices"}
    raise RuntimeError(f"no audio in response message.audio: {brief}")


def main() -> int:
    p = argparse.ArgumentParser(description="AudioX OpenAI chat client")
    p.add_argument("--task", required=True, choices=["t2a", "t2m", "v2a", "v2m", "tv2a", "tv2m"])
    p.add_argument("--prompt", "-p", default="", help="Text prompt (required for t2*/tv2*).")
    p.add_argument("--video", help="Video path or URL (required for v2*/tv2*).")
    p.add_argument("--output", "-o", default="audiox_out.wav")
    p.add_argument("--server", "-s", default="http://localhost:8099")
    p.add_argument("--model", default="zhangj1an/AudioX")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seconds-total", type=float, default=10.0)
    p.add_argument("--seconds-start", type=float, default=0.0)
    p.add_argument("--sigma-min", type=float, default=0.3)
    p.add_argument("--sigma-max", type=float, default=500.0)
    args = p.parse_args()

    if args.task in VIDEO_TASKS and not args.video:
        print(f"ERROR: task {args.task!r} requires --video", file=sys.stderr)
        return 2
    if args.task in TEXT_TASKS and not args.prompt.strip() and args.task not in {"v2a", "v2m"}:
        print(f"ERROR: task {args.task!r} requires --prompt", file=sys.stderr)
        return 2

    content: list[dict] = [{"type": "text", "text": args.prompt}]
    if args.task in VIDEO_TASKS:
        vurl = args.video if args.video.startswith(("http://", "https://")) else _to_data_url(args.video)
        content.append({"type": "video_url", "video_url": {"url": vurl}})

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "extra_args": {
            "audiox_task": args.task,
            "seconds_start": args.seconds_start,
            "seconds_total": args.seconds_total,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
        },
    }

    print(f"POST {args.server}/v1/chat/completions  task={args.task} steps={args.steps}")
    r = requests.post(
        f"{args.server}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    audio, sr = _decode_audio_from_response(r.json())
    _save_wav(audio, Path(args.output), sr)
    dur = audio.shape[-1] / sr
    print(f"saved {args.output}  sr={sr}Hz  duration={dur:.2f}s  channels={audio.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
