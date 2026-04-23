#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online client for Kimi-Audio covering three task modes:
``audio2text`` (non-streaming chat), ``audio2audio`` and ``text2audio``
(SSE-streamed base64-WAV chunks concatenated to ``--out``). Launch the
server with ``vllm_omni/deploy/kimi_audio.yaml``; toggle that file's
``async_chunk`` flag for sub-second TTFB on the audio-out tasks."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

OUTPUT_SAMPLE_RATE = 24000
MARY_HAD_LAMB = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
# Default audio URL for the audio2text task. Originally taken from the
# MiniMax TTS-Multilingual test set (sample 10), mirrored to Google Drive
# for a stable link. Direct-download form of
# https://drive.google.com/file/d/1RHz6uUSbAR_N3Li1Bjh8dPknykw4IVio/view?usp=sharing.
AUDIO2TEXT_DEFAULT_URL = "https://drive.google.com/uc?export=download&id=1RHz6uUSbAR_N3Li1Bjh8dPknykw4IVio"

TASK_CHOICES = ("audio2text", "audio2audio", "text2audio")

TASK_DEFAULT_QUESTION = {
    "audio2text": "Please transcribe the audio.",
    "audio2audio": "Answer in audio. Briefly summarize.",
    "text2audio": 'Please say the following in audio: "Hello, my name is Kimi."',
}

TASK_DEFAULT_AUDIO_URL = {
    "audio2text": AUDIO2TEXT_DEFAULT_URL,
    "audio2audio": MARY_HAD_LAMB,
    "text2audio": None,
}


def _resolve_audio_source(audio_path: str | None, default_url: str | None) -> dict:
    if audio_path:
        audio_bytes = Path(audio_path).read_bytes()
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {"url": f"data:audio/wav;base64,{b64}"}
    return {"url": default_url}


def build_request(task: str, model: str, audio_path: str | None, question: str) -> dict:
    user_content: list[dict] = []
    if task != "text2audio":
        default_url = TASK_DEFAULT_AUDIO_URL[task]
        user_content.append({"type": "audio_url", "audio_url": _resolve_audio_source(audio_path, default_url)})
    user_content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": user_content}]

    if task == "audio2text":
        thinker = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 512,
            "seed": 42,
            "repetition_penalty": 1.0,
        }
        return {
            "model": model,
            "stream": False,
            "sampling_params_list": [thinker],
            "messages": messages,
        }

    # audio2audio, text2audio: two-stage audio-out pipeline, streaming.
    thinker = {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "max_tokens": 1024, "seed": 42}
    code2wav = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 8192,
        "seed": 42,
        "detokenize": False,
    }
    return {
        "model": model,
        "stream": True,
        "modalities": ["text", "audio"],
        "sampling_params_list": [thinker, code2wav],
        "messages": messages,
    }


def iter_sse(resp: requests.Response):
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def run_audio_out(url: str, body: dict, out_path: str) -> None:
    with requests.post(url, json=body, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        chunks: list[np.ndarray] = []
        for event in iter_sse(resp):
            choices = event.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue
                # Server sends each audio chunk as base64 in delta.content.
                try:
                    wav_bytes = base64.b64decode(content)
                except Exception:
                    continue
                audio, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                chunks.append(audio.flatten())

    if not chunks:
        print("No audio chunks received", file=sys.stderr)
        sys.exit(1)
    full = np.concatenate(chunks)
    sf.write(out_path, full, samplerate=OUTPUT_SAMPLE_RATE, format="WAV")
    print(f"Wrote {out_path} ({full.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s of audio)")


def run_audio_to_text(url: str, body: dict) -> None:
    resp = requests.post(url, json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content")
    if text is None:
        print("No text response received", file=sys.stderr)
        sys.exit(1)
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Kimi-Audio online client (audio2text, audio2audio, text2audio).")
    parser.add_argument("--task", "-t", choices=TASK_CHOICES, default="audio2audio")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument(
        "--audio-path",
        default=None,
        help="Path to a local audio file. Ignored for --task text2audio.",
    )
    parser.add_argument("--question", default=None, help="Text instruction. Default depends on --task.")
    parser.add_argument("--out", default="output.wav", help="Output WAV path (audio-out tasks only).")
    args = parser.parse_args()

    question = args.question or TASK_DEFAULT_QUESTION[args.task]
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    body = build_request(args.task, args.model, args.audio_path, question)

    if args.task == "audio2text":
        run_audio_to_text(url, body)
    else:
        run_audio_out(url, body, args.out)


if __name__ == "__main__":
    main()
