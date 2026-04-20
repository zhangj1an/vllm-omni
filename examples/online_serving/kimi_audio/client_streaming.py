#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming chat client for Kimi-Audio audio-out.

Opens a server-sent-event stream against ``/v1/chat/completions`` with
``stream=true`` and ``modalities=["audio"]``. The server (running with
``kimi_audio_async_chunk.yaml``) emits per-chunk base64-encoded WAV
fragments as the code2wav stage produces them. The client decodes and
concatenates them into ``output.wav``.
"""

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
MARY_HAD_LAMB = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
)


def build_request(model: str, audio_source: dict, question: str) -> dict:
    thinker = {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "max_tokens": 1024, "seed": 42}
    code2wav = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 8192, "seed": 42, "detokenize": False}
    return {
        "model": model,
        "stream": True,
        "modalities": ["text", "audio"],
        "sampling_params_list": [thinker, code2wav],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": audio_source},
                    {"type": "text", "text": question},
                ],
            }
        ],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--question", default="Answer in audio. Briefly summarize.")
    parser.add_argument("--out", default="output.wav")
    args = parser.parse_args()

    if args.audio_path:
        audio_bytes = Path(args.audio_path).read_bytes()
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        audio_source = {"url": f"data:audio/wav;base64,{b64}"}
    else:
        audio_source = {"url": MARY_HAD_LAMB}

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    body = build_request(args.model, audio_source, args.question)
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
                audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                if sr != OUTPUT_SAMPLE_RATE:
                    print(f"Warning: chunk sample rate {sr} != {OUTPUT_SAMPLE_RATE}", file=sys.stderr)
                chunks.append(audio.flatten())

    if not chunks:
        print("No audio chunks received", file=sys.stderr)
        sys.exit(1)
    full = np.concatenate(chunks)
    sf.write(args.out, full, samplerate=OUTPUT_SAMPLE_RATE, format="WAV")
    print(f"Wrote {args.out} ({full.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s of audio)")


if __name__ == "__main__":
    main()
