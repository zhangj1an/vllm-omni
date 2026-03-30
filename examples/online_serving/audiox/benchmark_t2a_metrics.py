#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AudioX t2a streaming metrics.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--task", type=str, default="t2a")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds-total", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def _extract_has_payload(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    if not delta:
        return False
    # Count any stream payload as an output event.
    return bool(delta.get("content") or delta.get("audio") or delta.get("text") or delta.get("tool_calls"))


def main() -> None:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "model": "audiox",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            }
        ],
        "extra_body": {
            "audiox_task": args.task,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "seconds_total": args.seconds_total,
        },
    }

    t0 = time.perf_counter()
    first_event_ts: float | None = None
    token_event_timestamps: list[float] = []
    completion_tokens: int | None = None
    saw_non_sse_json = False

    with requests.post(url, json=payload, stream=True, timeout=args.timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                data = raw[6:].strip()
            else:
                # Some AudioX responses are returned as a single non-SSE JSON payload
                # even when stream=True. Handle that shape too.
                data = raw.strip()
                saw_non_sse_json = True
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            now = time.perf_counter()

            if _extract_has_payload(chunk):
                if first_event_ts is None:
                    first_event_ts = now
                token_event_timestamps.append(now)
            elif saw_non_sse_json:
                # Non-stream fallback: count arrival of the payload as first event.
                if first_event_ts is None:
                    first_event_ts = now
                token_event_timestamps.append(now)

            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

    t_end = time.perf_counter()
    total_latency = t_end - t0
    ttft = None if first_event_ts is None else (first_event_ts - t0)
    generation_window = None if first_event_ts is None else (t_end - first_event_ts)
    if saw_non_sse_json:
        # In non-SSE fallback mode, only full-response latency is observable.
        generation_window = total_latency

    # Prefer server-reported completion token count.
    tokens = completion_tokens if completion_tokens is not None else len(token_event_timestamps)
    if tokens <= 0:
        raise RuntimeError("Could not derive output token count from stream usage/events.")

    # TPOT uses decode window from first output event to completion.
    tpot = None if generation_window is None else (generation_window / tokens)
    output_throughput = None if generation_window is None or generation_window <= 0 else (tokens / generation_window)
    total_throughput = tokens / total_latency if total_latency > 0 else None

    # ITL from event spacing if available; otherwise estimate from token count.
    itl = None
    if len(token_event_timestamps) >= 2:
        gaps = [
            token_event_timestamps[i] - token_event_timestamps[i - 1]
            for i in range(1, len(token_event_timestamps))
        ]
        itl = statistics.mean(gaps)
    elif generation_window is not None and tokens > 1:
        itl = generation_window / (tokens - 1)

    result = {
        "task": args.task,
        "prompt": args.prompt,
        "total_latency_s": total_latency,
        "ttft_s": ttft,
        "completion_tokens": tokens,
        "tpot_s_per_token": tpot,
        "itl_s_per_token": itl,
        "output_throughput_tokens_per_s": output_throughput,
        "total_throughput_tokens_per_s": total_throughput,
        "token_events_seen": len(token_event_timestamps),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
