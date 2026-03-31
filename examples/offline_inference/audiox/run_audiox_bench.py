import json
import os
import wave
from pathlib import Path

from examples.offline_inference.audiox.end2end import (
    _default_diffusion_attention_backend,
    run_single_inference,
)


def main() -> None:
    _default_diffusion_attention_backend()
    base = Path(__file__).resolve().parent
    out_dir = base / "output" / "vllm_omni"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = str(base / "audiox_weights")
    videos = base / "videos"

    cases = [
        (
            "t2a",
            "Fireworks burst twice, followed by a period of silence before a clock begins ticking.",
            "",
        ),
        ("t2m", "Uplifting ukulele tune for a travel vlog", ""),
        ("v2a", "", str(videos / "v2a.mp4")),
        ("v2m", "", str(videos / "v2m.mp4")),
        ("tv2a", "drum beating sound and human talking", str(videos / "tv2a.mp4")),
        ("tv2m", "Uplifting music matching the scene", str(videos / "tv2m.mp4")),
    ]
    # Comma/space-separated subset, e.g. AUDIOX_BENCH_TASKS=t2a
    filt = os.environ.get("AUDIOX_BENCH_TASKS", "").strip()
    if filt:
        allow = {t.strip().lower() for t in filt.replace(",", " ").split() if t.strip()}
        cases = [c for c in cases if c[0] in allow]

    results = []
    for task, prompt, video in cases:
        print(f"=== RUN {task} ===", flush=True)
        out = out_dir / f"{task}.wav"

        latency = run_single_inference(
            model_dir=weights,
            task=task,
            prompt=prompt,
            video_path=video,
            reference_audio_path="",
            output=str(out),
            sample_rate=48000,
            seed=42,
            num_inference_steps=250,
            guidance_scale=7.0,
            seconds_start=0.0,
            seconds_total=10.0,
            negative_prompt="",
            enable_profiler=False,
            enable_cpu_offload=False,
        )
        with wave.open(str(out), "rb") as wf:
            duration = wf.getnframes() / float(wf.getframerate())
        rtf = latency / duration if duration > 0 else None
        throughput = (1.0 / latency) if latency > 0 else None
        results.append(
            {
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
                "output_wav": str(out),
            }
        )

    summary = {
        "success_rate": f"100% ({len(results)}/{len(cases)} tasks)",
        "latency_s_avg": sum(r["latency_s"] for r in results) / len(results),
        "duration_s_avg": sum(r["duration_s"] for r in results) / len(results),
        "rtf_avg": sum(r["rtf"] for r in results) / len(results),
        "ttft_s_avg": sum(r["ttft_s"] for r in results) / len(results),
        "tpot_s_per_token_avg": sum(r["tpot_s_per_token"] for r in results) / len(results),
        "itl_s_per_token_avg": None,
        "output_throughput_avg": sum(r["output_throughput_tokens_per_s"] for r in results)
        / len(results),
        "total_throughput_avg": sum(r["total_throughput_tokens_per_s"] for r in results)
        / len(results),
        "efficiency_proxy_avg": sum(r["efficiency_proxy"] for r in results) / len(results),
    }

    obj = {"results": results, "summary": summary}
    out_json = out_dir / "metrics_vllm_omni.json"
    out_json.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {out_json}")


if __name__ == "__main__":
    main()
