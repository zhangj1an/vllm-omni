import argparse
import base64

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AudioX online serving client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8099, help="Server port")
    parser.add_argument("--task", type=str, default="t2a", choices=["t2a", "t2m", "v2a", "v2m", "tv2a", "tv2m"])
    parser.add_argument(
        "--prompt",
        type=str,
        default="Fireworks burst twice, followed by a period of silence before a clock begins ticking",
        help="Text prompt (required for t2*/tv2* tasks)",
    )
    parser.add_argument(
        "--video-url",
        type=str,
        default="",
        help="Video URL or base64 data URL (required for v2*/tv2* tasks)",
    )
    parser.add_argument("--steps", type=int, default=250, help="num_inference_steps")
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds-total", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="audiox_out.wav", help="Output WAV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY")

    content: list[dict] = []
    if args.video_url:
        content.append({"type": "video_url", "video_url": {"url": args.video_url}})
    if args.prompt:
        content.append({"type": "text", "text": args.prompt})
    if not content:
        raise ValueError("At least one content item is required (text and/or video_url).")

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": content}],
        model="audiox",
        extra_body={
            "audiox_task": args.task,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "seconds_total": args.seconds_total,
        },
    )

    audio_b64 = response.choices[0].message.audio.data
    with open(args.output, "wb") as f:
        f.write(base64.b64decode(audio_b64))
    print(f"Saved audio to {args.output}")


if __name__ == "__main__":
    main()
