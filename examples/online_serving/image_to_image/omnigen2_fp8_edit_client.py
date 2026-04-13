#!/usr/bin/env python3
"""
OmniGen2 FP8 image-edit client (OpenAI-compatible /v1/chat/completions).

Requires a server started with vLLM-Omni, e.g. run_server_omnigen2_fp8.sh.

Default: two images (person first, dress second) and a dress-swap prompt.
"""

import argparse
import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR / "example"

DEFAULT_PROMPT = (
    "Edit the first image: replace the clothing of the person with the dress "
    "from the second image. Keep the person's identity, face, hairstyle, pose, "
    "body shape, and background unchanged. Photorealistic."
)


def _encode_image_as_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    try:
        img = Image.open(BytesIO(image_bytes))
        mime_type = f"image/{img.format.lower()}" if img.format else "image/jpeg"
    except Exception:
        mime_type = "image/jpeg"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def run_edit_request(
    *,
    server_url: str,
    model: str,
    image1: Path,
    image2: Path,
    prompt: str,
    output_path: Path,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> None:
    if not image1.exists():
        raise FileNotFoundError(f"First image not found: {image1}")
    if not image2.exists():
        raise FileNotFoundError(f"Second image not found: {image2}")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image_as_data_url(image1)}},
                    {"type": "image_url", "image_url": {"url": _encode_image_as_data_url(image2)}},
                ],
            }
        ],
        "extra_body": {
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        },
    }

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", [])
    if not isinstance(content, list):
        raise RuntimeError(f"Unexpected response format: {data}")

    for item in content:
        if isinstance(item, dict) and "image_url" in item:
            img_url = item["image_url"].get("url", "")
            if img_url.startswith("data:image"):
                _, b64_data = img_url.split(",", 1)
                output_path.write_bytes(base64.b64decode(b64_data))
                return

    raise RuntimeError(f"No image found in response: {data}")


def _profile_request(server_url: str, endpoint: str, stages: list[int] | None = None) -> dict:
    payload = {"stages": stages} if stages else {}
    response = requests.post(
        f"{server_url}/{endpoint}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json() if response.content else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniGen2 FP8 image-edit client")
    parser.add_argument(
        "--image1",
        type=Path,
        default=_EXAMPLE_DIR / "girl.jpg",
        help="First input image path (person image); default: example/girl.jpg next to this script",
    )
    parser.add_argument(
        "--image2",
        type=Path,
        default=_EXAMPLE_DIR / "dress.jpg",
        help="Second input image path (dress reference); default: example/dress.jpg next to this script",
    )
    parser.add_argument("--output", "-o", default="omnigen2_fp8_edit_output.png", help="Output image path")
    parser.add_argument("--server", default="http://localhost:8092", help="Server URL")
    parser.add_argument("--model", default="OmniGen2/OmniGen2", help="Model name")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Edit prompt")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument(
        "--steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Inference / denoise steps (default: 30, or 2 when --profile for a fast trace)",
    )
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Call /start_profile before request and /stop_profile after request.",
    )
    parser.add_argument(
        "--profile-stages",
        type=int,
        nargs="+",
        default=None,
        help="Optional stage IDs for start/stop profile, e.g. --profile-stages 0",
    )

    args = parser.parse_args()

    if hasattr(args, "steps"):
        num_steps = args.steps
    elif args.profile:
        num_steps = 2
    else:
        num_steps = 30

    image1 = args.image1
    image2 = args.image2
    output_path = Path(args.output)

    print(f"Server: {args.server}")
    print(f"Model: {args.model}")
    print(f"Image1: {image1}")
    print(f"Image2: {image2}")
    print(f"Output: {output_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Inference steps: {num_steps}")

    profile_started = False
    try:
        if args.profile:
            start_resp = _profile_request(args.server, "start_profile", args.profile_stages)
            profile_started = True
            print(f"Profiler started: {start_resp}")

        run_edit_request(
            server_url=args.server,
            model=args.model,
            image1=image1,
            image2=image2,
            prompt=args.prompt,
            output_path=output_path,
            height=args.height,
            width=args.width,
            steps=num_steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
    finally:
        if args.profile and profile_started:
            stop_resp = _profile_request(args.server, "stop_profile", args.profile_stages)
            print(f"Profiler stopped: {stop_resp}")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
