#!/usr/bin/env python3
"""Demo: Load Cosmos3-Super with a LoRA adapter and generate output.

Usage:
    ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11 \
    python demo_lora_generate.py \
        --model /home/ma-user/work/z84450661/Cosmos3-Super \
        --lora-path /home/ma-user/work/z84450661/cosmos3-super-demo-lora \
        --prompt "A beautiful sunset over mountains, oil painting style" \
        --output ./lora_output.png
"""

import argparse
import base64
import json
import os
import sys
import time

# Ensure local vllm-omni takes precedence
sys.path.insert(0, "/home/ma-user/work/z84450661/vllm-omni")

os.environ.setdefault("MINDIE_SD_FA_TYPE", "ascend_laser_attention")
os.environ.setdefault("HF_HOME", "/home/ma-user/work/z84450661/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def serve_and_generate(model_path, lora_path, prompt, output_path, lora_scale=1.0):
    """Start an API server, generate an image with LoRA, and save it."""
    import subprocess
    import urllib.request

    port = 18000  # Use non-default port to avoid conflicts

    # Start server in background
    print(f"Starting vllm server on port {port}...")
    cmd = [
        sys.executable, "-m", "vllm_omni.entrypoints.openai.api_server",
        model_path,
        "--omni",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tensor-parallel-size", "8",
        "--model-class-name", "Cosmos3OmniDiffusersPipeline",
        "--lora-path", lora_path,
        "--lora-scale", str(lora_scale),
        "--no-guardrails",
        "--init-timeout", "1800",
        "--max-model-len", "65536",
    ]
    print(f"  Command: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for server to be ready
    print("  Waiting for server...")
    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            urllib.request.urlopen(req, timeout=5)
            print("  Server is ready!")
            break
        except Exception:
            time.sleep(5)
    else:
        print("  ERROR: Server failed to start")
        proc.kill()
        return False

    # Generate
    print(f"\nGenerating: '{prompt}'")
    payload = json.dumps({
        "model": model_path,
        "prompt": prompt,
        "size": "512x512",
        "num_inference_steps": 30,
        "guidance_scale": 7.0,
        "seed": 42,
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/images/generations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=600)
        result = json.loads(resp.read())
        img_data = result["data"][0].get("b64_json") or result["data"][0].get("url")
        if img_data and img_data.startswith("data:"):
            img_data = img_data.split(",", 1)[1]

        if img_data:
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(img_data))
            print(f"  ✅ Saved to {output_path}")
        else:
            print(f"  Raw response: {json.dumps(result, indent=2)[:500]}")
    except Exception as e:
        print(f"  Generation failed: {e}")
        return False
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ma-user/work/z84450661/Cosmos3-Super")
    parser.add_argument("--lora-path", default="/home/ma-user/work/z84450661/cosmos3-super-demo-lora")
    parser.add_argument("--prompt", default="A cute cat sitting on a cloud, anime style")
    parser.add_argument("--output", default="./lora_output_t2i.png")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print("Cosmos3-Super LoRA Generation Demo")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"LoRA:  {args.lora_path}")
    print(f"Scale: {args.lora_scale}")
    print(f"Prompt: {args.prompt}")
    print()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)
    if not os.path.exists(args.lora_path):
        print(f"ERROR: LoRA adapter not found at {args.lora_path}")
        sys.exit(1)

    success = serve_and_generate(
        args.model, args.lora_path, args.prompt, args.output, args.lora_scale
    )

    if success:
        print(f"\n✅ Done! Output: {args.output}")
        size_kb = os.path.getsize(args.output) / 1024
        print(f"   File size: {size_kb:.1f} KB")
    else:
        print("\n❌ Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
