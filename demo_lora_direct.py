#!/usr/bin/env python3
"""Direct LoRA generation for Cosmos3-Super — bypasses CLI, uses engine API.

Usage (inside container or on host with NPUs):
    ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11 \
    python demo_lora_direct.py
"""

import os
import sys
import time

# Use local repo code
sys.path.insert(0, "/home/ma-user/work/z84450661/vllm-omni")
os.environ.setdefault("MINDIE_SD_FA_TYPE", "ascend_laser_attention")

MODEL_PATH = "/home/ma-user/work/z84450661/Cosmos3-Super"
LORA_PATH = "/home/ma-user/work/z84450661/cosmos3-super-demo-lora"
OUTPUT = "/home/ma-user/work/z84450661/lora_output_t2i.png"


def main():
    print("=" * 60)
    print("Cosmos3-Super + LoRA Direct Generation")
    print("=" * 60)

    import torch
    import torch_npu
    print(f"NPU devices: {torch_npu.npu.device_count()}")

    # --- Load the model pipeline ---
    print("\n[1] Loading Cosmos3-Super pipeline...")
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    pipeline = Cosmos3OmniDiffusersPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="npu",
        trust_remote_code=True,
    )
    print(f"  Pipeline loaded: {type(pipeline).__name__}")

    # --- Load LoRA adapter ---
    print("\n[2] Loading LoRA adapter...")
    from peft import PeftModel

    transformer = pipeline.transformer
    print(f"  Transformer: {type(transformer).__name__}")
    print(f"  Params: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B")

    transformer = PeftModel.from_pretrained(transformer, LORA_PATH)
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"  LoRA params: {trainable / 1e6:.1f}M trainable")

    # Verify LoRA weights are non-zero
    lora_params = 0
    lora_nonzero = 0
    for name, param in transformer.named_parameters():
        if "lora_" in name:
            lora_params += param.numel()
            lora_nonzero += (param.abs() > 1e-8).sum().item()
    print(f"  LoRA non-zero: {lora_nonzero}/{lora_params} ({100*lora_nonzero/lora_params:.1f}%)")

    # --- Generate ---
    print("\n[3] Generating T2I with LoRA...")
    prompt = "A cute cat sitting on a cloud, anime watercolor painting style, soft pastel colors"
    print(f'  Prompt: "{prompt}"')

    t0 = time.time()
    with torch.no_grad():
        output = pipeline(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=7.0,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
    elapsed = time.time() - t0

    # Save output
    images = output.images if hasattr(output, "images") else output
    if isinstance(images, list):
        img = images[0]
    else:
        img = images

    img.save(OUTPUT)
    size_kb = os.path.getsize(OUTPUT) / 1024

    print(f"\n  ✅ Generated in {elapsed:.1f}s")
    print(f"  Output: {OUTPUT} ({size_kb:.1f} KB)")
    print(f"  Size: {img.size}")

    # Also generate WITHOUT LoRA for comparison
    print("\n[4] Generating WITHOUT LoRA (baseline)...")
    # Unload LoRA
    transformer = transformer.merge_and_unload()
    pipeline.transformer = transformer

    output_no_lora = pipeline(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )
    images_nl = output_no_lora.images if hasattr(output_no_lora, "images") else output_no_lora
    img_nl = images_nl[0] if isinstance(images_nl, list) else images_nl

    baseline_path = OUTPUT.replace(".png", "_baseline.png")
    img_nl.save(baseline_path)
    print(f"  Baseline: {baseline_path} ({os.path.getsize(baseline_path)/1024:.1f} KB)")

    print(f"\n=== Done! ===")
    print(f"  With LoRA:    {OUTPUT}")
    print(f"  Without LoRA: {baseline_path}")


if __name__ == "__main__":
    main()
