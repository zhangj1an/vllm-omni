#!/usr/bin/env python3
"""Create a test LoRA adapter for Cosmos3-Super using PEFT.

This script creates a minimal LoRA adapter targeting Cosmos3-Super linear layers.
The adapter weights are deliberately zero-initialized (so output is unchanged),
but the file structure is valid for testing the vLLM-omni LoRA loading path.

Usage:
    python create_test_lora.py --model-path /run/z84450661/Cosmos3-Super --output ./test-lora
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


def create_adapter_config(output_dir: str, r: int = 8, lora_alpha: int = 16) -> None:
    """Write a PEFT-compatible adapter_config.json for Cosmos3-Super."""
    config = {
        "auto_mapping": None,
        "base_model_name_or_path": "",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "middle_lora_rank": r,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "task_type": None,
        "use_dora": False,
        "use_rslora": False,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote adapter_config.json to {output_dir}")


def find_linear_layers(model_dir: str) -> list[dict]:
    """Find all target linear layers from the model's safetensors index."""
    index_path = os.path.join(model_dir, "transformer", "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.exists(index_path):
        # Try alternate paths
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"No safetensors index found at {model_dir}, using estimated module names")
        return _estimate_modules()

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    # Extract unique module+target layer names
    targets = {"to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj",
               "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"}
    modules = set()
    for full_name in weight_map:
        for target in targets:
            if full_name.endswith(f".{target}.weight"):
                # Keep full path including the target layer name so
                # create_adapter_weights can extract the layer type
                layer_path = full_name[:-len(".weight")]
                # Normalize separator to "." for PEFT lookup
                layer_path = layer_path.replace("[", ".").replace("]", "")
                modules.add(layer_path)
                break

    result = sorted(modules)
    print(f"Found {len(result)} LoRA-targetable linear layers:")
    for m in result[:5]:
        print(f"  {m}")
    if len(result) > 5:
        print(f"  ... and {len(result) - 5} more")
    return result


def _estimate_modules() -> list[str]:
    """Estimate module names for Cosmos3-Super (64 layers UND + 32 layers GEN)."""
    modules = []
    # UND layers
    for i in range(64):
        prefix = f"language_model.layers.{i}"
        for mod in ["self_attn.to_q", "self_attn.to_k", "self_attn.to_v", "self_attn.to_out",
                      "self_attn.add_q_proj", "self_attn.add_k_proj", "self_attn.add_v_proj",
                      "self_attn.to_add_out",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            modules.append(f"{prefix}.{mod}")
    # GEN layers
    for i in range(32):
        prefix = f"gen_layers.{i}"
        for mod in ["self_attn.to_q", "self_attn.to_k", "self_attn.to_v", "self_attn.to_out",
                      "self_attn.add_q_proj", "self_attn.add_k_proj", "self_attn.add_v_proj",
                      "self_attn.to_add_out",
                      "cross_attn.to_q", "cross_attn.to_k", "cross_attn.to_v", "cross_attn.to_out",
                      "cross_attn.add_q_proj", "cross_attn.add_k_proj", "cross_attn.add_v_proj",
                      "cross_attn.to_add_out",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            modules.append(f"{prefix}.{mod}")
    print(f"Estimated {len(modules)} LoRA-targetable linear layers")
    return modules


def create_adapter_weights(module_names: list[str], r: int, output_dir: str) -> None:
    """Create zero-initialized LoRA adapter weights (safetensors format).

    Uses zero-init for both A and B weights so the adapter initially has no effect.
    This is the standard PEFT initialization pattern.
    """
    # We need to know the hidden_size for each layer to create correct shapes.
    # For Cosmos3-Super: hidden_size=5120, head_dim=128, num_heads=64, num_kv_heads=8
    # to_q: 5120 -> 64*128 = 8192 (but TP=1 splitting differs)
    # For a test adapter we use the actual shapes from the model

    weights = {}
    hidden_size = 5120  # Cosmos3-Super hidden_size
    qkv_out = 64 * 128   # num_heads * head_dim = 8192
    kv_out = 8 * 128     # num_kv_heads * head_dim = 1024
    mlp_intermediate = 25600  # intermediate_size

    qkv_size_map = {
        "to_q": qkv_out,
        "to_k": kv_out,
        "to_v": kv_out,
        "to_out": hidden_size,
        "add_q_proj": qkv_out,
        "add_k_proj": kv_out,
        "add_v_proj": kv_out,
        "to_add_out": hidden_size,
        "gate_proj": mlp_intermediate,
        "up_proj": mlp_intermediate,
        "down_proj": hidden_size,
    }

    for module_name in module_names:
        layer_suffix = module_name.split(".")[-1]  # e.g. "to_q", "gate_proj"
        out_features = qkv_size_map.get(layer_suffix)
        if out_features is None:
            print(f"  WARNING: Unknown layer type {layer_suffix}, skipping {module_name}")
            continue

        # lora_A: (r, in_features), lora_B: (out_features, r)
        # Output projections: input feature size matches their paired QKV output
        # MLP down_proj: input is intermediate_size
        in_features = hidden_size
        if layer_suffix in ("to_out", "to_add_out"):
            in_features = qkv_out
        elif layer_suffix == "down_proj":
            in_features = mlp_intermediate

        key_a = f"base_model.model.{module_name}.lora_A.default.weight"
        key_b = f"base_model.model.{module_name}.lora_B.default.weight"

        weights[key_a] = torch.zeros(r, in_features, dtype=torch.bfloat16)
        weights[key_b] = torch.zeros(out_features, r, dtype=torch.bfloat16)

    output_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(weights, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Wrote {len(weights)} LoRA weight tensors ({size_mb:.1f} MB) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create test LoRA adapter for Cosmos3-Super")
    parser.add_argument("--model-path", default="/run/z84450661/Cosmos3-Super",
                        help="Path to the Cosmos3-Super model directory")
    parser.add_argument("--output", default="./cosmos3-super-test-lora",
                        help="Output directory for the LoRA adapter")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    args = parser.parse_args()

    print(f"Creating test LoRA adapter (rank={args.rank}) for Cosmos3-Super")
    print(f"Model path: {args.model_path}")
    print(f"Output: {args.output}")
    print()

    create_adapter_config(args.output, r=args.rank)
    modules = find_linear_layers(args.model_path)
    create_adapter_weights(modules, args.rank, args.output)

    print()
    print("Done! To use this adapter:")
    print(f"  vllm serve /run/z84450661/Cosmos3-Super --omni --lora-path {args.output} --lora-scale 1.0")


if __name__ == "__main__":
    main()
