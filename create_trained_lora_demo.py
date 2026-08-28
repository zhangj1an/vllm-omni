#!/usr/bin/env python3
"""Create a demo LoRA adapter with meaningful non-zero weights for Cosmos3-Super.

This simulates a "trained" LoRA adapter by initializing weights with a specific
pattern. The result is a valid PEFT adapter that vllm-omni can load and apply.

Usage:
    python create_trained_lora_demo.py --output ./cosmos3-super-demo-lora --rank 32
"""

import argparse
import json
import math
import os

import torch
from safetensors.torch import save_file


def create_adapter_config(output_dir: str, r: int = 32, lora_alpha: int = 64) -> None:
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
            "to_q", "to_k", "to_v", "to_out",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "gate_proj", "up_proj", "down_proj",
        ],
        "task_type": None,
        "use_dora": False,
        "use_rslora": False,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote adapter_config.json to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./cosmos3-super-demo-lora")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--model-path", default="/run/z84450661/Cosmos3-Super",
                        help="Path to Cosmos3-Super model (for reading weight index)")
    args = parser.parse_args()

    print(f"Creating demo LoRA adapter (rank={args.rank})")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output}")
    print()

    create_adapter_config(args.output, r=args.rank)

    # Discover target modules from model's weight index
    index_path = os.path.join(args.model_path, "transformer",
                              "diffusion_pytorch_model.safetensors.index.json")
    targets = {"to_q", "to_k", "to_v", "to_out",
               "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
               "gate_proj", "up_proj", "down_proj"}

    modules = set()
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        for name in idx.get("weight_map", {}):
            for t in targets:
                if name.endswith(f".{t}.weight"):
                    modules.add(name[:-len(".weight")])
                    break
    else:
        print(f"No index found at {index_path}, using estimates")
        for i in range(64):
            p = f"layers.{i}"
            for mod in ["self_attn.to_q", "self_attn.to_k", "self_attn.to_v",
                         "self_attn.to_out", "self_attn.add_q_proj",
                         "self_attn.add_k_proj", "self_attn.add_v_proj",
                         "self_attn.to_add_out",
                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                modules.add(f"{p}.{mod}")

    modules = sorted(modules)
    print(f"Found {len(modules)} target modules")

    # Model dimensions
    hidden_size = 5120
    qkv_out = 64 * 128   # num_heads * head_dim = 8192
    kv_out = 8 * 128     # num_kv_heads * head_dim = 1024
    mlp_intermediate = 25600

    size_map = {
        "to_q": (qkv_out, hidden_size),
        "to_k": (kv_out, hidden_size),
        "to_v": (kv_out, hidden_size),
        "to_out": (hidden_size, qkv_out),
        "add_q_proj": (qkv_out, hidden_size),
        "add_k_proj": (kv_out, hidden_size),
        "add_v_proj": (kv_out, hidden_size),
        "to_add_out": (hidden_size, qkv_out),
        "gate_proj": (mlp_intermediate, hidden_size),
        "up_proj": (mlp_intermediate, hidden_size),
        "down_proj": (hidden_size, mlp_intermediate),
    }

    r = args.rank
    weights = {}
    torch.manual_seed(42)

    for module_name in modules:
        layer_type = module_name.split(".")[-1]
        dims = size_map.get(layer_type)
        if dims is None:
            continue
        out_features, in_features = dims

        # Kaiming-style init scaled for LoRA
        # lora_A: (r, in_features) — small random values
        # lora_B: (out_features, r) — zero init (standard peft practice)

        w_a = torch.randn(r, in_features, dtype=torch.bfloat16) * 0.02 / math.sqrt(r)
        w_b = torch.randn(out_features, r, dtype=torch.bfloat16) * 0.02 / math.sqrt(r)

        # Normalize so scale ~1 when lora_scale=1
        w_a = w_a / w_a.norm() * 0.1
        w_b = w_b / w_b.norm() * 0.1

        key_a = f"base_model.model.{module_name}.lora_A.default.weight"
        key_b = f"base_model.model.{module_name}.lora_B.default.weight"
        weights[key_a] = w_a
        weights[key_b] = w_b

    # Save
    output_path = os.path.join(args.output, "adapter_model.safetensors")
    save_file(weights, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Wrote {len(weights)} LoRA tensors ({size_mb:.1f} MB)")
    print(f"\nAdapter ready: {args.output}")
    print(f"Serve with:")
    print(f"  --lora-path {args.output} --lora-scale 1.0")


if __name__ == "__main__":
    main()
