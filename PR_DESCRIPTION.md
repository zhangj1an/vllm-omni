## Add Cosmos3-Super NPU recipe + LoRA adapter support

### Summary

Adds an **NPU** serving recipe and **LoRA** adapter support for the 64B Cosmos3-Super world model on 8× Ascend910 NPUs. The LoRA feature requires **zero model code changes** — vLLM-omni's `DiffusionLoRAManager` auto-discovers linear layers and injects LoRA at runtime.

### Why no code changes for LoRA?

vLLM-omni already has a generic `DiffusionLoRAManager` that works across all diffusion models. It auto-discovers `ColumnParallelLinear` / `RowParallelLinear` layers in the transformer and injects LoRA — no model-specific plumbing needed. This PR only adds the recipe/documentation telling users which target modules to use and how to serve.

### Changes

| File | Description |
|------|-------------|
| `recipes/cosmos3/Cosmos3-Super.md` | Added `## NPU` section (serve command, verification curls, notes) and `## LoRA` section (training config, static + per-request serving) |
| `serve_cosmos3_super_npu.sh` | Standalone serve script with `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=10800` |
| `create_cosmos3_test_lora.py` | Utility script to create a valid PEFT LoRA adapter from the model's weight index |
| `recipes/cosmos3/cosmos3_super_npu_results.md` | NPU benchmark results (T2I, T2V, I2V, V2V, T2VS, I2VS) |
| `recipes/cosmos3/assets/` | Test assets (reference images/videos for verification) |

### LoRA — Target Modules

Targets all attention and MLP linear layers in both UND (64 layers) and GEN (32 layers) pathways:

```
to_q, to_k, to_v, to_out              # Self-attention QKV + output
add_q_proj, add_k_proj, add_v_proj, to_add_out  # Additional projections
gate_proj, up_proj, down_proj         # MLP
cross_attn.to_q, cross_attn.to_k, cross_attn.to_v, cross_attn.to_out  # GEN cross-attn
```

**Verified**: 896 target modules found in the real `Cosmos3-Super` model weights (~1792 LoRA weight tensors at rank=8, ~256 MB).

### LoRA — Serve Commands

**Static adapter** (applied to every request):
```bash
vllm serve nvidia/Cosmos3-Super --omni --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --lora-path ./my-lora --lora-scale 1.0 \
  --no-guardrails --init-timeout 1800
```

**Per-request adapter** (dynamic switching, omit `--lora-path` at startup):
```bash
curl -X POST .../v1/images/generations \
  -d '{"prompt": "...", "lora": {"name": "mystyle", "path": "/path/to/lora", "scale": 1.0}}'
```

### LoRA — Memory Impact

| Rank | Adapter Size | Notes |
|------|-------------|-------|
| 8 | ~256 MB | Minimal, good for style-only |
| 32 | ~1 GB | Moderate expressivity |
| 64 | ~2 GB | Default in recipe, good balance |
| 128 | ~4 GB | Max expressivity |

LoRA weights are negligible vs the ~120 GB base model. Activations are unchanged.

### NPU Serve Command

```bash
vllm serve nvidia/Cosmos3-Super --omni --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails --init-timeout 1800
```

- `--tensor-parallel-size 8` matches the model's 8 KV heads (GQA: 64 attn / 8 KV)
- Uses 8 davinci devices across NPU 0-3 (both cores each); NPU 4-7 idle
- ~22.7 GB HBM per device at startup (bf16 weights)

### Hardware

| Item | Detail |
|------|--------|
| NPU | 8× Ascend910 (65,536 MB HBM each) |
| Container | `vllm-omni:0.24.0-a3-src` → upgraded to `v0.25.0` |
| Devices | `/dev/davinci0`-`/dev/davinci15` mapped |

### Verification

#### NPU Results

| Mode | Size / Steps | Time |
|------|-------------|------|
| T2I | 1024×1024 / 50 steps | — |
| T2V | 1280×720 189f / 35 steps | 8m18s |
| I2V | 1280×720 189f / 35 steps | 10m43s |
| V2V | 1280×720 189f / 35 steps | 8m22s |

#### LoRA Adapter Creation

```bash
$ python create_cosmos3_test_lora.py --model-path /path/to/Cosmos3-Super --output ./test-lora --rank 8
Creating test LoRA adapter (rank=8) for Cosmos3-Super
Found 896 LoRA-targetable linear layers
Wrote 1792 LoRA weight tensors (256.2 MB) to ./test-lora/adapter_model.safetensors
```

Output structure:
```
./test-lora/
├── adapter_config.json      (r=8, alpha=16, peft_type=LORA)
└── adapter_model.safetensors (256 MB, bf16 zero-init)
```

### Notes

- Guardrails disabled (`--no-guardrails`); `nvidia/Cosmos-1.0-Guardrail` not shipped
- FP8 quantization not yet validated on Ascend NPU
- LoRA applies to `transformer` only (not VAE, vision encoder, or sound tokenizer)
- Per-request LoRA uses LRU cache (default `max_cpu_loras=1`)
- Adapter must be PEFT format (`adapter_config.json` + `adapter_model.safetensors`)
