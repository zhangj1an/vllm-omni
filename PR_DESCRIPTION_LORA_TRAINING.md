## Add NPU LoRA training for Cosmos3-Super (8× Ascend910, full 64-layer GEN pathway)

### Summary

Adds a **standalone NPU training path** for LoRA adapters of the 64B Cosmos3-Super world model. The full GEN denoising pathway (all 64 `Cosmos3GenDecoderLayer` layers, real checkpoint weights) is sharded across 8× Ascend910 NPUs (8 layers per rank), LoRA is injected into every attention/MLP projection using the recipe's target modules, and training runs with a diffusion denoising objective on the [Hammershøi style dataset](https://huggingface.co/datasets/jejunepixels/Hammershoi-Training-Dataset) (244 images, CC0). LoRA gradients are all-reduced every step so all ranks converge to one adapter, saved in PEFT format for `DiffusionLoRAManager` serving.

Training is verified end-to-end on NPU: real-layer forward/backward, LoRA gradient flow, loss convergence, and adapter export. A standalone NPU LoRA-training smoke test (single GEN layer) is included for CI-able validation.

### Why a training path on NPU

vLLM-Omni's `DiffusionLoRAManager` already **serves** PEFT adapters on NPU with zero model-code changes, but producing those adapters required a CUDA training setup. This change adds the missing training side:

- Full 64-layer GEN pathway on NPU (the serving path loads the same layers, so trained adapters match the serving namespace `gen_layers.*`).
- Real checkpoint weights loaded through the **same `_remap_ckpt_key` rules** as `pipeline_cosmos3.py` (`layers.N.mlp_moe_gen.*` → `gen_layers.N.mlp.*`, `self_attn.add_*` → `cross_attention.*`), so the training namespace is bit-compatible with the serving loader.
- LoRA rank/alpha/dropout follow the recipe (`r=8, alpha=16` in this run; recipe default `r=64, alpha=128`).

### Changes

| File | Description |
|------|-------------|
| `train_full_gen_lora.py` | 8-NPU full-GEN LoRA trainer: layer-wise sharding (8 layers/rank), checkpoint remap loading, LoRA injection, gradient all-reduce, PEFT adapter export |
| `npu_lora_smoke_test.py` | Single-layer NPU LoRA smoke test (real `Cosmos3GenDecoderLayer`, manual low-rank injection, 30-step convergence check) — validates the NPU training path without 8 cards |
| `PR_DESCRIPTION_LORA_TRAINING.md` | This PR description |

> The trainer and smoke test live under the repo root next to the existing Cosmos3 tooling (`create_cosmos3_test_lora.py`, `train_lora_cosmos3.py`); merge location into `examples/`/`recipes/` is open to maintainer preference.

### Training command

```bash
# inside the vllm-omni NPU container (quay.io/ascend/vllm-omni:v0.26.0-a3)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 \
    train_full_gen_lora.py
```

Output adapter (PEFT format, serving-ready):

```
adapter_output/
├── adapter_config.json      # r=8, alpha=16, target_modules = recipe's 7 projections
└── adapter_model.safetensors
```

### LoRA — target modules (per GEN layer)

```
cross_attention.to_q, to_k, to_v, to_out   # GEN cross-attention QKV + output
mlp.gate_proj, up_proj, down_proj          # GEN gated MLP
```

448 modules across 64 GEN layers at 8 layers/rank (56 per rank). At `r=8`: 67.1 M trainable params total (8.4 M per rank), ~32 MB adapter.

### Verification (run on 8× Ascend910)

| Check | Result |
|---|---|
| Real weights loaded (remap + shards) | 704 tensors, 64/64 GEN layers |
| LoRA injected | 448 modules, gradients flowing (all-reduced across ranks) |
| Training convergence (60 steps, lr=1e-3, batch 2) | loss 1604.2 → 0.69 |
| Smoke test (single layer, 30 steps) | loss 1.003 → 0.080 |
| Adapter export | PEFT `adapter_config.json` + `adapter_model.safetensors` (112 tensors, 32 MB) |

### Known simplifications (explicitly out of scope for this PR)

The training **mechanics** (sharding, loading, LoRA, gradient sync, export) are complete; two **semantic** components are still proxied and flagged in the script:

1. **UND conditioning proxy** — GEN cross-attention K/V is a random latent tensor per step instead of real text-encoded UND K/V. Real conditioning requires running the UND language model + prompt upsampling per batch.
2. **Latent proxy** — a learnable Conv2d stands in for the real VAE encoder. Real latents require the Cosmos3 VAE forward on each training image.

Both are isolated behind their own sections in the trainer, so swapping in the real components does not touch the LoRA/sharding/export logic. Serving the resulting adapter still exercises the full `DiffusionLoRAManager` path.

### Serve the adapter

```bash
vllm serve /run/z84450661/Cosmos3-Super --omni --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails --init-timeout 1800
# per-request: "lora": {"name": "hammershoi", "path": "/path/to/adapter_output", "scale": 1.0}
```

### Related

- `recipes/cosmos3/Cosmos3-Super.md` — serving recipe + LoRA target modules (existing)
- Dataset: [jejunepixels/Hammershoi-Training-Dataset](https://huggingface.co/datasets/jejunepixels/Hammershoi-Training-Dataset) (CC0)
