# [Bugfix] Bagel: restore per-sequence masking for batched-CFG (block-diagonal mask) + enable GQA on cuDNN

## Purpose

Bagel text-to-image generation was producing degraded, washed-out images.

**Root cause.** An earlier change replaced Bagel's `flash_attn_varlen` attention (which was block-diagonal per sequence via `cu_seqlens`) with a dense SDPA path that ignores per-sequence boundaries. The single-GPU CFG denoising path ("batched-CFG") packs all CFG branches (conditional / text-uncond / image-uncond) into **one** forward and relied on that implicit per-sequence masking. Once it was gone, the branches **cross-attended and contaminated each other**, blending the conditional and unconditional velocities every denoising step → washed-out output.

**Fix.** Add an explicit **block-diagonal attention mask** to the batched-CFG attention so each query token only attends to keys in its own CFG branch (including that branch's KV-cache segment). The mask is built once per step and reused across layers. Batched-CFG remains the default; `VLLM_OMNI_BAGEL_SEPARATE_CFG=1` opts into per-branch forwards (lower peak memory), and SP always uses per-branch forwards.

**Secondary fix (Blackwell / sm_120).** Bagel could not run on the default diffusion attention backend there (`CUDNN_ATTN`) because it called `scaled_dot_product_attention` without `enable_gqa`, so Bagel's grouped-query attention (28 query heads vs 4 KV heads) hit `"No available kernel"`. We pass `enable_gqa` in the cuDNN backend. This also makes the masked batched-CFG path memory-efficient on Blackwell: cuDNN's FMHA does not materialize the score matrix, whereas the SDPA `MATH` fallback (the only GQA-capable SDPA kernel there) does and OOMs at 1024².

## Test Plan

Text-to-image with the standard Bagel CFG settings, comparing output before and after the fix:

- Model: `ByteDance-Seed/BAGEL-7B-MoT`
- Prompt: `A cute cat`
- `seed=52`, `num_inference_steps=50`, `cfg_text_scale=4.0`, `cfg_img_scale=1.5`
- Resolution: `1024x1024`
- Hardware: NVIDIA RTX PRO 6000 (Blackwell, sm_120)

"Before" was generated from the pre-fix code (batched-CFG without the mask). Both are validated against the offline `bagel/end2end.py` reference pixels (and the upstream BAGEL reference image) via per-channel max delta at fixed coordinates.

## Test Result

| | Text prompt | Output (1024×1024, 50 steps, seed 52) | Pixel match vs reference (lower is better) |
|---|---|---|---|
| **Before fix** (batched-CFG, no mask) | `A cute cat` | ![before](docs/assets/bagel_cfg_fix/before_fix.png) | sum of per-channel maxΔ = **196** (washed-out / desaturated; CFG branches contaminated) |
| **After fix** (block-diagonal mask) | `A cute cat` | ![after](docs/assets/bagel_cfg_fix/after_fix.png) | sum of per-channel maxΔ = **98** (matches the upstream BAGEL reference, ≈97) |

After the fix, the generated image is sharp and correctly colored, and matches the upstream BAGEL output (which runs each CFG branch as a separate forward) to within ~1 pixel value. The default cuDNN backend on Blackwell now runs Bagel out of the box (no `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` workaround needed), and the batched-CFG path fits comfortably in memory there.

### Notes / scope
- Files changed: `vllm_omni/diffusion/models/bagel/bagel_transformer.py` (block-diagonal mask + batched/separate CFG routing), `vllm_omni/diffusion/attention/backends/cudnn_attn.py` (`enable_gqa`).
- Verified on RTX PRO 6000 (sm_120). Datacenter Blackwell (B200, sm_100) was not tested.
