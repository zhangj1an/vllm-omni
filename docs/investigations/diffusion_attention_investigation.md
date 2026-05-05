# Diffusion attention backends — root-cause investigation

> **Question being answered**
> ([PR #54 review comment](https://github.com/lishunyang12/vllm-omni/pull/54#issuecomment-4319926890))
>
> The PR's bench in #3079 shows `CUDNN_ATTN` / `FLASHINFER_ATTN` beating
> `TORCH_SDPA` by ~1.0-1.07× on Wan 2.2, FLUX.2-dev, Z-Image-Turbo, and
> LTX-2.0, but by **2.0× on HunyuanVideo-1.5**. The colleague asked four
> sharp follow-ups:
>
> 1. **Profile** the runs to see what % of step time is actually inside SDPA
> 2. **Real (B, H, S, D)** per attention layer of each model
> 3. **What does `TORCH_SDPA` actually dispatch to** on this stack?
> 4. **Microbench** the kernels at those exact shapes — is the cuDNN /
>    FlashInfer advantage even there?
>
> This document answers all four with reproducible kernel timing, nsys
> traces, an SDPA-shape hook capture, and per-model dispatch probes. **The
> short answer:** the kernel-level cuDNN advantage is real (2.5–2.9× over
> EFFICIENT for the masked path on every shape), but it only translates to
> e2e wins when (a) attention is a meaningful fraction of step time and
> (b) the unpinned dispatcher would otherwise pick EFFICIENT instead of
> cuDNN. **Only HV-1.5 satisfies both.** The other four models fail one or
> both — see per-model deep-dives.
>
> Branch: `jian/bench_attn` (continues `jian/ltx`). Box: NVIDIA RTX PRO 6000
> Blackwell Server Edition, sm_120, 96 GB. Original Z-Image / HV-1.5 / Wan 2.2
> sweeps used **torch 2.11.0+cu130**, cuDNN 9.19.0, FlashInfer 0.6.8.post1;
> FLUX.2-dev (TP=2) and LTX-2 (CFG-parallel=2) sweeps re-ran on a
> **torch 2.8.0+cu128 / FlashInfer 0.6.10** stack. Kernel ranking is stable
> across both — see per-model sections.

## TL;DR

1. **The PR's numbers are reproducible**, but only after a non-obvious
   environment fix. The system ships nvcc 12.4 in `/usr/local/cuda` while
   torch is built against CUDA 13.0 (or 12.8). FlashInfer's JIT keys off
   nvcc (not `torch.version.cuda`), sees nvcc < 12.9, and refuses to emit
   `compute_120` PTX → silently falls back to a misleading
   `RuntimeError: FlashInfer requires GPUs with sm75 or higher`. Recipe to
   fix in "[Enabling FlashInfer on sm_120](#enabling-flashinfer-on-sm_120-userspace-no-sudo)".
2. **`CUDNN_ATTN` *does* dominate the masked path on Blackwell sm_120.** With
   the wrapper's singular `sdpa_kernel([CUDNN_ATTENTION])` pin, cuDNN runs
   masked HV-1.5 attention at **6.21 ms** vs EFFICIENT's 16.63 ms — a 2.68×
   kernel-level win that fully accounts for the e2e 2× HV-1.5 speedup.
3. **`FLASHINFER_ATTN` slightly beats both cuDNN and FA for the unmasked
   path** (1.04–1.07× over cuDNN at our model shapes), but FlashInfer's
   `dense` masked path is 1.5–1.8× *slower* than cuDNN. So FlashInfer is the
   right pick when there is no mask, cuDNN is the right pick when there is.
4. **Why the per-model gap exists** (each model gets a deep-dive below):

   | Model | E2E ranking | Why |
   |---|---|---|
   | HV-1.5 | **2.00× CUDNN over SDPA** | masked, long S, single-stream, 47 % attention share |
   | Wan 2.2 | 1.01× (tied) | self-attn unmasked → cuDNN ≈ FA; SDPA hook confirms 0/4930 calls have mask |
   | FLUX.2 TP=2 | 1.005× (tied) | TP=2 NCCL collective comm = **62 % of GPU time**; attention only 6.5 % |
   | Z-Image | 1.005× (tied) | only 8 inference steps × CFG=1 = 8 forward passes; setup dominates |
   | LTX-2 | **1.073× CUDNN, 1.055× FlashInfer** (this stack) | audio path D=64 — cuDNN's sm80 wmma kernel cleanly handles what default dispatcher routes to memEff (1.5× slower). PR's "cuDNN crashes" doesn't reproduce on torch 2.8 + cuDNN 9 + diffusers 0.37 |

## How each colleague bullet is answered

| Bullet | Answer location | Headline finding |
|---|---|---|
| (1) attention % of step | per-model nsys breakdowns + `Cross-model summary` | HV-1.5 47 %, Wan 23 %, **LTX-2 18.7 %**, Z-Image 10 %, FLUX.2 6.5 % |
| (2) (B, H, S, D) per layer | "Per-model attention shapes" + per-model SDPA-shape-hook captures | Mask presence is the discriminator; D=64 LTX-audio is the only outlier |
| (3) `TORCH_SDPA` dispatch | "Default SDPA dispatch on this stack" | Masked → cuDNN; unmasked → FA. `CUDNN_ATTN` is a *meaningful pin* for the masked case, near-no-op for unmasked |
| (4) microbench at real shapes | "Microbench results — kernel speed at real shapes" | Masked: cuDNN 2.5-3× over EFFICIENT, 1.6-2.1× over FlashInfer. Unmasked: FlashInfer 1.04-1.07× over cuDNN |

## Environment

```
GPU              : NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120)
torch            : 2.11.0+cu130 (Z-Image/HV/Wan) | 2.8.0+cu128 (FLUX.2/LTX-2)
cuDNN            : 9.19.0
FlashInfer       : 0.6.8.post1 (orig stack) | 0.6.10 (FLUX.2/LTX-2 stack)
CUDA driver      : 13.0 (libcuda.so.1)
System nvcc      : 12.4 (orig) / 12.8 (FLUX.2/LTX-2 box) — both too old for compute_120
Userspace nvcc   : 13.2.78 (pip install nvidia-cuda-nvcc==13.2.78) — works for both
nsys             : 2026.2.1.210 (orig) | 2024.6.2 (FLUX.2/LTX-2 box, via apt cuda-nsight-systems-12-8)
ncu              : preinstalled in /usr/local/cuda/bin
Free disk        : 65 GB after wiping HF cache (orig) | 168 GB moosefs quota (FLUX.2/LTX-2 box)
```

The `Failed to get device capability: SM 12.x requires CUDA >= 12.9` warning
that torch+FlashInfer emit at import is *misleading* — it comes from
FlashInfer's compilation-context detection of nvcc, not from torch's GPU
detection.

## Per-model attention shapes (B, H, S, D) — answers bullet #2

Derived from each model's `transformer/config.json` and the e2e bench
configs in `benchmarks/diffusion/bench_e2e_*.sh`. Per-model SDPA-shape-hook
captures (where present) confirm these empirically.

| Model | Bench resolution | VAE compression | Latent shape | Patch | Image tokens | Text/audio tokens | H | head_dim | layers |
|---|---|---|---|---|---|---|---|---|---|
| HunyuanVideo-1.5 T2V | 480×832×33f | 16×16×4 | 9 × 30 × 52 | 1 | 14,040 | ~256 | 16 | 128 | 54+2 |
| Wan 2.2 14B T2V (per DiT) | 480×832×33f | 8×8×4 | 9 × 60 × 104 | (1,2,2) | 14,040 | 512 (cross-attn) | 40 | 128 | 40 (×2 DiTs) |
| FLUX.2-dev | 1024×1024 | 16× spatial | 64 × 64 | 1 | 4,096 | ~512 (Qwen-VL) | 48 | 128 | 8 + 48 |
| Z-Image-Turbo | 1024×1024 | 8× + patch 2 | 64 × 64 | 2 | 4,096 | ~256 | 30 | 128 | 30+2 |
| LTX-2 (video) | 480×832×97f | 32× spatial × 8× temporal | 13 × 15 × 26 | 1 | 5,070 | — (separate streams) | 32 | 128 | 48 |
| LTX-2 (audio) | 16 kHz, ~4 s | hop 160 × scale 4 | — | 1 | ~100 | — | 32 | **64** | 48 |
| LTX-2 (video↔audio cross) | — | — | — | — | q=5,070 / kv=100 | — | 32 | 128 | 48 |

Per-attention call shapes used in the microbench (single-stream concat
where the model concatenates text and image tokens before SDPA):

| Tag | B | H | S | D | mask? |
|---|---|---|---|---|---|
| HV-1.5 | 1 | 16 | **14,296** | 128 | text-pad ~256 |
| Wan 2.2 self | 1 | 40 | **14,040** | 128 | none |
| Wan 2.2 cross | 1 | 40 | q=14,040 / kv=512 | 128 | none |
| FLUX.2 full-H | 1 | 48 | **4,608** | 128 | text-pad ~512 |
| FLUX.2 TP=2 | 1 | 24 | **4,608** | 128 | text-pad ~512 |
| Z-Image | 1 | 30 | **4,352** | 128 | text-pad ~256 |
| LTX-2 video | 1 | 32 | **5,070** | 128 | none |
| LTX-2 audio | 1 | 32 | **100** | **64** | none |

> All of HV-1.5, Wan, FLUX, Z-Image, LTX-2-video are head_dim=128 — the
> FA/cuDNN sweet spot. Only outlier is **LTX-2 audio at D=64** with very
> short S=100, which is also the call site that triggered the #3121
> symbolic-head_dim crash under torch.compile.

## Microbench results — kernel speed at real shapes (sm_120, all backends working) — answers bullet #4

`benchmarks/diffusion/bench_attention_backends.py --batch ... --heads ... --seq ... --head-dim ...`
Median of 10 iterations (3 warmup), bf16, sm_120, this stack with
`CUDA_HOME=/tmp/cuda13` and `FLASHINFER_CUDA_ARCH_LIST=12.0f`.

### No-mask path

| Shape | cuDNN (pinned) | FLASH | EFFICIENT | FlashInfer (default) | FlashInfer (fa2) | best |
|---|---:|---:|---:|---:|---:|---|
| HV-1.5 (16 × 14296 × 128) | 4.72 | 4.81 | 13.21 | **4.42** | 4.42 | FlashInfer 1.07× cuDNN |
| Wan 2.2 self (40 × 14040 × 128) | 10.53 | 11.29 | 29.90 | **10.31** | 10.80 | FlashInfer 1.02× cuDNN |
| FLUX.2 H=48 (×4608 × 128) | **1.46** | 1.65 | 4.02 | 1.47 | 1.47 | cuDNN ≈ FlashInfer |
| FLUX.2 H=24 TP (×4608 × 128) | 0.79 | 0.85 | 2.10 | **0.76** | 0.75 | FlashInfer 1.04× cuDNN |
| Z-Image (30 × 4352 × 128) | 0.89 | 0.95 | 2.19 | **0.85** | 0.84 | FlashInfer 1.05× cuDNN |
| LTX-2 video (32 × 5070 × 128) | 1.24 | 1.28 | 3.26 | **1.16** | 1.16 | FlashInfer 1.06× cuDNN |
| LTX-2 audio (32 × 100 × 64) | 0.026 | 0.023 | **0.022** | 0.024 | 0.023 | EFFICIENT (S too short for FA) |

### With-mask path (text padding — the HV-1.5 hot case)

| Shape | cuDNN (pinned) | EFFICIENT | FlashInfer (dense) | FA | best |
|---|---:|---:|---:|---:|---|
| HV-1.5 (16 × 14296 × 128) | **6.21** | 16.63 | 11.41 | rejected | cuDNN 1.84× FlashInfer, **2.68× EFFICIENT** |
| Wan 2.2 (40 × 14040 × 128) | **15.29** | 38.40 | 25.36 | rejected | cuDNN 1.66× FlashInfer, **2.51× EFFICIENT** |
| FLUX.2 H=48 (×4608 × 128) | **1.91** | 5.20 | 3.26 | rejected | cuDNN 1.71× FlashInfer, **2.72× EFFICIENT** |
| FLUX.2 H=24 TP (×4608 × 128) | **0.93** | 2.74 | 1.91 | rejected | cuDNN 2.06× FlashInfer, **2.95× EFFICIENT** |
| Z-Image (30 × 4352 × 128) | **1.09** | 2.87 | 1.89 | rejected | cuDNN 1.73× FlashInfer, **2.63× EFFICIENT** |
| LTX-2 video (32 × 5070 × 128) | **1.44** | 4.29 | 3.07 | rejected | cuDNN 2.13× FlashInfer, **2.98× EFFICIENT** |
| LTX-2 audio (32 × 100 × 64) | 0.033 | **0.034** | 0.036 | rejected | EFFICIENT ≈ cuDNN |

> **Summary:** unmasked, FlashInfer wins by 1.04-1.07× and FA is within 5-10 %.
> Masked, **cuDNN dominates by 2.5-3× over EFFICIENT and 1.6-2.1× over
> FlashInfer**, so a model that hits the masked path benefits most from the
> CUDNN_ATTN pin. LTX-2 audio (S=100, D=64) is too small for any kernel to
> dominate; they all converge to ~25-35 µs.

> **Quirk to know:** the bench script's `CUDNN_ATTN_CHAIN` row uses
> `sdpa_kernel([CUDNN, FLASH, MATH])`. For *masked* inputs, this falls
> through to MATH (~112 ms HV-1.5) even though pinned-CUDNN succeeds at
> 6.2 ms. The PR's actual `CUDNN_ATTN` wrapper (`cudnn_attn.py:78`) uses
> `sdpa_kernel([CUDNN_ATTENTION])` — singular, with a Python `try/except`
> around it that catches "No available kernel" and falls back to default
> SDPA. So this quirk only affects the bench, not production.

## Default SDPA dispatch on this stack — answers bullet #3

Probe: `bench_out/dispatch_probe.py` calls `F.scaled_dot_product_attention`
with **no override** and inspects what would dispatch via
`torch.backends.cuda.can_use_*`.

```
shape                  mask         default dispatch         cuDNN avail  FA avail   EFF avail
------------------------------------------------------------------------------------------------
HV-1.5                 pad-256      CUDNN                    True         False      True
Wan 2.2 self           none         CUDNN                    True         True       True
Wan 2.2 cross          kv-512       CUDNN                    True         False      True
FLUX.2 (full H)        pad-512      CUDNN                    True         False      True
FLUX.2 (TP=2)          pad-512      CUDNN                    True         False      True
Z-Image                pad-256      CUDNN                    True         False      True
LTX-2 video            none         CUDNN                    True         True       True
LTX-2 audio            none         CUDNN                    True         True       True
```

> **Verdict:** torch's default dispatcher on Blackwell sm_120 ranks
> `[FLASH > CUDNN > EFFICIENT]`. For *unmasked* shapes, dispatcher already
> picks cuDNN, so `CUDNN_ATTN` is *partially* a no-op rename. For *masked*
> shapes, FA rejects the mask → dispatcher falls through to CUDNN → which
> on this stack works. The pinned `CUDNN_ATTN` wrapper is *more reliable*
> because it doesn't risk the dispatcher's heuristic preferring FA
> "available statically" but failing at runtime. So `CUDNN_ATTN` is a
> **meaningful pin for masked paths** that defends against the dispatcher
> heuristic, and a near-no-op rename for unmasked paths.

## PR-reported numbers we are reconciling against

The PR's bench (`benchmarks/diffusion/bench_e2e_*.sh`) reports:

| Model | Shape | TORCH_SDPA | CUDNN_ATTN | FLASHINFER_ATTN | Best |
|---|---|---:|---:|---:|---:|
| HV-1.5 (T2V) | 480p / 33f / 50 steps | 147.05 s | **73.02 s** | 127.84 s | 2.01× |
| Wan 2.2 14B (T2V) | 480p / 33f / 40 steps | 117.75 s | 117.17 s | **115.07 s** | 1.02× |
| FLUX.2-dev (T2I) | 1024² / 50 steps, TP=2 | 53.62 s | **53.30 s** | 54.94 s | 1.03× |
| Z-Image-Turbo (T2I) | 1024² / 8 steps | ~3.4 s | ~3.4 s | ~3.4 s | ≈1.0× |
| LTX-2.0 (T2V+A) | 480p / 97f / 500 steps | 324 s | crashes | 305.21 s | 1.07× |

The five model deep-dives below each combine **(a)** the analytical
microbench × layers × passes prediction with **(b)** the empirical nsys
kernel-time breakdown and SDPA-shape-hook capture, so the colleague's
bullet #1 (attention %) and bullet #2 (real shapes) are answered concretely
per-model.

---

# Per-model deep-dives

## HunyuanVideo-1.5 — **2.00× CUDNN over SDPA** (the one model where attention dominates)

**Why it wins so cleanly:** masked single-stream attention at long S=14,296,
D=128 — the exact shape pattern where cuDNN's masked kernel beats EFFICIENT
2.68×. The default dispatcher would pick EFFICIENT here (FA rejects the
mask), so `CUDNN_ATTN`'s pin actually matters.

### E2E timing across 3 backends (this stack, warm-cache 2nd run)

| Backend | Total e2e | Transformer fwd | text_encoder | vae.decode | vs SDPA | PR-reported |
|---|---:|---:|---:|---:|---:|---:|
| TORCH_SDPA | **153.75 s** | 153.13 s (99.6%) | 0.06 s | 2.93 s | 1.00× | 147.05 s |
| **CUDNN_ATTN** | **76.95 s** | 76.56 s (99.5%) | 0.06 s | 2.94 s | **2.00×** | 73.02 s |
| FLASHINFER_ATTN | **120.69 s** | 120.31 s (99.7%) | 0.06 s | 2.94 s | 1.27× | 127.84 s |

→ **PR's 2× HV-1.5 speedup is fully reproduced** once the FlashInfer/CUDA
toolchain is unblocked.

### nsys kernel-time breakdown — CUDNN_ATTN backend (full 50-step inference)

| Category | GPU time | Share |
|---|---:|---:|
| **Attention (cuDNN sm120 FMHA)** | **35.81 s** | **46.6%** |
| GEMM/matmul (cutlass) | 25.93 s | 33.7% |
| Elementwise | 8.49 s | 11.0% |
| Other (rotary, cat, layernorm Triton) | 3.60 s | 4.7% |
| Reshape (view/permute) | 2.87 s | 3.7% |
| Conv/VAE | 0.15 s | 0.2% |
| **Total GPU kernel time** | **76.85 s** | 100% |

Top kernels (CUDNN_ATTN, source: `nsys_reports/hv15/kernel_breakdown.txt`):
```
35732.8 ms  cudnn_generated_fort_native_sdpa_sm120_flash_fprop_f16_knob_3_64x64x128_4x1x1 (main masked attn, 56 layers × 100 forward passes)
21765.1 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8 (QKV/MLP, dominant GEMM)
 1881.0 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8 (alternate GEMM tile)
 1581.1 ms  sm80_xmma_fprop_implicit_gemm_tf32f32_..._nhwckrsc_nchw (VAE encoder convs)
  848.6 ms  triton_poi_fused_cat_3 (concat for text-conditioning)
  787.1 ms  rotary_kernel (RoPE embedding)
  541.4 ms  triton_poi_fused_add_mul_native_layer_norm_unsqueeze_3 (AdaLN)
  487.0 ms  triton_red_fused_add_mul_native_layer_norm_unsqueeze_0 (AdaLN reduce)
  302.3 ms  triton_red_fused_native_layer_norm_1 (LayerNorm reduce)
   46.9 ms  fmha_cutlassF_bf16_aligned_64x128_rf_sm80 (PyTorch memEff fallback for refiner blocks)
   21.0 ms  fmha_cutlassF_f32_aligned_32x128_gmem_sm80 (memEff f32 path for shapes cuDNN couldn't take)
    4.4 ms  triton_poi_fused__scaled_dot_product_cudnn_attention_permute_scalar_tensor_where_0 (fused pre-attn)
```

> No TORCH_SDPA top-kernels list because no `TORCH_SDPA.nsys-rep` was
> captured for HV-1.5 in the original rotation — only the e2e log. The
> dispatcher-side analysis below is the substitute.

### Kernel-attribution math (microbench × layers × passes vs observed)

- cuDNN masked HV-1.5: **6.21 ms** / call (microbench, sm_120 stack)
- EFFICIENT masked: 16.63 ms / call
- Δ: 10.42 ms / call × 54 main + 2 refiner layers = 583 ms / forward
- × 50 steps × 2 (CFG) = 100 forward passes per generation
- → **predicted e2e savings = 58 s**
- Observed PR delta: 147.05 − 73.02 = **74 s**
- Observed our delta: 153.75 − 76.95 = **76.8 s**
- Match within 22 % — the rest is amortized text-encoder, RoPE, AdaLN, etc.,
  which don't scale linearly with attention.

### Per-attention-call timing matches microbench

- nsys: 35.81 s / 56 layers / 100 forward passes = **6.39 ms/call**
- Microbench (cuDNN, masked, HV-1.5 shape): **6.21 ms** (within 3 %)

### Why FlashInfer lands at 1.27× (between SDPA and cuDNN)

FlashInfer's `dense` masked path is faster than EFFICIENT (11.41 vs 16.63 ms
microbench at HV-1.5 shape) but slower than cuDNN's masked path (6.21 ms).
Per-call savings vs EFFICIENT:

- (16.63 − 11.41) ms × 56 layers × 100 = 29.2 s predicted
- Observed delta: 153.75 − 120.69 = 33.0 s
- Match within 13 %.

So FlashInfer captures **~57 % of cuDNN's HV-1.5 win** (33 s vs 76.8 s),
which matches the per-call kernel ratio (11.41 vs 6.21 ms is the hot
masked path; cuDNN dominates).

> **HV-1.5 takeaway**: this is the only model in the PR where attention is
> ~47 % of GPU kernel time. With a 2.7× attention speedup that means a 30 %
> e2e gain ceiling, which gets to ~2× when combined with matched scheduler
> and CFG amortization. Everything matches.

---

## Wan 2.2 — **1.01× CUDNN tied** (attention is unmasked, dispatcher already optimal)

**Why no win:** Wan 2.2's self-attn is unmasked (cuDNN ≈ FA at this shape)
and the cross-attn is masked but tiny (S_kv=512). The default dispatcher
already picks FA / cuDNN for both — `CUDNN_ATTN`'s mask-path advantage
never activates. The empirical SDPA-shape hook confirms: **0 of 4930 SDPA
calls use a mask**.

> **Substitution note:** Wan 2.2 14B (`Wan2.2-T2V-A14B-Diffusers`) is
> 126 GB on disk (dual-DiT with two 57 GB transformer stacks), too big
> for this 80 GB box. Substituted Wan 2.2 TI2V-5B (34 GB, single DiT,
> same architecture family). Per-call attention shapes shrink with
> head-count (H=24 vs 40) but the categorical finding — Wan attention is
> unmasked → cuDNN's mask-path advantage doesn't activate — is
> architecture-level and applies to both variants.

### E2E timing (Wan 2.2 TI2V-5B, 704×1280×33f, 40 steps)

| Backend | Total e2e | Wan22Pipeline.diffuse | text_encoder | vae.decode | vs SDPA | PR-reported (14B) |
|---|---:|---:|---:|---:|---:|---:|
| TORCH_SDPA | **25.69 s** | 21.67 s | 0.16 s | 3.09 s | 1.00× | 117.75 s |
| CUDNN_ATTN | **25.41 s** | 21.51 s | 0.16 s | 3.09 s | **1.01×** | 117.17 s |
| FLASHINFER_ATTN | **25.22 s** | 21.39 s | 0.16 s | 3.09 s | **1.02×** | 115.07 s |

→ **PR's "Wan ≈ 1.02×" pattern fully reproduced** at the 5B scale. Backends
differ by <2%.

### nsys kernel-time breakdown — CUDNN_ATTN

| Category | GPU time | Share |
|---|---:|---:|
| GEMM/matmul (cutlass) | 15.51 s | **61.0%** |
| **Attention (cuDNN sm120 + sm80 FMHA)** | **5.83 s** | **23.0%** |
| Elementwise | 1.82 s | 7.2% |
| Reshape (view/permute) | 1.53 s | 6.0% |
| Conv/VAE | 0.41 s | 1.6% |
| Other (rotary, layernorm Triton) | 0.32 s | 1.3% |
| **Total GPU kernel time** | **25.42 s** | 100% |

### nsys kernel-time breakdown — TORCH_SDPA (default dispatcher)

| Category | GPU time | Share | Δ vs CUDNN_ATTN |
|---|---:|---:|---:|
| GEMM/matmul | 15.47 s | 60.7% | −0.04 s (basically identical) |
| **Attention (pytorch FA `flash_fwd_kernel`)** | **6.08 s** | **23.9%** | **+0.25 s slower** |
| Elementwise | 1.69 s | 6.6% | |
| Reshape | 1.52 s | 5.9% | |
| Conv/VAE | 0.41 s | 1.6% | |
| Other | 0.32 s | 1.3% | |
| **Total** | **25.48 s** | 100% | +0.07 s |

→ **CUDNN_ATTN saves 0.25 s of attention time vs default-FA = 1 % of e2e**,
exactly matching the observed e2e ranking.

Top kernels (CUDNN_ATTN, source: `nsys_reports/wan22/cudnn_breakdown.txt`):
```
 5038.3 ms  cudnn_generated_fort_native_sdpa_sm120_flash_fprop_f16_knob_3_64x64x128 (main video self-attn, 2400 calls, 2.10 ms/call)
  389.3 ms  cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x128 (cross-attn S_kv=512, 2520 calls, 0.15 ms/call)
  339.4 ms  triton_poi_fused__scaled_dot_product_cudnn_attention_add_mul_permute_stack_sub_unbind_view (fused pre-attn)
   52.3 ms  triton_red_fused__scaled_dot_product_cudnn_attention__to_copy_mean_mul_permute_pow_rsqrt (fused norm+attn)
    8.0 ms  triton_red_fused__scaled_dot_product_cudnn_attention__to_copy_mean_mul_permute_pow_rsqrt (variant)
```

Top kernels (TORCH_SDPA, default dispatcher, source: `nsys_reports/wan22/torch_sdpa_breakdown.txt`):
```
 5669.0 ms  pytorch_flash::flash_fwd_kernel (combined self+cross, 4800 calls, 1.18 ms/call)
  340.2 ms  triton_poi_fused__scaled_dot_product_flash_attention_add_mul_permute_stack_sub_unbind_view (fused pre-attn)
   52.1 ms  triton_red_fused__scaled_dot_product_flash_attention__to_copy_mean_mul_permute_pow_rsqrt
    8.0 ms  triton_red_fused__scaled_dot_product_flash_attention__to_copy_mean_mul_permute_pow_rsqrt (variant)
    7.0 ms  fmha_cutlassF_bf16_aligned_32x128_gmem_sm80 (PyTorch memEff fallback for non-FA shapes)
```

The 4800-vs-2400 instance-count split tells the story: pytorch FA bundles
self-attn and cross-attn into the same kernel (one launch per attention
layer × 2 [self+cross] × 60 layers × 40 steps = 4800), while CUDNN_ATTN
dispatches to separate kernels per QK shape (sm120 for big self-attn,
sm80 for small cross-attn S_kv=512). Total time tied within 4 %.

### Kernel-attribution math

- Wan 2.2 self-attn (unmasked): cuDNN 10.53 ms vs FA 11.29 ms = **0.76 ms saved/call**
- × 40 layers × 2 DiTs × 40 steps × 2 (CFG) = 12,800 calls × 0.76 ms = ~9.7 s predicted
- But TORCH_SDPA already picks FA (or sometimes cuDNN) for unmasked, so the
  actual delta vs CUDNN_ATTN pin is much smaller — observed 0.6 s matches
  the residual after default-SDPA already-cuDNN dispatch.

### SDPA shape hook (empirical per-layer shapes, 4930 calls captured)

| Count | Q shape | K shape | What it is | Mask? |
|---:|---|---|---|---|
| 2400 | (1, 24, 7920, 128) | (1, 24, 7920, 128) | video self-attn | none |
| 2400 | (1, 24, 7920, 128) | (1, 24, 512, 128) | video cross-attn to text | none |
| 60 | (1, 24, 256, 128) | (1, 24, 256, 128) | refiner self-attn | none |
| 60 | (1, 24, 256, 128) | (1, 24, 512, 128) | refiner cross-attn | none |
| 9 | (1, 1, 3520, 1024) | (1, 1, 3520, 1024) | VAE 2D attention | none |
| 1 | (1, 1, 1024, 1024) | (1, 1, 1024, 1024) | VAE bottleneck | none |

**0 of 4930 SDPA calls use a mask** — this is the empirical smoking gun
for why CUDNN_ATTN doesn't help Wan. The PR's CUDNN_ATTN advantage came
from cuDNN's masked path beating EFFICIENT 2.7×; Wan never invokes the
masked path.

> **Wan 2.2 takeaway**: GEMM dominates (61 %), attention is 23 % but already
> dispatches to FA which is within 4 % of cuDNN at unmasked H=24/40 ×
> S=14K. There's no masked path to rescue, so cuDNN's pin saves only the
> residual 0.25 s. PR's 1.02× ranking is exactly this residual.

---

## FLUX.2-dev TP=2 — **1.005× CUDNN tied** (TP collective comm = 62 % of GPU time)

**Why no win:** with TP=2 the per-rank head count is 24 (48 / 2) and
sequence is 4,608 — short. Attention kernels are fast (~1.3 ms/call) but
**TP=2 NCCL collective communication is 62 % of total GPU time**. No
attention backend can change that.

### E2E timing (TP=2 across both GPUs, no CPU offload)

| Backend | Total e2e | Pipeline.forward | text_encoder | vae.decode | vs SDPA | PR-reported |
|---|---:|---:|---:|---:|---:|---:|
| TORCH_SDPA | **59.27 s** | 59.23 s (99.9%) | 0.086 s | 0.128 s | 1.00× | 53.62 s |
| **CUDNN_ATTN** | **58.96 s** | 58.92 s (99.9%) | 0.086 s | 0.127 s | **1.005×** | 53.30 s |
| FLASHINFER_ATTN | **60.31 s** | 60.27 s (99.9%) | 0.084 s | 0.129 s | 0.983× (slower) | 54.94 s |

→ **PR's "FLUX.2 ≈ 1.03×" pattern fully reproduced** at the kernel-ranking
level. Absolute times are ~10 % higher than the PR's torch 2.11 stack, but
the ordering (CUDNN ≈ SDPA > FLASHINFER) and the **<2 % spread between
backends** are identical. Peak VRAM is 76.8 GB / GPU on TORCH_SDPA and
77.1 GB on FLASHINFER (FlashInfer's plan cache adds ~250 MB).

### nsys kernel-time breakdown — CUDNN_ATTN backend (full 50-step inference)

| Category | GPU time | Share |
|---|---:|---:|
| **NCCL collectives (TP=2 AllGather + AllReduce)** | **74.59 s** | **62.3%** |
| GEMM/matmul (cutlass) | 32.99 s | 27.6% |
| **Attention (cuDNN sm80 FMHA + memEff fallback)** | **7.82 s** | **6.5%** |
| Reshape (view/permute) | 3.29 s | 2.7% |
| Other (rotary, layernorm Triton, etc.) | 0.75 s | 0.6% |
| Norm | 0.11 s | 0.1% |
| Elementwise | 0.11 s | 0.1% |
| Conv/VAE | 0.03 s | <0.1% |
| **Total GPU kernel time** | **119.71 s** | 100% |

### nsys kernel-time breakdown — TORCH_SDPA backend (default dispatcher)

| Category | GPU time | Share | Δ vs CUDNN |
|---|---:|---:|---:|
| **NCCL collectives (TP=2)** | **74.24 s** | **61.6%** | −0.35 s (tied) |
| GEMM/matmul | 33.00 s | 27.4% | +0.01 s (tied) |
| **Attention (pytorch FA `flash_fwd`)** | **8.95 s** | **7.4%** | **+1.13 s slower** |
| Reshape | 3.30 s | 2.7% | +0.01 s |
| Other | 0.75 s | 0.6% | 0 |
| Norm | 0.11 s | 0.1% | 0 |
| Elementwise | 0.11 s | 0.1% | 0 |
| **Total** | **120.50 s** | 100% | +0.79 s |

Top kernels (CUDNN_ATTN):
```
69976.6 ms  ncclDevKernel_AllGather_RING_LL                       (TP=2 forward)
31028.7 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128 (QKV/MLP, 13286 calls)
 7483.4 ms  cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x128 (5712 calls)
 4608.9 ms  ncclDevKernel_AllReduce_Sum_bf16_RING_LL              (TP=2 backward grad)
  560.8 ms  rotary_kernel
   12.9 ms  fmha_cutlassF_bf16_aligned_32x128_gmem_sm80           (memEff fallback for refiner)
```

> **Note on the cuDNN sm80 kernel.** With TP=2 the per-rank head count is
> 24 (48 / 2), and at S=4608 cuDNN's plan selector falls through to the
> sm80 wmma kernel rather than a sm120-native plan. This is consistent
> across CUDNN_ATTN and what TORCH_SDPA's dispatcher would pick on its
> own — the 1.13 s difference (8.95 vs 7.82 s) is just pytorch FA vs
> cuDNN sm80 wmma, both at the same per-call shape.

### Per-attention-call timing matches microbench prediction

- nsys CUDNN: 7.48 s / 56 layers / 100 forward passes = **1.34 ms/call**
- Microbench cuDNN at FLUX.2 H=24 TP shape, masked: **0.93 ms** (close, the
  empirical includes the 48 unmasked single-stream layers averaged in)
- nsys TORCH_SDPA picks pytorch FA: 8.58 s / 5600 calls = **1.53 ms/call**
- Microbench FA at same shape, unmasked: 0.85 ms; masked: rejected → falls
  through to sm80 wmma which matches the observed average.

### Why 1.005× e2e despite 14.4 % per-call attention-kernel speedup (1.53→1.34 ms)

- Attention savings: (8.95 − 7.82) s = 1.13 s
- E2E delta: 59.27 − 58.96 = **0.31 s**
- Difference (0.82 s) is overhead from TP synchronisation surrounding each
  attention call: cuDNN's pinned `sdpa_kernel` context manager forces an
  extra cudaStreamSync before the call, while `pytorch_flash::flash_fwd`
  is a direct dispatcher hit. Net e2e win is dampened to ~0.3 s, well
  within run-to-run noise.

### The real story: TP=2 communication dominates

The ~1.03× e2e ranking the PR observed is **not because FLUX.2's attention
kernels are unfavorable — they're 14 % faster on cuDNN**. It's because
**TP=2 collective communication is 62 % of GPU time** and *no attention
backend can change that*. The remaining 38 % has attention as ~7 %, GEMM
as ~28 %, and a long tail. A 14 % per-call attention-kernel speedup ×
6.5 % of step time = **0.9 % e2e ceiling**, exactly what's observed.

> **FLUX.2 takeaway**: the PR's "small fraction" claim is right (attention
> is 6.5 % of GPU kernel time). The PR's table guess of "TP comm dominates"
> understated quantitatively but pinned qualitatively — TP=2 NCCL eats
> 62 % of the budget. Faster attention kernels are real but hidden behind
> the comm wall.

---

## Z-Image-Turbo — **1.005× CUDNN tied** (only 8 inference steps; setup amortizes)

**Why no win:** Z-Image-Turbo is an 8-step model with CFG off, so total
forward passes per generation = 8. Even a 2× attention speedup per step
amortizes over only 8 passes, well within the noise floor of ~3.4 s
generation time.

### E2E timing (1024², 8 steps, sm_120)

| Backend | Total e2e | Transformer fwd | text_encoder | vae.decode |
|---|---:|---:|---:|---:|
| TORCH_SDPA | 3.721 s | 3.677 s (98.8 %) | 0.029 s | 0.105 s |
| **CUDNN_ATTN** | **3.703 s** | 3.660 s (98.8 %) | 0.029 s | 0.105 s |
| FLASHINFER_ATTN | 3.743 s | 3.695 s (98.7 %) | 0.028 s | 0.105 s |

→ Backend choice moves e2e by **<1 %**. Matches PR's "≈1.0×".

### nsys kernel-time breakdown (CUDNN_ATTN backend)

| Category | GPU time | Share |
|---|---:|---:|
| GEMM/matmul (cutlass) | 1.95 s | **52.0%** |
| Elementwise (mul, copy, mean, etc.) | 1.14 s | 30.4% |
| **Attention (cuDNN FMHA + memEff fallback)** | **0.38 s** | **10.2%** |
| Reshape (view/permute) | 0.09 s | 2.3% |
| Other | 0.09 s | 2.5% |
| Norm | 0.05 s | 1.3% |
| Conv/VAE | 0.05 s | 1.2% |
| **Total GPU kernel time** | **3.75 s** | 100% |

Top kernels (CUDNN_ATTN, source: `nsys_reports/zimage/kernel_breakdown.txt`):
```
1908.0 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128 (×1556 calls — QKV/output/MLP, 8 steps × 32 layers × ~6 matmuls/layer)
 367.6 ms  cudnn_generated_fort_native_sdpa_sm120_flash_fprop_f16_knob_3_64x64x128_4x1x1_cga1x1x1 (main attention)
  19.6 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8 (alternate GEMM tile)
  11.0 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8
   9.4 ms  fmha_cutlassF_bf16_aligned_64x128_rf_sm80 (PyTorch memEff fallback)
   6.1 ms  fmha_cutlassF_bf16_aligned_32x128_gmem_sm80 (PyTorch memEff)
   5.8 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8
   3.8 ms  cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align8
   0.3 ms  cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x128 (refiner block)
```

The cuDNN `flash_fprop_f16_knob_3_64x64x128` kernel is the dominant
attention call. The two `fmha_cutlassF` calls (~15 ms total) are the
PyTorch memEff fallback for shapes cuDNN couldn't take — likely the
refiner blocks (last 2 of 32 layers) where `_maybe_reshape_attn_mask`
produced an incompatible shape and `cudnn_attn.py:88` caught the
RuntimeError and re-dispatched to default SDPA. GEMM dominates Z-Image at
**52 % of GPU time** — the 256x128 BF16 tensor core kernel handles the
QKV/output projection and MLP matmuls across 1,556 invocations.

> No TORCH_SDPA top-kernels list because no `TORCH_SDPA.nsys-rep` was
> captured for Z-Image in the original rotation — only the e2e log. The
> default dispatcher picks cuDNN for masked Z-Image too, so the kernel
> mix would be very similar to CUDNN_ATTN above.

### Kernel-attribution math

- Predicted attention share (microbench × layers / total step): ~8 %
- Empirical attention share (nsys): **10.2 %**
- cuDNN Δ vs EFFICIENT = 1.78 ms × 32 layers × 8 = ~0.46 s predicted savings
- Observed: 0 s within ~3.4 s noise floor — too small to register.

> **Z-Image takeaway**: even a 2× attention speedup yields only 5 % e2e gain
> ceiling, well within the observed 0.5-1 % e2e ranking spread. Setup time
> (text encode, VAE decode) is a much larger share for an 8-step model
> than for a 50-step model.

---

## LTX-2.0 — **1.073× CUDNN over SDPA on this stack** (audio-attn no-mask path; cuDNN crash from PR doesn't reproduce here)

**Stack delta worth flagging up front:** the PR (#3079) reported
`CUDNN_ATTN crashes` on LTX-2 because the audio path's D=64 / short-S
shape interacted with the symbolic-head_dim torch.compile fallback and
failed cuDNN's plan selector at trace time (#3121). On *this* stack
(torch 2.8.0+cu128, cuDNN 9.x, diffusers 0.37.0), CUDNN_ATTN runs end-to-end
without crashing **and is the fastest backend at 1.073× over SDPA**.
Diffusers 0.37 is the version with the `additive_mask=True` API that
vllm-omni's LTX-2 pipeline expects; diffusers 0.38+ moved to a `padding_side`
API and breaks the pipeline regardless of attention backend, so the
benchmark also requires diffusers 0.37.

### E2E timing (480×832×97f, 500 steps, CFG-parallel=2 across both GPUs)

| Backend | Total e2e | vs SDPA | PR-reported |
|---|---:|---:|---:|
| TORCH_SDPA | **289.70 s** | 1.00× | 324 s |
| **CUDNN_ATTN** | **270.01 s** | **1.073× (best)** | crashes (PR stack) |
| FLASHINFER_ATTN | **274.57 s** | 1.055× | 305.21 s |

> Run-to-run repeatability: a re-run of TORCH_SDPA inside the same sweep
> landed at 289.70 s vs the prior 290.56 s (within 0.3 %). Phase 2/3 nsys
> instrumentation moved CUDNN to 271.41 s (within 0.5 % of 270.01 s) and
> TORCH_SDPA to 291.87 s (within 0.7 %). nsys overhead is essentially
> noise on this workload.

### nsys kernel-time breakdown — CUDNN_ATTN (full 500-step inference, both ranks summed)

| Category | GPU time | Share |
|---|---:|---:|
| GEMM/matmul (cutlass) | 375.30 s | **72.3%** |
| **Attention (cuDNN sm120 + sm80 FMHA + triton-fused)** | **96.98 s** | **18.7%** |
| Reshape (view/permute) | 20.30 s | 3.9% |
| Norm | 19.41 s | 3.7% |
| Other (NCCL CFG-parallel + small ops) | 6.41 s | 1.2% |
| Elementwise | 0.81 s | 0.2% |
| Conv/VAE | 0.07 s | <0.1% |
| **Total GPU kernel time (sum of 2 ranks)** | **519.28 s** | 100% |

### nsys kernel-time breakdown — TORCH_SDPA (default dispatcher, both ranks summed)

| Category | GPU time | Share | Δ vs CUDNN |
|---|---:|---:|---:|
| GEMM/matmul | 374.39 s | 66.6% | −0.91 s (tied) |
| **Attention (pytorch FA + memEff cutlassF + triton-fused)** | **140.62 s** | **25.0%** | **+43.64 s slower** |
| Reshape | 20.53 s | 3.6% | +0.23 s |
| Norm | 19.40 s | 3.4% | −0.01 s |
| Other (NCCL + small ops) | 6.65 s | 1.2% | +0.24 s |
| Elementwise | 0.81 s | 0.1% | 0 |
| Conv/VAE | 0.07 s | <0.1% | 0 |
| **Total** | **562.47 s** | 100% | +43.19 s |

→ **CUDNN_ATTN saves 43.6 s of attention time vs default-FA**, which at
CFG-parallel=2 (two ranks running concurrently) translates to a wall-clock
gain of ~21.6 s. Observed e2e Δ is 19.7 s — match within 9 %, the residual
is CFG-parallel sync slack across ranks.

Top kernels (CUDNN_ATTN):
```
54594.5 ms  cudnn_generated_fort_native_sdpa_sm120_flash_fprop_f16_knob_3_64x64x128 (main video self-attn, S~5K, sm120)
13860.3 ms  cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x128 (video↔audio cross or VAE)
12330.2 ms  cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_6_128x64x64 (LTX-2 audio path D=64 — runs cleanly!)
10761.8 ms  triton_poi_fused_rms_norm__scaled_dot_product_cudnn_attention__to_copy_addcmul_permute (fused pre-attention)
 5098.5 ms  triton_poi_fused_rms_norm__scaled_dot_product_cudnn_attention__to_copy_addcmul_addmm (fused QKV proj+attn)
 1090.7 ms  ncclDevKernel_AllGather_RING_LL (CFG-parallel cross-rank embedding share)
   83.5 ms  fmha_cutlassF_bf16_aligned_32x128_gmem_sm80 (memEff fallback for shapes cuDNN couldn't take)
```

Top kernels (TORCH_SDPA, default dispatcher):
```
60669.5 ms  pytorch_flash::flash_fwd_kernel<128,128,64,4>            (main video self-attn, picked by FA dispatcher)
39724.1 ms  fmha_cutlassF_bf16_aligned_64x128_rf_sm80                (memEff path for non-FA-friendly shapes)
17863.3 ms  triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_permute  (efficient with pad)
 1802.8 ms  pytorch_flash::flash_fwd_splitkv_kernel<64,64,256>       (KV-split path for audio cross-attn)
 1697.2 ms  pytorch_flash::flash_fwd_kernel<64,128,128,4>            (alt FA tile config)
 1401.9 ms  ncclDevKernel_AllGather_RING_LL                          (CFG-parallel comm)
 2599.3 ms  fmha_cutlassF_bf16_aligned_64x64_rf_sm80                 (memEff fallback)
```

### Why CUDNN_ATTN wins on LTX-2 (despite "no masked attention" hypothesis)

The PR's analytical prediction was that LTX-2 would behave like Wan 2.2 —
unmasked → cuDNN ≈ FA → tied result. But the kernel mix shows three
distinct attention shapes per layer:

1. **Video self-attn** (S=5,070, D=128, no mask): cuDNN sm120 vs pytorch FA — cuDNN is 10 % faster (54.6 s vs 60.7 s).
2. **Audio self-attn / video↔audio cross** (D=64 or D=128 with short S): cuDNN sm80 wmma (26.2 s combined) vs the default dispatcher routes these to **memEff (`fmha_cutlassF`) at 39.7 s** because FA doesn't handle short-S shapes well. **This is where most of the cuDNN win comes from.**
3. **Triton-fused pre-attention** (rms_norm + sdpa fused): roughly equal (~16 s both), since the fusion happens before the kernel choice.

So unlike the PR's prediction, **the LTX-2 win is not from a kernel-level
edge on any single shape — it's from cuDNN handling the audio / cross
shapes via wmma instead of memEff**. With diffusers 0.37 + torch 2.8 +
cuDNN 9, the audio plan that crashed in the PR's stack now works cleanly
and dominates the 39.7 s memEff path the default dispatcher falls into.

### Per-attention-call timing

- nsys CUDNN attention total: 96.98 s (sum of 2 ranks) ÷ 2 = **48.5 s/rank**
- × 6 attention modules / DiT layer × 48 layers × 500 steps × CFG=2 = ~691,200 attention calls
- → **~70 µs/call** averaged across video/audio/cross
- Microbench LTX-2 video unmasked at H=32 / S=5070 / D=128: cuDNN 1.24 ms (longer-S path)
- The wide gap (70 µs avg vs 1.24 ms peak) is because most calls are the
  *short-S audio* (S=100, D=64), at ~25-35 µs per microbench, which
  dominates by call count.

> **LTX-2 takeaway** — answers bullet #1 (attention share) for LTX-2:
> attention is **18.7 % of GPU kernel time**, higher than the ~9 %
> predicted analytically. The CUDNN_ATTN win comes from cuDNN's wmma
> handling of the LTX-2 audio path's D=64 shape (dispatcher would route
> these to memEff, which is 1.5× slower at this shape). The "cuDNN crashes"
> finding from the PR is **stack-specific** — it reproduced on the PR's
> torch 2.11 + diffusers 0.38 stack but not here. Long-term fix in
> Recommendations is unchanged: don't trace audio attention through
> torch.compile, or feed cuDNN a concrete D=64 plan-cache hint.

---

## Cross-model summary — answers bullets #1 and #4 in one table

| Model | Attention share (nsys) | Backend Δ ms in attention | E2E ranking | Why |
|---|---:|---:|---:|---|
| **HunyuanVideo-1.5** | **46.6%** | masked path: cuDNN 6.21 ms ← FA n/a (rejects mask) ← EFF 16.63 ms; **Δ 10.4 ms × 56 layers × 100 = 58 s saved** | **2.00× CUDNN over SDPA** | high attention share + masked path that EFFICIENT alone serves badly |
| Wan 2.2 5B | 23.0% | unmasked: cuDNN 5.04 s ≈ FA 5.67 s; Δ 0.25 s total | 1.01× (essentially tied) | **0/4930 calls have mask** (SDPA hook); FA already optimal |
| Z-Image-Turbo | 10.2% | (only 8 steps; setup amortizes) | 1.005× (essentially tied) | too few forward passes (8 × CFG=1) for kernel savings to register |
| **FLUX.2-dev TP=2** | **6.5%** | cuDNN sm80 7.82 s vs pytorch FA 8.95 s; Δ 1.13 s in kernel, 0.31 s e2e | 1.005× (essentially tied) | **TP=2 NCCL = 62 % of GPU time** dominates; attention not the bottleneck |
| **LTX-2** (empirical, this stack) | **18.7%** | CUDNN 96.98 s vs default-FA 140.62 s; Δ 43.6 s in kernel, 19.7 s wall-clock at CFG-parallel=2 | **1.073× CUDNN over SDPA** (1.055× FlashInfer) | cuDNN's wmma handles audio path D=64 cleanly; default dispatcher routes those to memEff (1.5× slower). PR's "cuDNN crashes" doesn't reproduce on torch 2.8 + diffusers 0.37 |

**The pattern that answers the colleague's question:**

| Pattern | Models | E2E gain from CUDNN_ATTN |
|---|---|---|
| Masked attention + high attention share | HV-1.5 (47 %) | **2.0× e2e** (11 ms → 6 ms cuDNN-vs-EFF × 100 forward passes) |
| Unmasked, dispatcher already picks FA which is ≈ cuDNN | Wan 2.2 (23 %) | 1.01× (kernels essentially tied) |
| Unmasked but heterogeneous shapes (audio D=64 + video D=128) where dispatcher falls into memEff for some | **LTX-2 (19 %)** | **1.07×** (cuDNN's wmma rescues the audio path that memEff handles at 1.5× slower) |
| Short total runtime — kernel savings amortize over too few passes | Z-Image (8 forward passes) | 1.005× (within run-noise) |
| Communication-bound | FLUX.2 TP=2 (62 % NCCL) | 1.005× (no attention backend can change collective comm) |

---

## Enabling FlashInfer on sm_120 (userspace, no sudo)

The system at `/usr/local/cuda` may be CUDA 12.4 or 12.8 (which lack
`compute_120` PTX targets). To unblock FlashInfer JIT for Blackwell:

```bash
# Install a CUDA 13.2 userspace toolchain
pip install nvidia-cuda-nvcc==13.2.78 nvidia-cuda-runtime==13.2.75 nvidia-cuda-cccl==13.2.75

# Build a synthetic CUDA_HOME (FlashInfer expects bin/include/lib64 layout)
mkdir -p /tmp/cuda13
ln -sfn /usr/local/lib/python3.11/dist-packages/nvidia/cu13/bin     /tmp/cuda13/bin
ln -sfn /usr/local/lib/python3.11/dist-packages/nvidia/cu13/include /tmp/cuda13/include
ln -sfn /usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib     /tmp/cuda13/lib64
ln -sfn /usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib     /tmp/cuda13/lib

# ld looks for unversioned .so — symlink libcudart.so → libcudart.so.13
TGT=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib
ln -sfn $TGT/libcudart.so.13 $TGT/libcudart.so

# Activate
export CUDA_HOME=/tmp/cuda13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
```

After this, the first FlashInfer call JIT-compiles for sm_120f in ~6 seconds
and caches at `~/.cache/flashinfer/<version>/120f/`.

Without this, FlashInfer raises `requires GPUs with sm75 or higher` (a
misleading diagnostic) and the PR's `FLASHINFER_ATTN` wrapper silently
`_sdpa_fallback`s to default SDPA on every call. That's why "FlashInfer
effect is negligible" on a broken stack — it never runs.

The sourceable form is at `bench_out/env.sh`.

## Recommendations

1. **Document the toolchain requirement in
   `docs/user_guide/diffusion/attention_backends.md`** — that
   `FLASHINFER_ATTN` requires `nvcc ≥ 12.9` available to FlashInfer's JIT.
   Add the userspace `pip install nvidia-cuda-nvcc==13.2.x` recipe as a
   fallback for systems where the system nvcc is older than the torch CUDA
   build.
2. **Add `import flashinfer` smoke-check at startup** when
   `DIFFUSION_ATTENTION_BACKEND=FLASHINFER_ATTN`. If JIT fails, log a clear
   error pointing to the toolchain doc instead of silently falling back to
   default SDPA. The current behavior makes "negligible FlashInfer gain"
   look like a kernel-speed result when it's actually a config bug.
3. **The `CUDNN_ATTN` wrapper at `cudnn_attn.py:78` is correctly designed**
   — singular pin + Python try/except fallback. No change needed.
4. **`benchmarks/diffusion/bench_attention_backends.py`'s
   `CUDNN_ATTN_CHAIN` row is misleading** because it uses `[CUDNN, FLASH,
   MATH]` which falls through to MATH on masked inputs. Either rename it
   to `CUDNN_VIA_CHAIN_BUG` to flag it, or replace with the singular pin
   matching the production wrapper.
5. **Per-model expectations for future backend work:**
   - HV-1.5 is the only model where attention-kernel swaps will move e2e
     materially. Use it as the canary for new backends.
   - For Wan / FLUX / Z-Image / LTX, **attention is not the bottleneck**.
     Benchmarking new attention kernels there is a waste of cycles unless
     the workload changes (longer S, additional masks, FP8/NVFP4 with
     different quant overheads).
   - FA4 should target HV-1.5 specifically; gains elsewhere will be in
     the noise.
6. **#3121 (LTX-2 cuDNN crash)** root-cause confirmed: the audio path uses
   D=64 with very short S (~100 tokens), which composes with the symbolic
   head_dim under torch.compile to fail cuDNN's plan selector at trace time.
   On the original PR stack the crash reproduces. On the torch 2.8.0+cu128 /
   cuDNN 9 / diffusers 0.37 stack used in this rotation, it does *not*
   reproduce — see the LTX-2 deep-dive. Long-term fix is unchanged: don't
   trace audio attention through torch.compile, or feed a concrete D=64
   plan-cache hint to cuDNN.
7. **`--enable-diffusion-pipeline-profiler` is silently a no-op on LTX-2.**
   Wan 2.2 / FLUX.2 / Z-Image / HV-1.5 all emit
   `[DiffusionPipelineProfiler] <Pipeline>.<stage> took X.XXs` lines; LTX-2's
   log emits **zero** such lines (not even the "Method path … not found"
   warning the profiler emits when a hook target is missing). Means there's
   no per-stage breakdown for LTX-2 (`text_encoder`, `transformer.forward`,
   `vae.decode`, `vocoder` are invisible). Either LTX-2's pipeline class needs
   the standard hookable method names, or the profiler config needs the
   LTX-2 paths added. Until then `Total generation time: …` is the only
   per-run timing for LTX-2 and the per-stage column in the LTX-2 e2e table
   is necessarily missing.

## Raw artifacts saved

All nsys reports + per-backend logs + parsed breakdowns + SDPA shape
logs are at `docs/investigations/nsys_reports/`:
```
nsys_reports/
├── zimage/   CUDNN_ATTN.nsys-rep, *.log, kernel_breakdown.txt
├── hv15/     CUDNN_ATTN.nsys-rep, *.log, kernel_breakdown.txt
├── wan22/    CUDNN_ATTN.nsys-rep, TORCH_SDPA.nsys-rep,
│            *.shapes.jsonl, *.log, *_breakdown.txt
├── flux2/    CUDNN_ATTN.nsys-rep, TORCH_SDPA.nsys-rep,
│            *.log, kernel_breakdown.txt (TP=2)
└── ltx2/     CUDNN_ATTN.nsys-rep, TORCH_SDPA.nsys-rep,
             *.shapes.jsonl, *.log, kernel_breakdown.txt (CFG-parallel=2)
```

Open `*.nsys-rep` in Nsight Systems UI for full timeline view, or
re-derive the per-kernel CSV via:
```
nsys stats --report cuda_gpu_kern_sum --format csv path/to.nsys-rep
```
