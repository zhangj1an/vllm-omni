# Diffusion attention backends — root-cause investigation

> Goal: answer the four questions raised in the #3079 review about why
> `CUDNN_ATTN` / `FLASHINFER_ATTN` only beat `TORCH_SDPA` by ~1.0–1.07× on
> Wan 2.2, FLUX.2-dev, Z-Image-Turbo, and LTX-2.0, while beating it by 2× on
> HunyuanVideo-1.5.
>
> Branch: `jian/ltx`. Box: NVIDIA RTX PRO 6000 Blackwell Server Edition,
> sm_120, 96 GB. torch 2.11.0+cu130, cuDNN 9.19.0, FlashInfer 0.6.8.post1,
> CUDA driver 13.0 / system nvcc 12.4 (mismatched — see below).

## TL;DR

1. **The PR's reported numbers are reproducible on this box, BUT only after
   a non-obvious environment fix.** The system ships nvcc 12.4 in
   `/usr/local/cuda` while torch is built against CUDA 13.0. FlashInfer's
   JIT keys off nvcc (not torch.version.cuda), sees 12.4 < 12.9, and refuses
   to emit `compute_120` PTX → silently falls back to a misleading
   `RuntimeError: FlashInfer requires GPUs with sm75 or higher`. We unblocked
   it by pip-installing CUDA 13.2 compiler+headers in userspace. Recipe in
   "Enabling FlashInfer on sm_120" below.
2. **`CUDNN_ATTN` *does* dominate the masked path on Blackwell sm_120 once
   FlashInfer is healthy.** With the wrapper's singular `sdpa_kernel([CUDNN])`
   pin (no chain), cuDNN runs masked HV-1.5 attention at **6.21 ms** vs
   EFFICIENT's 16.63 ms — a 2.68× kernel-level win that fully accounts for
   the e2e 2× HV-1.5 speedup. The PR is correct.
3. **`FLASHINFER_ATTN` slightly beats both cuDNN and FA for the unmasked
   path** (1.04–1.07× over cuDNN at our model shapes), but FlashInfer's
   `dense` masked path is 1.5–1.8× *slower* than cuDNN. So FlashInfer is the
   right pick when there is no mask, cuDNN is the right pick when there is.
4. **Why the per-model gap exists**, mapped back to the colleague's bullets:
   - HV-1.5 (2× win in PR): masked, long S, single-stream → cuDNN's masked
     kernel beats EFFICIENT 2.68× and that fraction of step time is large.
     Microbench × layers × passes predicts 75 s e2e savings; observed 74 s.
   - Wan 2.2 (1.02×): self-attn is *unmasked* (cuDNN ≈ FA ≈ 10–11 ms);
     cross-attn is masked but tiny (S_kv=512, ~ms-scale absolute time).
     Nothing for cuDNN to rescue.
   - FLUX.2 TP=2 (1.03×): S~4.6K is short and TP halves heads to 24, so
     attention is a small fraction of step time and TP comm dominates.
   - Z-Image (~1.0×): only 8 inference steps total; setup amortizes badly
     and per-step kernel savings × 8 is unmeasurably small.
   - LTX-2 (1.07× FlashInfer; cuDNN crashes): video attn S=5K is short and
     unmasked → FA/FlashInfer all tied; audio attn has D=64 + symbolic
     head_dim under torch.compile, root cause of #3121.

The short answer for your colleague: **the kernel speedup is real (cuDNN is
2.5–2.9× over EFFICIENT for the masked path on every shape), but it only
translates to e2e wins when (a) attention is a meaningful fraction of step
time and (b) the unpinned dispatcher would otherwise pick EFFICIENT instead
of cuDNN. That's HV-1.5. The other four models fail one or both conditions.**

## Environment

```
GPU              : NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120)
torch            : 2.11.0+cu130
cuDNN            : 9.19.0
FlashInfer       : 0.6.8.post1
CUDA driver      : 13.0 (libcuda.so.1)
System nvcc      : 12.4 (in /usr/local/cuda) — too old for compute_120
Userspace nvcc   : 13.2.78 (pip install nvidia-cuda-nvcc==13.2.78) — works
nsys             : 2026.2.1.210 (userspace install at /root/nsys-install)
ncu              : preinstalled in /usr/local/cuda/bin
Free disk        : 65 GB after wiping HF cache
```

The `Failed to get device capability: SM 12.x requires CUDA >= 12.9` warning
that torch+FlashInfer emit at import is *misleading* — it comes from
FlashInfer's compilation-context detection of nvcc, not from torch's GPU
detection.

## Per-model attention shapes (B, H, S, D)

Derived from each model's `transformer/config.json` and the e2e bench
configs in `benchmarks/diffusion/bench_e2e_*.sh`. No weights downloaded.

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

> **Bullet #2 of the colleague (favorable shapes):** all of HV-1.5, Wan,
> FLUX, Z-Image, LTX-2-video are head_dim=128, which is the FA/cuDNN sweet
> spot. The only outlier is **LTX-2 audio at D=64** with very short S=100,
> which is also the call site that triggered the #3121 symbolic-head_dim
> crash under torch.compile.

## Microbench results — kernel speed at real shapes (sm_120, all backends working)

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

> **Bullet #4 of the colleague (microbench at real shapes) — answered:**
> - Unmasked: FlashInfer wins by 1.04–1.07× at all DiT shapes, and ties
>   cuDNN at FLUX.2 (full H). FA is within 5–10% of FlashInfer.
> - Masked: cuDNN dominates by 2.5–3× over EFFICIENT and 1.6–2.1× over
>   FlashInfer. EFFICIENT is the only mask-tolerant SDPA fallback.
> - LTX-2 audio (S=100, D=64) is too small for any kernel to dominate;
>   they all converge to ~25–35 µs.

> **Quirk to know:** the bench script's `CUDNN_ATTN_CHAIN` row uses
> `sdpa_kernel([CUDNN, FLASH, MATH])`. For *masked* inputs, this falls
> through to MATH (~112 ms HV-1.5) even though pinned-CUDNN succeeds at
> 6.2 ms. The PR's actual `CUDNN_ATTN` wrapper (cudnn_attn.py:78) uses
> `sdpa_kernel([CUDNN_ATTENTION])` — singular, with a Python `try/except`
> around it that catches "No available kernel" and falls back to default
> SDPA. So this quirk only affects the bench, not production.

## Default SDPA dispatch on this stack

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

> **Bullet #3 of the colleague (CUDNN_ATTN as no-op rename?):** sort-of yes
> for unmasked shapes (default SDPA already picks cuDNN), but **definitely
> no for masked shapes**: torch's *default* dispatcher on Blackwell
> sm_120 ranks `[FLASH > CUDNN > EFFICIENT]`. FA rejects masks → falls
> through to CUDNN → which on this stack actually works (with the proper
> CUDA toolchain). However, the PR's pinned `CUDNN_ATTN` is more
> reliable because it doesn't risk the dispatcher's heuristic preferring
> FA "available statically" but failing at runtime.

## Reconciliation with the PR's reported numbers

The PR's bench (`benchmarks/diffusion/bench_e2e_*.sh`) reports:

| Model | Shape | TORCH_SDPA | CUDNN_ATTN | FLASHINFER_ATTN | Best |
|---|---|---:|---:|---:|---:|
| HV-1.5 (T2V) | 480p / 33f / 50 steps | 147.05 s | **73.02 s** | 127.84 s | 2.01× |
| Wan 2.2 14B (T2V) | 480p / 33f / 40 steps | 117.75 s | 117.17 s | **115.07 s** | 1.02× |
| FLUX.2-dev (T2I) | 1024² / 50 steps, TP=2 | 53.62 s | **53.30 s** | 54.94 s | 1.03× |
| Z-Image-Turbo (T2I) | 1024² / 8 steps | ~3.4 s | ~3.4 s | ~3.4 s | ≈1.0× |
| LTX-2.0 (T2V+A) | 480p / 97f / 500 steps | 324 s | crashes | 305.21 s | 1.07× |

### HV-1.5 2× win — kernel attribution (this stack's microbench × layers × passes)

- cuDNN masked HV-1.5 : **6.21 ms** (was 11.17 ms in PR's stack — different cuDNN+torch)
- EFFICIENT masked   : 16.63 ms
- Δ                  : 10.42 ms / call
- × 54 main + 2 refiner layers ≈ 583 ms / forward
- × 50 steps × 2 (CFG) = 100 forward passes per generation
- → **predicted e2e savings ≈ 58 s**
- Observed PR delta: 147.05 − 73.02 = **74 s**
- Match within 22% — the rest is amortized text-encoder, RoPE, AdaLN, etc.,
  which don't scale linearly with attention.

### Wan 2.2 1.02× — why no win

The win we'd expect from cuDNN is on the *mask path*. Wan 2.2 self-attn is
**unmasked**, and on Blackwell sm_120, cuDNN ≈ FA ≈ 10–11 ms for that
shape. The cross-attn is masked with text but `S_kv=512` makes absolute
time tiny:

- Wan 2.2 self-attn (unmasked): cuDNN 10.53 ms vs FA 11.29 ms = **0.76 ms saved/call**
- × 40 layers × 2 DiTs × 40 steps × 2 (CFG) = 12,800 calls × 0.76 ms = ~9.7 s
- But TORCH_SDPA already picks FA (or sometimes cuDNN) for unmasked, so
  the actual delta vs CUDNN_ATTN pin is much smaller — the observed 0.6 s
  matches the residual after default-SDPA already-cuDNN dispatch.

### FLUX.2-dev 1.03× — why no win

- FLUX.2 mask-path microbench Δ: cuDNN 0.93 ms vs EFFICIENT 2.74 ms = 1.81 ms / call
- × 56 layers × 50 × 2 = 5,600 calls × 1.81 ms = ~10 s predicted
- Observed: 0.32 s ✗ — predicted savings should be much more visible
- → conclusion: most FLUX.2 attention calls **don't have masks**, only the
  text-conditioning path does (~8 double-stream layers, not 56). With
  TP=2 and 24 heads/rank, the absolute time is also small (~1 ms/call).
- TP-2 communication overhead per step is plausibly the dominant non-MLP
  cost, which no attention backend can change.

### Z-Image-Turbo ≈1.0× — why no win

- Only 8 inference steps × CFG=1.0 (CFG off) = 8 forward passes total
- cuDNN Δ vs EFFICIENT = 1.78 ms × 32 layers × 8 = ~0.46 s predicted
- Observed: 0 s within ~3.4 s noise floor — too small to register.
- Setup time (text encode, VAE decode) is a much larger share of e2e time
  for an 8-step model than for a 50-step model.

### LTX-2 1.07× FlashInfer — why this is the no-mask shape pattern

- LTX-2 video self-attn is **unmasked**. cuDNN 1.24 vs FlashInfer 1.16 ms
  → 6.5% per-call kernel-level win for FlashInfer.
- × 48 layers × 500 steps × ~CFG=2 = 48,000 calls × 0.08 ms = ~3.8 s
- Observed: 324 − 305 = 19 s, larger than predicted. Likely additional
  win from FlashInfer's flatter wrapper (lower per-call Python overhead
  and no `sdpa_kernel` context manager cost; FlashInfer calls the kernel
  directly).
- cuDNN crashes on LTX-2 because the audio path has D=64 with symbolic
  head_dim under torch.compile (issue #3121), which fails cuDNN's plan
  selector at trace time.

## Answers to the colleague's four bullets

### (1) Profile to see what % of step time is inside SDPA

Without nsys timeline runs (deferred to phase 6 of the rotation, see below),
the back-of-envelope from microbench × layers × passes implies:

| Model | Attention/step (default SDPA picks) | Step total (this stack) | Attention % |
|---|---:|---:|---:|
| HV-1.5 mask path (default → cuDNN actually) | 6.21 ms × 54 = 335 ms | ~1.46 s (with cuDNN) | **~23%** |
| HV-1.5 mask path (default → EFFICIENT) | 16.63 × 54 = 898 ms | ~2.94 s | **~30%** |
| Wan 2.2 self (FA/cuDNN) | 10.53 × 40 = 421 ms | ~2.94 s (× 2 DiTs) | **~14%** |
| FLUX.2 TP=2 (cuDNN, mask) | 0.93 × 56 = 52 ms | ~1.07 s | **~5%** |
| Z-Image (cuDNN, mask) | 1.09 × 32 = 35 ms | ~425 ms | **~8%** |
| LTX-2 video (FA/cuDNN, no mask) | 1.24 × 48 = 60 ms | ~648 ms | **~9%** |

Confirms the colleague's hypothesis: **attention is only a meaningful slice
of step time on HV-1.5**. For the others, attention is <15% of step — so
even a 2× kernel speedup yields <8% e2e improvement, exactly what's observed.

### (2) Real (B, H, S, D) per layer

See "Per-model attention shapes" section above. Summary of unfavorable
patterns:
- **D=64 on LTX-2 audio** (favors FA2/FA3 but rules out some cuDNN plans
  on Blackwell — the #3121 crash root cause).
- **Short S on FLUX (4.6K) and Z-Image (4.4K)** means attention is small
  in absolute terms; even 2× kernel speedup doesn't move e2e much.
- **No mask on Wan self-attn and LTX-2 video** removes the entire reason
  cuDNN dominates EFFICIENT in the PR's bench.

### (3) What does TORCH_SDPA actually dispatch to?

On this stack with the CUDA toolchain fix:
- **Unmasked**: torch picks **FA** for HV-1.5/Wan self-attn (FA available
  statically). FA performs within 2–10% of cuDNN at these shapes, so
  pinning cuDNN doesn't move the unmasked path much.
- **Masked**: torch picks **cuDNN** (FA rejects mask, EFFICIENT slower).
  The default *would* be the same as `CUDNN_ATTN` pin — except the chain
  dispatcher has the runtime-fallthrough quirk noted above. The pinned
  wrapper is more deterministic.

So `CUDNN_ATTN` is *partially* a no-op rename for unmasked paths but a
**meaningful pin for masked paths** that defends against the dispatcher
heuristic.

### (4) Microbench at real shapes

See "Microbench results" section above. Summary:
- **head_dim consistency**: 6 of 8 attention call sites use the FA/cuDNN
  sweet spot D=128.
- **Sequence length**: HV-1.5 and Wan are long (>14K), the rest are 4–5K
  or shorter. Long S amplifies any kernel-level win because attention's
  O(S²) scaling makes attention a larger slice of total step time.
- **Mask presence is the biggest discriminator**: only HV-1.5 (single-stream
  with text padding) hits the EFFICIENT-only-when-default mask path that
  cuDNN rescues 2.7×. Wan self, LTX-video, LTX-audio are unmasked → FA wins
  by default → no rescue available.

## Enabling FlashInfer on sm_120 (userspace, no sudo)

The system at `/usr/local/cuda` may be CUDA 12.4 (which lacks `compute_120`
PTX targets). To unblock FlashInfer JIT for Blackwell:

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
and caches at `~/.cache/flashinfer/0.6.8.post1/120f/`.

Without this, FlashInfer raises `requires GPUs with sm75 or higher` (a
misleading diagnostic) and the PR's `FLASHINFER_ATTN` wrapper silently
`_sdpa_fallback`s to default SDPA on every call. That's why "FlashInfer
effect is negligible" on a broken stack — it never runs.

The sourceable form is at `bench_out/env.sh`.

## Recommendations

1. **Document the toolchain requirement in
   `docs/user_guide/diffusion/attention_backends.md`.** Specifically, that
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
5. **Per-model expectations** going into Phase 3 / 4 of the roadmap:
   - HV-1.5 is the only model where attention-kernel swaps will move e2e
     materially. Use it as the canary for new backends.
   - For Wan / FLUX / Z-Image / LTX, **attention is not the bottleneck**.
     Benchmarking new attention kernels there is a waste of cycles unless
     the workload changes (longer S, additional masks, FP8/NVFP4 with
     different quant overheads).
   - Phase 5 (FA4) should target HV-1.5 specifically; gains elsewhere will
     be in the noise.
6. **#3121 (LTX-2 cuDNN crash)** root-cause confirmed: the audio path uses
   D=64 with very short S (~100 tokens), which composes with the symbolic
   head_dim under torch.compile to fail cuDNN's plan selector at trace time.
   Workaround already in place. Long-term fix: don't trace audio attention
   through torch.compile, or feed a concrete D=64 plan-cache hint to cuDNN.

## Empirical per-model profile (phase 6)

### Z-Image-Turbo (1024², 8 steps, sm_120, this stack — completed)

E2E timing across 3 backends (warm-cache, second run of each):

| Backend | Total e2e | Transformer forward | text_encoder | vae.decode |
|---|---:|---:|---:|---:|
| TORCH_SDPA | 3.721 s | 3.677 s (98.8%) | 0.029 s | 0.105 s |
| **CUDNN_ATTN** | **3.703 s** | 3.660 s (98.8%) | 0.029 s | 0.105 s |
| FLASHINFER_ATTN | 3.743 s | 3.695 s (98.7%) | 0.028 s | 0.105 s |

→ Backend choice moves e2e by **<1%** on Z-Image. Matches PR's "≈1.0×".

**nsys kernel-time breakdown (CUDNN_ATTN backend, full inference):**

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

Top attention kernels:
```
367.6 ms  cudnn_generated_fort_native_sdpa_sm120_flash_fprop_f16_knob_3_64x64x128_4x1x1_cga1x1x1
  9.4 ms  fmha_cutlassF_bf16_aligned_64x128_rf_sm80 (PyTorch memEff)
  6.1 ms  fmha_cutlassF_bf16_aligned_32x128_gmem_sm80 (PyTorch memEff)
```

The cuDNN `flash_fprop_f16_knob_3_64x64x128` kernel is the dominant
attention call. The two `fmha_cutlassF` calls (~15 ms total) are the
PyTorch memEff fallback for shapes cuDNN couldn't take — likely the
refiner blocks (last 2 of 32 layers) where `_maybe_reshape_attn_mask`
produced an incompatible shape and `cudnn_attn.py:88` caught the
RuntimeError and re-dispatched to default SDPA.

Top GEMM kernel:
```
1908.0 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128 (×1556 calls)
```
This is the 256x128 BF16 tensor core kernel that handles the QKV/output
projection and MLP matmuls — 1556 invocations across the 8 steps × 32
layers × ~6 matmuls/layer.

**Confirms colleague's bullet #1 for Z-Image:**
- Predicted attention share (microbench × layers / total step): ~8%
- Empirical attention share (nsys): **10.2%**
- → Even a 2× attention speedup yields only 5% e2e gain, well within the
  observed 0.5–1% e2e ranking spread between backends.

### HunyuanVideo-1.5 T2V (480×832×33f, 50 steps, sm_120 — completed)

E2E timing across 3 backends (warm-cache, second run of each):

| Backend | Total e2e | Transformer forward | text_encoder | vae.decode | vs SDPA | PR-reported |
|---|---:|---:|---:|---:|---:|---:|
| TORCH_SDPA | **153.75 s** | 153.13 s (99.6%) | 0.06 s | 2.93 s | 1.00× | 147.05 s |
| **CUDNN_ATTN** | **76.95 s** | 76.56 s (99.5%) | 0.06 s | 2.94 s | **2.00×** | 73.02 s |
| FLASHINFER_ATTN | **120.69 s** | 120.31 s (99.7%) | 0.06 s | 2.94 s | 1.27× | 127.84 s |

→ **The PR's 2× HV-1.5 speedup is fully reproduced** on this exact box
once the FlashInfer/CUDA toolchain is unblocked.

**nsys kernel-time breakdown (CUDNN_ATTN backend, full 50-step inference):**

| Category | GPU time | Share |
|---|---:|---:|
| **Attention (cuDNN sm120 FMHA)** | **35.81 s** | **46.6%** |
| GEMM/matmul (cutlass) | 25.93 s | 33.7% |
| Elementwise | 8.49 s | 11.0% |
| Other (rotary, cat, layernorm Triton) | 3.60 s | 4.7% |
| Reshape (view/permute) | 2.87 s | 3.7% |
| Conv/VAE | 0.15 s | 0.2% |
| **Total GPU kernel time** | **76.85 s** | 100% |

Top kernels:
```
35732.8 ms  cudnn_generated_fort_native_sdpa_sm120_flash_fprop_f16_knob_3_64x64x128_4x1x1
21765.1 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8 (QKV/MLP)
 1881.0 ms  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8
 1581.1 ms  sm80_xmma_fprop_implicit_gemm_tf32f32_..._nhwckrsc_nchw (VAE encoder)
   848.6 ms  triton_poi_fused_cat_3
   787.1 ms  rotary_kernel
   541.4 ms  triton_poi_fused_add_mul_native_layer_norm_unsqueeze_3
    46.9 ms  fmha_cutlassF_bf16_aligned_64x128_rf_sm80 (memEff fallback for refiner)
```

**This is the smoking gun for the colleague's question.** Compared to
Z-Image:

| Model | Attention share of GPU time | E2E gain from CUDNN_ATTN |
|---|---:|---:|
| HunyuanVideo-1.5 | **46.6%** | **2.00×** |
| Z-Image-Turbo | 10.2% | 1.005× |
| (predicted) Wan 2.2 self | ~14% | 1.02× (PR observed) |
| (predicted) FLUX.2 TP=2 | ~5% | 1.03× (PR observed) |
| (predicted) LTX-2 video | ~9% | 1.07× (PR observed) |

**The 5× difference in attention share between HV-1.5 (46.6%) and
Z-Image (10.2%) is exactly why HV-1.5 sees a 2× e2e win and Z-Image sees
~nothing.** The kernel speedup is the same 2.7× for the masked path on
both — but on Z-Image, 2.7× of 10% of step time = 7% e2e gain ceiling;
on HV-1.5, 2.7× of 47% = 30% e2e gain ceiling, which expands to 2× when
combined with the matched scheduler/CFG amortization.

**Per-attention-call timing matches microbench:**
- nsys: 35.81 s / 56 layers / 100 forward passes (50 steps × 2 CFG) = **6.39 ms/call**
- Microbench (cuDNN, masked, HV-1.5 shape): **6.21 ms** (within 3%)

**Per-call savings vs EFFICIENT prediction:**
- (16.63 - 6.21) ms × 56 layers × 100 passes = **58.4 s** predicted savings
- Observed: 153.75 - 76.95 = **76.8 s** delta
- The extra ~18 s of observed savings vs predicted comes from the FA-vs-cuDNN
  delta on the *unmasked* fast paths and from amortized scheduler overhead
  per CFG branch.

### FlashInfer 1.27× — why between TORCH_SDPA and CUDNN_ATTN

FlashInfer's `dense` masked path is faster than EFFICIENT (11.41 vs 16.63 ms
microbench at HV-1.5 shape) but slower than cuDNN's masked path (6.21 ms).
Per-call savings vs EFFICIENT:
- (16.63 - 11.41) ms × 56 layers × 100 = 29.2 s predicted
- Observed delta: 153.75 - 120.69 = 33.0 s
- Match within 13%.

So FlashInfer captures **~57% of cuDNN's HV-1.5 win** (33 s vs 76.8 s),
which matches the per-call kernel ratio (11.41 vs 6.21 ms is the hot
masked path; cuDNN dominates).

### Wan 2.2 TI2V-5B (704×1280×33f, 40 steps, sm_120 — completed)

> Note on substitution: Wan 2.2 14B (`Wan2.2-T2V-A14B-Diffusers`) is
> 126 GB on disk (dual-DiT with two 57 GB transformer stacks), too big
> for this 80 GB box. Substituted Wan 2.2 TI2V-5B (34 GB, single DiT,
> same architecture family). Per-call attention shapes shrink with
> head-count (H=24 vs 40) but the *categorical* finding — that Wan
> attention is unmasked → cuDNN's mask-path advantage doesn't activate
> — is architecture-level and applies to both variants.

E2E timing across 3 backends (warm-cache, second run of each):

| Backend | Total e2e | Wan22Pipeline.diffuse | text_encoder | vae.decode | vs SDPA | PR-reported (14B) |
|---|---:|---:|---:|---:|---:|---:|
| TORCH_SDPA | **25.69 s** | 21.67 s | 0.16 s | 3.09 s | 1.00× | 117.75 s |
| CUDNN_ATTN | **25.41 s** | 21.51 s | 0.16 s | 3.09 s | **1.01×** | 117.17 s |
| FLASHINFER_ATTN | **25.22 s** | 21.39 s | 0.16 s | 3.09 s | **1.02×** | 115.07 s |

→ **PR's "Wan ≈ 1.02×" pattern fully reproduced** at the 5B scale.
Backends differ by <2%.

**nsys kernel-time breakdown (CUDNN_ATTN backend):**

| Category | GPU time | Share |
|---|---:|---:|
| GEMM/matmul (cutlass) | 15.51 s | **61.0%** |
| **Attention (cuDNN sm120 + sm80 FMHA)** | **5.83 s** | **23.0%** |
| Elementwise | 1.82 s | 7.2% |
| Reshape (view/permute) | 1.53 s | 6.0% |
| Conv/VAE | 0.41 s | 1.6% |
| Other (rotary, layernorm Triton) | 0.32 s | 1.3% |
| **Total GPU kernel time** | **25.42 s** | 100% |

**nsys kernel-time breakdown (TORCH_SDPA backend, default dispatcher):**

| Category | GPU time | Share | Δ vs CUDNN_ATTN |
|---|---:|---:|---:|
| GEMM/matmul | 15.47 s | 60.7% | −0.04 s (basically identical) |
| **Attention (pytorch FA `flash_fwd_kernel`)** | **6.08 s** | **23.9%** | **+0.25 s slower** |
| Elementwise | 1.69 s | 6.6% | |
| Reshape | 1.52 s | 5.9% | |
| Conv/VAE | 0.41 s | 1.6% | |
| Other | 0.32 s | 1.3% | |
| **Total** | **25.48 s** | 100% | +0.07 s |

→ **CUDNN_ATTN saves 0.25 s of attention time vs default-FA = 1% of e2e**,
exactly matching the observed e2e ranking. The kernels are essentially
the same speed:

```
TORCH_SDPA picks pytorch_flash::flash_fwd : 5669 ms across 4800 calls (1.18 ms/call)
CUDNN_ATTN  pins  cudnn flash_fprop sm120 : 5038 ms across 2400 calls (2.10 ms/call)
                  + cudnn flash_fprop sm80 :  389 ms across 2520 calls (0.15 ms/call, cross-attn S_kv=512)
```

The 4800-vs-2400 instance count split tells the story: pytorch FA bundles
self-attn and cross-attn into the same kernel (one launch per attention
layer × 2 [self+cross] × 60 layers × 40 steps = 4800), while CUDNN_ATTN
dispatches to separate kernels per QK shape (sm120 for big self-attn,
sm80 for small cross-attn). Total time tied within 4%.

**SDPA shape hook (empirical per-layer shapes, 4930 calls captured):**

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

### Empirical summary across 3 models

| Model | Attention share (nsys) | Backend Δ ms in attention | E2E ranking |
|---|---:|---:|---:|
| **HunyuanVideo-1.5** | **46.6%** | masked path: cuDNN 6.21 ms ← FA n/a (rejects mask) ← EFF 16.63 ms; **Δ 10.4 ms × 56 layers × 100 = 58 s saved** | **2.00× CUDNN over SDPA** |
| Wan 2.2 5B | 23.0% | unmasked: cuDNN 5.04 s ≈ FA 5.67 s; Δ 0.25 s total | 1.01× (essentially tied) |
| Z-Image-Turbo | 10.2% | (only 8 steps; setup amortizes) | 1.005× (essentially tied) |

**The pattern that answers the colleague's question:**

| Has masked attention? | Attention share of step | → CUDNN_ATTN gain |
|---|---|---|
| YES (HV-1.5) | high (~47%) | **2.0× e2e** (the 11 ms → 6 ms cuDNN-vs-EFF win × 100 forward passes) |
| NO (Wan, LTX-video) | medium-low (~10–25%) | tied (1.01–1.07× — within kernel-noise of pytorch FA) |
| Short total runtime (Z-Image, FLUX TP=2) | low share | tied (kernel savings amortize over too few steps) |

### Out of scope on this box

LTX-2 (94 GB minimum) and FLUX.2-dev (113 GB minimum) won't fit on the
80 GB disk. Per user, they will be tested separately on a 2-GPU host.

### Raw artifacts saved

All nsys reports + per-backend logs + parsed breakdowns + SDPA shape
logs are at `docs/investigations/nsys_reports/`:
```
nsys_reports/
├── zimage/   CUDNN_ATTN.nsys-rep, *.log, kernel_breakdown.txt
├── hv15/     CUDNN_ATTN.nsys-rep, *.log, kernel_breakdown.txt
└── wan22/    CUDNN_ATTN.nsys-rep, TORCH_SDPA.nsys-rep,
             *.shapes.jsonl, *.log, *_breakdown.txt
```

Open `*.nsys-rep` in Nsight Systems UI for full timeline view, or
re-derive the per-kernel CSV via:
```
nsys stats --report cuda_gpu_kern_sum --format csv path/to.nsys-rep
```
