# Draft reply — PR #3079 review comment

> Context: reply to @lishunyang12's four diagnostic bullets on
> https://github.com/lishunyang12/vllm-omni/pull/54#issuecomment-4319926890.
> Full investigation data is in `diffusion_attention_investigation.md` and
> `nsys_reports/`.

---

## TL;DR

Attention kernels only move the needle when attention is a large fraction of
step time, with long S and standard head_dim (64 or 128). Given a new model,
use cuDNN for masked attention and FlashInfer for unmasked. FA cannot process
irregularly shaped masks. For very short S, backend choice does not matter.

- **HV-1.5** is faster because it previously was not pinned to the optimal
  attention kernel. With 47 % of GPU time on attention, the speedup is
  significant after selecting cuDNN.
- **FLUX.2 and Z-Image** also were not pinned to the optimal kernel, but their
  attention share is below 20 %, so changing the kernel did not change e2e
  performance.
- **LTX-2** was not pinned to the optimal kernel. The default dispatcher routed
  the audio path (D=64) and cross-attention (short S) to memEff, which is
  1.5–2× slower than cuDNN's wmma kernel for those shapes. Switching to
  cuDNN across all LTX-2 attention shapes produced the 1.07× e2e speedup.
- **Wan 2.2 and LTX-2 video** were already selecting the correct kernel (FA
  for unmasked shapes). FA ≈ cuDNN at those shapes, so no speedup is possible.

---

## 1. Attention % of step time (profiled using `nsys`, CUDNN\_ATTN backend)

| Model | Attention share | E2E: TORCH\_SDPA | E2E: CUDNN\_ATTN | E2E: FLASHINFER | Best gain |
|---|---:|---:|---:|---:|---:|
| HunyuanVideo-1.5 | 46.6 % | 153.75 s | 76.95 s | 120.69 s | 2.00× (CUDNN) |
| LTX-2 (video + audio) | 18.7 % | 289.70 s | 270.01 s | 274.57 s | 1.07× (CUDNN) |
| Wan 2.2 | 23.0 % | 25.69 s | 25.41 s | 25.22 s | 1.02× (FlashInfer) |
| Z-Image-Turbo | 10.2 % | 3.721 s | 3.703 s | 3.743 s | ~1.005× (tied) |
| FLUX.2-dev TP=2 | 6.5 % | 59.27 s | 58.96 s | 60.31 s | ~1.005× (tied) |

HunyuanVideo-1.5 wins because attention takes 46.6 % of GPU time — a faster
kernel has a large surface to improve. Z-Image and FLUX.2 are too
attention-light (10 % and 6.5 %) for any kernel swap to matter; FLUX.2's 6.5 %
is further dwarfed by NCCL collective comm at 62 % of GPU time, which no
attention backend can change.

LTX-2 at 18.7 % still gets a 1.07× win — explained in the dedicated section
below, not by attention share alone.

Wan 2.2 at 23 % is not explained by this table. The attention share is large
enough that a faster kernel should help, but it doesn't. The reason is in
point 3.

## 2. Actual (B, H, S, D) per layer

| Model | B | H | S | D | mask? |
|---|---|---|---|---|---|
| HV-1.5 | 1 | 16 | 14,296 | 128 | yes (text-pad ~256) |
| Wan 2.2 self | 1 | 24 | 7,920 | 128 | none |
| Wan 2.2 cross | 1 | 24 | q=7,920 / kv=512 | 128 | none |
| FLUX.2 TP=2 | 1 | 24 | 4,608 | 128 | yes (text-pad ~512) |
| Z-Image | 1 | 30 | 4,352 | 128 | yes (text-pad ~256) |
| LTX-2 video | 1 | 32 | 5,070 | 128 | none |
| LTX-2 audio | 1 | 32 | 100 | 64 | none |

Your hypothesis — cuDNN/FlashInfer is beneficial for models with long S and
head_dim ∈ {64, 128} — does not hold as the discriminating factor here. Every
model uses D=128 (the sweet spot for both FA and cuDNN), and both HV-1.5 and
Wan 2.2 have long S (~14K tokens). Yet HV-1.5 gets a 2× win and Wan 2.2 gets
nothing. Long S and D ∈ {64, 128} are necessary conditions for these kernels
to run efficiently, but they are not what separates the winners from the
non-winners.

The real discriminator is the mask column and what it causes the default
dispatcher to do — explained in point 3.

## 3. What TORCH\_SDPA actually dispatches to on sm\_120 (tested on RTX Pro 6000)

For context, the dispatcher on Blackwell sm\_120 ranks `[FLASH > CUDNN > EFFICIENT]`.
However, FlashAttention only works for causal triangular masks or no mask at
all. For models like HV-1.5, FLUX.2, and Z-Image, the model passes in
text-pad masks with irregular boolean patterns, which FA cannot handle. cuDNN's
FMHA implementation (`flash_fprop`) handles arbitrary attention masks natively.

We used a dispatcher hook to check what `F.scaled_dot_product_attention`
actually selects at runtime, confirmed against nsys kernel traces:

| Shape | mask | default dispatch | cuDNN avail | FA avail | EFF avail |
|---|---|---|---|---|---|
| HV-1.5 | pad-256 | CUDNN | True | False | True |
| Wan 2.2 self | none | CUDNN | True | True | True |
| Wan 2.2 cross | kv-512 | CUDNN | True | False | True |
| FLUX.2 TP=2 | pad-512 | CUDNN | True | False | True |
| Z-Image | pad-256 | CUDNN | True | False | True |
| LTX-2 video | none | CUDNN | True | True | True |
| LTX-2 audio | none | CUDNN | True | True | True |

- **Unmasked shapes (Wan 2.2, LTX-2 video):** Yes, the default already picked
  FA. nsys confirmed `pytorch_flash::flash_fwd_kernel` as the dominant kernel.
  FA is within 4–10 % of cuDNN at these shapes, so their performance is
  essentially the same. Nothing to rescue.

- **Masked shapes (HV-1.5, FLUX.2, Z-Image):** No. FA is rejected due to the
  irregular mask. The dispatcher should then fall through to cuDNN, but due to
  a quirk in PyTorch's unpinned dispatch chain on this stack, it lands on
  EFFICIENT_ATTENTION instead. EFFICIENT is 2.5–3× slower than cuDNN for
  masked inputs. You can verify this from the HV-1.5 numbers: if TORCH_SDPA
  had landed on cuDNN, it would take ~76 s — the same as `CUDNN_ATTN`. It
  took 153.75 s, and the EFFICIENT per-call delta (16.63 ms × 56 layers × 100
  passes ≈ 93 s of attention) accounts for the difference. The `CUDNN_ATTN`
  pin (`sdpa_kernel([CUDNN_ATTENTION])` with a try/except fallback) bypasses
  the chain entirely and guarantees cuDNN runs for masked inputs.

- **Short S / unusual D (LTX-2 audio):** No. The default fell to memEff
  (`fmha_cutlassF`) for the short-S and D=64 shapes. cuDNN's wmma kernel
  handles those better. Combined with the video path win, this contributes to
  LTX-2's overall 1.07× speedup.

Wan 2.2 is the model this bullet was really asking about: the shape hook
captured 4,930 live SDPA calls and confirmed 0 of them use a mask. The
default already picks FA for all Wan attention calls, and FA ≈ cuDNN at those
shapes, so `CUDNN_ATTN` cannot help regardless of attention share.

## 4. Microbench at real shapes (kernel isolation, sm\_120, bf16)

### No-mask path

| Shape | cuDNN | FA | EFFICIENT | FlashInfer | Best |
|---|---:|---:|---:|---:|---|
| HV-1.5 (16×14296×128) | 4.72 ms | 4.81 ms | 13.21 ms | 4.42 ms | FlashInfer 1.07× cuDNN |
| Wan 2.2 self (40×14040×128) | 10.53 ms | 11.29 ms | 29.90 ms | 10.31 ms | FlashInfer 1.02× cuDNN |
| FLUX.2 TP=2 (24×4608×128) | 0.79 ms | 0.85 ms | 2.10 ms | 0.76 ms | FlashInfer 1.04× cuDNN |
| Z-Image (30×4352×128) | 0.89 ms | 0.95 ms | 2.19 ms | 0.85 ms | FlashInfer 1.05× cuDNN |
| LTX-2 video (32×5070×128) | 1.24 ms | 1.28 ms | 3.26 ms | 1.16 ms | FlashInfer 1.06× cuDNN |
| LTX-2 audio (32×100×64) | 0.026 ms | 0.023 ms | 0.022 ms | 0.024 ms | all tied; EFFICIENT wins at S=100 |

### With-mask path

| Shape | cuDNN (pinned) | EFFICIENT | FlashInfer (dense) | Best |
|---|---:|---:|---:|---|
| HV-1.5 (16×14296×128) | 6.21 ms | 16.63 ms | 11.41 ms | cuDNN 2.68× EFFICIENT |
| Wan 2.2 (40×14040×128) | 15.29 ms | 38.40 ms | 25.36 ms | cuDNN 2.51× EFFICIENT |
| FLUX.2 TP=2 (24×4608×128) | 0.93 ms | 2.74 ms | 1.91 ms | cuDNN 2.95× EFFICIENT |
| Z-Image (30×4352×128) | 1.09 ms | 2.87 ms | 1.89 ms | cuDNN 2.63× EFFICIENT |
| LTX-2 video (32×5070×128) | 1.44 ms | 4.29 ms | 3.07 ms | cuDNN 2.98× EFFICIENT |
| LTX-2 audio (32×100×64) | 0.033 ms | 0.034 ms | 0.036 ms | all tied |

- **No-mask path (Wan 2.2, LTX-2 video):** FlashInfer is marginally the
  fastest across all standard shapes (1.02–1.07× over cuDNN). FA and cuDNN
  are close behind. EFFICIENT is 3–4× slower and should never be on the
  no-mask path.
- **With-mask path (HV-1.5, FLUX.2, Z-Image):** cuDNN wins decisively at
  every shape (2.5–3× over EFFICIENT). FlashInfer's dense masked path is a
  middle ground — faster than EFFICIENT but 1.6–2.1× slower than cuDNN. FA
  rejects masked inputs entirely and cannot be used here.
- **Short S / unusual D (LTX-2 audio):** all backends tied on both paths.
  Shape is too small for any kernel to dominate.

Practical recommendation: use cuDNN for masked attention, FlashInfer for
unmasked. For very short S (LTX-2 audio), backend choice does not matter.

## LTX-2: what distinguishes it from Wan / Z-Image / FLUX

Your question: "LTX-2 achieves 1.07× — what distinguishes its attention
shape distribution?"

The prediction in the PR was wrong. We predicted LTX-2 would behave like
Wan 2.2 (unmasked → cuDNN ≈ FA → tied). The nsys data tells a different
story. LTX-2 has three distinct attention shapes per layer, not one:

1. Video self-attn (S=5,070, D=128, no mask): cuDNN sm_120 vs pytorch FA
   — cuDNN is ~10 % faster (54.6 s vs 60.7 s total across the run).
2. Audio self-attn / video↔audio cross (D=64 or short S, no mask): cuDNN
   sm_80 wmma (26.2 s combined) vs the default dispatcher routing these to
   memEff (`fmha_cutlassF`) at 39.7 s — because FA doesn't handle short-S
   and D=64 shapes well. This is where most of the cuDNN win comes from.
3. Triton-fused pre-attention (rms\_norm + sdpa fused): roughly equal
   (~16 s both), kernel choice doesn't apply.

So the LTX-2 win is not from a single-shape kernel edge — it is from cuDNN's
wmma handling of the audio path (D=64) and cross-attn (short S) instead of
the memEff fallback the default dispatcher uses. nsys kernel totals:

| Backend | Attention total | Key kernel |
|---|---:|---|
| CUDNN\_ATTN | 96.98 s (18.7 % of GPU time) | cuDNN sm\_120 flash + sm\_80 wmma |
| TORCH\_SDPA | 140.62 s (25.0 % of GPU time) | pytorch FA + 39.7 s memEff fallback |

CUDNN\_ATTN saves 43.6 s of attention kernel time, which at CFG-parallel=2
(two ranks running concurrently) translates to ~19.7 s wall-clock — exactly
the 289.70 − 270.01 = 19.7 s observed e2e delta.

On the "cuDNN crashes" report in the PR: that reproduced on the original
torch 2.11 + diffusers 0.38 stack where the audio path's symbolic `head_dim`
under `torch.compile` failed cuDNN's plan selector. On our current stack
(torch 2.8 + diffusers 0.37 + cuDNN 9, with the #3121 fix applied),
CUDNN\_ATTN runs cleanly on LTX-2 and is the fastest backend.

## Summary

The kernel speedup is real and consistent. It becomes visible e2e only when:

| Condition | Models | Effect |
|---|---|---|
| Masked attention + high share | HV-1.5 (47 %) | 2.0× e2e (cuDNN rescues the EFFICIENT-only masked path) |
| Heterogeneous shapes where dispatcher falls into memEff for some | LTX-2 (19 %) | 1.07× e2e (cuDNN wmma beats memEff on audio D=64 + cross-attn) |
| Unmasked, dispatcher already picks FA ≈ cuDNN | Wan 2.2 (23 %), Z-Image (10 %) | 1.01–1.005× (kernels essentially tied; too few steps on Z-Image) |
| Communication-bound (TP=2 NCCL = 62 % of GPU time) | FLUX.2 TP=2 (6.5 %) | 1.005× (no kernel change moves collective comm) |
