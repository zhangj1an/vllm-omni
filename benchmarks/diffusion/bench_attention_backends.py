# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone diagnostic for the diffusion attention backends added in #3079.

Exercises the same synthetic attention shape through each backend and each
torch SDPA sub-kernel, with and without an attn_mask, so we can pinpoint why
CUDNN_ATTN underperforms on some SKUs. Two typical causes:

  (a) cuDNN lacks a tuned kernel for this (SM, head_dim, seq) combination
      and silently falls back to MATH.
  (b) attn_mask is non-None and the cuDNN SDPA dispatch rejects it,
      walking CUDNN -> FLASH -> MATH.

Run:
    python benchmarks/diffusion/bench_attention_backends.py --preset hv15
    python benchmarks/diffusion/bench_attention_backends.py --preset wan22
    python benchmarks/diffusion/bench_attention_backends.py \
        --batch 1 --heads 24 --seq 14336 --head-dim 128

The table at the end is the data we want on the PR — surface the row where a
backend is >1.5x the SDPA baseline, that's the one to gate off the auto-route.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Presets mirror the DiT hot-path attention call (single-stream block) at the
# shapes we run in the PR's validation configs.
_PRESETS = {
    # HunyuanVideo-1.5 480p/33f: (30 * 52 * 9) latent + ~256 text tokens.
    "hv15": {"batch": 1, "heads": 24, "seq": 14336, "head_dim": 128},
    # Wan 2.2 480p/33f rough estimate; adjust when we measure real shapes.
    "wan22": {"batch": 1, "heads": 40, "seq": 16384, "head_dim": 128},
    # Smaller image-gen shape for quick smoke tests.
    "flux": {"batch": 1, "heads": 24, "seq": 4096, "head_dim": 128},
}

_SDPA_BACKENDS = [
    ("CUDNN_ATTENTION", [SDPBackend.CUDNN_ATTENTION]),
    ("FLASH_ATTENTION", [SDPBackend.FLASH_ATTENTION]),
    ("EFFICIENT_ATTENTION", [SDPBackend.EFFICIENT_ATTENTION]),
    ("MATH", [SDPBackend.MATH]),
    # The PR's CUDNN_ATTN impl uses this priority chain; keep it here to show
    # what torch actually picks when multiple are allowed.
    ("CUDNN_ATTN_CHAIN", [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]),
]


def _env_report() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this diagnostic needs a GPU.")
    cc = torch.cuda.get_device_capability()
    print("=" * 72)
    print(f"GPU           : {torch.cuda.get_device_name()}")
    print(f"SM capability : sm_{cc[0]}{cc[1]}")
    print(f"torch         : {torch.__version__}")
    print(f"cuDNN version : {torch.backends.cudnn.version()}")
    print("flashinfer    : ", end="")
    try:
        import flashinfer

        print(getattr(flashinfer, "__version__", "present"))
    except Exception as e:
        print(f"not installed ({type(e).__name__})")
    print("=" * 72)


def _make_qkv(batch: int, heads: int, seq: int, head_dim: int, device: str, dtype: torch.dtype):
    # Layout: (B, S, H, D) — what the PR's backends accept as input.
    q = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
    return q, k, v


def _make_mask(batch: int, seq: int, device: str, dtype: torch.dtype, pad_tokens: int = 128) -> torch.Tensor:
    # Mirrors a text-encoder padding mask: last `pad_tokens` positions masked.
    mask = torch.zeros(batch, 1, seq, seq, device=device, dtype=dtype)
    if pad_tokens > 0:
        mask[..., -pad_tokens:] = float("-inf")
    return mask


def _time_call(fn, *args, warmup: int = 3, iters: int = 10) -> tuple[float, str]:
    """Return (median ms, error string). Skips on runtime errors."""
    try:
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(*args)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        return times[len(times) // 2], ""
    except RuntimeError as e:
        return float("nan"), type(e).__name__


def _run_sdpa_variants(q, k, v, attn_mask, scale: float) -> list[tuple[str, float, str]]:
    # q/k/v come in (B, S, H, D); F.sdpa wants (B, H, S, D).
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    rows: list[tuple[str, float, str]] = []
    for name, backends in _SDPA_BACKENDS:

        def _call(mask=attn_mask):
            with sdpa_kernel(backends):
                return torch.nn.functional.scaled_dot_product_attention(
                    q_t, k_t, v_t, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale
                )

        ms, err = _time_call(_call)
        rows.append((name, ms, err))
    return rows


def _run_flashinfer(q, k, v, scale: float) -> tuple[float, str]:
    try:
        from flashinfer.prefill import single_prefill_with_kv_cache
    except Exception as e:
        return float("nan"), f"import-{type(e).__name__}"

    def _call():
        # FlashInfer dense prefill: (S, H, D) per batch item.
        out = single_prefill_with_kv_cache(q[0], k[0], v[0], sm_scale=scale, causal=False, return_lse=False)
        return out.unsqueeze(0)

    return _time_call(_call)


def _print_table(title: str, rows: list[tuple[str, float, str]]) -> None:
    print(f"\n{title}")
    print("-" * 72)
    print(f"{'backend':<24} {'median (ms)':>14}    status")
    print("-" * 72)
    for name, ms, err in rows:
        ms_str = f"{ms:>14.3f}" if ms == ms else f"{'n/a':>14}"  # NaN check
        status = "ok" if not err else f"FAILED ({err})"
        print(f"{name:<24} {ms_str}    {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=list(_PRESETS.keys()), default="hv15")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    shape = dict(_PRESETS[args.preset])
    for k in ("batch", "heads", "seq", "head_dim"):
        v = getattr(args, k if k != "head_dim" else "head_dim")
        if v is not None:
            shape[k] = v
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    _env_report()
    print(
        f"Shape         : batch={shape['batch']} heads={shape['heads']} seq={shape['seq']} head_dim={shape['head_dim']}"
    )
    print(f"dtype         : {args.dtype}")

    torch.manual_seed(0)
    q, k, v = _make_qkv(**shape, device=args.device, dtype=dtype)
    scale = 1.0 / (shape["head_dim"] ** 0.5)

    # --- No mask (Wan-like hot path) ---
    rows_nomask = _run_sdpa_variants(q, k, v, attn_mask=None, scale=scale)
    fi_ms, fi_err = _run_flashinfer(q, k, v, scale)
    rows_nomask.append(("FLASHINFER (dense)", fi_ms, fi_err))
    _print_table("No attention mask", rows_nomask)

    # --- With padding mask (HV-1.5 hot path) ---
    mask = _make_mask(shape["batch"], shape["seq"], device=args.device, dtype=dtype, pad_tokens=256)
    rows_mask = _run_sdpa_variants(q, k, v, attn_mask=mask, scale=scale)
    rows_mask.append(("FLASHINFER (dense)", float("nan"), "mask-not-supported"))
    _print_table("With attention mask (pad 256 tokens)", rows_mask)

    print("\nInterpretation:")
    print("  * If CUDNN_ATTENTION shows FAILED on the mask run → cuDNN rejects the mask")
    print("    shape, and CUDNN_ATTN_CHAIN falls to FLASH or MATH. That's why the PR's")
    print("    auto-route hurts HV-1.5: it pins the chain where SDPA would have picked")
    print("    FLASH directly.")
    print("  * If CUDNN_ATTENTION runs but is >1.5x FLASH_ATTENTION → cuDNN has no tuned")
    print("    kernel for this (SM, shape) pair. Restrict the auto-route to datacenter")
    print("    Blackwell (sm_100/103) until cuDNN 9.x adds consumer Blackwell support.")


if __name__ == "__main__":
    main()
