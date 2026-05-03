"""Parse an nsys cuda_gpu_kern_sum CSV and bucket kernels into attention,
gemm/matmul, elementwise, norm, conv, reshape, other.

Usage:
  nsys stats --report cuda_gpu_kern_sum --format csv path/to.nsys-rep \\
    | python bench_out/parse_nsys.py [--top N]
"""
import argparse
import csv
import sys


ATTN_PATTERNS = (
    "flash_fprop",         # cuDNN sm120/sm80
    "flash_fwd",           # pytorch_flash::flash_fwd
    "fmha",                # PyTorch memEff (cutlassF)
    "single_prefill",      # FlashInfer
    "prefill_kernel",      # FlashInfer JIT
    "scaled_dot_product",  # any sdpa-named triton kernel
    "_attention_",
    "sdpa_",
)
GEMM_PATTERNS = ("gemm", "xmma", "cublas")
NORM_PATTERNS = ("rowwisemoments", "groupnorm", "layernorm", "rms_")
CONV_PATTERNS = ("fprop", "conv", "nchw", "nhwc")  # _fprop appears in cuDNN sdpa too — order matters
ELEM_PATTERNS = (
    "elementwise",
    "vectorized_elementwise",
    "reduce_kernel",
    "copy_kernel",
)
RESHAPE_PATTERNS = ("view", "permute", "transpose")


def bucket(name: str) -> str:
    nl = name.lower()
    # Attention check first because some attn kernels include "fprop"
    if any(p in nl for p in ATTN_PATTERNS):
        return "attention"
    if any(p in nl for p in GEMM_PATTERNS):
        return "gemm"
    if any(p in nl for p in NORM_PATTERNS):
        return "norm"
    if any(p in nl for p in CONV_PATTERNS):
        return "conv"
    if any(p in nl for p in ELEM_PATTERNS):
        return "elem"
    if any(p in nl for p in RESHAPE_PATTERNS):
        return "reshape"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8, help="Top N kernels per bucket to print")
    args = ap.parse_args()

    totals = {k: 0 for k in ("attention", "gemm", "elem", "norm", "conv", "reshape", "other")}
    by_bucket: dict[str, list[tuple[str, float]]] = {k: [] for k in totals}
    grand = 0.0

    for i, row in enumerate(csv.reader(sys.stdin)):
        if i == 0 or len(row) < 8:
            continue
        try:
            t = float(row[1])
        except ValueError:
            continue
        name = row[-1]
        b = bucket(name)
        totals[b] += t
        by_bucket[b].append((name, t))
        grand += t

    print(f"Total GPU kernel time: {grand/1e9:.3f} s")
    print()
    for b in ("attention", "gemm", "elem", "norm", "conv", "reshape", "other"):
        v = totals[b]
        print(f"  {b:<11} : {v/1e9:>7.3f} s ({100*v/grand:>5.1f}%)")
    print()
    print(f"Top {args.top} attention kernels:")
    for n, t in sorted(by_bucket["attention"], key=lambda x: -x[1])[: args.top]:
        print(f"  {t/1e6:>8.1f} ms  {n[:96]}")
    if by_bucket["other"]:
        print()
        print(f"Top {args.top} 'other' kernels (sanity check that we're not mis-bucketing):")
        for n, t in sorted(by_bucket["other"], key=lambda x: -x[1])[: args.top]:
            print(f"  {t/1e6:>8.1f} ms  {n[:96]}")


if __name__ == "__main__":
    main()
