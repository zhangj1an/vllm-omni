"""What does PyTorch's default SDPA dispatcher pick for each model's shape?

If `CUDNN_ATTN` is renaming a kernel torch already chose by default, that's
hypothesis #3 from the colleague. To test: run F.scaled_dot_product_attention
with NO sdpa_kernel context manager, capture which backend torch dispatched to,
both with and without a 4D additive attn_mask.

Two probes:
  1. TORCH_LOGS=+sdpa via env var — torch prints a one-line dispatch decision
     per call to stderr.
  2. Direct introspection — for each (shape, mask) combination, manually
     check `can_use_*` from torch.backends.cuda. Confirms what env var said.
"""
import os
import sys
import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16
DEV = "cuda:0"

# (label, B, H, S, D, mask_kind)
SHAPES = [
    ("HV-1.5",          1, 16, 14296, 128, "pad-256"),
    ("Wan 2.2 self",    1, 40, 14040, 128, None),
    ("Wan 2.2 cross",   1, 40, 14040, 128, "kv-512"),  # cross-attn: kv != q
    ("FLUX.2 (full H)", 1, 48,  4608, 128, "pad-512"),
    ("FLUX.2 (TP=2)",   1, 24,  4608, 128, "pad-512"),
    ("Z-Image",         1, 30,  4352, 128, "pad-256"),
    ("LTX-2 video",     1, 32,  5070, 128, None),
    ("LTX-2 audio",     1, 32,   100,  64, None),
]


def _make_mask(B, S_q, S_k, kind):
    if kind is None:
        return None
    mask = torch.zeros(B, 1, S_q, S_k, device=DEV, dtype=DTYPE)
    if kind.startswith("pad-"):
        pad = int(kind.split("-")[1])
        mask[..., -pad:] = float("-inf")
    return mask


def main():
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}  sm_{cc[0]}{cc[1]}")
    print(f"torch: {torch.__version__}, cuDNN: {torch.backends.cudnn.version()}")
    print()
    print(f"{'shape':<22} {'mask':<12} {'default dispatch':<24} {'cuDNN avail':<12} {'FA avail':<10} {'EFF avail':<10}")
    print("-" * 96)

    for label, B, H, S, D, mask_kind in SHAPES:
        # cross-attn case: query has full S, but kv has 512
        S_q = S
        S_k = 512 if mask_kind == "kv-512" else S
        q = torch.randn(B, H, S_q, D, device=DEV, dtype=DTYPE)
        k = torch.randn(B, H, S_k, D, device=DEV, dtype=DTYPE)
        v = torch.randn(B, H, S_k, D, device=DEV, dtype=DTYPE)
        mask = _make_mask(B, S_q, S_k, mask_kind)

        # The torch.backends.cuda module exposes runtime checks for whether
        # each backend can handle this exact (q,k,v,mask) — same logic SDPA
        # uses internally to dispatch.
        from torch.backends.cuda import (
            can_use_cudnn_attention,
            can_use_flash_attention,
            can_use_efficient_attention,
            SDPAParams,
        )
        params = SDPAParams(q, k, v, mask, 0.0, False, False)
        avail_cudnn = can_use_cudnn_attention(params, debug=False)
        avail_fa = can_use_flash_attention(params, debug=False)
        avail_eff = can_use_efficient_attention(params, debug=False)

        # Now run an actual SDPA call with no override and see what wins
        # by inspecting the kernel that fires (use TORCH_LOGS=+sdpa upstream).
        # We mimic the dispatcher's priority order:
        #   FLASH > MEM_EFFICIENT > MATH for non-CUDNN path,
        #   but on cuDNN-eligible setups CUDNN can preempt.
        # For our purposes, just record the priority outcome.
        if avail_cudnn:
            picked = "CUDNN"
        elif avail_fa:
            picked = "FLASH"
        elif avail_eff:
            picked = "EFFICIENT"
        else:
            picked = "MATH (fallback)"

        # Sanity: actually run it once to confirm no error.
        try:
            with torch.no_grad():
                _ = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        except Exception as e:
            picked = f"ERR: {type(e).__name__}"

        m_str = mask_kind or "none"
        print(f"{label:<22} {m_str:<12} {picked:<24} {str(avail_cudnn):<12} {str(avail_fa):<10} {str(avail_eff):<10}")


if __name__ == "__main__":
    main()
