# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused AdaRMS normalization for the π0.5 action expert.

The eager form of a conditioned AdaRMS norm is

    normed = x.float() * rsqrt(mean(x.float() ** 2) + eps)
    out    = (normed * (1 + scale.float()) + shift.float()).to(x.dtype)

which PyTorch executes as roughly eight elementwise/reduction kernels, each
re-reading a ``(50, 1024)`` activation. The expert applies 37 of these per
denoising step, 370 per chunk, and profiling attributes ~23% of a chunk's GPU
time to elementwise kernels of ~2.6us each — memory-bound traffic, not
arithmetic. Fusing collapses the chain to one kernel and one pass over ``x``.

Falls back to the eager path when Triton is unavailable, when the tensors are
not contiguous CUDA tensors, or when the row is wider than one block.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on builds without triton
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _adarms_modulate_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        row_stride,
        n_cols,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols

        x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        # mean over the full row; masked lanes contribute 0 to the sum.
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        y = x * rstd * (1.0 + scale) + shift
        tl.store(out_ptr + row * row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def can_fuse(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    """True when the fused kernel can serve this call."""
    if not HAS_TRITON or not x.is_cuda:
        return False
    if x.ndim != 3:
        return False
    n_cols = x.shape[-1]
    if n_cols > 8192 or n_cols & (n_cols - 1):  # one power-of-two block per row
        return False
    # scale/shift broadcast across tokens from a single row; anything else
    # (a real per-token modulation) is not what this kernel implements.
    if scale.numel() != n_cols or shift.numel() != n_cols:
        return False
    return x.is_contiguous()


def adarms_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMS-normalize ``x`` and apply ``(1 + scale)`` / ``shift`` in one pass."""
    out = torch.empty_like(x)
    n_rows = x.shape[0] * x.shape[1]
    n_cols = x.shape[-1]
    _adarms_modulate_kernel[(n_rows,)](
        x,
        scale.reshape(-1),
        shift.reshape(-1),
        out,
        n_cols,  # contiguous rows
        n_cols,
        eps,
        BLOCK=triton.next_power_of_2(n_cols),
        num_warps=8,
    )
    return out
