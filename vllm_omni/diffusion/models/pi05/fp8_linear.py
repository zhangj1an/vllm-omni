# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP8 linear layers for the π0.5 prefix.

Ada (sm_89) and newer have FP8 tensor cores worth ~2x bfloat16 on large GEMMs.
The win depends entirely on shape, because an FP8 linear must quantize its
activations before it can use them:

    M=968 (prefix, 3 images + 200 text tokens)   1.079ms -> 0.603ms   1.79x
    M=50  (suffix, one action chunk at batch 1)  0.014ms -> 0.060ms   0.24x

At M=50 quantizing a (50, 1024) activation costs 0.037ms against a 0.039ms
GEMM, so the denoising loop is left in bfloat16 — quantizing it is a 4x loss,
regardless of which steps are chosen. Only the prefix is converted.

Scaling is per-tensor on both sides. Three variants were measured on
pi05_libero (chunk time / max deviation from the bfloat16 result):

    per-tensor                    116.8 ms   2.27e-1
    per-channel weights, fp32 out 129.9 ms   1.24e-1
    per-channel weights, bf16 out 119.4 ms   3.04e-1

Per-channel scales halve the error but need a float32 output tensor to hold the
unscaled GEMM result, and that (968, 16384) tensor's traffic costs more than the
GEMM saves. Keeping the output in bfloat16 is worse than per-tensor: with
scale_b=1 the unscaled result spans a range bfloat16's 7 mantissa bits cannot
hold. Per-tensor is therefore the operating point, and the error is what it is.

Accuracy: the prefix runs once per chunk and produces a KV cache, so error here
perturbs the conditioning a single time rather than feeding the 10-step Euler
integrator. Even so, 2.27e-1 is 4.5x this model's stated 5e-2 gate, which is why
``fp8_prefix`` is off by default and must be validated against task success --
not tensor deltas -- before use.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


class Fp8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` using an FP8 tensor-core GEMM.

    The weight is quantized once at conversion time; activations are quantized
    per call from their own amax, so no calibration pass is required.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.data
        self.out_features, self.in_features = weight.shape
        self.orig_dtype = weight.dtype

        # Per-output-channel weight scales. _scaled_mm's row-wise mode would
        # pair these with row-wise activation scales, which measured slower than
        # bfloat16 on these shapes; instead the GEMM runs in tensor-wise mode
        # with scale_b=1 and the per-channel factors are applied to the output,
        # one (M, N) multiply. Same accuracy benefit, tensor-wise speed.
        scale = (weight.float().abs().amax() / FP8_MAX).clamp(min=1e-12)
        # (out, in) fp8; .t() gives the (in, out) column-major operand _scaled_mm wants.
        self.register_buffer("weight_fp8", (weight.float() / scale).to(FP8_DTYPE), persistent=False)
        self.register_buffer("weight_scale", scale.reshape(()).float(), persistent=False)
        self.bias = linear.bias
        # Kept so a failed call can fall back, and so callers that inspect
        # ``.weight.dtype`` (e.g. the dtype-matching helper in modeling_pi05)
        # still see the layer's compute dtype rather than float8.
        self._fallback_weight = weight
        self._disabled = False

    @property
    def weight(self) -> torch.Tensor:
        """The original (unquantized) weight, for dtype queries and fallback."""
        return self._fallback_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._disabled:
            return torch.nn.functional.linear(x.to(self._fallback_weight.dtype), self._fallback_weight, self.bias)

        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        # Quantize in the activation's own dtype. Widening to float32 first
        # allocates and re-reads a tensor twice the size, which on the
        # (968, 16384) down_proj input costs 1.05ms against 0.21ms -- more than
        # the GEMM this is meant to accelerate.
        scale = (x2d.abs().amax().float() / FP8_MAX).clamp(min=1e-12).reshape(())
        x8 = (x2d * (1.0 / scale).to(x2d.dtype)).to(FP8_DTYPE)
        try:
            out = torch._scaled_mm(
                x8,
                self.weight_fp8.t(),
                scale_a=scale,
                scale_b=self.weight_scale,
                out_dtype=self.orig_dtype,
            )
        except RuntimeError as exc:
            logger.warning("Fp8Linear: %s; falling back to eager for this layer.", str(exc)[:120])
            self._disabled = True
            return torch.nn.functional.linear(x.to(self._fallback_weight.dtype), self._fallback_weight, self.bias)

        out = out.reshape(*shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_linears_to_fp8(module: nn.Module, min_numel: int = 1 << 20) -> int:
    """Replace ``nn.Linear`` children with :class:`Fp8Linear`, recursively.

    ``min_numel`` skips small projections, where the quantization cost is not
    amortized. Returns how many layers were converted.
    """
    converted = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if child.weight.numel() < min_numel or not child.weight.is_cuda:
                continue
            setattr(module, name, Fp8Linear(child))
            converted += 1
        else:
            converted += convert_linears_to_fp8(child, min_numel)
    return converted
