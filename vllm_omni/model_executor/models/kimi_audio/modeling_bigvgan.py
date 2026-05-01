# SPDX-License-Identifier: MIT
# Adapted from NVIDIA BigVGAN via Kimi-Audio's vendored copy.
# Trimmed to the single configuration the Kimi-Audio-7B vocoder uses:
#     resblock="1", activation="snakebeta", snake_logscale=true,
#     use_tanh_at_final=false, use_bias_at_final=false.
#
# This file consolidates the previously-nested
# ``vocoder/{bigvgan,activations,utils}.py`` plus
# ``vocoder/alias_free_activation/torch/{act,filter,resample}.py`` and
# the thin ``KimiBigVGAN`` wrapper that lived at the parent level.
"""Kimi-Audio BigVGAN-v2 vocoder. Mel (B, n_mels, T) -> wav (B, 1, T*hop)."""
import json
import logging
import math
import os

import torch
from torch import Tensor, nn, sin
from torch.nn import Conv1d, ConvTranspose1d, Parameter
from torch.nn import functional as F
from torch.nn.utils import weight_norm

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# config / weight helpers
# -------------------------------------------------------------------------
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    return torch.load(filepath, map_location=device, weights_only=True)


def init_weights(m, mean=0.0, std=0.01):
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


# -------------------------------------------------------------------------
# anti-aliased activation (alias-free-torch)
#   forward = downsample(activation(upsample(x)))
# -------------------------------------------------------------------------
def _kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> Tensor:
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        return torch.zeros_like(time).view(1, 1, kernel_size)
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ = filter_ / filter_.sum()
    return filter_.view(1, 1, kernel_size)


class _LowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", _kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: Tensor) -> Tensor:
        _, C, _ = x.shape
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)


class _UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            _kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C,
        )
        return x[..., self.pad_left : -self.pad_right]


class _DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.lowpass = _LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.lowpass(x)


class SnakeBeta(nn.Module):
    """SnakeBeta ∶= x + 1/exp(b) * sin^2(x * exp(a)); both trainable, log-scale."""

    def __init__(self, in_features: int):
        super().__init__()
        self.alpha = Parameter(torch.zeros(in_features))
        self.beta = Parameter(torch.zeros(in_features))

    def forward(self, x: Tensor) -> Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return x + (1.0 / (beta + 1e-9)) * sin(x * alpha).pow(2)


class _Activation1d(nn.Module):
    """Anti-aliased SnakeBeta: 2x upsample → SnakeBeta → 2x downsample."""

    def __init__(self, channels: int):
        super().__init__()
        self.upsample = _UpSample1d(ratio=2, kernel_size=12)
        self.act = SnakeBeta(channels)
        self.downsample = _DownSample1d(ratio=2, kernel_size=12)

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(self.act(self.upsample(x)))


# -------------------------------------------------------------------------
# BigVGAN body
# -------------------------------------------------------------------------
class _AMPBlock1(nn.Module):
    """Two parallel conv stacks with anti-aliased SnakeBeta activations."""

    def __init__(self, channels: int, kernel_size: int, dilation: tuple):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                               dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                               dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations = nn.ModuleList(
            [_Activation1d(channels) for _ in range(2 * len(dilation))]
        )

    def forward(self, x: Tensor) -> Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = c1(a1(x))
            xt = c2(a2(xt))
            x = xt + x
        return x


class BigVGAN(nn.Module):
    """conv_pre → 7 upsample stages × (ConvTranspose1d + 4 AMPBlock1)
    → SnakeBeta → conv_post (no bias) → clamp(-1, 1)."""

    def __init__(self, h: AttrDict):
        super().__init__()
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Each upsample stage is wrapped in a ModuleList so the checkpoint
        # keys (``ups.{i}.0.weight*``) load unchanged.
        self.ups = nn.ModuleList([
            nn.ModuleList([
                weight_norm(ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2,
                ))
            ])
            for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes))
        ])
        for stage in self.ups:
            stage.apply(init_weights)

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(_AMPBlock1(ch, k, d))

        self.activation_post = _Activation1d(ch)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=False))
        self.conv_post.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
            x = xs / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0)


# -------------------------------------------------------------------------
# Kimi-Audio wrapper: load weights from the checkpoint dir + mel→wav helper
# -------------------------------------------------------------------------
class KimiBigVGAN(BigVGAN):
    def decode_mel(self, mel: Tensor) -> Tensor:
        """[T, num_mels] mel → [1, T] wav."""
        # ``conv_pre`` is wrapped with weight_norm; on torch>=2.10 the
        # ``.weight`` parametrization may report a stale device while the
        # actual parameters (weight_v/weight_g) live on the configured
        # device. Use the first parameter's device instead.
        target_device = next(self.parameters()).device
        mel = mel.transpose(0, 1).unsqueeze(0).to(target_device)
        return self(mel).squeeze(0)

    @classmethod
    def load_from_local(cls, model_path: str, device) -> "KimiBigVGAN":
        """Load from ``{model_path}/vocoder/{config.json,model.pt}``."""
        config_path = os.path.join(model_path, "vocoder", "config.json")
        ckpt_path = os.path.join(model_path, "vocoder", "model.pt")
        with open(config_path) as f:
            h = AttrDict(json.load(f))
        model = cls(h)
        state = load_checkpoint(ckpt_path, "cpu")
        model.load_state_dict(state["generator"])
        logger.info(">>> Loaded vocoder from %s", ckpt_path)
        return model.to(device).eval()
