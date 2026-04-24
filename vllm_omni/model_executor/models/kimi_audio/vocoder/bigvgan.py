# SPDX-License-Identifier: MIT
# Adapted from NVIDIA BigVGAN via
# Kimi-Audio/kimia_infer/models/detokenizer/vocoder/bigvgan.py.
# The HF-mirror / PyTorchModelHubMixin paths are dropped — vllm-omni loads
# weights from the local ``{model_path}/vocoder/{config.json,model.pt}``.
import json

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .activations import Snake, SnakeBeta
from .alias_free_activation.torch.act import Activation1d as TorchActivation1d
from .utils import AttrDict, get_padding, init_weights, load_checkpoint


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        return AttrDict(json.loads(f.read()))


class AMPBlock1(nn.Module):
    """Two parallel conv stacks with snake / snakebeta anti-aliased activations."""

    def __init__(self, h: AttrDict, channels: int, kernel_size: int = 3,
                 dilation: tuple = (1, 3, 5), activation: str = None):
        super().__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                               dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                               dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in range(len(dilation))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)

        Activation1d = TorchActivation1d
        if activation == "snake":
            self.activations = nn.ModuleList([
                Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == "snakebeta":
            self.activations = nn.ModuleList([
                Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError(f"unknown activation: {activation}")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(nn.Module):
    """Single-stack variant of AMPBlock1 (no second fixed-dilation conv)."""

    def __init__(self, h: AttrDict, channels: int, kernel_size: int = 3,
                 dilation: tuple = (1, 3, 5), activation: str = None):
        super().__init__()
        self.h = h

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                               dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)

        Activation1d = TorchActivation1d
        if activation == "snake":
            self.activations = nn.ModuleList([
                Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == "snakebeta":
            self.activations = nn.ModuleList([
                Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError(f"unknown activation: {activation}")

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(nn.Module):
    """BigVGAN-v2 vocoder: mel (B, n_mels, T) -> wav (B, 1, T*hop)."""

    def __init__(self, h: AttrDict):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"unknown resblock class: {h.resblock}")

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList([
                    weight_norm(
                        ConvTranspose1d(
                            h.upsample_initial_channel // (2**i),
                            h.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    )
                ])
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for (k, d) in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        activation_post = (
            Snake(ch, alpha_logscale=h.snake_logscale) if h.activation == "snake"
            else SnakeBeta(ch, alpha_logscale=h.snake_logscale) if h.activation == "snakebeta"
            else None
        )
        if activation_post is None:
            raise NotImplementedError(f"unknown activation: {h.activation}")
        self.activation_post = TorchActivation1d(activation=activation_post)

        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)
        return x

    def remove_weight_norm(self):
        try:
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            pass

    @classmethod
    def from_local(cls, config_path: str, ckpt_path: str) -> "BigVGAN":
        """Load a BigVGAN from ``config.json`` + ``model.pt`` (the layout
        inside ``{kimi-audio-7b}/vocoder/``)."""
        h = load_hparams_from_json(config_path)
        model = cls(h)
        state = load_checkpoint(ckpt_path, "cpu")
        model.load_state_dict(state["generator"])
        return model
