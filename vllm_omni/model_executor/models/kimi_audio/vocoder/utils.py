# SPDX-License-Identifier: MIT
# Ported from Kimi-Audio kimia_infer/models/detokenizer/vocoder/utils.py
# (``get_melspec`` and its librosa dep dropped — not used by vllm-omni's
# one-way mel→wav decode path).
import os

import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    return torch.load(filepath, map_location=device, weights_only=True)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
