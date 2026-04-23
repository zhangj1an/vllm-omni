"""BigVGAN for Kimi-Audio built on HF's ``Qwen2_5OmniToken2WavBigVGANModel``.
The subclass skips Qwen-Omni's mel preprocessor (Kimi's mel is already
normalized) and keeps the output on-device as (B, 1, T). Weights are loaded
from a pre-fused HF checkpoint (zhangj1an/kimi-audio-bigvgan-hf): the
original Moonshot checkpoint stored weight_g/weight_v shards and alias-free
filter buffers; the HF mirror has those fused / stripped so load_state_dict
is a direct call."""

import json
import logging

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniBigVGANConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniToken2WavBigVGANModel,
)

logger = logging.getLogger(__name__)

HF_VOCODER_REPO = "zhangj1an/kimi-audio-bigvgan-hf"


class KimiBigVGAN(Qwen2_5OmniToken2WavBigVGANModel):

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_pre(mel_spectrogram)
        for layer_index in range(self.num_upsample_layers):
            hidden = self.ups[layer_index][0](hidden)
            residual = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](
                    hidden
                )
                for block_index in range(self.num_residual_blocks)
            )
            hidden = residual / self.num_residual_blocks
        hidden = self.activation_post(hidden)
        output = self.conv_post(hidden)
        return torch.clamp(output, min=-1.0, max=1.0)

    def decode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """[T, num_mels] mel -> [1, T] wav."""
        mel = mel.transpose(0, 1).unsqueeze(0).to(self.device)
        wav = self(mel)
        return wav.squeeze(0)

    @classmethod
    def load_from_hf(cls, device) -> "KimiBigVGAN":
        config_path = hf_hub_download(HF_VOCODER_REPO, "config.json")
        weights_path = hf_hub_download(HF_VOCODER_REPO, "model.safetensors")
        with open(config_path) as f:
            kimi_h = json.load(f)
        hf_config = Qwen2_5OmniBigVGANConfig(
            mel_dim=kimi_h["num_mels"],
            upsample_initial_channel=kimi_h["upsample_initial_channel"],
            resblock_kernel_sizes=kimi_h["resblock_kernel_sizes"],
            resblock_dilation_sizes=kimi_h["resblock_dilation_sizes"],
            upsample_rates=kimi_h["upsample_rates"],
            upsample_kernel_sizes=kimi_h["upsample_kernel_sizes"],
        )
        vocoder = cls(hf_config)
        vocoder.load_state_dict(load_file(weights_path))
        logger.info(">>> Loaded vocoder from %s", HF_VOCODER_REPO)
        return vocoder.to(device).eval()
