"""BigVGAN for Kimi-Audio built on HF's ``Qwen2_5OmniToken2WavBigVGANModel``.
The subclass skips Qwen-Omni's mel preprocessor (Kimi's mel is already
normalized) and keeps the output on-device as (B, 1, T). The state-dict
adapter fuses ``weight_norm``-factored conv weights into plain weights and
drops alias-free filter buffers (regenerated at construction)."""

import json
import logging
import os

import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniBigVGANConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniToken2WavBigVGANModel,
)

logger = logging.getLogger(__name__)


def _kimi_to_hf_bigvgan_config(kimi_h: dict) -> Qwen2_5OmniBigVGANConfig:
    """Translate Kimi's ``vocoder/config.json`` into an HF config. HF
    hardcodes several axes; we assert first so a divergent future
    checkpoint fails loudly instead of producing wrong audio."""
    _assert_hf_compatible(kimi_h)
    return Qwen2_5OmniBigVGANConfig(
        mel_dim=kimi_h["num_mels"],
        upsample_initial_channel=kimi_h["upsample_initial_channel"],
        resblock_kernel_sizes=kimi_h["resblock_kernel_sizes"],
        resblock_dilation_sizes=kimi_h["resblock_dilation_sizes"],
        upsample_rates=kimi_h["upsample_rates"],
        upsample_kernel_sizes=kimi_h["upsample_kernel_sizes"],
    )


def _assert_hf_compatible(h: dict) -> None:
    # Any divergence would produce wrong audio without a shape mismatch.
    if h.get("activation", "snakebeta") != "snakebeta":
        raise ValueError(
            f"HF Qwen-Omni BigVGAN hardcodes SnakeBeta; got activation="
            f"{h.get('activation')!r}"
        )
    if h.get("resblock", "1") != "1":
        raise ValueError(
            f"HF Qwen-Omni BigVGAN hardcodes AMPBlock1 (resblock='1'); got "
            f"resblock={h.get('resblock')!r}"
        )
    if h.get("use_tanh_at_final", False):
        raise ValueError(
            "HF Qwen-Omni BigVGAN applies torch.clamp(-1, 1); config requires "
            "use_tanh_at_final=false"
        )
    if h.get("use_bias_at_final", False):
        raise ValueError(
            "HF Qwen-Omni BigVGAN's conv_post has bias=False; config requires "
            "use_bias_at_final=false"
        )


def _adapt_kimi_state_dict(kimi_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert NVIDIA-BigVGAN keys to HF Qwen-Omni BigVGAN keys: fuse
    ``weight_g``/``weight_v`` pairs into ``weight`` (as ``weight_norm``
    would) and drop alias-free filter buffers (HF regenerates them)."""
    out: dict[str, torch.Tensor] = {}

    pending_g: dict[str, torch.Tensor] = {}
    pending_v: dict[str, torch.Tensor] = {}

    for k, v in kimi_sd.items():
        if k.endswith(".filter") and (
            ".upsample." in k or ".downsample." in k
        ):
            continue
        if k.endswith(".weight_g"):
            pending_g[k[: -len(".weight_g")]] = v
        elif k.endswith(".weight_v"):
            pending_v[k[: -len(".weight_v")]] = v
        else:
            out[k] = v

    prefixes = set(pending_g) | set(pending_v)
    for prefix in prefixes:
        if prefix not in pending_g or prefix not in pending_v:
            raise ValueError(
                f"Dangling weight_norm shard for {prefix!r}: "
                f"has_g={prefix in pending_g} has_v={prefix in pending_v}"
            )
        g = pending_g[prefix]
        w_v = pending_v[prefix]
        norm_dims = list(range(1, w_v.ndim))
        fused = w_v * (g / w_v.norm(dim=norm_dims, keepdim=True))
        out[f"{prefix}.weight"] = fused

    return out


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
    def load_kimi_checkpoint(cls, model_config, ckpt_path, device) -> "KimiBigVGAN":
        with open(model_config) as f:
            kimi_h = json.load(f)
        hf_config = _kimi_to_hf_bigvgan_config(kimi_h)
        vocoder = cls(hf_config)

        assert os.path.isfile(ckpt_path), ckpt_path
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        hf_sd = _adapt_kimi_state_dict(checkpoint["generator"])
        missing, unexpected = vocoder.load_state_dict(hf_sd, strict=False)
        # HF alias-free filter buffers are persistent=False and regenerate
        # at construction, so they're "missing" from a saved dict.
        real_missing = [
            k for k in missing
            if not (
                k.endswith(".filter")
                and (".upsample." in k or ".downsample." in k)
            )
        ]
        if real_missing or unexpected:
            raise RuntimeError(
                f"State-dict adapter failed to cover all keys. "
                f"missing={real_missing}, unexpected={unexpected}"
            )

        logger.info(">>> Load vocoder from %s", ckpt_path)
        return vocoder.to(device).eval()
