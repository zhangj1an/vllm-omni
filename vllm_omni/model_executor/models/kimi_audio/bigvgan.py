"""Kimi-Audio BigVGAN wrapper. Delegates to the real Moonshot BigVGAN
(vendored under ``.vocoder``) with snake + anti-aliased activations, and
loads weights from the checkpoint-local ``{model_path}/vocoder/`` —
matching upstream ``KimiAudio.detokenize_audio`` bit-for-bit.

The earlier HF ``Qwen2_5OmniToken2WavBigVGANModel`` fork dropped the
anti-alias filters, producing mels→wavs with the right content/pace but
audible aliasing noise ("fuzzy" output). This module replaces that path."""

import logging
import os

import torch

from .vocoder.bigvgan import BigVGAN as _BigVGAN

logger = logging.getLogger(__name__)


class KimiBigVGAN(_BigVGAN):
    def decode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """[T, num_mels] mel -> [1, T] wav. Matches ``BigVGANWrapper.decode_mel``."""
        # ``conv_pre`` is wrapped with weight_norm; on torch>=2.10 ``.weight``
        # is recomputed via parametrization and may report a stale device,
        # while the actual parameters (weight_v/weight_g) live on the
        # configured device. Use the first parameter's device instead.
        target_device = next(self.parameters()).device
        mel = mel.transpose(0, 1).unsqueeze(0).to(target_device)
        wav = self(mel)
        return wav.squeeze(0)

    @classmethod
    def load_from_local(cls, model_path: str, device) -> "KimiBigVGAN":
        """Load vocoder from ``{model_path}/vocoder/{config.json,model.pt}``."""
        config_path = os.path.join(model_path, "vocoder", "config.json")
        ckpt_path = os.path.join(model_path, "vocoder", "model.pt")
        model = cls.from_local(config_path, ckpt_path)
        logger.info(">>> Loaded vocoder from %s", ckpt_path)
        return model.to(device).eval()
