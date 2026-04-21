"""BigVGAN wrapper for Kimi-Audio.

The BigVGAN generator itself is HuggingFace's ``Qwen2_5OmniToken2WavBigVGANModel``
from ``transformers.models.qwen2_5_omni`` — architecturally identical to NVIDIA
BigVGAN on the axes Kimi's checkpoint exercises (SnakeBeta activations, no
final tanh, no final bias, alias-free up/down filters). We subclass it to:

1. Skip ``process_mel_spectrogram`` — that's a Qwen-Omni-specific input
   normalization (amplitude→dB→[-1,1] scale). Kimi's BigVGAN consumes the mel
   directly; routing through Qwen's preprocessor would double-normalize.
2. Drop the ``.squeeze().cpu()`` at the end of HF's forward — the streaming
   detokenizer needs ``(B, 1, T)`` tensors on the source device.

And a state-dict adapter fuses Kimi's ``weight_norm``-factored conv weights
(``weight_g`` + ``weight_v``) into plain ``weight`` tensors before loading
into HF, plus drops the alias-free filter buffers (HF registers those as
``persistent=False``, so they regenerate from kernel_size at construction).
"""

import json
import logging
import os

import librosa
import torch
from librosa.filters import mel as librosa_mel_fn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniBigVGANConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniToken2WavBigVGANModel,
)

logger = logging.getLogger(__name__)


_mel_basis_cache: dict[str, torch.Tensor] = {}
_hann_window_cache: dict[str, torch.Tensor] = {}


def _get_melspec(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None = None,
    center: bool = False,
) -> torch.Tensor:
    """Log-mel spectrogram with slaney-norm librosa filterbank + Hann STFT.

    Mirrors NVIDIA BigVGAN's ``get_melspec``; kept local so we can drop the
    vendored ``vocoder/utils.py``.
    """
    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"
    if key not in _mel_basis_cache:
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        _mel_basis_cache[key] = torch.from_numpy(mel_basis).float().to(device)
        _hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = _mel_basis_cache[key]
    hann_window = _hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    return torch.log(torch.clamp(mel_spec, min=1e-5))  # spectral_normalize


def _kimi_to_hf_bigvgan_config(kimi_h: dict) -> Qwen2_5OmniBigVGANConfig:
    """Translate Kimi's ``vocoder/config.json`` into an HF config.

    HF hardcodes the axes Kimi also selects (``snakebeta``, no tanh-at-final,
    no bias-at-final, AMPBlock1). We assert that before constructing so a
    future checkpoint with different choices fails loudly rather than
    silently producing wrong audio.
    """
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
    # HF's port hardcodes these choices; any divergence would produce wrong
    # audio without a shape mismatch to catch it.
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
    """Convert a NVIDIA-BigVGAN-style state dict to HF Qwen-Omni BigVGAN keys.

    Two transforms:

    * Fuse every ``<prefix>.weight_g`` + ``<prefix>.weight_v`` pair into
      ``<prefix>.weight`` via ``w = v * (g / ||v||_per_out_channel)``. That's
      how ``torch.nn.utils.weight_norm`` materializes ``weight`` from its
      factored form.
    * Drop ``<...>.upsample.filter`` and ``<...>.downsample.lowpass.filter``
      buffers. HF registers those as ``persistent=False``, so they're
      regenerated deterministically from ``kernel_size`` at construction.
    """
    out: dict[str, torch.Tensor] = {}

    # First pass: collect weight_g / weight_v pairs by their shared prefix.
    pending_g: dict[str, torch.Tensor] = {}
    pending_v: dict[str, torch.Tensor] = {}

    for k, v in kimi_sd.items():
        # Drop alias-free filter buffers regardless of NVIDIA's nested
        # naming (``.downsample.lowpass.filter``) vs HF's flat naming
        # (``.downsample.filter``). HF registers these ``persistent=False``
        # and regenerates them deterministically from ``kernel_size``.
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

    # Second pass: fuse each pair.
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
    """HF Qwen-Omni BigVGAN wired up to consume Kimi-style log-mel inputs.

    Overrides ``forward`` to (1) skip ``process_mel_spectrogram`` and (2) keep
    the output on-device as ``(B, 1, T)`` rather than ``.squeeze().cpu()``.
    Everything else — layer topology, SnakeBeta, final clamp, alias-free
    filters — is inherited unchanged.
    """

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


class BigVGANWrapper:
    def __init__(
        self,
        vocoder: KimiBigVGAN,
        device: torch.device,
        h: dict,
        dtype=None,
    ) -> None:
        self.vocoder = vocoder.to(device)
        if dtype is not None:
            self.vocoder = self.vocoder.to(dtype)
        self.vocoder = self.vocoder.eval()
        self.device = device
        self.h = h

    def to_dtype(self, dtype):
        self.vocoder = self.vocoder.to(dtype)

    def extract_mel_from_wav(self, wav_path=None, wav_data=None):
        """
        params:
            wav_path: str, path of the wav, should be 24k
            wav_data: torch.tensor or numpy array, shape [T], wav data, should be 24k
        return:
            mel: [T, num_mels], torch.tensor
        """
        if wav_data is None:
            wav_data, _ = librosa.load(wav_path, sr=self.h["sampling_rate"])

        wav_data = torch.tensor(wav_data).unsqueeze(0)

        mel = _get_melspec(
            y=wav_data,
            n_fft=self.h["n_fft"],
            num_mels=self.h["num_mels"],
            sampling_rate=self.h["sampling_rate"],
            hop_size=self.h["hop_size"],
            win_size=self.h["win_size"],
            fmin=self.h["fmin"],
            fmax=self.h["fmax"],
        )
        return mel.squeeze(0).transpose(0, 1)

    @torch.inference_mode()
    def extract_mel_from_wav_batch(self, wav_data):
        """
        params:
            wav_data: torch.tensor or numpy array, shape [Batch, T], wav data, should be 24k
        return:
            mel: [Batch, T, num_mels], torch.tensor
        """
        wav_data = torch.tensor(wav_data)
        mel = _get_melspec(
            y=wav_data,
            n_fft=self.h["n_fft"],
            num_mels=self.h["num_mels"],
            sampling_rate=self.h["sampling_rate"],
            hop_size=self.h["hop_size"],
            win_size=self.h["win_size"],
            fmin=self.h["fmin"],
            fmax=self.h["fmax"],
        )
        return mel.transpose(1, 2)

    def decode_mel(self, mel):
        """
        params:
            mel: [T, num_mels], torch.tensor
        return:
            wav: [1, T], torch.tensor
        """
        mel = mel.transpose(0, 1).unsqueeze(0).to(self.device)
        wav = self.vocoder(mel)
        return wav.squeeze(0)

    def decode_mel_batch(self, mel):
        """
        params:
            mel: [B, T, num_mels], torch.tensor
        return:
            wav: [B, 1, T], torch.tensor
        """
        mel = mel.transpose(1, 2).to(self.device)
        wav = self.vocoder(mel)
        return wav

    @classmethod
    def from_pretrained(cls, model_config, ckpt_path, device):
        with open(model_config) as f:
            kimi_h = json.load(f)

        hf_config = _kimi_to_hf_bigvgan_config(kimi_h)
        vocoder = KimiBigVGAN(hf_config)

        assert os.path.isfile(ckpt_path), ckpt_path
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        kimi_sd = checkpoint["generator"]
        hf_sd = _adapt_kimi_state_dict(kimi_sd)
        missing, unexpected = vocoder.load_state_dict(hf_sd, strict=False)
        # The HF port registers alias-free filter buffers as persistent=False;
        # they're regenerated at construction, so they count as "missing" from
        # a saved-state-dict perspective. Tolerate exactly those.
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
        return cls(vocoder, device, kimi_h)
