"""Offline inference example for MOSS-TTS variants.

Covers all five models:
  - OpenMOSS-Team/MOSS-TTS           (8B, general TTS)
  - OpenMOSS-Team/MOSS-TTSD-v1.0     (8B, dialogue TTS)
  - OpenMOSS-Team/MOSS-SoundEffect   (8B, sound effect synthesis)
  - OpenMOSS-Team/MOSS-TTS-Realtime  (1.7B, low-latency streaming)
  - OpenMOSS-Team/MOSS-VoiceGenerator(1.7B, voice design)

Usage
-----
# Standard TTS with voice cloning:
python end2end.py \\
    --model OpenMOSS-Team/MOSS-TTS \\
    --text "Hello, this is a MOSS-TTS test." \\
    --ref-audio path/to/reference.wav \\
    --output output.wav
"""

from __future__ import annotations

import argparse
import gc
import sys

import soundfile as sf
import torch


def _load_ref_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load and resample reference audio to (1, T) at target_sr."""
    data, sr = sf.read(path, always_2d=True)  # (T, C)
    wav = torch.from_numpy(data.T.astype("float32"))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav  # (1, T)


def _build_unified_codes(
    model_id: str,
    text: str,
    ref_audio: torch.Tensor | None,
    *,
    instruction: str | None = None,
    tokens: int | None = None,
    quality: str | None = None,
    sound_event: str | None = None,
    ambient_sound: str | None = None,
    language: str | None = None,
) -> tuple[list[int], torch.Tensor, int]:
    """Build the upstream MossTTSDelay unified-codes prompt.

    Returns (text_token_ids, audio_codes_LxNQ, n_vq) where
    ``text_token_ids`` is the prefill prompt for the talker and
    ``audio_codes_LxNQ`` is the matching delay-pattern audio code grid
    (``audio_pad_code`` everywhere except inside the reference-audio block).
    """
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    reference: list[torch.Tensor] | None = None
    if ref_audio is not None:
        # Pre-encode so the processor doesn't need torchaudio.load (which
        # now requires torchcodec on recent torchaudio releases).
        # encode_audios_from_wav expects (C, T) per item.
        wav_2d = ref_audio if ref_audio.dim() == 2 else ref_audio.unsqueeze(0)
        codes_list = proc.encode_audios_from_wav(
            [wav_2d], sampling_rate=24000, n_vq=proc.model_config.n_vq
        )
        reference = [codes_list[0]]

    user_msg_kwargs: dict = {"text": text}
    if reference is not None:
        user_msg_kwargs["reference"] = reference
    if instruction is not None:
        user_msg_kwargs["instruction"] = instruction
    if tokens is not None:
        user_msg_kwargs["tokens"] = tokens
    if quality is not None:
        user_msg_kwargs["quality"] = quality
    if sound_event is not None:
        user_msg_kwargs["sound_event"] = sound_event
    if ambient_sound is not None:
        user_msg_kwargs["ambient_sound"] = ambient_sound
    if language is not None:
        user_msg_kwargs["language"] = language

    user_msg = proc.build_user_message(**user_msg_kwargs)
    batch = proc(conversations=[[user_msg]], mode="generation")
    unified = batch["input_ids"][0]  # (L, 1+n_vq)
    n_vq = int(unified.shape[-1] - 1)
    text_ids = unified[:, 0].tolist()
    audio_codes = unified[:, 1:].contiguous().to(torch.int64)  # (L, n_vq)

    # Free the processor (incl. ~1.6B-param audio_tokenizer on CPU) before
    # we instantiate Omni — we only needed it for the one-shot encode.
    del proc
    gc.collect()

    return text_ids, audio_codes, n_vq


def run_tts(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    from vllm_omni import Omni

    deploy_map = {
        "OpenMOSS-Team/MOSS-TTS": "moss_tts",
        "OpenMOSS-Team/MOSS-TTSD-v1.0": "moss_ttsd",
        "OpenMOSS-Team/MOSS-SoundEffect": "moss_sound_effect",
        "OpenMOSS-Team/MOSS-TTS-Realtime": "moss_tts_realtime",
        "OpenMOSS-Team/MOSS-VoiceGenerator": "moss_voice_generator",
    }
    deploy_config = deploy_map.get(args.model, "moss_tts")

    is_sound_effect = "SoundEffect" in args.model
    is_voice_gen = "VoiceGenerator" in args.model

    ref_audio = None
    builder_kwargs: dict = {}

    if is_sound_effect:
        if not args.ambient_sound:
            sys.exit("MOSS-SoundEffect requires --ambient-sound")
        builder_kwargs["ambient_sound"] = args.ambient_sound
        if args.duration_tokens:
            builder_kwargs["tokens"] = args.duration_tokens
        text = args.text or ""
    elif is_voice_gen:
        text = args.text or ""
    else:
        if not args.ref_audio:
            sys.exit("Voice cloning requires --ref-audio")
        ref_audio = _load_ref_audio(args.ref_audio)
        text = args.text or ""

    text_ids, audio_codes, n_vq = _build_unified_codes(
        args.model, text, ref_audio, **builder_kwargs
    )
    print(f"Prefill prompt: {len(text_ids)} tokens, {n_vq}-quantizer audio block")

    omni = Omni(model=args.model, deploy_config=deploy_config, stage_init_timeout=600)

    talker_sp = SamplingParams(
        temperature=1.7,
        top_p=0.8,
        top_k=25,
        max_tokens=2048,
        seed=42,
        detokenize=False,
    )
    codec_sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=65536,
        seed=42,
        detokenize=True,
    )

    request = {
        "prompt_token_ids": text_ids,
        "additional_information": {
            # Reference-audio codes consumed by the talker prefill (additive
            # embedding); shape is (L, n_vq) where pad_code marks non-audio
            # positions. The OmniPayload schema reserves ``codes.ref`` for
            # exactly this kind of conditioning input.
            "codes": {"ref": audio_codes},
        },
    }

    chunks: list[torch.Tensor] = []
    for stage_outputs in omni.generate(request, [talker_sp, codec_sp]):
        # Each yielded ``stage_outputs`` is an OmniRequestOutput; the codec
        # stage attaches its waveform under ``multimodal_output["model_outputs"]``.
        # Avoid ``a or b`` when ``a`` may be a tensor (truthiness raises).
        mm = stage_outputs.multimodal_output or {}
        wav = mm.get("audio")
        if wav is None:
            wav = mm.get("model_outputs")
        if wav is None:
            continue
        if isinstance(wav, list):
            chunks.extend(w.reshape(-1) for w in wav if isinstance(w, torch.Tensor) and w.numel() > 0)
        elif isinstance(wav, torch.Tensor) and wav.numel() > 0:
            chunks.append(wav.reshape(-1))

    if not chunks:
        print("No audio in output — check model loading logs.")
        return

    wav_t = torch.cat(chunks).float().cpu()
    sf.write(args.output, wav_t.numpy(), 24000)
    print(f"Saved {len(wav_t)/24000:.2f}s of audio to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSS-TTS offline inference example")
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS",
        help="HuggingFace model ID",
    )
    parser.add_argument("--text", default="", help="Text to synthesise (TTS variants)")
    parser.add_argument("--ref-audio", default=None, help="Reference audio for voice cloning")
    parser.add_argument(
        "--ambient-sound",
        default=None,
        help="Sound description for MOSS-SoundEffect",
    )
    parser.add_argument(
        "--duration-tokens",
        type=int,
        default=None,
        help="Target duration in tokens (1 s ≈ 12.5 tokens) for MOSS-SoundEffect",
    )
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
