"""Offline inference example for MOSS-TTS variants.

Covers all five models:
  - OpenMOSS-Team/MOSS-TTS           (8B, general TTS)
  - OpenMOSS-Team/MOSS-TTSD-v1.0     (8B, dialogue TTS)
  - OpenMOSS-Team/MOSS-SoundEffect   (8B, sound effect synthesis)
  - OpenMOSS-Team/MOSS-TTS-Realtime  (1.7B, low-latency streaming)
  - OpenMOSS-Team/MOSS-VoiceGenerator(1.7B, voice design)

Usage
-----
# Standard TTS with voice cloning (MOSS-TTS, MOSS-TTSD, MOSS-TTS-Realtime):
python end2end.py \\
    --model OpenMOSS-Team/MOSS-TTS \\
    --text "Hello, this is a MOSS-TTS test." \\
    --ref-audio path/to/reference.wav \\
    --output output.wav

# Sound effect generation (no ref audio):
python end2end.py \\
    --model OpenMOSS-Team/MOSS-SoundEffect \\
    --ambient-sound "Thunder rumbling, rain pattering." \\
    --output thunder.wav

# Voice design (no ref audio):
python end2end.py \\
    --model OpenMOSS-Team/MOSS-VoiceGenerator \\
    --text "Hello, I am a generated voice." \\
    --output voice.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch


def _load_ref_audio(path: str, target_sr: int = 24000) -> list[torch.Tensor]:
    """Load and resample reference audio to target sample rate."""
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return [wav.squeeze(0)]


def run_tts(args: argparse.Namespace) -> None:
    from vllm_omni import AsyncOmniEngine  # type: ignore[import]
    from vllm_omni.config import OmniEngineArgs  # type: ignore[import]

    deploy_map = {
        "OpenMOSS-Team/MOSS-TTS": "moss_tts",
        "OpenMOSS-Team/MOSS-TTSD-v1.0": "moss_ttsd",
        "OpenMOSS-Team/MOSS-SoundEffect": "moss_sound_effect",
        "OpenMOSS-Team/MOSS-TTS-Realtime": "moss_tts_realtime",
        "OpenMOSS-Team/MOSS-VoiceGenerator": "moss_voice_generator",
    }
    deploy_config = deploy_map.get(args.model, "moss_tts")

    engine_args = OmniEngineArgs(
        model=args.model,
        stage_config=deploy_config,
    )
    engine = AsyncOmniEngine.from_engine_args(engine_args)

    is_sound_effect = "SoundEffect" in args.model
    is_voice_gen = "VoiceGenerator" in args.model

    if is_sound_effect:
        if not args.ambient_sound:
            sys.exit("MOSS-SoundEffect requires --ambient-sound")
        additional_info = {
            "task_type": ["sound_effect"],
            "ambient_sound": [args.ambient_sound],
        }
        if args.duration_tokens:
            additional_info["tokens"] = [args.duration_tokens]
    elif is_voice_gen:
        additional_info = {
            "task_type": ["voice_generator"],
            "text": [args.text or ""],
        }
    else:
        if not args.ref_audio:
            sys.exit("Voice cloning requires --ref-audio")
        ref_wavs = _load_ref_audio(args.ref_audio)
        additional_info = {
            "task_type": ["voice_clone"],
            "text": [args.text or ""],
            "prompt_audio_array": ref_wavs,
            "mode": ["voice_clone"],
        }

    import asyncio

    async def _generate() -> None:
        outputs = await engine.generate(
            inputs={"prompt_token_ids": [1], "additional_information": additional_info},
            sampling_params=None,
            request_id="moss-tts-0",
        )
        mm = outputs.outputs[0].multimodal_output or {}
        wav = mm.get("audio") or mm.get("model_outputs")
        if wav is None:
            print("No audio in output — check model loading logs.")
            return
        if isinstance(wav, list):
            import torch as _torch
            wav = _torch.cat([w.reshape(-1) for w in wav])
        wav_np = wav.float().cpu().numpy()
        out_path = args.output
        sf.write(out_path, wav_np, 24000)
        print(f"Saved {len(wav_np)/24000:.2f}s of audio to {out_path}")

    asyncio.run(_generate())


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
