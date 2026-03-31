"""
Text-to-audio using the upstream AudioX package (github.com/ZeyueT/AudioX), not vLLM-Omni.

Same default prompt and generation settings as ``end2end.py infer`` / ``run_single_inference`` for ``t2a`` (when aligned to that flow).

Uses the same local bundle as the offline example: ``audiox_weights/`` (config + ``model.ckpt`` + ``VAE.ckpt``).

Install (into your env, e.g. ``/root/.venv``)::

    pip install 'git+https://github.com/ZeyueT/AudioX.git'

Run::

    export PYTHONPATH=/path/to/vllm-omni
    python examples/offline_inference/audiox/run_upstream_audiox_t2a.py
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import torch
from einops import rearrange

from audiox.inference.generation import generate_diffusion_cond
from audiox.models.factory import create_model_from_config
from audiox.models.utils import load_ckpt_state_dict
from audiox.training.utils import copy_state_dict

# Match offline AudioX t2a defaults (see end2end.py / AudioXPipeline)
PROMPT = (
    "Fireworks burst twice, followed by a period of silence before a clock begins ticking."
)
SEED = 42
STEPS = 250
CFG_SCALE = 7.0
SECONDS_TOTAL = 10.0
SECONDS_START = 0.0


def _fix_vae_ckpt_paths(obj: object, weights_dir: Path) -> None:
    """config.json references ``./model/VAE.ckpt``; point at the downloaded ``VAE.ckpt`` next to config."""
    vae = weights_dir / "VAE.ckpt"
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "pretransform_ckpt_path" and isinstance(v, str) and "VAE" in v:
                obj[k] = str(vae)
            else:
                _fix_vae_ckpt_paths(v, weights_dir)
    elif isinstance(obj, list):
        for item in obj:
            _fix_vae_ckpt_paths(item, weights_dir)


def main() -> None:
    base = Path(__file__).resolve().parent
    weights_dir = base / "audiox_weights"
    out_dir = base / "output" / "upstream_audiox"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "t2a.wav"

    cfg_path = weights_dir / "config.json"
    ckpt_path = weights_dir / "model.ckpt"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        raise SystemExit(
            f"Missing {cfg_path} or {ckpt_path}. "
            "Download weights first (see examples/offline_inference/audiox/README.md)."
        )

    with open(cfg_path, encoding="utf-8") as f:
        model_config = json.load(f)
    _fix_vae_ckpt_paths(model_config, weights_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(str(ckpt_path)))
    model.to(device).eval().requires_grad_(False)

    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])
    target_fps = int(model_config.get("video_fps", 5))

    video_tensors = torch.zeros(int(target_fps * SECONDS_TOTAL), 3, 224, 224)
    audio_tensor = torch.zeros(2, int(sample_rate * SECONDS_TOTAL))
    sync_features = torch.zeros(1, 240, 768, device=device)

    seconds_input = sample_size / float(sample_rate)

    conditioning = [
        {
            "video_prompt": {
                "video_tensors": video_tensors.unsqueeze(0),
                "video_sync_frames": sync_features,
            },
            "text_prompt": PROMPT,
            "audio_prompt": audio_tensor.unsqueeze(0).to(device),
            "seconds_start": SECONDS_START,
            "seconds_total": seconds_input,
        }
    ]

    output = generate_diffusion_cond(
        model,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        conditioning=conditioning,
        sample_size=sample_size,
        seed=SEED,
        device=device,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.3,
        sigma_max=500,
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )
    # Avoid torchaudio.save (may require torchcodec in newer torchaudio builds).
    pcm = output.numpy()
    if pcm.ndim != 2:
        raise RuntimeError(f"Expected 2D audio, got {pcm.shape}")
    nch, nframes = pcm.shape
    interleaved = pcm.T.reshape(-1)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())
    print(f"Saved {out_wav}")


if __name__ == "__main__":
    main()
