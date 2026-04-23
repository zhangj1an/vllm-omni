# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming (async-chunk) variant of the Kimi-Audio audio-out pipeline.

Covers the two audio-out task modes:

  - ``audio2audio`` audio in, audio out (spoken response).
  - ``text2audio``  text in, audio out (TTS-style).

The thinker (stage 0) emits audio tokens chunk-by-chunk; ``code2wav``
starts synthesizing waveform as soon as the first chunk lands. End TTFB
is bounded by ``codec_chunk_frames`` from ``vllm_omni/deploy/kimi_audio.yaml``
(25 frames * 480 samples / 24000 Hz ~ 0.5 s of audio per chunk).

For ``audio2text`` (no streaming audio output) see ``end2end.py``.
"""

import os
from typing import NamedTuple

import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42
AUDIO_PLACEHOLDER = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
OUTPUT_SAMPLE_RATE = 24000

TASK_CHOICES = ("audio2audio", "text2audio")

TASK_DEFAULTS = {
    "audio2audio": {
        "question": "Answer in audio. Briefly summarize what was said.",
        "output_dir": "./output_audio_streaming",
    },
    "text2audio": {
        "question": "Please say the following in audio: \"Hello, my name is Kimi.\"",
        "output_dir": "./output_tts_streaming",
    },
}


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def _build_prompt(question: str, with_audio: bool) -> str:
    placeholder = AUDIO_PLACEHOLDER if with_audio else ""
    return (
        f"<|im_kimia_user_msg_start|>{placeholder}{question}"
        f"<|im_msg_end|><|im_kimia_assistant_msg_start|>"
    )


def get_query(task: str, question: str, audio_path: str | None, sr: int) -> QueryResult:
    if task == "text2audio":
        return QueryResult(
            inputs={"prompt": _build_prompt(question, with_audio=False)},
            limit_mm_per_prompt={},
        )

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)
        audio_data = load_audio(audio_path, sr=sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate[0]
    return QueryResult(
        inputs={
            "prompt": _build_prompt(question, with_audio=True),
            "multi_modal_data": {"audio": [audio_data]},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def main(args):
    defaults = TASK_DEFAULTS[args.task]
    question = args.question or defaults["question"]
    output_dir = args.output_dir or defaults["output_dir"]

    omni = Omni(
        model=args.model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.enable_stats,
        log_file=("omni_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )
    thinker_sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=50, max_tokens=args.max_tokens, seed=SEED)
    code2wav_sp = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens * 8, seed=SEED, detokenize=False
    )
    query = get_query(args.task, question, args.audio_path, args.sampling_rate)
    omni_outputs = omni.generate([query.inputs] * args.num_prompts, [thinker_sp, code2wav_sp])

    os.makedirs(output_dir, exist_ok=True)
    for stage_outputs in omni_outputs:
        output = stage_outputs.request_output
        request_id = output.request_id
        if stage_outputs.final_output_type == "audio":
            audio_data = output.outputs[0].multimodal_output.get("audio")
            if audio_data is None:
                continue
            # In streaming mode multimodal_output['audio'] arrives as a list
            # of per-chunk tensors; concatenate before writing.
            if isinstance(audio_data, list):
                audio_tensor = torch.cat(audio_data, dim=-1)
            else:
                audio_tensor = audio_data
            audio_numpy = audio_tensor.float().detach().cpu().numpy().reshape(-1)
            out_wav = os.path.join(output_dir, f"{request_id}.wav")
            sf.write(out_wav, audio_numpy, samplerate=OUTPUT_SAMPLE_RATE, format="WAV")
            print(f"Request {request_id}: streaming audio -> {out_wav}")


def parse_args():
    parser = FlexibleArgumentParser(description="Kimi-Audio streaming audio-out example (audio2audio, text2audio).")
    parser.add_argument(
        "--task",
        "-t",
        choices=TASK_CHOICES,
        default="audio2audio",
        help="Which Kimi-Audio audio-out task to run.",
    )
    parser.add_argument("--model-name", "-m", default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--question", "-q", default=None)
    parser.add_argument(
        "--audio-path",
        "-a",
        default=None,
        help="Path to a local audio file. Ignored for --task text2audio.",
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument(
        "--stage-configs-path",
        default="../../../vllm_omni/deploy/kimi_audio.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--enable-stats", action="store_true")
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=5000)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
