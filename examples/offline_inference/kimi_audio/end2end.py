# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Kimi-Audio-7B-Instruct, Slice 1 (text out).

Audio-in, text-out (ASR / audio question-answering). Wraps the user's audio
in Kimi's chat template and runs a single-stage fused-thinker pipeline.

Slice 2 will add ``end2end_audio_out.py`` (chat with audio response) and
``end2end_async_chunk.py`` (streaming variant).
"""

import os
from typing import NamedTuple

from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42

# Kimi-Audio audio placeholder (mirrors upstream vLLM PR 36127's
# AUDIO_PLACEHOLDER / chat template).
AUDIO_PLACEHOLDER = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def _build_prompt(question: str) -> str:
    """Wrap a single audio + text turn in the Kimi-Audio chat template."""
    return f"<|im_kimia_user_msg_start|>{AUDIO_PLACEHOLDER}{question}<|im_msg_end|><|im_kimia_assistant_msg_start|>"


def get_audio_query(
    question: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    if question is None:
        question = "Please transcribe the audio."

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_data = load_audio(audio_path, sr=sampling_rate)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate[0]

    return QueryResult(
        inputs={
            "prompt": _build_prompt(question),
            "multi_modal_data": {"audio": [audio_data]},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def main(args):
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

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.0,
    )

    query = get_audio_query(
        question=args.question,
        audio_path=args.audio_path,
        sampling_rate=args.sampling_rate,
    )
    prompts = [query.inputs for _ in range(args.num_prompts)]

    omni_outputs = omni.generate(prompts, [sampling_params])

    os.makedirs(args.output_dir, exist_ok=True)
    for stage_outputs in omni_outputs:
        output = stage_outputs.request_output
        if stage_outputs.final_output_type != "text":
            continue
        request_id = output.request_id
        text = output.outputs[0].text
        out_txt = os.path.join(args.output_dir, f"{request_id}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("Prompt:\n")
            f.write(str(output.prompt) + "\n\n")
            f.write("vllm_text_output:\n")
            f.write(str(text).strip() + "\n")
        print(f"Request ID: {request_id}, text saved to {out_txt}")
        print("--- response ---")
        print(text)
        print("----------------")


def parse_args():
    parser = FlexibleArgumentParser(description="Offline inference for Kimi-Audio-7B-Instruct (text-out).")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="moonshotai/Kimi-Audio-7B-Instruct",
        help="HuggingFace repo or local path to the Kimi-Audio checkpoint.",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Text instruction accompanying the audio (default: transcribe).",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to a local audio file. Falls back to a bundled asset.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Resample input audio to this rate before tokenization.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max output tokens.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of identical prompts to dispatch (debugging).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default="../../../vllm_omni/model_executor/stage_configs/kimi_audio.yaml",
        help="Path to the stage config YAML.",
    )
    parser.add_argument("--output-dir", default="./output_text")
    parser.add_argument("--enable-stats", action="store_true")
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=5000)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
