# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Kimi-Audio-7B-Instruct, Slice 2 (audio out).

Audio-in, audio-out (spoken response). Two-stage pipeline:
  Stage 0: fused thinker (Whisper + VQ-Adaptor + Qwen2 + MIMO branch)
  Stage 1: code2wav (PrefixStreamingFlowMatchingDetokenizer + BigVGAN)

Slice 3 will add ``end2end_async_chunk.py`` for streaming audio output
with sub-second TTFB.
"""

import os
from typing import NamedTuple

import soundfile as sf
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42
AUDIO_PLACEHOLDER = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
OUTPUT_SAMPLE_RATE = 24000


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def _build_prompt(question: str) -> str:
    """Wrap a single audio + text turn in the Kimi-Audio chat template,
    asking the assistant to respond in audio."""
    return f"<|im_kimia_user_msg_start|>{AUDIO_PLACEHOLDER}{question}<|im_msg_end|><|im_kimia_assistant_msg_start|>"


def get_audio_query(
    question: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    if question is None:
        question = "Answer in audio. Briefly summarize what was said."

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

    thinker_sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=args.max_tokens,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.0,
    )
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens * 8,
        seed=SEED,
        detokenize=False,
    )

    query = get_audio_query(
        question=args.question,
        audio_path=args.audio_path,
        sampling_rate=args.sampling_rate,
    )
    prompts = [query.inputs for _ in range(args.num_prompts)]
    omni_outputs = omni.generate(
        prompts,
        [thinker_sampling_params, code2wav_sampling_params],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for stage_outputs in omni_outputs:
        output = stage_outputs.request_output
        request_id = output.request_id
        if stage_outputs.final_output_type == "text":
            text = output.outputs[0].text
            out_txt = os.path.join(args.output_dir, f"{request_id}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("Prompt:\n")
                f.write(str(output.prompt) + "\n\n")
                f.write("vllm_text_output:\n")
                f.write(str(text).strip() + "\n")
            print(f"Request {request_id}: text -> {out_txt}")
        elif stage_outputs.final_output_type == "audio":
            audio_tensor = output.outputs[0].multimodal_output.get("audio")
            if audio_tensor is None:
                print(f"Request {request_id}: no audio emitted (text-only response)")
                continue
            audio_numpy = audio_tensor.float().detach().cpu().numpy().reshape(-1)
            out_wav = os.path.join(args.output_dir, f"{request_id}.wav")
            sf.write(out_wav, audio_numpy, samplerate=OUTPUT_SAMPLE_RATE, format="WAV")
            print(f"Request {request_id}: audio -> {out_wav}")


def parse_args():
    parser = FlexibleArgumentParser(description="Offline inference for Kimi-Audio-7B-Instruct (audio out).")
    parser.add_argument("--model-name", "-m", default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--question", "-q", default=None)
    parser.add_argument("--audio-path", "-a", default=None)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument(
        "--stage-configs-path",
        default="../../../vllm_omni/model_executor/stage_configs/kimi_audio_audio_out.yaml",
    )
    parser.add_argument("--output-dir", default="./output_audio")
    parser.add_argument("--enable-stats", action="store_true")
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=5000)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
