# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Kimi-Audio-7B-Instruct covering three
task modes: ``audio2text`` (ASR / audio QA), ``audio2audio``, and
``text2audio``. All three use ``vllm_omni/deploy/kimi_audio.yaml``
(two-stage audio-out pipeline); ``audio2text`` ignores the stage-1
audio output. For low-TTFB streaming see ``end2end_async_chunk.py``."""

import os
from typing import NamedTuple

import soundfile as sf
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media import MediaConnector
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42

# Mirrors the AUDIO_PLACEHOLDER token sequence Kimi's chat template expects.
AUDIO_PLACEHOLDER = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
OUTPUT_SAMPLE_RATE = 24000

TASK_CHOICES = ("audio2text", "audio2audio", "text2audio")

# Default sample audio for the audio2text task. Originally taken from the
# MiniMax TTS-Multilingual test set (sample 10), mirrored to Google Drive
# for a stable link. The original share URL is
# https://drive.google.com/file/d/1RHz6uUSbAR_N3Li1Bjh8dPknykw4IVio/view?usp=sharing;
# we rewrite to the direct-download form so the fetcher receives bytes
# instead of the HTML preview page.
AUDIO2TEXT_DEFAULT_URL = "https://drive.google.com/uc?export=download&id=1RHz6uUSbAR_N3Li1Bjh8dPknykw4IVio"

TASK_DEFAULTS = {
    "audio2text": {
        "stage_configs_path": "../../../vllm_omni/deploy/kimi_audio.yaml",
        "question": "Please transcribe the audio.",
        "output_dir": "./output_text",
        "default_audio_url": AUDIO2TEXT_DEFAULT_URL,
    },
    "audio2audio": {
        "stage_configs_path": "../../../vllm_omni/deploy/kimi_audio.yaml",
        "question": "Answer in audio. Briefly summarize what was said.",
        "output_dir": "./output_audio",
        "default_audio_url": None,
    },
    "text2audio": {
        "stage_configs_path": "../../../vllm_omni/deploy/kimi_audio.yaml",
        "question": 'Please say the following in audio: "Hello, my name is Kimi."',
        "output_dir": "./output_tts",
        "default_audio_url": None,
    },
}


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def _build_prompt(question: str, with_audio: bool, output_type: str = "text") -> str:
    """Wrap a single user turn in the Kimi-Audio chat template.

    ``with_audio=True`` inserts the audio placeholder before the text
    instruction (used for tasks that take audio input). ``False`` builds
    a text-only user turn (used for ``text2audio``).

    For ``output_type="both"`` on a TEXT user message the reference
    tokenize_message does NOT append ``kimia_speech_ctd_id`` — that
    token is only emitted for ``message_type="audio"`` user turns.
    Verified against dump_reference_paired_streams.json.
    """
    del output_type
    placeholder = AUDIO_PLACEHOLDER if with_audio else ""
    return f"<|im_kimia_user_msg_start|>{placeholder}{question}<|im_msg_end|><|im_kimia_assistant_msg_start|>"


def get_audio_input_query(
    question: str,
    audio_path: str | None,
    sampling_rate: int,
    default_audio_url: str | None,
    output_type: str = "text",
) -> QueryResult:
    audio_data = _load_audio_or_default(audio_path, sampling_rate, default_audio_url)
    return QueryResult(
        inputs={
            "prompt": _build_prompt(question, with_audio=True, output_type=output_type),
            "multi_modal_data": {"audio": [audio_data]},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_text_input_query(
    question: str,
    audio_path: str | None,
    sampling_rate: int,
    default_audio_url: str | None,
    output_type: str = "text",
) -> QueryResult:
    # No audio input; the placeholder + multi_modal_data are omitted.
    del audio_path, sampling_rate, default_audio_url
    return QueryResult(
        inputs={"prompt": _build_prompt(question, with_audio=False, output_type=output_type)},
        limit_mm_per_prompt={},
    )


def _load_audio_or_default(audio_path: str | None, sampling_rate: int, default_audio_url: str | None):
    """Resolve a local path or remote URL through vLLM's MediaConnector; resample to ``sampling_rate``."""
    source = audio_path or default_audio_url
    if source is None:
        return AudioAsset("mary_had_lamb").audio_and_sample_rate[0]

    connector = MediaConnector(allowed_local_media_path="/")
    audio, src_sr = connector.fetch_audio(source)
    if int(src_sr) != sampling_rate:
        from vllm.multimodal.audio import resample_audio_scipy

        audio = resample_audio_scipy(audio.astype("float32"), orig_sr=int(src_sr), target_sr=sampling_rate)
    return audio


query_map = {
    "audio2text": get_audio_input_query,
    "audio2audio": get_audio_input_query,
    "text2audio": get_text_input_query,
}


def build_sampling_params(task: str, max_tokens: int) -> list[SamplingParams]:
    if task == "audio2text":
        return [
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=max_tokens,
                seed=SEED,
                detokenize=True,
                repetition_penalty=1.0,
            )
        ]

    # Two-stage audio-out pipelines (audio2audio, text2audio) need one
    # SamplingParams per stage.
    # Text head mirrors Kimi-Audio upstream defaults (text_temperature=0.0,
    # text_top_k=5). The model's config.json uses the non-standard plural
    # `eos_token_ids: [151644, 151645]` so vLLM parses `eos_token_id=None`
    # and has no stop token by default. For output_type="both" the upstream
    # loop terminates only on audio-head EOD (not text-stream EOS) — so we
    # stop on [151644, 151645] (which compute_logits boosts once the MIMO
    # head samples msg_end/media_end) and deliberately NOT on 151667, which
    # fires on the text stream after only ~7 tokens and would cut off the
    # audio stream before it produces real codec tokens.
    thinker = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=5,
        max_tokens=max_tokens,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.0,
        stop_token_ids=[151644, 151645],
    )
    code2wav = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens * 8,
        seed=SEED,
        detokenize=False,
    )
    return [thinker, code2wav]


def write_outputs(omni_outputs, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for stage_outputs in omni_outputs:
        output = stage_outputs.request_output
        request_id = output.request_id

        if stage_outputs.final_output_type == "text":
            text = output.outputs[0].text
            out_txt = os.path.join(output_dir, f"{request_id}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("Prompt:\n")
                f.write(str(output.prompt) + "\n\n")
                f.write("vllm_text_output:\n")
                f.write(str(text).strip() + "\n")
            print(f"Request {request_id}: text -> {out_txt}")
            print("--- response ---")
            print(text)
            print("----------------")
        elif stage_outputs.final_output_type == "audio":
            audio_data = output.outputs[0].multimodal_output.get("audio")
            if audio_data is None:
                print(f"Request {request_id}: no audio emitted (text-only response)")
                continue
            # In async-chunk streaming mode the audio arrives as a list of
            # per-chunk tensors; concatenate before writing.
            if isinstance(audio_data, list):
                import torch
                audio_tensor = torch.cat(audio_data, dim=-1)
            else:
                audio_tensor = audio_data
            audio_numpy = audio_tensor.float().detach().cpu().numpy().reshape(-1)
            out_wav = os.path.join(output_dir, f"{request_id}.wav")
            sf.write(out_wav, audio_numpy, samplerate=OUTPUT_SAMPLE_RATE, format="WAV")
            print(f"Request {request_id}: audio -> {out_wav}")


def main(args):
    defaults = TASK_DEFAULTS[args.task]
    stage_configs_path = args.stage_configs_path or defaults["stage_configs_path"]
    question = args.question or defaults["question"]
    output_dir = args.output_dir or defaults["output_dir"]

    omni = Omni(
        model=args.model_name,
        stage_configs_path=stage_configs_path,
        log_stats=args.enable_stats,
        log_file=("omni_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    sampling_params = build_sampling_params(args.task, args.max_tokens)
    query_func = query_map[args.task]
    output_type = "both" if args.task in ("audio2audio", "text2audio") else "text"
    query = query_func(
        question,
        args.audio_path,
        args.sampling_rate,
        defaults["default_audio_url"],
        output_type=output_type,
    )
    prompts = [query.inputs for _ in range(args.num_prompts)]
    omni_outputs = omni.generate(prompts, sampling_params)
    write_outputs(omni_outputs, output_dir)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Offline inference for Kimi-Audio-7B-Instruct (audio2text, audio2audio, text2audio).",
    )
    parser.add_argument(
        "--task",
        "-t",
        choices=TASK_CHOICES,
        default="audio2text",
        help="Which Kimi-Audio task to run.",
    )
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
        help="Text instruction. Default depends on --task.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to a local audio file. Ignored for --task text2audio. "
        "Falls back to a bundled asset for audio-input tasks.",
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
        help="Max output tokens for the thinker stage. Audio-out tasks scale this by 8x for the code2wav stage.",
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
        default=None,
        help="Override the per-task default stage config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the per-task default output directory.",
    )
    parser.add_argument("--enable-stats", action="store_true")
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=5000)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
