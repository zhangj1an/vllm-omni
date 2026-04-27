# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Kimi-Audio-7B-Instruct covering the
three upstream demo tasks from MoonshotAI/Kimi-Audio's ``infer.py``:

  * ``asr``       — audio in, text out (ASR over ``asr_example.wav``).
  * ``qa``        — audio in, audio + text out (spoken QA over
                    ``qa_example.wav``, no text instruction in the user
                    turn).
  * ``multiturn`` — multi-turn audio chat (q1 → assistant audio+text a1
                    → q2). NOT YET IMPLEMENTED in vllm-omni; assistant
                    audio history requires the ``_split_prefill`` mask
                    fix in ``kimi_audio_thinker.py`` and a prompt builder
                    that pre-tokenizes prior assistant audio with the
                    GLM-4-Voice tokenizer. See the integration plan.

All tasks load ``vllm_omni/deploy/kimi_audio.yaml`` (two-stage audio-out
pipeline); ``asr`` ignores the stage-1 audio output. For low-TTFB
streaming see ``end2end_async_chunk.py``.
"""

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

TASK_CHOICES = ("asr", "qa", "multiturn")

# Default sample audio for audio-input tasks. Pulled from the upstream
# Kimi-Audio repo's ``test_audios/`` directory so this example mirrors
# the canonical demos in MoonshotAI/Kimi-Audio's ``infer.py``.
_KIMI_TEST_AUDIOS = (
    "https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/test_audios"
)
ASR_DEFAULT_URL = f"{_KIMI_TEST_AUDIOS}/asr_example.wav"
QA_DEFAULT_URL = f"{_KIMI_TEST_AUDIOS}/qa_example.wav"
# Multi-turn case2 mirrors the README quick-start in upstream's repo:
# user q1 → assistant audio+text a1 ("当然可以，这很简单。一二三四五六七八九十。")
# → user q2 → assistant generates next reply.
MULTITURN_CASE = "case2"
MULTITURN_Q1_URL = f"{_KIMI_TEST_AUDIOS}/multiturn/{MULTITURN_CASE}/multiturn_q1.wav"
MULTITURN_A1_URL = f"{_KIMI_TEST_AUDIOS}/multiturn/{MULTITURN_CASE}/multiturn_a1.wav"
MULTITURN_A1_TEXT = "当然可以，这很简单。一二三四五六七八九十。"
MULTITURN_Q2_URL = f"{_KIMI_TEST_AUDIOS}/multiturn/{MULTITURN_CASE}/multiturn_q2.wav"

TASK_DEFAULTS = {
    "asr": {
        # Text-only single-stage YAML: drops stage 1 and disables the
        # MIMO branch on stage 0 (kimia_generate_audio: false). Avoids
        # routing unwanted audio codes through the flow-matching
        # detokenizer, whose ~900-token KV cache budget overflows under
        # vLLM's profile_run.
        "stage_configs_path": "../../../vllm_omni/deploy/kimi_audio_asr_single_gpu.yaml",
        "question": "请将音频内容转换为文字。",
        "output_dir": "./output_asr",
        "default_audio_url": ASR_DEFAULT_URL,
    },
    "qa": {
        "stage_configs_path": "../../../vllm_omni/deploy/kimi_audio.yaml",
        # Empty question -> audio-only user turn, matching upstream
        # ``infer.py`` for ``qa_example.wav`` (no text instruction).
        "question": "",
        "output_dir": "./output_qa",
        "default_audio_url": QA_DEFAULT_URL,
    },
    "multiturn": {
        "stage_configs_path": "../../../vllm_omni/deploy/kimi_audio.yaml",
        "question": "",
        "output_dir": "./output_multiturn",
        "default_audio_url": None,  # uses 3 URLs above, not a single default
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
    "asr": get_audio_input_query,
    "qa": get_audio_input_query,
    # ``multiturn`` is dispatched separately in ``main`` because it
    # needs a multi-message prompt builder, not the single-turn helpers.
}


def build_sampling_params(task: str, max_tokens: int) -> list[SamplingParams]:
    if task == "asr":
        # ASR's YAML keeps stage 1 defined (orchestrator requires two
        # stages) but stage 0 has kimia_generate_audio=false, so stage 1
        # never sees audio codes. Pass a placeholder SamplingParams for
        # stage 1; it will not actually run.
        # stop_token_ids:
        #   151644 - [EOS]
        #   151645 - <|im_msg_end|>
        #   151667 - <|im_kimia_text_eos|>  (text-stream EOS)
        # The pipeline default omits 151667 because for audio-out tasks
        # the text head naturally fires it before the audio stream is
        # done (and that would truncate audio). For text-only ASR we
        # MUST include it — otherwise the text head emits 151667 once
        # transcription finishes, then runs to max_tokens repeating
        # the last text token (since the audio EOD path that normally
        # halts generation never fires when kimia_generate_audio=false).
        thinker = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            seed=SEED,
            detokenize=True,
            repetition_penalty=1.0,
            stop_token_ids=[151644, 151645, 151667],
        )
        code2wav_placeholder = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            seed=SEED,
            detokenize=False,
        )
        return [thinker, code2wav_placeholder]

    # Two-stage audio-out pipelines (qa, multiturn) need one
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
            token_ids = list(output.outputs[0].token_ids)
            out_txt = os.path.join(output_dir, f"{request_id}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("Prompt:\n")
                f.write(str(output.prompt) + "\n\n")
                f.write("vllm_text_output:\n")
                f.write(str(text).strip() + "\n\n")
                f.write(f"token_ids ({len(token_ids)}):\n")
                f.write(", ".join(str(t) for t in token_ids) + "\n")
            print(f"Request {request_id}: text -> {out_txt}")
            print("--- response ---")
            print(text)
            print("----------------")
            print(f"first 30 ids: {token_ids[:30]}")
            print(f"last 30 ids:  {token_ids[-30:]}")
            print(f"total ids: {len(token_ids)}")
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
            # Skip writing empty WAVs (44-byte header only). Happens when
            # the upstream stage produces no audio tokens — e.g. ASR runs
            # with kimia_generate_audio=false where stage 1 still fires
            # but receives an empty token buffer.
            if audio_numpy.size == 0:
                print(f"Request {request_id}: audio output empty (text-only task); skipping wav write")
                continue
            out_wav = os.path.join(output_dir, f"{request_id}.wav")
            sf.write(out_wav, audio_numpy, samplerate=OUTPUT_SAMPLE_RATE, format="WAV")
            print(f"Request {request_id}: audio -> {out_wav}")


def main(args):
    if args.task == "multiturn":
        # Multi-turn requires:
        #   1. _split_prefill in kimi_audio_thinker.py to route IDs
        #      >= KIMIA_TOKEN_OFFSET (152064) to the audio stream.
        #   2. A prompt builder that pre-tokenizes prior assistant audio
        #      with the GLM-4-Voice tokenizer and emits parallel
        #      audio/text streams aligned by kimia_text_audiodelaytokens.
        # Neither is in place yet — see the integration plan.
        raise NotImplementedError(
            "multiturn task is not yet wired through the vllm-omni "
            "Kimi-Audio integration. Pending work: (a) extend "
            "_split_prefill in kimi_audio_thinker.py to route discrete "
            "audio code IDs (>= 152064) to the audio stream, and "
            "(b) add a multi-turn prompt builder that pre-tokenizes "
            "prior assistant audio with GLM-4-Voice. Until then, only "
            "asr and qa run end-to-end."
        )

    defaults = TASK_DEFAULTS[args.task]
    stage_configs_path = args.stage_configs_path or defaults["stage_configs_path"]
    question = args.question if args.question is not None else defaults["question"]
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
    output_type = "both" if args.task == "qa" else "text"
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
        description="Offline inference for Kimi-Audio-7B-Instruct (asr, qa, multiturn).",
    )
    parser.add_argument(
        "--task",
        "-t",
        choices=TASK_CHOICES,
        default="asr",
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
        help="Path to a local audio file. Falls back to the upstream "
        "Kimi-Audio test_audios sample (asr_example.wav for asr, "
        "qa_example.wav for qa).",
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
