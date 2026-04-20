# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for Kimi-Audio: thinker -> code2wav handoff.

Slice 2 ships ``kimi2code2wav`` (sync, collect-all). Slice 3 will add
``kimi2code2wav_async_chunk`` for streaming TTFB.
"""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

# Mirrors KimiAudioCode2Wav.KIMIA_TOKEN_OFFSET — audio tokens occupy IDs in
# [KIMIA_TOKEN_OFFSET, vocab_size). Anything below is text and is dropped on
# the way to the code2wav stage.
KIMIA_TOKEN_OFFSET = 152064


def _extract_audio_tokens(
    output_token_ids: list[int] | torch.Tensor,
    multimodal_output: dict[str, Any] | None,
) -> list[int]:
    """Pull audio token IDs out of one thinker output.

    Prefers an explicit ``multimodal_output['audio_tokens']`` payload (set by
    :class:`KimiAudioFusedThinker` when audio decoding fires). Falls back to
    filtering the flat token-id stream for IDs >= ``KIMIA_TOKEN_OFFSET`` so
    we still work even if the thinker only emitted them via the text stream
    (single-head fallback path during Slice-2 bring-up).
    """
    if multimodal_output and "audio_tokens" in multimodal_output:
        codes = multimodal_output["audio_tokens"]
        if isinstance(codes, torch.Tensor):
            return codes.reshape(-1).to(torch.long).tolist()
        return list(codes)

    if isinstance(output_token_ids, torch.Tensor):
        flat = output_token_ids.reshape(-1).tolist()
    else:
        flat = list(output_token_ids)
    return [t for t in flat if t >= KIMIA_TOKEN_OFFSET]


def kimi2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync handoff: collect every audio token, hand it to code2wav at once.

    Stage 0 (fused thinker) emits a per-request mixed text/audio token stream
    plus an optional ``multimodal_outputs['audio_tokens']`` tensor. We strip
    the text portion and forward the audio token IDs (still in the model's
    global vocab, the code2wav stage subtracts the offset) as the next
    stage's ``prompt_token_ids``.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    thinker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs: list[OmniTokensPrompt] = []

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        audio_token_ids = _extract_audio_tokens(
            output_token_ids=getattr(output, "token_ids", None) or [],
            multimodal_output=getattr(output, "multimodal_output", None),
        )

        if not audio_token_ids:
            request_id = getattr(thinker_output, "request_id", f"unknown_{i}")
            logger.warning(
                "Skipping request %s: thinker emitted no audio tokens. "
                "Either the user prompt requested text-only or the audio "
                "head fired below threshold. Code2Wav will receive empty.",
                request_id,
            )
            audio_token_ids = []

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs


def kimi2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Streaming handoff: emit per-chunk audio tokens as the thinker decodes.

    Mirrors :func:`vllm_omni.model_executor.stage_input_processors.qwen3_omni
    .talker2code2wav_async_chunk`. Accumulates audio tokens (either from an
    explicit ``multimodal_outputs['audio_tokens']`` slot or by filtering the
    pooling_output's token stream for IDs >= ``KIMIA_TOKEN_OFFSET``) in the
    connector's per-request buffer; flushes a chunk of
    ``codec_chunk_frames`` once enough have arrived, or at request end.

    Returns:
        A dict with ``code_predictor_codes`` (the flat token list for this
        chunk, still in the global vocab), ``left_context_size``, and
        ``finished`` — the code2wav stage consumes these as its prompt.
        Returns ``None`` to signal "wait for more tokens".
    """
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    # Pull new audio tokens out of this step's pooling output.
    audio_tokens_obj = pooling_output.get("audio_tokens") if pooling_output else None
    new_tokens: list[int]
    if audio_tokens_obj is not None:
        if isinstance(audio_tokens_obj, torch.Tensor):
            new_tokens = audio_tokens_obj.reshape(-1).to(torch.long).tolist()
        else:
            new_tokens = [int(t) for t in audio_tokens_obj]
    else:
        # Fallback: scan the full token stream for this step.
        step_token_ids = pooling_output.get("token_ids") or [] if pooling_output else []
        if isinstance(step_token_ids, torch.Tensor):
            step_token_ids = step_token_ids.reshape(-1).tolist()
        new_tokens = [int(t) for t in step_token_ids if int(t) >= KIMIA_TOKEN_OFFSET]

    request_id = request.external_req_id
    buffer = transfer_manager.code_prompt_token_ids[request_id]
    if new_tokens:
        buffer.extend(new_tokens)

    length = len(buffer)
    if length == 0:
        # Nothing to hand off yet. If the request has finished, tell the
        # downstream stage so it can close cleanly; otherwise wait.
        if is_finished:
            return {
                "code_predictor_codes": [],
                "left_context_size": 0,
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return None

    chunk_remainder = length % chunk_size
    if chunk_remainder != 0 and not is_finished:
        return None

    context_length = chunk_remainder if chunk_remainder != 0 else chunk_size
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)
    codes = buffer[-end_index:]

    return {
        "code_predictor_codes": list(codes),
        "left_context_size": left_context_size,
        "codec_chunk_frames": chunk_size,
        "codec_left_context_frames": left_context_size_config,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }
