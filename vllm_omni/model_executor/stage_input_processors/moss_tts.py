# coding=utf-8
# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Stage input processors: MOSS-TTS talker (Stage 0) → codec (Stage 1)."""

from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TokensPrompt as OmniTokensPrompt
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_audio_codes(stage_output: Any) -> torch.Tensor | None:
    """Pull audio codes from a Stage-0 OmniOutput or raw tensor."""
    if stage_output is None:
        return None

    # OmniOutput
    mm = getattr(stage_output, "multimodal_outputs", None)
    if mm is not None:
        codes_dict = mm.get("codes", {})
        if isinstance(codes_dict, dict):
            ac = codes_dict.get("audio")
            if isinstance(ac, torch.Tensor):
                return ac

    return None


# ---------------------------------------------------------------------------
# Non-streaming (sync): called once after Stage 0 finishes
# ---------------------------------------------------------------------------

def talker2codec(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Convert all talker codes to a single Stage-1 token sequence.

    Stage 0 output contains ``codes["audio"]`` shaped ``(T, NQ)`` where T is
    the number of generated audio frames and NQ is n_vq.  We flatten to
    ``[NQ * T]`` as the Stage-1 ``input_ids`` so the codec can reshape back
    to ``(NQ, T)`` for decoding.
    """
    results: list[Any] = []

    for src_idx in engine_input_source:
        if src_idx >= len(stage_list):
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        stage_out = stage_list[src_idx]
        audio_codes = _extract_audio_codes(stage_out)

        if audio_codes is None or audio_codes.numel() == 0:
            logger.warning("talker2codec: no audio codes in stage output %d; emitting silence.", src_idx)
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        # audio_codes: (T, NQ) → flatten to [NQ, T] → list[int]
        t, nq = audio_codes.shape[0], audio_codes.shape[1]
        codes_nq_t = audio_codes.transpose(0, 1).contiguous()  # (NQ, T)
        flat = codes_nq_t.reshape(-1).tolist()

        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data={"codes": {"audio": codes_nq_t}},
            )
        )

    return results


# ---------------------------------------------------------------------------
# Streaming (async chunk): called each time Stage 0 emits a new chunk
# ---------------------------------------------------------------------------

def talker2codec_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Emit accumulated audio codes to Stage 1 as they arrive from Stage 0.

    State is maintained in ``transfer_manager`` keyed by request ID.
    A chunk is forwarded to Stage 1 when either:
      (a) ``is_finished`` is True (flush all remaining codes), or
      (b) the accumulated frame count reaches ``chunk_frames`` (default 25).

    Returns a dict compatible with the Stage-1 input format, or None to
    signal "not enough data yet — wait for more frames".
    """
    req_id: str = str(getattr(request, "request_id", id(request)))

    # Initialise per-request accumulation state
    if not hasattr(transfer_manager, "_moss_tts_state"):
        transfer_manager._moss_tts_state = {}
    state = transfer_manager._moss_tts_state

    if req_id not in state:
        state[req_id] = {
            "accumulated": None,   # (T_acc, NQ) tensor or None
            "total_emitted": 0,
        }
    req_state = state[req_id]

    # Extract new codes from this chunk
    if pooling_output is not None:
        mm = pooling_output.get("multimodal_outputs", {}) or {}
        codes_dict = mm.get("codes", {}) or {}
        new_codes = codes_dict.get("audio")
        if isinstance(new_codes, torch.Tensor) and new_codes.numel() > 0:
            if req_state["accumulated"] is None:
                req_state["accumulated"] = new_codes.cpu()
            else:
                req_state["accumulated"] = torch.cat(
                    [req_state["accumulated"], new_codes.cpu()], dim=0
                )

    acc = req_state["accumulated"]
    if acc is None or acc.numel() == 0:
        if is_finished:
            del state[req_id]
        return None

    # Determine chunk threshold (default 25 frames for low latency)
    chunk_frames: int = getattr(transfer_manager, "codec_chunk_frames", 25)
    left_context: int = getattr(transfer_manager, "codec_left_context_frames", 0)

    t_acc = int(acc.shape[0])
    should_emit = is_finished or (t_acc - req_state["total_emitted"] >= chunk_frames)

    if not should_emit:
        return None

    # Determine the slice to emit
    emit_start = max(0, req_state["total_emitted"] - left_context)
    chunk_codes = acc[emit_start:]  # (T_chunk, NQ)
    req_state["total_emitted"] = t_acc

    if is_finished:
        del state[req_id]

    nq = int(chunk_codes.shape[1])
    codes_nq_t = chunk_codes.transpose(0, 1).contiguous()  # (NQ, T_chunk)

    return {
        "codes": {"audio": codes_nq_t},
        "meta": {
            "left_context_size": left_context,
            "finished": is_finished,
        },
    }


__all__ = ["talker2codec", "talker2codec_async_chunk"]
