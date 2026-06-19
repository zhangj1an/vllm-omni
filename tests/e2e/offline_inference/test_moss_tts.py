# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for the MOSS-TTS two-stage pipeline (delay model).

Tests the MossTTSDelayModel path using MOSS-VoiceGenerator (1.7B) — the
smallest MossTTSDelayModel variant — so the test runs on a single A10G or L4
without requiring an 80 GB GPU.

The MOSS-TTS-Realtime variant lives in the sibling file
test_moss_tts_realtime.py.  Use a single module-scoped engine per file; do NOT
add a second engine fixture here — that triggers mid-module teardown races
(see skill invariant I4).
"""

from __future__ import annotations

import os

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni import Omni

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Selected by the nightly "TTS · Function Test" (-m "full_model and L4 and tts").
# Without the tts + run-level marks these tests are collected but never run.
pytestmark = [pytest.mark.full_model, pytest.mark.tts]

SAMPLE_RATE = 24_000  # All MOSS-TTS full variants output 24 kHz

# Voice description for the MOSS-VoiceGenerator (``voice_generator`` variant):
# it conditions on text (what to say) + instruction (how the voice sounds), not
# a reference clip.
_DELAY_INSTRUCTION = "A clear, natural speaking voice."

_DEFAULT_SAMPLING = SamplingParams(
    temperature=1.7,
    top_p=0.8,
    top_k=25,
    max_tokens=512,  # short for tests (~20 s at 24 kHz, ~12.5 tokens/s)
    seed=42,
    detokenize=False,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_from_mm(mm: dict | None) -> torch.Tensor | None:
    """Return a 1-D CPU waveform from one output's multimodal_output, or None.

    Audio arrives under ``audio`` or (pre-consolidation) ``model_outputs``, as a
    tensor or a list of per-request tensors. Never use ``a or b`` on tensor
    values — a multi-element tensor raises on truthiness; check ``is None``.
    """
    if not mm:
        return None
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if audio is None:
        return None
    if isinstance(audio, list):
        parts = [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0]
        if not parts:
            return None
        audio = torch.cat(parts, dim=0)
    if not isinstance(audio, torch.Tensor):
        return None
    return audio.reshape(-1).cpu()


def _sr_from_mm(mm: dict | None) -> int | None:
    """Return the integer sample rate from a multimodal_output, or None."""
    if not mm:
        return None
    sr = mm.get("sr")
    if isinstance(sr, list):
        sr = sr[0] if sr else None
    if sr is None:
        return None
    return int(sr.item()) if isinstance(sr, torch.Tensor) else int(sr)


def _collect_audio(omni: Omni, request: dict, sampling: list | None = None) -> tuple[torch.Tensor, int]:
    """Run one request; return (waveform_cpu, sample_rate).

    ``generate()`` returns a flat list of ``OmniRequestOutput`` (one or more
    per request as audio streams); each exposes ``.multimodal_output`` directly
    (``.request_output`` is a single inner ``RequestOutput``, not an iterable).
    Concatenate every non-empty audio chunk in arrival order.

    ``sampling`` is the PER-STAGE sampling-params list (length == num_stages);
    when omitted, ``_DEFAULT_SAMPLING`` is replicated across stages.
    """
    chunks: list[torch.Tensor] = []
    sr = SAMPLE_RATE
    # sampling_params_list is PER STAGE (its length must equal num_stages),
    # not per request. MOSS-TTS is a two-stage pipeline (talker + codec), so a
    # single SamplingParams is rejected — replicate it across stages.
    if sampling is None:
        sampling = [_DEFAULT_SAMPLING] * omni.num_stages
    for omni_out in omni.generate(request, sampling):
        mm = omni_out.multimodal_output
        sr_val = _sr_from_mm(mm)
        if sr_val is not None:
            sr = sr_val
        chunk = _audio_from_mm(mm)
        if chunk is not None and chunk.numel() > 0:
            chunks.append(chunk)
    assert chunks, "No audio received across generate() outputs"
    return torch.cat(chunks, dim=0), sr


# ---------------------------------------------------------------------------
# MossTTSDelayModel — MOSS-VoiceGenerator 1.7B
# (tests the delay-pattern generation path on a small GPU)
# ---------------------------------------------------------------------------

_DELAY_MODEL = os.environ.get("MOSS_VOICEGEN_MODEL", "OpenMOSS-Team/MOSS-VoiceGenerator")


@pytest.fixture(scope="module")
def delay_engine():
    # The two stages (talker + codec) share one GPU, so cap per-stage memory —
    # the default 0.92 lets stage 0 claim the whole card and stage 1 fails its
    # memory check. Disable prefix caching so two identical-seed runs are
    # bit-reproducible (the cached-prefill path can perturb logits enough to
    # shift sampling/EOS) — required by ``test_moss_tts_delay_deterministic``.
    omni = Omni(
        model=_DELAY_MODEL,
        stage_init_timeout=300,
        stage_overrides={
            "0": {"gpu_memory_utilization": 0.4, "enable_prefix_caching": False},
            "1": {"gpu_memory_utilization": 0.4},
        },
    )
    yield omni
    # Tear down the stage engines so their GPU memory is released before the
    # next module-scoped engine starts; otherwise the leaked StageEngineCoreProc
    # processes hold the card and the next engine fails its startup memory check
    # ("StageEngineCoreProc died during READY", issue #4087).
    omni.shutdown()


@pytest.fixture(scope="module")
def delay_processor():
    """Upstream MossTTSProcessor used to build conditioned prompts.

    The delay talker conditions on ``prompt_token_ids`` (tokenised text) and
    ``codes.ref`` (the unified audio-code grid) — NOT a bare assistant prompt
    with the text stuffed in ``additional_information`` (nothing tokenises that
    offline, so the talker would ignore it and every request would produce the
    same audio). Mirror the online speech server: run the processor.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(_DELAY_MODEL, trust_remote_code=True)


def _build_voicegen_request(processor, text: str, instruction: str = _DELAY_INSTRUCTION, seed: int = 42) -> dict:
    """Build a MOSS-VoiceGenerator request the way ``serving_speech`` does.

    ``build_user_message`` + ``processor(..., mode="generation")`` yield a
    unified ``(L, 1+n_vq)`` grid; column 0 is the text/special token stream and
    columns ``1..n_vq`` are the delay-pattern audio-code grid. We forward column
    0 as ``prompt_token_ids`` and the rest as ``codes.ref`` — so different text
    produces different prompts (hence different audio), which is what makes the
    batch regression test a genuine guard rather than an RNG-noise artifact.
    """
    msg = processor.build_user_message(text=text, instruction=instruction)
    batch = processor(conversations=[[msg]], mode="generation")
    unified = batch["input_ids"][0]  # (L, 1 + n_vq)
    return {
        "prompt_token_ids": unified[:, 0].tolist(),
        "additional_information": {
            "codes": {"ref": unified[:, 1:].contiguous().to(torch.int64)},
            "seed": [seed],
        },
    }


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_english(delay_engine, delay_processor):
    """MossTTSDelayModel: English text produces non-empty 24 kHz audio."""
    req = _build_voicegen_request(delay_processor, "Hello, this is a MOSS-TTS generation test.")
    audio, sr = _collect_audio(delay_engine, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_chinese(delay_engine, delay_processor):
    """MossTTSDelayModel: Chinese input produces non-empty audio."""
    req = _build_voicegen_request(delay_processor, "你好，这是语音合成测试。")
    audio, sr = _collect_audio(delay_engine, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_deterministic(delay_engine, delay_processor):
    """MossTTSDelayModel: same seed yields identical waveforms.

    Audio codes are sampled with ``torch.multinomial`` outside vLLM's seeded
    sampler; the talker re-seeds a per-(seed, frame) generator from the request
    seed so two same-seed runs reproduce exactly (without this they diverge).
    """
    # Build a FRESH request per run: the engine consumes ``codes.ref`` /
    # ``audio_state`` from ``additional_information`` in place, so reusing one
    # dict would let the first run's mutations leak into the second.
    audio1, _ = _collect_audio(
        delay_engine, _build_voicegen_request(delay_processor, "Reproducibility check.", seed=99)
    )
    audio2, _ = _collect_audio(
        delay_engine, _build_voicegen_request(delay_processor, "Reproducibility check.", seed=99)
    )

    assert audio1.shape == audio2.shape, "Shapes differ across identical seeds"
    assert torch.allclose(audio1, audio2, atol=1e-4), "Waveforms differ across identical seeds"


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_batch(delay_engine, delay_processor):
    """MossTTSDelayModel: two concurrent requests with different text return
    non-empty AND distinct audio.

    This is the regression guard for the batch concurrency bugs: when audio
    codes are broadcast (``audio_codes_list[0]`` for every request) or hidden
    states are sliced with a uniform ``rows_per_req`` division, request 1
    receives request 0's audio. Asserting the waveforms differ catches that.

    The requests must be built through the processor (``_build_voicegen_request``)
    so the differing text actually reaches the talker as ``prompt_token_ids`` —
    a bare assistant prompt would leave every request identical and the bug
    invisible. Audio sampling is seeded (deterministic), so the distinctness
    here reflects genuine per-request content, not RNG noise.

    Coverage limitation — this does NOT exercise the mixed prefill+decode case.
    Both requests are submitted together via ``generate(requests, ...)``, so the
    scheduler prefills then decodes them in lockstep: every step is either
    all-prefill or all-decode, and each request contributes the same number of
    rows. The uniform ``rows_per_req`` fallback in ``_request_row_spans`` happens
    to be exact there, so it would still pass even if ``query_start_loc`` recovery
    were broken. The genuinely dangerous case — request A in decode (1 row) while
    request B prefills (N rows) in the SAME batch, where a uniform split
    misattributes A's hidden states — requires staggered submission (start A,
    let it reach decode, then submit B). The synchronous ``Omni.generate()`` batch
    API cannot inject a request into an in-flight batch, so that path is covered
    only indirectly by the ``query_start_loc`` recovery in ``_request_row_spans``
    and its unit-level reasoning, not by this end-to-end test.
    """
    requests = [
        _build_voicegen_request(delay_processor, "First sentence."),
        _build_voicegen_request(
            delay_processor, "A completely different second utterance, much longer than the first."
        ),
    ]
    # generate() returns a flat list of OmniRequestOutput, one or more per
    # request as audio streams. Each carries .request_id ("<idx>_<uuid>") and
    # exposes .multimodal_output directly. Group audio chunks per request index
    # so a per-request mix-up (the broadcast / row-misalignment bugs) is
    # observable as identical waveforms across the two requests.
    # sampling_params_list is PER STAGE (length == num_stages), independent of
    # the number of requests; the two prompts above are the per-request batch.
    sampling = [_DEFAULT_SAMPLING] * delay_engine.num_stages
    chunks_by_req: dict[int, list[torch.Tensor]] = {}
    for omni_out in delay_engine.generate(requests, sampling):
        idx = int(omni_out.request_id.split("_", 1)[0])
        chunk = _audio_from_mm(omni_out.multimodal_output)
        if chunk is not None and chunk.numel() > 0:
            chunks_by_req.setdefault(idx, []).append(chunk)

    assert set(chunks_by_req) == {0, 1}, f"Expected audio for requests 0 and 1, got {sorted(chunks_by_req)}"
    results = [torch.cat(chunks_by_req[i], dim=0) for i in (0, 1)]
    for i, audio in enumerate(results):
        assert audio.numel() > 0, f"Audio[{i}] is empty"

    # Different text must not yield identical audio. Identical output here is
    # the signature of the broadcast / row-misalignment concurrency bugs.
    a, b = results
    if a.numel() == b.numel():
        assert not torch.equal(a, b), "Both requests returned identical audio (batch concurrency bug)"
