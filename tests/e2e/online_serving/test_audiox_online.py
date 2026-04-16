# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving test for AudioX text-to-audio diffusion.

Mirrors `tests/e2e/online_serving/test_sd3_expansion.py` for image diffusion: spin up
`vllm-omni` with `--model-class-name AudioXPipeline`, send an OpenAI chat-completions
request with the AudioX-specific extra_body (`audiox_task=t2a`), and verify that the
response carries a non-empty WAV-encoded audio payload (`message.audio.data`).

`OpenAIClientHandler.send_diffusion_request` only validates image responses
(`assert_audio_diffusion_response` is `NotImplementedError`), so this test calls the
underlying OpenAI client directly to inspect the audio field.
"""

import base64
import os
import wave
from io import BytesIO

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.utils import hardware_marks

# Tiny / random checkpoint usable in CI; override to a real bundle locally.
AUDIOX_TEST_MODEL = os.environ.get("AUDIOX_TEST_MODEL", "linyueqian/audiox_random")
T2A_PROMPT = "A quiet living room with soft fabric rustle and gentle cat breathing."

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"})


def _audiox_server_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--model-class-name", "AudioXPipeline"],
            ),
            id="t2a",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _audiox_server_cases(AUDIOX_TEST_MODEL), indirect=True)
def test_audiox_t2a_online(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """AudioX text-to-audio: chat completion returns a non-empty WAV in `message.audio.data`."""
    messages = dummy_messages_from_mix_data(content_text=T2A_PROMPT)
    completion = openai_client.client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        extra_body={
            "num_inference_steps": 4,
            "guidance_scale": 6.0,
            "seed": 42,
            "audiox_task": "t2a",
            "seconds_start": 0.0,
            "seconds_total": 2.0,
        },
    )

    assert completion.choices, "No choices in completion response"
    audio = getattr(completion.choices[0].message, "audio", None)
    assert audio is not None, "Response message has no `audio` field (AudioX should emit audio)"
    assert audio.data, "Response audio payload is empty"

    wav_bytes = base64.b64decode(audio.data)
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnframes() > 0, "Decoded WAV has zero frames"
        assert wav_file.getframerate() > 0, "Decoded WAV has invalid sample rate"
