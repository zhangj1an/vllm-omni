# Kimi-Audio-7B for ASR and audio chat

## Summary

- Vendor: moonshotai
- Model: `moonshotai/Kimi-Audio-7B-Instruct`
- Task: ASR (audio in, text out) and single-turn audio chat (audio in, text + spoken audio out)
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`moonshotai/Kimi-Audio-7B-Instruct` with vLLM-Omni in one of two modes:

- **ASR / audio-to-text** — transcribe an audio clip with `"modalities": ["text"]`.
- **Audio chat** — respond to an audio prompt with text and synthesized
  speech using `"modalities": ["text", "audio"]`.

The pipeline runs the fused thinker (Whisper-large-v3 + VQ-Adaptor +
Qwen2-7B + 6-layer MIMO branch) on stage 0 and a flow-matching DiT +
BigVGAN code2wav module on stage 1 that turns MIMO-emitted semantic
codes into a 24 kHz waveform.

## References

- Upstream model: [`MoonshotAI/Kimi-Audio`](https://github.com/MoonshotAI/Kimi-Audio)
- For offline inference (including the multi-turn case that requires a
  custom `prompt_token_ids` builder) and a Python streaming client, see
  `examples/offline_inference/kimi_audio/` and
  `examples/online_serving/kimi_audio/`.

## Hardware Support

This recipe documents reference GPU configurations for the two-stage
audio-out deployment. Other hardware and configurations are welcome
as community validation lands.

## GPU

### 2x H100 80GB — async-chunked audio chat

The bundled `kimi_audio.yaml` runs the fused thinker on `cuda:0` and
the code2wav stage on `cuda:1`, connected by `SharedMemoryConnector`
for sub-second TTFB. Adjust `devices` in the YAML to match your
hardware.

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA driver / runtime: 12.4+
- vLLM: 0.20.0
- vLLM-Omni: 0.20.0rc1 or built from this checkout (`pip install -e .`)

#### Command

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct \
    --omni \
    --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/kimi_audio.yaml
```

For sync (non-streaming) operation, edit the YAML to set
`async_chunk: false`. For text-out-only deployment, point at
`kimi_audio_asr_single_gpu.yaml` and request `"modalities": ["text"]`
only.

#### Verification

ASR — transcribe a clip:

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "moonshotai/Kimi-Audio-7B-Instruct",
      "modalities": ["text"],
      "sampling_params_list": [
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 512, "seed": 42}
      ],
      "messages": [
        {"role": "user", "content": [
          {"type": "audio_url", "audio_url": {"url": "https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/test_audios/asr_example.wav"}},
          {"type": "text", "text": "请将音频内容转换为文字。"}
        ]}
      ]
    }' | jq -r '.choices[0].message.content'
```

Audio chat — spoken response (saves to `kimi_response.wav`):

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "moonshotai/Kimi-Audio-7B-Instruct",
      "modalities": ["text", "audio"],
      "sampling_params_list": [
        {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "max_tokens": 2048, "seed": 42},
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 16384, "seed": 42, "detokenize": false}
      ],
      "messages": [
        {"role": "user", "content": [
          {"type": "audio_url", "audio_url": {"url": "https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/test_audios/qa_example.wav"}}
        ]}
      ]
    }' | jq -r '.choices[0].message.audio.data' | base64 -d > kimi_response.wav
```

The reply should be ~5 s of clean speech (rms ≈ 0.045, peak ≈ 0.35).

Spoken response from text input ("text-to-audio" mode):

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "moonshotai/Kimi-Audio-7B-Instruct",
      "modalities": ["text", "audio"],
      "sampling_params_list": [
        {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "max_tokens": 2048, "seed": 42},
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 16384, "seed": 42, "detokenize": false}
      ],
      "messages": [
        {"role": "user", "content": [
          {"type": "text", "text": "Please say the following in audio: \"Hello, my name is Kimi.\""}
        ]}
      ]
    }' | jq -r '.choices[0].message.audio.data' | base64 -d > kimi_tts.wav
```

#### Notes

- **Two stage YAMLs ship**:
  - `kimi_audio.yaml` — async-chunked, 2-GPU (default)
  - `kimi_audio_single_gpu.yaml` — sync, both stages on `cuda:0`
  - `kimi_audio_asr_single_gpu.yaml` — text-out only, single stage
- **`max_tokens=2048`** on the thinker is the right setting for
  audio-out tasks — the audio head needs room to reach its natural
  EOD (`<|im_msg_end|>` / `<|im_media_end|>`). Smaller caps truncate
  the speech mid-sentence.
- **Audio sampling caveat**: the MIMO audio head samples via greedy
  argmax inside the model and does not flow through vLLM's sampler.
  The thinker's `temperature` / `top_p` / `top_k` only affect the
  text path; audio tokens are deterministic given the text prefix.
  Changing audio quality requires editing
  `_run_mimo_branch` in `kimi_audio_thinker.py`.
- **Multi-turn audio history** (where a prior assistant turn includes
  spoken audio) is not yet wired through the OpenAI chat-completions
  path. For multi-turn, use the offline Python entry point under
  `examples/offline_inference/kimi_audio/` (see `--task multiturn`),
  which builds the prompt manually with GLM-4-Voice tokenization of
  prior assistant audio.
