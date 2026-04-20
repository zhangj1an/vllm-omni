# Kimi-Audio-7B online serving

vLLM-Omni exposes Kimi-Audio through the OpenAI-compatible
`/v1/chat/completions` endpoint. Both audio-in and audio-out flow over
chat (no `/v1/audio/speech` handler is needed — Kimi is a chat model, not
a TTS service).

## Launch the server

Text-out only (Slice 1 stage config — single-stage, lighter):

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/kimi_audio.yaml
```

Audio-in / audio-out (Slice 2 — two-stage, needs 2 GPUs by default):

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/kimi_audio_audio_out.yaml
```

Streaming audio output (Slice 3 — async-chunk, lower TTFB):

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/kimi_audio_async_chunk.yaml
```

The audio-out configs require `kimia_infer` from
<https://github.com/MoonshotAI/Kimi-Audio> for the flow-matching
detokenizer + BigVGAN vocoder.

## Send a request

### Audio-in / text-out (ASR or audio QA)

```bash
bash run_curl_audio_in_text_out.sh
```

Backend: stage 0 returns `final_output_type: text`, so the response uses
the standard `choices[0].message.content` field.

### Audio-in / audio-out

```bash
bash run_curl_audio_in_audio_out.sh
```

Backend: stage 1 returns `final_output_type: audio`. The response uses
`choices[0].message.audio.data` (base64-encoded WAV at 24 kHz mono).
The `modalities` field on the request must include `"audio"`.

### Streaming audio output

```bash
python client_streaming.py \
    --host localhost --port 8091 \
    --audio-path /path/to/clip.wav
```

The client opens a server-sent-event stream against `/v1/chat/completions`,
collects `content` chunks (each is a base64-encoded audio chunk), and
writes the concatenated waveform to `output.wav`.

## Endpoint summary

| Modalities       | Stage config                      | Final output type |
|------------------|-----------------------------------|-------------------|
| audio in, text out | `kimi_audio.yaml`                 | `text`            |
| audio in, audio out | `kimi_audio_audio_out.yaml`       | `audio`           |
| audio in, streaming audio out | `kimi_audio_async_chunk.yaml` | `audio` (chunked) |
| audio in, audio out (TP=2) | `kimi_audio_tp.yaml`         | `audio`           |

## Multi-GPU (TP)

For tighter memory headroom, shard the fused thinker across two GPUs and
keep the code2wav stage on a third:

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/kimi_audio_tp.yaml
```

The Qwen2 backbone (28 layers) shards via vLLM's standard TP path. The
6-layer MIMO branch is small enough to replicate on every rank — audio
sampling is greedy so ranks stay in sync without extra collectives.
