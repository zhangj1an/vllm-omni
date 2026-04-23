# Kimi-Audio-7B online serving

vLLM-Omni exposes Kimi-Audio through the OpenAI-compatible
`/v1/chat/completions` endpoint. All three task modes — audio-in /
text-out, audio-in / audio-out, and text-in / audio-out — flow over
chat (no `/v1/audio/speech` handler is needed; Kimi is a chat model,
not a TTS service).

## Launch the server

```bash
# Two-stage audio-out pipeline. Defaults to async-chunk streaming
# (sub-second TTFB) on 2 GPUs; pass --no-async-chunk for the legacy
# single-GPU sync path. Also serves audio->text if the client ignores
# the stage-1 audio output.
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/deploy/kimi_audio.yaml
```

The flow-matching detokenizer + BigVGAN vocoder used by the audio-out
configs is vendored in-tree under
`vllm_omni/model_executor/models/kimi_audio/` (see `detokenizer.py`,
`flow_matching.py`, `bigvgan.py`). It still needs `flash_attn`
installed at runtime for the DiT prefix model.

## Send a request

### Unified Python client (`client_streaming.py`)

```bash
# audio -> text (non-streaming)
python client_streaming.py --task audio2text

# audio -> audio (streaming, writes output.wav)
python client_streaming.py --task audio2audio --audio-path /path/to/clip.wav

# text -> audio (streaming, writes output.wav)
python client_streaming.py --task text2audio \
    --question "Please say the following in audio: \"Hello, my name is Kimi.\""
```

For audio-out tasks the client opens a server-sent-event stream against
`/v1/chat/completions`, collects `delta.content` chunks (each is a
base64-encoded audio chunk), and writes the concatenated waveform to
`--out` (default `output.wav`).

### Unified curl script (`run_curl.sh`)

```bash
TASK=audio2text  bash run_curl.sh
TASK=audio2audio bash run_curl.sh   # writes response.wav
TASK=text2audio  bash run_curl.sh   # writes response.wav
```

Overrides: `PORT`, `HOST`, `OUT_FILE`, `QUESTION`.

## Endpoint summary

All three tasks load `vllm_omni/deploy/kimi_audio.yaml`. Streaming behavior is
controlled by `async_chunk` in that file (default `true`).

| Task          | Stream? (when `async_chunk: true`) | Response field                                         |
|---------------|------------------------------------|--------------------------------------------------------|
| `audio2text`  | no                                 | `choices[0].message.content`                           |
| `audio2audio` | yes                                | `choices[0].message.audio.data` (WAV b64, 24 kHz mono) |
| `text2audio`  | yes                                | `choices[0].message.audio.data` (WAV b64, 24 kHz mono) |

For audio-out responses the request must set `modalities: ["text", "audio"]`.

## Multi-GPU (TP)

To shard the fused thinker across two GPUs, edit stage 0 in
`vllm_omni/deploy/kimi_audio.yaml` to set `tensor_parallel_size: 2`,
`devices: "0,1"`, and `distributed_executor_backend: "mp"`; move stage 1
onto a free device (e.g. `devices: "2"`). The Qwen2 backbone shards via
vLLM's standard TP path; the 6-layer MIMO branch is replicated on every
rank (greedy sampling keeps ranks in sync without extra collectives).
