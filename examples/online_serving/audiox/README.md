# AudioX online serving

This example shows how to serve AudioX with `vllm-omni` using the OpenAI-compatible chat endpoint.

## Prerequisites

1. Install dependencies from repo root:

```bash
pip install -e ".[audiox]"
```

2. Prepare AudioX sharded weights (same as offline flow):

```bash
cd examples/offline_inference/audiox
python end2end.py run --skip-download-assets
```

This creates/uses:

`examples/offline_inference/audiox/audiox_weights`

## Start server

```bash
cd examples/online_serving/audiox
bash run_server.sh
```

Environment overrides:

- `MODEL` (default: `examples/offline_inference/audiox/audiox_weights`)
- `PORT` (default: `8099`)
- `DIFFUSION_ATTENTION_BACKEND` (default: `TORCH_SDPA`)

## Send requests

### 1) curl (`t2a`)

```bash
cd examples/online_serving/audiox
bash run_curl_t2a.sh
```

This writes `audiox_t2a.wav` by default.

### 2) Python client (all AudioX tasks)

```bash
cd examples/online_serving/audiox
python openai_chat_client.py \
  --task t2a \
  --prompt "Fireworks burst twice, followed by a period of silence before a clock begins ticking" \
  --output t2a.wav
```

For video-conditioned tasks:

```bash
python openai_chat_client.py \
  --task tv2a \
  --video-url "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4" \
  --prompt "Add matching foley and ambience for this clip." \
  --output tv2a.wav
```

## Request shape

`audiox_task` is passed through `extra_body`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video_url", "video_url": {"url": "https://.../sample.mp4"}},
        {"type": "text", "text": "Add matching foley and ambience."}
      ]
    }
  ],
  "extra_body": {
    "audiox_task": "tv2a",
    "num_inference_steps": 250,
    "guidance_scale": 7.0,
    "seed": 42,
    "seconds_total": 10.0
  }
}
```
