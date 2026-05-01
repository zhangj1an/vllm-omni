# Kimi-Audio

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/kimi_audio>.

For a hardware-tagged step-by-step guide, see the
[Kimi-Audio recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/moonshotai/Kimi-Audio-7B-Instruct.md).

## Launch the server

Two-stage audio-out pipeline (default async-chunk streaming on 2 GPUs):

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct \
    --omni \
    --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/kimi_audio.yaml
```

For sync (non-streaming) operation, edit the YAML to set
`async_chunk: false`. For text-out only deployment, point at
`kimi_audio_asr_single_gpu.yaml` and request `"modalities": ["text"]`.

## Curl examples

The example dir ships a unified curl script:

```bash
TASK=audio2text  bash examples/online_serving/kimi_audio/run_curl.sh
TASK=audio2audio bash examples/online_serving/kimi_audio/run_curl.sh
TASK=text2audio  bash examples/online_serving/kimi_audio/run_curl.sh
```

A Python streaming client mirroring the same three modes lives at
`examples/online_serving/kimi_audio/client_streaming.py`.
