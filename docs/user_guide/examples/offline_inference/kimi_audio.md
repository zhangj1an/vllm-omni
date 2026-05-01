# Kimi-Audio

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/kimi_audio>.

For a hardware-tagged step-by-step guide, see the
[Kimi-Audio recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/moonshotai/Kimi-Audio-7B-Instruct.md).

## Setup

The pipeline runs as two stages: a fused thinker (Whisper-large-v3 +
VQ-Adaptor + Qwen2-7B + 6-layer MIMO branch) and a code2wav stage
(flow-matching DiT + BigVGAN). See the
[stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/)
to size GPU memory for your setup.

## Run examples

```bash
cd examples/offline_inference/kimi_audio
```

ASR (audio in, text out) — uses the single-GPU text-only YAML
(`kimi_audio_asr_single_gpu.yaml`):

```bash
python end2end.py --task asr
```

Single-turn audio chat (audio in, text + spoken audio out):

```bash
python end2end.py --task qa \
    --stage-configs-path ../../../vllm_omni/model_executor/stage_configs/kimi_audio_single_gpu.yaml \
    --max-tokens 2048
```

Multi-turn audio chat (q1 audio → a1 audio + text → q2 audio):

```bash
python end2end.py --task multiturn \
    --stage-configs-path ../../../vllm_omni/model_executor/stage_configs/kimi_audio_single_gpu.yaml \
    --max-tokens 2048
```

Async-chunked streaming (sub-second TTFB on 2 GPUs):

```bash
python end2end_async_chunk.py --task qa
```

Outputs land in `output_asr/`, `output_qa/`, or `output_multiturn/`
(one `.txt` transcript and one `.wav` per request).
