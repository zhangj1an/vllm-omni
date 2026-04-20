# Kimi-Audio-7B offline inference

Slice 1: audio-in / text-out (ASR and audio question-answering). Audio output
arrives in Slice 2.

## Run

```bash
cd examples/offline_inference/kimi_audio

# ASR with a bundled asset
python end2end.py \
    --model-name moonshotai/Kimi-Audio-7B-Instruct \
    --question "Please transcribe the audio."

# Audio QA over your own file
python end2end.py \
    --model-name moonshotai/Kimi-Audio-7B-Instruct \
    --audio-path /path/to/clip.wav \
    --question "Summarize what the speaker is saying."
```

Outputs land in `./output_text/<request_id>.txt`.

## Stage config

Single-stage `fused_thinker` (Whisper-large-v3 encoder + VQ-Adaptor + Qwen2-7B
LLM + text LM head). See
`vllm_omni/model_executor/stage_configs/kimi_audio.yaml`.

## Audio-out (Slice 2)

```bash
# Audio in, audio out (uses the two-stage pipeline)
python end2end_audio_out.py \
    --model-name moonshotai/Kimi-Audio-7B-Instruct \
    --audio-path /path/to/clip.wav \
    --question "Answer in audio. Briefly summarize what was said."
```

Outputs: `./output_audio/<request_id>.txt` (text) and
`./output_audio/<request_id>.wav` (24 kHz mono audio).

Requires the `kimia_infer` package (from
<https://github.com/MoonshotAI/Kimi-Audio>) for the flow-matching detokenizer
and BigVGAN vocoder. The two-stage config lives at
`vllm_omni/model_executor/stage_configs/kimi_audio_audio_out.yaml` and
expects 2 GPUs by default. It sets `hf_overrides.kimia_generate_audio: true`
on stage 0 so the MIMO audio-out branch is built and runs; text-only YAMLs
omit that flag and skip the branch.

### Audio sampling caveat

The MIMO audio head samples via greedy argmax inside the model and does
not go through vLLM's sampler (vLLM expects one distribution per step, and
text+audio heads run in parallel). So the `default_sampling_params` in the
stage YAML — `temperature`, `top_p`, `top_k` — only affect the text path.
Audio tokens are deterministic for a given text prefix. Changing
generation quality of the audio requires editing `_run_mimo_branch` in
`kimi_audio_thinker.py`.

Slice 3 will add `end2end_async_chunk.py` (streaming, low TTFB).
