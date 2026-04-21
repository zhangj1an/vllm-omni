# Kimi-Audio-7B offline inference

Unified `end2end.py` supports three task modes via the `--task` flag:

| `--task`       | Input | Output | Stage config                          |
|----------------|-------|--------|---------------------------------------|
| `audio2text`   | audio | text   | `kimi_audio.yaml` (single-stage)      |
| `audio2audio`  | audio | audio  | `kimi_audio_audio_out.yaml` (2 stages)|
| `text2audio`   | text  | audio  | `kimi_audio_audio_out.yaml` (2 stages)|

The script picks the right stage config and per-stage sampling params
automatically; `--stage-configs-path` / `--output-dir` / `--question`
can override the defaults.

## Run

```bash
cd examples/offline_inference/kimi_audio

# audio -> text (ASR with a bundled asset)
python end2end.py --task audio2text

# audio -> text (audio QA over your own file)
python end2end.py --task audio2text \
    --audio-path /path/to/clip.wav \
    --question "Summarize what the speaker is saying."

# audio -> audio (spoken response)
python end2end.py --task audio2audio \
    --audio-path /path/to/clip.wav \
    --question "Answer in audio. Briefly summarize what was said."

# text -> audio (TTS-style)
python end2end.py --task text2audio \
    --question "Please say the following in audio: \"Hello, my name is Kimi.\""
```

Outputs land under the per-task default directory
(`./output_text`, `./output_audio`, `./output_tts`) or under
`--output-dir` if provided.

## Stage configs

- `kimi_audio.yaml`: single-stage `fused_thinker` (Whisper-large-v3
  encoder + VQ-Adaptor + Qwen2-7B LLM + text LM head).
- `kimi_audio_audio_out.yaml`: two-stage pipeline adding the MIMO
  audio-out branch on stage 0 and `code2wav`
  (PrefixStreamingFlowMatchingDetokenizer + BigVGAN) as stage 1.
  The detokenizer is vendored in-tree under
  `vllm_omni/model_executor/models/kimi_audio/kimia_detokenizer/`;
  `flash_attn` is still required at runtime. Expects 2 GPUs by
  default and sets `hf_overrides.kimia_generate_audio: true` on
  stage 0 so the MIMO branch is built and runs.

## Streaming audio output

`end2end_async_chunk.py` runs the same two audio-out tasks
(`audio2audio`, `text2audio`) but with the async-chunk stage config
`kimi_audio_async_chunk.yaml` (sub-second TTFB). Example:

```bash
python end2end_async_chunk.py --task text2audio \
    --question "Please say the following in audio: \"Hello, my name is Kimi.\""
```

## Audio sampling caveat

The MIMO audio head samples via greedy argmax inside the model and does
not go through vLLM's sampler (vLLM expects one distribution per step,
and text+audio heads run in parallel). So the `default_sampling_params`
in the stage YAML — `temperature`, `top_p`, `top_k` — only affect the
text path. Audio tokens are deterministic for a given text prefix.
Changing generation quality of the audio requires editing
`_run_mimo_branch` in `kimi_audio_thinker.py`.
