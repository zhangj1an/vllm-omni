# Kimi-Audio-7B offline inference

Unified `end2end.py` supports three task modes via the `--task` flag,
mirroring the canonical demos in MoonshotAI/Kimi-Audio's `infer.py`:

  * `asr` — audio in, text out (ASR over upstream `asr_example.wav`)
  * `qa` — audio in, audio + text out (spoken QA over upstream
    `qa_example.wav`, audio-only user turn — no text instruction)
  * `multiturn` — multi-turn audio chat (q1 → assistant audio+text a1 →
    q2). **Not yet implemented in vllm-omni** — assistant audio history
    requires the `_split_prefill` mask fix in `kimi_audio_thinker.py`
    plus a prompt builder that pre-tokenizes prior assistant audio with
    the GLM-4-Voice tokenizer. Tracked separately.

All tasks load `vllm_omni/model_executor/stage_configs/kimi_audio.yaml`. Per-stage sampling
params come from that file; `--stage-configs-path` / `--output-dir` /
`--question` can override the defaults.

## Run

Defaults pull audio from
[upstream Kimi-Audio's `test_audios/`](https://github.com/MoonshotAI/Kimi-Audio/tree/master/test_audios)
so the audio-input modes work out of the box. Pass `--audio-path` /
`--question` to override either.

```bash
cd examples/offline_inference/kimi_audio

# ASR over upstream asr_example.wav (Chinese transcription)
python end2end.py --task asr

# ASR over your own clip
python end2end.py --task asr \
    --audio-path /path/to/clip.wav \
    --question "Summarize what the speaker is saying."

# Spoken QA over upstream qa_example.wav (audio-only user turn)
python end2end.py --task qa
```

Outputs land under the per-task default directory
(`./output_asr`, `./output_qa`, `./output_multiturn`) or under
`--output-dir` if provided.

For single-GPU hosts, pass
`--stage-configs-path ../../../vllm_omni/model_executor/stage_configs/kimi_audio_single_gpu.yaml`
(both stages on `cuda:0`, `async_chunk: false`).

## Deploy config

`vllm_omni/model_executor/stage_configs/kimi_audio.yaml` is a 2-stage audio-out pipeline. Stage 0
is the fused thinker (Whisper-large-v3 + VQ-Adaptor + Qwen2-7B + 6-layer
MIMO branch) with `hf_overrides.kimia_generate_audio: true`. Stage 1 is
`code2wav` (PrefixStreamingFlowMatchingDetokenizer + BigVGAN). The
detokenizer is vendored in-tree under
`vllm_omni/model_executor/models/kimi_audio/` (see `detokenizer.py`,
`flow_matching.py`, `bigvgan.py`) and uses vLLM's bundled
`vllm.vllm_flash_attn` for attention — no separate `flash-attn` install
is needed.

Defaults to async-chunk streaming on 2 GPUs (sub-second TTFB). Set
`async_chunk: false` in the YAML and put both stages on `devices: "0"`
to run sync mode on a single GPU. For text-out only, drop stage 1 and
set `kimia_generate_audio: false` on stage 0.

## Streaming audio output

`end2end_async_chunk.py` runs the same two audio-out tasks
(`audio2audio`, `text2audio`) using the same deploy YAML (`async_chunk:
true` is the default there, so streaming is on). Example:

```bash
python end2end_async_chunk.py --task text2audio \
    --question "Please say the following in audio: \"Hello, my name is Kimi.\""
```

## Audio sampling caveat

The MIMO audio head samples via greedy argmax inside the model and does
not go through vLLM's sampler (vLLM expects one distribution per step,
and text+audio heads run in parallel). So the `default_sampling_params`
in the deploy YAML — `temperature`, `top_p`, `top_k` — only affect the
text path. Audio tokens are deterministic for a given text prefix.
Changing generation quality of the audio requires editing
`_run_mimo_branch` in `kimi_audio_thinker.py`.
