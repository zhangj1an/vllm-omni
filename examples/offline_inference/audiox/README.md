# AudioX offline inference (vLLM-Omni)

This folder runs [AudioX](https://github.com/ZeyueT/AudioX) through the `AudioXPipeline` integration in vLLM-Omni.
The example is intentionally trimmed to a single built-in **animal** sample flow.

## Quick start

Place a vLLM-Omni **sharded** weight bundle under `./audiox_weights` (or set `AUDIOX_MODEL` to another directory), then:

```bash
cd examples/offline_inference/audiox
./run_audiox_sample_task.sh
```

Equivalent:

```bash
python end2end.py run
```

`run` uses an inlined default config in `end2end.py` (formerly `configs/animal.json`).

To fetch weights, download a pre-sharded bundle **before** running the example (the script does not download model files for you), for example:

```bash
huggingface-cli download zhangj1an/AudioX --local-dir ./audiox_weights
```

## Prerequisites

1. vLLM + vLLM-Omni with diffusion enabled.
2. AudioX extras:

```bash
pip install -e ".[audiox]"
```

3. `ffmpeg` on `PATH` for sample video download and video tasks (`v2*`, `tv2*`).

## Commands

- `python end2end.py run`: animal sample bundle; optional Pexels video download for video tasks.
- `python end2end.py infer ...`: single-task debugging with explicit CLI flags.

Example infer command:

```bash
python end2end.py infer --model ./audiox_weights --task t2a \
  --prompt "A busy city street with horns and footsteps."
```

## Environment overrides

- `DIFFUSION_ATTENTION_BACKEND`: defaults to `TORCH_SDPA` if unset.
- `AUDIOX_MODEL`: local weights directory (overrides default `./audiox_weights`).
- `AUDIOX_VIDEO`: override sample video path for `v2*` / `tv2*`.
- `AUDIOX_TASKS` / `AUDIOX_TASKS_FILE`: subset of `t2a t2m v2a v2m tv2a tv2m`.
- `AUDIOX_STEPS`, `AUDIOX_SECONDS`: generation overrides.
- `AUDIOX_OUT_DIR` or `AUDIOX_TASK_OUTPUT_ROOT`: output location overrides.
- `AUDIOX_PR_MODEL_SLUG`: output slug under task output root.

## Outputs

Default layout:

```text
audiox_task_outputs/<model_slug>/animal/{t2a,t2m,v2a,v2m,tv2a,tv2m}.wav
```

## Notes

- Weights must already be in the vLLM-Omni sharded layout; use a Hub bundle such as [zhangj1an/AudioX](https://huggingface.co/zhangj1an/AudioX).
- Sample video is fetched from [Pexels](https://www.pexels.com/license/).
