# AudioX offline inference (vLLM-Omni)

This folder runs [AudioX](https://github.com/ZeyueT/AudioX) through the `AudioXPipeline` integration: **local weight bundles**, optional **sample stock videos** (Pexels), and **`Omni()`** inference. Inference code is **vendored** under `vllm_omni.diffusion.models.audiox` — you do **not** need a separate `AudioX` git clone or `AUDIOX_SRC`.

## Prerequisites

1. **vLLM** and **vLLM-Omni** (editable install from this repo) with diffusion enabled.
2. **AudioX Python extras** (PyPI packages not in the core `requirements/common.txt`):

   ```bash
   # from repository root
   pip install -e ".[audiox]"
   ```

3. **Diffusion attention backend** — `end2end.py` defaults to `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` if unset, so a plain `python end2end.py …` works on GPUs where **fa3-fwd** / Flash reports errors such as *“This flash attention build does not support FP16”*. To force Flash when your stack supports it:

   ```bash
   export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
   ```

4. **`ffmpeg`** on your **PATH** if you use `run` with video tasks (`v2*`, `tv2*`) or Pexels sample download (see `end2end.py`). Text-only `infer --task t2a` does not need ffmpeg.

## Weight layout

`AudioXPipeline` loads **only** vLLM-Omni **component-sharded** weights (`transformer/diffusion_pytorch_model.safetensors` and `conditioners/diffusion_pytorch_model.safetensors`, plus `vae/` when the config has a pretransform).

Obtain weights by **downloading a pre-sharded bundle** (for example [zhangj1an/AudioX](https://huggingface.co/zhangj1an/AudioX) on Hugging Face):

```bash
huggingface-cli download zhangj1an/AudioX --local-dir ./audiox_weights
```

Earlier docs and upstream used **`HKUSTAudio/AudioX`**; the layout and remap path in vLLM-Omni are aligned with the pre-sharded bundle published as **`zhangj1an/AudioX`** for now.

## Quick start

From the vLLM-Omni repo root (or anywhere), with `PYTHONPATH` pointing at the repo if you are not using an editable install:

```bash
cd examples/offline_inference/audiox

# Populate ./audiox_weights (sharded safetensors), then:
./run_audiox_sample_task.sh
```

Equivalent without the shell helper:

```bash
python end2end.py run
```

## `end2end.py`

| Subcommand | Purpose |
|------------|---------|
| `run` | Use inlined default config in `end2end.py`; optional Pexels video download; run task list. |
| `infer …` | Single-task inference (CLI flags for task, prompt, video, steps, etc.). |

Examples:

```bash
# Built-in sample batch (expects weights under ./audiox_weights or AUDIOX_MODEL)
python end2end.py run

# Use custom local weight bundle
export AUDIOX_MODEL=/data/audiox_weights
python end2end.py run

# Subset of tasks
export AUDIOX_TASKS="t2a tv2a"
python end2end.py run

# Single-shot debugging
python end2end.py infer --model ./audiox_weights --task t2a \
  --prompt "A busy city street with horns and footsteps."
```

Skip **asset** downloads only (you already have sample video paths):

```bash
python end2end.py run --skip-download-assets
```

## Built-in run config

`end2end.py run` uses an inlined default config (`DEFAULT_RUN_CONFIG`) with:

- **`weights`**: `local_dir` (default `audiox_weights` under this example folder).
- **`assets`**: `local_dir`, `trim_seconds`, `download_if_missing`.
- **`run`**: task list, `output_root`, `num_inference_steps`, `seconds_total`, `guidance_scale`, etc.

Use environment variables below to override these defaults without editing code.

## Environment variables

| Variable | Effect |
|----------|--------|
| `DIFFUSION_ATTENTION_BACKEND` | `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, etc. Unset in `end2end.py` defaults to `TORCH_SDPA`. |
| `AUDIOX_CONFIG` | Path to JSON config (shell default: `configs/<AUDIOX_CLIP>.json`). |
| `AUDIOX_CLIP` | `animal` or `human`; selects default config when `AUDIOX_CONFIG` is unset. |
| `AUDIOX_MODEL` | Absolute path to weight directory (overrides `local_dir`). |
| `AUDIOX_VIDEO` | Override video path for `v2*` / `tv2*` tasks. |
| `AUDIOX_TASKS` | Comma/space-separated subset of `t2a t2m v2a v2m tv2a tv2m`. |
| `AUDIOX_TASKS_FILE` | Same tasks, one per line (`#` comments allowed). |
| `AUDIOX_STEPS` | `num_inference_steps` override. |
| `AUDIOX_SECONDS` | `seconds_total` override. |
| `AUDIOX_OUT_DIR` | Exact output directory (no `model_slug/clip` subdirs). |
| `AUDIOX_TASK_OUTPUT_ROOT` | Root under this folder for `audiox_task_outputs/<slug>/<clip>/`. |
| `AUDIOX_PR_MODEL_SLUG` | Folder name under the output root (default: weight directory name). |

## Outputs

Default layout:

```text
audiox_task_outputs/<model_slug>/<animal|human>/{t2a,t2m,v2a,v2m,tv2a,tv2m}.wav
```

## Troubleshooting

- **Import errors** (`dac`, `einops_exts`, …): install `.[audiox]` as above.
- **protobuf / vLLM errors** after installing AudioX deps: re-run `pip install -e ".[audiox]"` to enforce the protobuf floor.
- **Flash attention / FP16 / dummy run failed**: use default `TORCH_SDPA` (already the default in `end2end.py`) or fix your Flash / fa3-fwd build for your GPU.
- **Missing `transformer/config.json`**: some tooling expects `transformer/config.json` (can be `{}`) under the bundle; add it if something complains.
- **`tv2a` / `tv2m`**: require a **non-empty** prompt. **`v2*` / `tv2*`**: require **`--video-path`** (or a config that supplies a video file).
- **VRAM**: shorten `seconds_total`, reduce steps, shorter video clips, or `enable_cpu_offload: true` in config / `--enable-cpu-offload` for `infer`.

## License

Sample videos are fetched from [Pexels](https://www.pexels.com/license/) download URLs; cache them under `assets/` for local use.
