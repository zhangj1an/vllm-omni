# AudioX offline inference (vLLM-Omni)

This folder runs [AudioX](https://github.com/ZeyueT/AudioX) through the `AudioXPipeline` integration: **Hugging Face weights**, optional **sample stock videos** (Pexels), and **`Omni()`** inference. Inference code is **vendored** under `vllm_omni.diffusion.models.audiox` — you do **not** need a separate `AudioX` git clone or `AUDIOX_SRC`.

## Prerequisites

1. **vLLM** and **vLLM-Omni** (editable install from this repo) with diffusion enabled.
2. **AudioX Python extras** (PyPI packages not in the core `requirements/common.txt`):

   ```bash
   # from repository root
   pip install -e ".[audiox]"
   ```

   Or: `pip install -r requirements/audiox.txt`.

   The protobuf floor is included directly in the AudioX dependency set because
   **descript-audio-codec** pulls **descript-audiotools** (old protobuf pin), while
   **vLLM 0.18+** needs `protobuf>=5.29.6`.

3. **Diffusion attention backend** — `end2end.py` defaults to `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` if unset, so a plain `python end2end.py …` works on GPUs where **fa3-fwd** / Flash reports errors such as *“This flash attention build does not support FP16”*. To force Flash when your stack supports it:

   ```bash
   export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
   ```

4. **`ffmpeg`** on your **PATH** if you use `run --config` with video tasks (`v2*`, `tv2*`) or Pexels sample download (see `end2end.py`). Text-only `infer --task t2a` does not need ffmpeg.

## Weight layout

`AudioXPipeline` loads **only** vLLM-Omni **component-sharded** weights (`transformer/diffusion_pytorch_model.safetensors` and `conditioners/diffusion_pytorch_model.safetensors`, plus `vae/` when the config has a pretransform). Hugging Face repos ship a flat `model.ckpt`; **`end2end.py` converts that checkpoint in place** after download (or on first `infer` / `run` if you copied `model.ckpt` yourself). Manual conversion: `python -m vllm_omni.diffusion.models.audiox.audiox_weights --input-dir DIR --output-dir DIR`.

## Quick start

From the vLLM-Omni repo root (or anywhere), with `PYTHONPATH` pointing at the repo if you are not using an editable install:

```bash
cd examples/offline_inference/audiox

# One command: HF weights + assets (if missing) + all six tasks
./run_audiox_sample_task.sh
```

Equivalent without the shell helper:

```bash
python end2end.py run --config configs/animal.json
```

## `end2end.py`

| Subcommand | Purpose |
|------------|---------|
| `run --config PATH` | Load JSON config; download weights/assets when configured and missing; run the task list. |
| `infer …` | Single-task inference (CLI flags for task, prompt, video, steps, etc.). |

Examples:

```bash
# Config-driven batch (animal clip bundle)
python end2end.py run --config configs/animal.json

# Human clip, custom HF bundle directory (skips download)
export AUDIOX_MODEL=/data/audiox_weights
python end2end.py run --config configs/human.json

# Subset of tasks
export AUDIOX_TASKS="t2a tv2a"
python end2end.py run --config configs/animal.json

# Single-shot debugging
python end2end.py infer --model ./audiox_weights --task t2a \
  --prompt "A busy city street with horns and footsteps."
```

Skip downloads (you already have weights and videos):

```bash
python end2end.py run --config configs/animal.json \
  --skip-download-weights --skip-download-assets
```

## JSON config

See `configs/animal.json` and `configs/human.json`. Main sections:

- **`weights`**: `hf_model` (`maf-mmdit`), `local_dir`, `full` (VAE + synchformer), `download_if_missing`, optional `repo_id` (full Hugging Face id).
- **`assets`**: `local_dir`, `trim_seconds`, `download_if_missing`.
- **`run`**: `clip` (`animal` \| `human`), optional `tasks` list, `output_root`, `num_inference_steps`, `seconds_total`, etc.

Paths in the config are relative to this directory unless absolute.

## Environment variables

| Variable | Effect |
|----------|--------|
| `DIFFUSION_ATTENTION_BACKEND` | `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, etc. Unset in `end2end.py` defaults to `TORCH_SDPA`. |
| `AUDIOX_CONFIG` | Path to JSON config (shell default: `configs/<AUDIOX_CLIP>.json`). |
| `AUDIOX_CLIP` | `animal` or `human`; selects default config when `AUDIOX_CONFIG` is unset. |
| `AUDIOX_MODEL` | Weight directory; if set, HF download is skipped. |
| `AUDIOX_VIDEO` | Override video path for `v2*` / `tv2*` tasks. |
| `AUDIOX_TASKS` | Comma/space-separated subset of `t2a t2m v2a v2m tv2a tv2m`. |
| `AUDIOX_TASKS_FILE` | Same tasks, one per line (`#` comments allowed). |
| `AUDIOX_STEPS` | `num_inference_steps` override. |
| `AUDIOX_SECONDS` | `seconds_total` override. |
| `AUDIOX_OUT_DIR` | Exact output directory (no `model_slug/clip` subdirs). |
| `AUDIOX_TASK_OUTPUT_ROOT` | Root under this folder for `audiox_task_outputs/<slug>/<clip>/`. |
| `AUDIOX_PR_MODEL_SLUG` | Folder name under the output root (default: `hf_model`). |

## Model (Hugging Face)

| Key | Repo |
|-----|------|
| `maf-mmdit` | `HKUSTAudio/AudioX-MAF-MMDiT` |

## Outputs

Default layout:

```text
audiox_task_outputs/<model_slug>/<animal|human>/{t2a,t2m,v2a,v2m,tv2a,tv2m}.wav
```

## Troubleshooting

- **Import errors** (`k_diffusion`, `dac`, `einops_exts`, …): install `.[audiox]` (or `requirements/audiox.txt`) as above.
- **protobuf / vLLM errors** after installing AudioX deps: re-run `pip install -e ".[audiox]"` (or `pip install -r requirements/audiox.txt`) to enforce the protobuf floor.
- **Flash attention / FP16 / dummy run failed**: use default `TORCH_SDPA` (already the default in `end2end.py`) or fix your Flash / fa3-fwd build for your GPU.
- **Missing `transformer/config.json`**: re-run `end2end.py run` with downloads enabled, or add `transformer/config.json` containing `{}` under the weight bundle.
- **`tv2a` / `tv2m`**: require a **non-empty** prompt. **`v2*` / `tv2*`**: require **`--video-path`** (or a config that supplies a video file).
- **VRAM**: shorten `seconds_total`, reduce steps, shorter video clips, or `enable_cpu_offload: true` in config / `--enable-cpu-offload` for `infer`.

## License

Sample videos are fetched from [Pexels](https://www.pexels.com/license/) download URLs; cache them under `assets/` for local use.
