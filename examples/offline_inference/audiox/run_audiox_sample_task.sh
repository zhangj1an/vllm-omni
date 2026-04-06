#!/usr/bin/env bash
# AudioX offline sample: set PYTHONPATH, pick Python, run end2end.py.
#
# Prerequisites:
#   - vLLM-Omni with diffusion (AudioX code is vendored under vllm_omni)
#   - pip install -e ".[audiox]"
#   - ffmpeg (system) for video tasks / Pexels asset download in run
#
# Diffusion attention: defaults to TORCH_SDPA inside end2end.py if DIFFUSION_ATTENTION_BACKEND is unset
# (fa3-fwd / Flash may error on FP16 on some GPUs). Override: export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
#
# Usage:
#   ./run_audiox_sample_task.sh
#
# Weights (optional overrides):
#   export AUDIOX_MODEL=/path/to/bundle     # use this directory instead of ./audiox_weights
#
# Tasks (default: all six, from config unless overridden):
#   export AUDIOX_TASKS="t2a,t2m,v2a,v2m,tv2a,tv2m"
#   export AUDIOX_TASKS_FILE=./my_tasks.txt
#
# Other env (see README.md):
#   AUDIOX_VIDEO, AUDIOX_STEPS, AUDIOX_SECONDS, AUDIOX_OUT_DIR, AUDIOX_TASK_OUTPUT_ROOT,
#   AUDIOX_PR_MODEL_SLUG, AUDIOX_ASSETS_DIR
#
# Python: repo .venv when present, else python3. Override: export AUDIOX_PYTHON=/path/to/python

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${AUDIOX_PYTHON:-}" ]]; then
  PYTHON="$AUDIOX_PYTHON"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

exec "$PYTHON" "$ROOT/end2end.py" run
