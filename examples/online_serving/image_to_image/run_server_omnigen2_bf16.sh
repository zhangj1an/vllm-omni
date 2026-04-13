#!/usr/bin/env bash
# Start OmniGen2 without FP8 quantization (BF16 weights / default precision).
# Pair with omnigen2_fp8_edit_client.py for apples-to-apples vs run_server_omnigen2_fp8.sh.
#
# Optional torch profiling:
#   ENABLE_TORCH_PROFILER=1 bash run_server_omnigen2_bf16.sh
#   ENABLE_TORCH_PROFILER=1 PROFILE_DIR=./my_traces bash run_server_omnigen2_bf16.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)/.venv/bin:${PATH}"

MODEL="${MODEL:-OmniGen2/OmniGen2}"
PORT="${PORT:-8092}"

echo "Starting OmniGen2 server (no FP8 quantization)..."
echo "Model: ${MODEL}"
echo "Port: ${PORT}"

EXTRA_ARGS=()
if [[ "${ENABLE_TORCH_PROFILER:-0}" == "1" ]]; then
  PROFILE_DIR="${PROFILE_DIR:-./vllm_profile_omnigen2_bf16}"
  PROFILER_CONFIG="${PROFILER_CONFIG:-{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\"}}"
  echo "Torch profiler: enabled (${PROFILER_CONFIG})"
  EXTRA_ARGS+=(--profiler-config "${PROFILER_CONFIG}")
else
  echo "Torch profiler: disabled (set ENABLE_TORCH_PROFILER=1 to enable)"
fi

vllm-omni serve "${MODEL}" --omni \
  "${EXTRA_ARGS[@]}" \
  --port "${PORT}"
