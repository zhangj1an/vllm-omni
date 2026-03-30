#!/bin/bash
# AudioX online serving startup script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL="${MODEL:-${REPO_ROOT}/examples/offline_inference/audiox/audiox_weights}"
PORT="${PORT:-8099}"
DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-TORCH_SDPA}"

echo "Starting AudioX server..."
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "DIFFUSION_ATTENTION_BACKEND: ${DIFFUSION_ATTENTION_BACKEND}"

export DIFFUSION_ATTENTION_BACKEND
vllm serve "${MODEL}" --omni --port "${PORT}"
