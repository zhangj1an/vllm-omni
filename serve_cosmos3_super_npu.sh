#!/bin/bash
# Serve Cosmos3-Super (64B) on 8 Ascend910 NPUs with tensor-parallel=8
#
# Model: /run/z84450661/Cosmos3-Super
# NPUs: 8 × Ascend910 (16 davinci devices, using TP=8 across 8 NPU chips)
# KV heads: 8 → tensor-parallel-size=8
#
# Usage: docker exec -i vllm-omni-24 bash < serve_cosmos3_super_npu.sh
# Or run it inside the container directly.

set -euo pipefail

MODEL_PATH="/run/z84450661/Cosmos3-Super"

echo "=== Serving Cosmos3-Super on Ascend NPU (TP=8) ==="
echo "Model: ${MODEL_PATH}"
echo ""

# Ensure HF_HOME is set
export HF_HOME=/home/ma-user/work/z84450661/hf_cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
# HF_TOKEN: 如需下载 gated 模型,在容器外 export HF_TOKEN 后 docker exec 传入
# Extend video generation timeout (default 600s is too short for 720p/189f)
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=10800

vllm serve "${MODEL_PATH}" \
    --omni \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --no-guardrails \
    --init-timeout 1800 \
    --max-model-len 65536
