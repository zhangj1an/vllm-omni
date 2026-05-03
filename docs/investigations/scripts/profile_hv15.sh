#!/usr/bin/env bash
# Wraps the HV-1.5 e2e bench with torch.profiler + nsys timeline.
# HV-1.5 is the 2× e2e win case in PR #3079 — this confirms attention
# is a large fraction of step time.

set -euo pipefail
cd "$(dirname "$0")/.."

source bench_out/env.sh
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace read token}"

MODEL="${MODEL:-hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v}"
PROMPT="${PROMPT:-A cat walks through a sunlit garden, flowers swaying gently in the breeze.}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
FRAMES="${FRAMES:-33}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-6.0}"
SEED="${SEED:-42}"

OUT_DIR="bench_out/hv15_e2e"
mkdir -p "$OUT_DIR"

BACKENDS=(TORCH_SDPA CUDNN_ATTN FLASHINFER_ATTN)

echo '########################################'
echo '# Phase 1: e2e timings + pipeline profiler'
echo '########################################'
for BACKEND in "${BACKENDS[@]}"; do
  LOG="$OUT_DIR/${BACKEND}.log"
  VID="$OUT_DIR/${BACKEND}.mp4"
  echo "=== $BACKEND ==="
  CUDA_VISIBLE_DEVICES=0 DIFFUSION_ATTENTION_BACKEND="$BACKEND" \
    python examples/offline_inference/text_to_video/text_to_video.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-frames "$FRAMES" \
      --num-inference-steps "$STEPS" \
      --guidance-scale "$GUIDANCE" \
      --seed "$SEED" \
      --tensor-parallel-size 1 \
      --enable-diffusion-pipeline-profiler \
      --output "$VID" 2>&1 | tee "$LOG"
done

echo '########################################'
echo '# Phase 2: nsys timeline of CUDNN_ATTN (the win case)'
echo '########################################'
NSYS_REP="$OUT_DIR/CUDNN_ATTN.nsys-rep"
CUDA_VISIBLE_DEVICES=0 DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN \
  nsys profile \
    --output "$NSYS_REP" \
    --force-overwrite=true \
    --trace cuda,nvtx,osrt \
    --sample none \
    --capture-range=none \
  python examples/offline_inference/text_to_video/text_to_video.py \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$FRAMES" \
    --num-inference-steps "$STEPS" \
    --guidance-scale "$GUIDANCE" \
    --seed "$SEED" \
    --tensor-parallel-size 1 \
    --output "$OUT_DIR/CUDNN_ATTN_nsys.mp4" 2>&1 | tee "$OUT_DIR/CUDNN_ATTN_nsys.log"

echo '########################################'
echo '# Summary'
echo '########################################'
for BACKEND in "${BACKENDS[@]}"; do
  TOTAL=$(grep -oE "Total generation time: [0-9.]+" "$OUT_DIR/${BACKEND}.log" | awk '{print $4}' | tail -1 || echo "?")
  echo "$BACKEND : ${TOTAL}s"
done
