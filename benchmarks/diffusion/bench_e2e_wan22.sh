#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end attention-backend comparison on Wan 2.2 T2V 14B for the #3079 PR.
# Single-GPU; Wan 2.2 has a dual-DiT structure (high-noise + low-noise stages)
# but each fits on one 96 GB Blackwell. Useful video-DiT data point alongside
# HunyuanVideo-1.5 — Wan 2.2's mask topology differs (light masks), so the
# CUDNN_ATTN advantage seen on HV-1.5 mostly collapses here.
#
# Usage (from repo root):
#   bash benchmarks/diffusion/bench_e2e_wan22.sh
#   STEPS=20 bash benchmarks/diffusion/bench_e2e_wan22.sh   # quicker variant

set -euo pipefail

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PROMPT="${PROMPT:-A cat walks through a sunlit garden, flowers swaying gently in the breeze.}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
FRAMES="${FRAMES:-33}"
STEPS="${STEPS:-40}"
GUIDANCE="${GUIDANCE:-4.0}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
FPS="${FPS:-24}"
SEED="${SEED:-42}"

BACKENDS=(
  TORCH_SDPA
  CUDNN_ATTN
  FLASHINFER_ATTN
)

OUT_DIR="${OUT_DIR:-bench_wan22_out}"
mkdir -p "$OUT_DIR"

declare -A TOTALS
declare -A PER_STEP
declare -A PEAK_MEM

for BACKEND in "${BACKENDS[@]}"; do
  LOG="$OUT_DIR/wan22_${BACKEND}.log"
  VID="$OUT_DIR/wan22_${BACKEND}.mp4"
  echo "======================================================================"
  echo "Running $BACKEND ..."
  echo "======================================================================"
  CUDA_VISIBLE_DEVICES=0 DIFFUSION_ATTENTION_BACKEND="$BACKEND" \
    python examples/offline_inference/text_to_video/text_to_video.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-frames "$FRAMES" \
      --guidance-scale "$GUIDANCE" \
      --boundary-ratio "$BOUNDARY_RATIO" \
      --flow-shift "$FLOW_SHIFT" \
      --num-inference-steps "$STEPS" \
      --fps "$FPS" \
      --seed "$SEED" \
      --output "$VID" 2>&1 | tee "$LOG"

  TOTAL=$(grep -oE "Total generation time: [0-9.]+" "$LOG" | awk '{print $4}' | tail -1)
  STEP=$(grep -oE "[0-9]+\.[0-9]+s/it" "$LOG" | tail -1 | sed 's/s\/it//')
  if [[ -z "$STEP" ]]; then
    RATE=$(grep -oE "[0-9]+\.[0-9]+it/s" "$LOG" | tail -1 | sed 's/it\/s//')
    if [[ -n "$RATE" ]]; then
      STEP=$(awk -v r="$RATE" 'BEGIN { printf "%.3f", 1.0/r }')
    fi
  fi
  MEM=$(grep -oE "Peak GPU memory.*reserved" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+ GB" | head -1)

  TOTALS[$BACKEND]="${TOTAL:-?}"
  PER_STEP[$BACKEND]="${STEP:-?}"
  PEAK_MEM[$BACKEND]="${MEM:-?}"
done

echo
echo "======================================================================"
echo "Wan 2.2 14B T2V e2e ranking (${HEIGHT}x${WIDTH}, ${FRAMES} frames, ${STEPS} steps, seed ${SEED})"
echo "======================================================================"
printf "%-18s %14s %14s %18s\n" "backend" "total (s)" "s/step" "peak VRAM"
printf "%-18s %14s %14s %18s\n" "------------------" "--------------" "--------------" "------------------"
for BACKEND in "${BACKENDS[@]}"; do
  printf "%-18s %14s %14s %18s\n" \
    "$BACKEND" "${TOTALS[$BACKEND]}" "${PER_STEP[$BACKEND]}" "${PEAK_MEM[$BACKEND]}"
done
