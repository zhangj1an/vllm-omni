#!/usr/bin/env bash
# Wraps the Z-Image-Turbo e2e bench with torch.profiler + nsys timeline so we
# can answer "what % of step time is in SDPA" empirically.
#
# Output:
#   bench_out/zimage_e2e/<BACKEND>.log         — full stdout (Total time + per-step rate)
#   bench_out/zimage_e2e/<BACKEND>.nsys-rep    — nsys timeline
#   bench_out/zimage_e2e/<BACKEND>.png         — generated image
#
# Sourcing env.sh first activates CUDA 13.2 + FlashInfer sm_120f.

set -euo pipefail
cd "$(dirname "$0")/.."

source bench_out/env.sh
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace read token}"

MODEL="${MODEL:-Tongyi-MAI/Z-Image-Turbo}"
PROMPT="${PROMPT:-A warm morning kitchen close-up of a woman and a man in their 30s standing across from each other at the counter, both holding mugs, cinematic, deadpan, golden morning light, shallow depth of field.}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-8}"
CFG_SCALE="${CFG_SCALE:-1.0}"
SEED="${SEED:-42}"

OUT_DIR="bench_out/zimage_e2e"
mkdir -p "$OUT_DIR"

BACKENDS=(TORCH_SDPA CUDNN_ATTN FLASHINFER_ATTN)

# Phase 1: plain e2e timings across all 3 backends
echo '########################################'
echo '# Phase 1: e2e timings'
echo '########################################'
for BACKEND in "${BACKENDS[@]}"; do
  LOG="$OUT_DIR/${BACKEND}.log"
  IMG="$OUT_DIR/${BACKEND}.png"
  echo "=== $BACKEND ==="
  CUDA_VISIBLE_DEVICES=0 DIFFUSION_ATTENTION_BACKEND="$BACKEND" \
    python examples/offline_inference/text_to_image/text_to_image.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-inference-steps "$STEPS" \
      --cfg-scale "$CFG_SCALE" \
      --seed "$SEED" \
      --tensor-parallel-size 1 \
      --enable-diffusion-pipeline-profiler \
      --output "$IMG" 2>&1 | tee "$LOG"
done

# Phase 2: nsys timeline of CUDNN_ATTN run (the masked-path winner)
# Only profile one backend — nsys overhead is ~2× and Z-Image is short.
echo '########################################'
echo '# Phase 2: nsys timeline of CUDNN_ATTN'
echo '########################################'
NSYS_REP="$OUT_DIR/CUDNN_ATTN.nsys-rep"
CUDA_VISIBLE_DEVICES=0 DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN \
  nsys profile \
    --output "$NSYS_REP" \
    --force-overwrite=true \
    --trace cuda,nvtx,osrt \
    --sample none \
    --capture-range=none \
  python examples/offline_inference/text_to_image/text_to_image.py \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-inference-steps "$STEPS" \
    --cfg-scale "$CFG_SCALE" \
    --seed "$SEED" \
    --tensor-parallel-size 1 \
    --output "$OUT_DIR/CUDNN_ATTN_nsys.png" 2>&1 | tee "$OUT_DIR/CUDNN_ATTN_nsys.log"

echo '########################################'
echo '# Phase 3: nsys stats summary'
echo '########################################'
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --format csv "$NSYS_REP" \
  2>&1 | head -200 | tee "$OUT_DIR/CUDNN_ATTN_nsys_stats.csv"

echo '########################################'
echo '# Summary'
echo '########################################'
for BACKEND in "${BACKENDS[@]}"; do
  TOTAL=$(grep -oE "Total generation time: [0-9.]+" "$OUT_DIR/${BACKEND}.log" | awk '{print $4}' | tail -1 || echo "?")
  echo "$BACKEND : ${TOTAL}s"
done
