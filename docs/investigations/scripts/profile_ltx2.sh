#!/usr/bin/env bash
# Wraps the Wan 2.2 14B T2V e2e bench:
#   Phase 1: e2e timings + pipeline profiler + SDPA shape hook (3 backends)
#   Phase 2: nsys timeline of CUDNN_ATTN run
#   Phase 3: nsys timeline of TORCH_SDPA run (so we see what default dispatcher picked)

set -euo pipefail
cd "$(dirname "$0")/.."

source bench_out/env.sh
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace read token}"

MODEL="${MODEL:-Lightricks/LTX-2}"
PROMPT="${PROMPT:-A cat walks through a sunlit garden, flowers swaying gently in the breeze.}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
# LTX-2 PR bench uses 97 frames + 500 steps. That's ~5 min/backend; for the
# rotation profile we shorten to 33 frames + 50 steps to bound total time
# while keeping the attention-shape mix the same (video=5070 tokens, audio=100).
FRAMES="${FRAMES:-33}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-2.0}"
SEED="${SEED:-42}"

OUT_DIR="bench_out/ltx2_e2e"
mkdir -p "$OUT_DIR"

BACKENDS=(TORCH_SDPA CUDNN_ATTN FLASHINFER_ATTN)

run_driver() {
  local backend="$1"
  local out_video="$2"
  local log="$3"
  local sdpa_log="${4:-}"
  local nsys_rep="${5:-}"

  local prefix=""
  local cmd_pre=""
  if [[ -n "$nsys_rep" ]]; then
    prefix="nsys profile --output $nsys_rep --force-overwrite=true --trace cuda,nvtx,osrt --sample none --capture-range=none"
  fi

  local extra_env=""
  if [[ -n "$sdpa_log" ]]; then
    extra_env="SDPA_SHAPE_LOG=$sdpa_log"
  fi

  local script=examples/offline_inference/text_to_video/text_to_video.py
  local pyargs=(
    --model "$MODEL"
    --prompt "$PROMPT"
    --height "$HEIGHT"
    --width "$WIDTH"
    --num-frames "$FRAMES"
    --num-inference-steps "$STEPS"
    --guidance-scale "$GUIDANCE"
    --seed "$SEED"
    --tensor-parallel-size 1
    --enable-diffusion-pipeline-profiler
    --output "$out_video"
  )

  CUDA_VISIBLE_DEVICES=0 DIFFUSION_ATTENTION_BACKEND="$backend" $extra_env \
    $prefix python bench_out/run_with_hook.py "$script" "${pyargs[@]}" 2>&1 | tee "$log"
}

echo '########################################'
echo '# Phase 1: e2e timings + pipeline profiler + SDPA shape hook'
echo '########################################'
for BACKEND in "${BACKENDS[@]}"; do
  echo "=== $BACKEND ==="
  run_driver "$BACKEND" "$OUT_DIR/${BACKEND}.mp4" "$OUT_DIR/${BACKEND}.log" \
    "$OUT_DIR/${BACKEND}.shapes.jsonl"
done

echo '########################################'
echo '# Phase 2: nsys timeline of CUDNN_ATTN'
echo '########################################'
run_driver "CUDNN_ATTN" "$OUT_DIR/CUDNN_ATTN_nsys.mp4" "$OUT_DIR/CUDNN_ATTN_nsys.log" \
  "" "$OUT_DIR/CUDNN_ATTN.nsys-rep"

echo '########################################'
echo '# Phase 3: nsys timeline of TORCH_SDPA (default dispatcher)'
echo '########################################'
run_driver "TORCH_SDPA" "$OUT_DIR/TORCH_SDPA_nsys.mp4" "$OUT_DIR/TORCH_SDPA_nsys.log" \
  "" "$OUT_DIR/TORCH_SDPA.nsys-rep"

echo '########################################'
echo '# Summary'
echo '########################################'
for BACKEND in "${BACKENDS[@]}"; do
  TOTAL=$(grep -oE "Total generation time: [0-9.]+" "$OUT_DIR/${BACKEND}.log" | awk '{print $4}' | tail -1 || echo "?")
  echo "$BACKEND : ${TOTAL}s"
done

# SDPA shape summaries
echo
echo '--- SDPA shape distribution per backend ---'
for BACKEND in "${BACKENDS[@]}"; do
  echo
  echo "### $BACKEND"
  python bench_out/sdpa_shape_logger.py --summarize "$OUT_DIR/${BACKEND}.shapes.jsonl" 2>&1 | head -30
done
