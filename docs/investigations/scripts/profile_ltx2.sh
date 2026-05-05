#!/usr/bin/env bash
# LTX-2.0 (T2V + audio) e2e attention-backend profile to match PR #3079:
#   Phase 1: e2e timings + pipeline profiler + SDPA shape hook (3 backends)
#   Phase 2: nsys timeline of CUDNN_ATTN run
#   Phase 3: nsys timeline of TORCH_SDPA run (so we see what default dispatcher picked)
# CFG_PARALLEL_SIZE=2 matches the PR's two-GPU CFG-parallel setup. Defaults match
# benchmarks/diffusion/bench_e2e_ltx2.sh (97 frames, 500 steps); override via env
# vars if you want to bound profiling time, e.g. FRAMES=33 STEPS=50.

set -euo pipefail
cd "$(dirname "$0")/.."

source bench_out/env.sh
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace read token}"

MODEL="${MODEL:-Lightricks/LTX-2}"
PROMPT="${PROMPT:-A warm morning kitchen. The camera opens on a tight close-up of a woman and a man in their 30s standing across from each other at the counter, mugs in hand. The woman says low and serious: \"We said one. Just one.\" The man exhales, glancing guiltily at the floor, and mutters, \"It got lonely.\" The camera slowly pans right to reveal every surface of the kitchen covered in identical potted houseplants. He is still holding one more behind his back. Tone: deadpan, affectionate, and quietly tragic.}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
FRAMES="${FRAMES:-97}"
STEPS="${STEPS:-500}"
GUIDANCE="${GUIDANCE:-4.0}"
FRAME_RATE="${FRAME_RATE:-48}"
FPS="${FPS:-48}"
SEED="${SEED:-42}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-2}"

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
  if [[ -n "$nsys_rep" ]]; then
    prefix="nsys profile --output $nsys_rep --force-overwrite=true --trace cuda,nvtx,osrt --sample none --capture-range=none"
  fi

  local extra_env=""
  if [[ -n "$sdpa_log" ]]; then
    extra_env="SDPA_SHAPE_LOG=$sdpa_log"
  fi

  local script=/root/vllm-omni/examples/offline_inference/text_to_video/text_to_video.py
  local pyargs=(
    --model "$MODEL"
    --prompt "$PROMPT"
    --height "$HEIGHT"
    --width "$WIDTH"
    --num-frames "$FRAMES"
    --num-inference-steps "$STEPS"
    --guidance-scale "$GUIDANCE"
    --frame-rate "$FRAME_RATE"
    --fps "$FPS"
    --seed "$SEED"
    --tensor-parallel-size 1
    --cfg-parallel-size "$CFG_PARALLEL_SIZE"
    --enable-diffusion-pipeline-profiler
    --output "$out_video"
  )

  # `|| true` keeps the sweep going if the example's downstream video-save
  # logic fails (we only care about the inference-time line in the log).
  env CUDA_VISIBLE_DEVICES=0,1 DIFFUSION_ATTENTION_BACKEND="$backend" $extra_env \
    $prefix python bench_out/run_with_hook.py "$script" "${pyargs[@]}" 2>&1 | tee "$log" || true
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
