#!/usr/bin/env bash
# FLUX.2-dev e2e attention-backend profile to match the PR #3079 bench config:
#   Phase 1: e2e timings + pipeline profiler + SDPA shape hook (3 backends)
#   Phase 2: nsys timeline of CUDNN_ATTN run
#   Phase 3: nsys timeline of TORCH_SDPA run (so we see what default dispatcher picked)
# TP=2 across both GPUs matches benchmarks/diffusion/bench_e2e_flux2.sh; FLUX.2-dev
# is too big for a single 96 GB card without CPU offload, and offload changes the
# critical path enough that the kernel ranking would no longer be comparable.

set -euo pipefail
cd "$(dirname "$0")/.."

source bench_out/env.sh
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var to your HuggingFace read token}"

MODEL="${MODEL:-black-forest-labs/FLUX.2-dev}"
PROMPT="${PROMPT:-A warm morning kitchen close-up of a woman and a man in their 30s standing across from each other at the counter, both holding mugs, cinematic, deadpan, golden morning light, shallow depth of field.}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
SEED="${SEED:-42}"
TP="${TP:-2}"

OUT_DIR="bench_out/flux2_e2e"
mkdir -p "$OUT_DIR"

BACKENDS=(TORCH_SDPA CUDNN_ATTN FLASHINFER_ATTN)

run_driver() {
  local backend="$1"
  local out_image="$2"
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

  local script=/root/vllm-omni/examples/offline_inference/text_to_image/text_to_image.py
  local pyargs=(
    --model "$MODEL"
    --prompt "$PROMPT"
    --height "$HEIGHT"
    --width "$WIDTH"
    --num-inference-steps "$STEPS"
    --cfg-scale "$CFG_SCALE"
    --seed "$SEED"
    --tensor-parallel-size "$TP"
    --enable-diffusion-pipeline-profiler
    --output "$out_image"
  )

  env CUDA_VISIBLE_DEVICES=0,1 DIFFUSION_ATTENTION_BACKEND="$backend" $extra_env \
    $prefix python bench_out/run_with_hook.py "$script" "${pyargs[@]}" 2>&1 | tee "$log"
}

echo '########################################'
echo '# Phase 1: e2e timings + pipeline profiler + SDPA shape hook'
echo '########################################'
for BACKEND in "${BACKENDS[@]}"; do
  echo "=== $BACKEND ==="
  run_driver "$BACKEND" "$OUT_DIR/${BACKEND}.png" "$OUT_DIR/${BACKEND}.log" \
    "$OUT_DIR/${BACKEND}.shapes.jsonl"
done

echo '########################################'
echo '# Phase 2: nsys timeline of CUDNN_ATTN'
echo '########################################'
run_driver "CUDNN_ATTN" "$OUT_DIR/CUDNN_ATTN_nsys.png" "$OUT_DIR/CUDNN_ATTN_nsys.log" \
  "" "$OUT_DIR/CUDNN_ATTN.nsys-rep"

echo '########################################'
echo '# Phase 3: nsys timeline of TORCH_SDPA (default dispatcher)'
echo '########################################'
run_driver "TORCH_SDPA" "$OUT_DIR/TORCH_SDPA_nsys.png" "$OUT_DIR/TORCH_SDPA_nsys.log" \
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
