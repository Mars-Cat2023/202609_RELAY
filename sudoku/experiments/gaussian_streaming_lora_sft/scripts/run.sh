#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/nvme-data2/qilong/202609_RELAY}"
ACTION="${1:-all}"

case "$ACTION" in
  train)
    bash "$ROOT/sudoku/experiments/gaussian_streaming_lora_sft/scripts/train_3seeds.sh"
    ;;
  evaluate)
    bash "$ROOT/sudoku/experiments/gaussian_streaming_lora_sft/scripts/evaluate_3seeds.sh"
    ;;
  summarize)
    PY="${PY:-/data/qilong/miniconda3/envs/relay-sudoku/bin/python}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-$ROOT/logs/gaussian_streaming_lora_sft_tconf2_sigma1_r32_a64_trainseed}"
    EVAL_DIR_NAME="${EVAL_DIR_NAME:-matched_evaluation}"
    "$PY" "$ROOT/sudoku/experiments/gaussian_streaming_lora_sft/summarize_3seeds.py" \
      --output-prefix "$OUTPUT_PREFIX" \
      --evaluation-directory "$EVAL_DIR_NAME" \
      --train-seeds 1 2 3 \
      --eval-seeds 1 2 3 \
      --sigmas 0 1
    ;;
  all)
    bash "$ROOT/sudoku/experiments/gaussian_streaming_lora_sft/scripts/train_3seeds.sh"
    bash "$ROOT/sudoku/experiments/gaussian_streaming_lora_sft/scripts/evaluate_3seeds.sh"
    ;;
  *)
    echo "Usage: bash $0 {train|evaluate|summarize|all}" >&2
    exit 2
    ;;
esac
