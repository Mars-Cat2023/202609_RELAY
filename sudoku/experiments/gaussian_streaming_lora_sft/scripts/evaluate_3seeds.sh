#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/nvme-data2/qilong/202609_RELAY}"
PY="${PY:-/data/qilong/miniconda3/envs/relay-sudoku/bin/python}"
GPU="${GPU:-1}"
CKPT="${CKPT:-$ROOT/logs/sudoku_extreme_relay_bptt_steps2_300k_tied_seed1/checkpoints/40-300000.ckpt}"
TRAIN_SEEDS="${TRAIN_SEEDS:-1 2 3}"
TRAIN_SIGMA="${TRAIN_SIGMA:-1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$ROOT/logs/gaussian_streaming_lora_sft_tconf2_sigma1_r32_a64_trainseed}"
EVAL_DIR_NAME="${EVAL_DIR_NAME:-matched_evaluation}"

cd "$ROOT"

if [[ ! -f "$CKPT" ]]; then
  echo "Pretrained checkpoint not found: $CKPT" >&2
  exit 1
fi

for train_seed in $TRAIN_SEEDS; do
  train_output="${OUTPUT_PREFIX}${train_seed}"
  adapter="$train_output/adapter_final.pt"
  eval_output="$train_output/$EVAL_DIR_NAME"

  if [[ ! -f "$adapter" ]]; then
    echo "[train seed $train_seed] adapter not found: $adapter" >&2
    echo "Run train_3seeds.sh first." >&2
    exit 1
  fi

  echo "[train seed $train_seed] evaluating at hidden sigma=0 and 1"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
    sudoku/experiments/streaming_lora_sft/evaluate.py \
    --checkpoint "$CKPT" \
    --adapter "$adapter" \
    --adapter-method gaussian_streaming_lora_sft \
    --expected-training-hidden-noise-sigma "$TRAIN_SIGMA" \
    --output "$eval_output" \
    --weights ema \
    --dev-size 2000 \
    --eval-seeds 1 2 3 \
    --sigmas 0 1 \
    --confidence-temperature 2 \
    --token-temperature 0.3 \
    --threshold 0.15 \
    --group-size 8 \
    --max-rollout-steps 64 \
    --eval-prompt-batch 32 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --precision bf16 \
    --device cuda:0
done

"$PY" sudoku/experiments/gaussian_streaming_lora_sft/summarize_3seeds.py \
  --output-prefix "$OUTPUT_PREFIX" \
  --evaluation-directory "$EVAL_DIR_NAME" \
  --train-seeds $TRAIN_SEEDS \
  --eval-seeds 1 2 3 \
  --sigmas 0 1
