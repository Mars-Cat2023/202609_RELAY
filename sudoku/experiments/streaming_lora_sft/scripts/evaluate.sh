#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/nvme-data2/qilong/202609_RELAY}"
PY="${PY:-/data/qilong/miniconda3/envs/relay-sudoku/bin/python}"
GPU="${GPU:-1}"
CKPT="${CKPT:-$ROOT/logs/sudoku_extreme_relay_bptt_steps2_300k_tied_seed1/checkpoints/40-300000.ckpt}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-$ROOT/logs/streaming_lora_sft_tconf2_r32_a64_trainseed1}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$TRAIN_OUTPUT/matched_evaluation}"

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" sudoku/experiments/streaming_lora_sft/evaluate.py \
  --checkpoint "$CKPT" \
  --adapter "$TRAIN_OUTPUT/adapter_final.pt" \
  --output "$EVAL_OUTPUT" \
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
