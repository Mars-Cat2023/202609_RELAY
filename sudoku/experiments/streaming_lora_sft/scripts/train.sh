#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/nvme-data2/qilong/202609_RELAY}"
PY="${PY:-/data/qilong/miniconda3/envs/relay-sudoku/bin/python}"
GPU="${GPU:-1}"
CKPT="${CKPT:-$ROOT/logs/sudoku_extreme_relay_bptt_steps2_300k_tied_seed1/checkpoints/40-300000.ckpt}"
OUTPUT="${OUTPUT:-$ROOT/logs/streaming_lora_sft_tconf2_r32_a64_trainseed1}"

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" sudoku/experiments/streaming_lora_sft/train.py \
  --checkpoint "$CKPT" \
  --output "$OUTPUT" \
  --weights ema \
  --train-size 5000 \
  --dev-size 2000 \
  --completed-trajectories 5000 \
  --streaming-batch 32 \
  --max-optimizer-steps 100000 \
  --confidence-temperature 2 \
  --threshold 0.15 \
  --learning-rate 1e-5 \
  --warmup-steps 50 \
  --weight-decay 0 \
  --max-grad-norm 1 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --seed 1 \
  --precision bf16 \
  --device cuda:0 \
  --save-every-completed 1000
