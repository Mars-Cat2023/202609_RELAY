#!/usr/bin/env bash
set -euo pipefail

# Train three independent one-step Streaming LoRA-SFT adapters sequentially.
# Override GPU, CKPT, PY, SEEDS, or OUTPUT_PREFIX from the shell if needed.

ROOT="${ROOT:-/nvme-data2/qilong/202609_RELAY}"
PY="${PY:-/data/qilong/miniconda3/envs/relay-sudoku/bin/python}"
GPU="${GPU:-1}"
CKPT="${CKPT:-$ROOT/logs/sudoku_extreme_relay_bptt_steps2_300k_tied_seed1/checkpoints/40-300000.ckpt}"
SEEDS="${SEEDS:-1 2 3}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$ROOT/logs/streaming_lora_sft_tconf2_r32_a64_trainseed}"

cd "$ROOT"

if [[ ! -f "$CKPT" ]]; then
  echo "Pretrained checkpoint not found: $CKPT" >&2
  exit 1
fi

for seed in $SEEDS; do
  output="${OUTPUT_PREFIX}${seed}"
  final_adapter="$output/adapter_final.pt"

  if [[ -f "$final_adapter" ]]; then
    echo "[train seed $seed] complete adapter already exists; skipping: $final_adapter"
    continue
  fi

  resume_args=()
  if [[ -d "$output" ]]; then
    latest_file="$output/latest_checkpoint.txt"
    if [[ ! -s "$latest_file" ]]; then
      echo "[train seed $seed] output exists but has no resumable checkpoint: $output" >&2
      echo "Move that incomplete directory aside, or set a different OUTPUT_PREFIX." >&2
      exit 1
    fi
    resume_checkpoint="$(<"$latest_file")"
    if [[ ! -f "$resume_checkpoint" ]]; then
      echo "[train seed $seed] resume checkpoint not found: $resume_checkpoint" >&2
      exit 1
    fi
    resume_args=(--resume "$resume_checkpoint")
    echo "[train seed $seed] resuming from $resume_checkpoint"
  else
    echo "[train seed $seed] starting a new run at $output"
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
    sudoku/experiments/streaming_lora_sft/train.py \
    --checkpoint "$CKPT" \
    --output "$output" \
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
    --seed "$seed" \
    --precision bf16 \
    --device cuda:0 \
    --save-every-completed 1000 \
    "${resume_args[@]}"
done

echo "All requested training seeds are complete."
