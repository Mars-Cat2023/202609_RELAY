#!/usr/bin/env bash
# Dump Sudoku decode trajectories for MLM-uniform vs BPTT+loopholing+PUMA checkpoints, then pair.
# Run from repository root, or from anywhere (script cds to repo root).
#
# Required env:
#   CKPT_MLM   — path to Lightning checkpoint (e.g. logs/.../last.ckpt)
#   CKPT_BPTT  — path to Lightning checkpoint for loopholing BPTT+PUMA run
# Optional env:
#   SEED=1
#   LIMIT_VAL_BATCHES=20
#   MAX_EXAMPLES=200
#   OUT_DIR    — default: visualization/outputs

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"

: "${CKPT_MLM:?Set CKPT_MLM to your MLM checkpoint path}"
: "${CKPT_BPTT:?Set CKPT_BPTT to your BPTT+loopholing+PUMA checkpoint path}"

SEED="${SEED:-1}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
OUT_DIR="${OUT_DIR:-$ROOT/visualization/outputs}"
mkdir -p "$OUT_DIR"

MLM_JSON="$OUT_DIR/trajectories_sudoku_mlm.jsonl"
BPTT_JSON="$OUT_DIR/trajectories_sudoku_bptt_puma.jsonl"
PAIRED_JSON="$OUT_DIR/paired_sudoku_mlm_vs_bptt.jsonl"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "Writing dumps to $OUT_DIR"

python -m doublebackprop.dump_decode_trajectories \
  experiment=sudoku_extreme_mlm_uniform \
  +generation.ckpt_path="$CKPT_MLM" \
  seed="$SEED" \
  per_device_batch_size=1 \
  global_batch_size=1 \
  +dump.output_path="$MLM_JSON" \
  +dump.max_examples="$MAX_EXAMPLES" \
  +dump.limit_val_batches="$LIMIT_VAL_BATCHES"

python -m doublebackprop.dump_decode_trajectories \
  experiment=sudoku_extreme_loopholing_bptt_puma \
  +generation.ckpt_path="$CKPT_BPTT" \
  seed="$SEED" \
  per_device_batch_size=1 \
  global_batch_size=1 \
  +dump.output_path="$BPTT_JSON" \
  +dump.max_examples="$MAX_EXAMPLES" \
  +dump.limit_val_batches="$LIMIT_VAL_BATCHES"

python -m doublebackprop.pair_trajectory_dumps \
  --mlm "$MLM_JSON" \
  --bptt "$BPTT_JSON" \
  --out "$PAIRED_JSON" \
  --only-bptt-wins

echo "Done."
echo "  MLM:    $MLM_JSON"
echo "  BPTT:   $BPTT_JSON"
echo "  Paired: $PAIRED_JSON"
