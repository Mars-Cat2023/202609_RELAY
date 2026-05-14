#!/usr/bin/env bash
# Dump Sudoku decode trajectories for an MLM-uniform vs Relay-BPTT checkpoint pair, then pair.
# Run from repository root, or from anywhere (script cds to repo root).
#
# Required env:
#   CKPT_MLM    - path to Lightning checkpoint (e.g. logs/.../last.ckpt)
#   CKPT_RELAY  - path to Lightning checkpoint for the relay BPTT run
# Optional env:
#   SEED=1
#   LIMIT_VAL_BATCHES=20
#   MAX_EXAMPLES=200
#   OUT_DIR     - default: visualization/outputs

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"

: "${CKPT_MLM:?Set CKPT_MLM to your MLM checkpoint path}"
: "${CKPT_RELAY:?Set CKPT_RELAY to your relay BPTT checkpoint path}"

SEED="${SEED:-1}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
OUT_DIR="${OUT_DIR:-$ROOT/visualization/outputs}"
mkdir -p "$OUT_DIR"

MLM_JSON="$OUT_DIR/trajectories_sudoku_mlm.jsonl"
RELAY_JSON="$OUT_DIR/trajectories_sudoku_relay_bptt.jsonl"
PAIRED_JSON="$OUT_DIR/paired_sudoku_mlm_vs_relay.jsonl"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "Writing dumps to $OUT_DIR"

python -m relay.dump_decode_trajectories \
  experiment=sudoku_extreme_mlm_uniform \
  +generation.ckpt_path="$CKPT_MLM" \
  seed="$SEED" \
  per_device_batch_size=1 \
  global_batch_size=1 \
  +dump.output_path="$MLM_JSON" \
  +dump.max_examples="$MAX_EXAMPLES" \
  +dump.limit_val_batches="$LIMIT_VAL_BATCHES"

python -m relay.dump_decode_trajectories \
  experiment=sudoku_extreme_relay_bptt \
  +generation.ckpt_path="$CKPT_RELAY" \
  seed="$SEED" \
  per_device_batch_size=1 \
  global_batch_size=1 \
  +dump.output_path="$RELAY_JSON" \
  +dump.max_examples="$MAX_EXAMPLES" \
  +dump.limit_val_batches="$LIMIT_VAL_BATCHES"

python -m relay.pair_trajectory_dumps \
  --mlm "$MLM_JSON" \
  --bptt "$RELAY_JSON" \
  --out "$PAIRED_JSON" \
  --only-bptt-wins

echo "Done."
echo "  MLM:    $MLM_JSON"
echo "  RELAY:  $RELAY_JSON"
echo "  Paired: $PAIRED_JSON"
