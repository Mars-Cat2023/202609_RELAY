#!/usr/bin/env bash
# Submit Sudoku Extreme 300k MLP/CAB loopholing PUMA+BPTT jobs.
#
# Usage (from repository root):
#   ./submit_sudoku_mlp_cab_300k_sweep.sh
# Dry-run (print SLURM scripts, do not submit):
#   DO=print ./submit_sudoku_mlp_cab_300k_sweep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DO="${DO:-submit}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TRAIN_COMPILE="${TRAIN_COMPILE:-false}"
TRAIN_PRECISION="${TRAIN_PRECISION:-bf16-mixed}"
HARDWARE="${HARDWARE:-1_node_1_gpu}"
SLURM_TIME="${SLURM_TIME:-36:00:00}"
USE_JOB_NAME_AS_ID="${USE_JOB_NAME_AS_ID:-false}"
SLURM_RESERVATION="${SLURM_RESERVATION:-dhruvesh}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-300000}"
TRAIN_VAL_INTERVAL="${TRAIN_VAL_INTERVAL:-5000}"
SLURM_CONSTRAIN_PUMA="${SLURM_CONSTRAIN_PUMA:-vram80,bf16}"
TAG_SWEEP="${TAG_SWEEP:-sudoku_extreme_mlp_cab_300k}"

_submit() {
  local job_name=$1
  local model_name=$2
  local embed_tying=$3
  local objective=$4
  shift 4
  python slurm_scripts/submit_train.py \
    "do=${DO}" \
    "job_name=${job_name}" \
    "train.experiment=sudoku_extreme_loopholing_bptt_puma" \
    "train.batch_size=${BATCH_SIZE}" \
    "train.compile=${TRAIN_COMPILE}" \
    "train.precision=${TRAIN_PRECISION}" \
    "hardware=${HARDWARE}" \
    "slurm.constrain=\"${SLURM_CONSTRAIN_PUMA}\"" \
    "slurm.time=${SLURM_TIME}" \
    "use_job_name_as_id=${USE_JOB_NAME_AS_ID}" \
    "++slurm.reservation=${SLURM_RESERVATION}" \
    "---" \
    "model=${model_name}" \
    "trainer.max_steps=${TRAIN_MAX_STEPS}" \
    "trainer.val_check_interval=${TRAIN_VAL_INTERVAL}" \
    "+tags.sweep=${TAG_SWEEP}" \
    "+tags.embed_tying=${embed_tying}" \
    "+tags.objective=${objective}" \
    "loss.with_loopholing=true" \
    "loss.stop_grad_h_s=false" \
    "loss.num_steps=2" \
    "predictor.with_loopholing=true" \
    "$@"
}

# --- Untied (no weight tying) ---
_submit "sudoku_extreme_loopholing_mlp_puma_bptt_steps2_300k_untied" \
  "rotary_transformer_xtiny_loopholing_mlp" \
  "untied" \
  "puma_loopholing_bptt_mlp"

_submit "sudoku_extreme_loopholing_cab_puma_bptt_steps2_300k_untied" \
  "rotary_transformer_xtiny_loopholing_cab" \
  "untied" \
  "puma_loopholing_bptt_cab"

# --- Tied vocab embeddings / unembeddings ---
_submit "sudoku_extreme_loopholing_mlp_puma_bptt_steps2_300k_tied" \
  "rotary_transformer_xtiny_loopholing_mlp" \
  "tied" \
  "puma_loopholing_bptt_mlp" \
  "++model.tie_embeddings=true"

_submit "sudoku_extreme_loopholing_cab_puma_bptt_steps2_300k_tied" \
  "rotary_transformer_xtiny_loopholing_cab" \
  "tied" \
  "puma_loopholing_bptt_cab" \
  "++model.tie_embeddings=true"
