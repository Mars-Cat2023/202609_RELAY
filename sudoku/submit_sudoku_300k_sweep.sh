#!/usr/bin/env bash
# Submit the eight Sudoku Extreme 300k training jobs described in SUDOKU_COMMANDS.md.
# Common options are set once; per-job args add experiment-specific overrides.
#
# Usage (from repository root):
#   ./submit_sudoku_300k_sweep.sh
# Dry-run (print SLURM scripts, do not submit):
#   DO=print ./submit_sudoku_300k_sweep.sh
# Optional: override the shared W&B `sweep` tag (default: sudoku_extreme_300k):
#   TAG_SWEEP=my_group ./submit_sudoku_300k_sweep.sh
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
# MLM uniform uses 40GB-class GPUs; PUMA / loopholing use 80GB in the hand-written commands.
SLURM_CONSTRAIN_MLM="${SLURM_CONSTRAIN_MLM:-vram40,bf16}"
SLURM_CONSTRAIN_PUMA="${SLURM_CONSTRAIN_PUMA:-vram80,bf16}"
# W&B: added as Hydra `tags` entries → run tags like `sweep=sudoku_extreme_300k` (see
# `xlm-core/.../loggers/wandb.yaml`, resolver `dict_to_list`).
#   objective: mlm_uniform | puma_no_loopholing | puma_loopholing_stop_grad | puma_loopholing_bptt
#   embed_tying: untied | tied
TAG_SWEEP="${TAG_SWEEP:-sudoku_extreme_300k}"

# Invoke submit_train.py with shared outer args; inner Hydra overrides go after ---.
# Usage: _submit <job_name> <experiment> <slurm_constrain> <embed_tying: untied|tied> <objective tag> [extra inner args...]
#   <objective tag> is a short name for the training recipe (for W&B filters).
_submit() {
  local job_name=$1
  local experiment=$2
  local slurm_constrain=$3
  local embed_tying=$4
  local objective=$5
  shift 5
  python slurm_scripts/submit_train.py \
    "do=${DO}" \
    "job_name=${job_name}" \
    "train.experiment=${experiment}" \
    "train.batch_size=${BATCH_SIZE}" \
    "train.compile=${TRAIN_COMPILE}" \
    "train.precision=${TRAIN_PRECISION}" \
    "hardware=${HARDWARE}" \
    "slurm.constrain=\"${slurm_constrain}\"" \
    "slurm.time=${SLURM_TIME}" \
    "use_job_name_as_id=${USE_JOB_NAME_AS_ID}" \
    "++slurm.reservation=${SLURM_RESERVATION}" \
    "---" \
    "trainer.max_steps=${TRAIN_MAX_STEPS}" \
    "trainer.val_check_interval=${TRAIN_VAL_INTERVAL}" \
    "+tags.sweep=${TAG_SWEEP}" \
    "+tags.embed_tying=${embed_tying}" \
    "+tags.objective=${objective}" \
    "$@"
}

# --- Untied (no weight tying) ---
_submit "sudoku_extreme_mlm_uniform_300k_untied" "sudoku_extreme_mlm_uniform" \
  "${SLURM_CONSTRAIN_MLM}" "untied" "mlm_uniform"

_submit "sudoku_extreme_puma_only_300k_untied" "sudoku_extreme_loopholing_bptt_puma" \
  "${SLURM_CONSTRAIN_PUMA}" "untied" "puma_no_loopholing" \
  "model=rotary_transformer_xtiny" \
  "loss.with_loopholing=false" \
  "loss.stop_grad_h_s=true" \
  "predictor.with_loopholing=false"

_submit "sudoku_extreme_loopholing_puma_no_bptt_300k_untied" "sudoku_extreme_loopholing_bptt_puma" \
  "${SLURM_CONSTRAIN_PUMA}" "untied" "puma_loopholing_stop_grad" \
  "loss.with_loopholing=true" \
  "loss.stop_grad_h_s=true" \
  "predictor.with_loopholing=true"

_submit "sudoku_extreme_loopholing_puma_bptt_steps2_300k_untied" "sudoku_extreme_loopholing_bptt_puma" \
  "${SLURM_CONSTRAIN_PUMA}" "untied" "puma_loopholing_bptt" \
  "loss.with_loopholing=true" \
  "loss.stop_grad_h_s=false" \
  "loss.num_steps=2" \
  "predictor.with_loopholing=true"

# --- Tied vocab embeddings / unembeddings ---
_submit "sudoku_extreme_mlm_uniform_300k_tied" "sudoku_extreme_mlm_uniform" \
  "${SLURM_CONSTRAIN_MLM}" "tied" "mlm_uniform" \
  "++model.tie_embeddings=true"

_submit "sudoku_extreme_puma_only_300k_tied" "sudoku_extreme_loopholing_bptt_puma" \
  "${SLURM_CONSTRAIN_PUMA}" "tied" "puma_no_loopholing" \
  "model=rotary_transformer_xtiny" \
  "++model.tie_embeddings=true" \
  "loss.with_loopholing=false" \
  "loss.stop_grad_h_s=true" \
  "predictor.with_loopholing=false"

_submit "sudoku_extreme_loopholing_puma_no_bptt_300k_tied" "sudoku_extreme_loopholing_bptt_puma" \
  "${SLURM_CONSTRAIN_PUMA}" "tied" "puma_loopholing_stop_grad" \
  "++model.tie_embeddings=true" \
  "loss.with_loopholing=true" \
  "loss.stop_grad_h_s=true" \
  "predictor.with_loopholing=true"

_submit "sudoku_extreme_loopholing_puma_bptt_steps2_300k_tied" "sudoku_extreme_loopholing_bptt_puma" \
  "${SLURM_CONSTRAIN_PUMA}" "tied" "puma_loopholing_bptt" \
  "++model.tie_embeddings=true" \
  "loss.with_loopholing=true" \
  "loss.stop_grad_h_s=false" \
  "loss.num_steps=2" \
  "predictor.with_loopholing=true"
