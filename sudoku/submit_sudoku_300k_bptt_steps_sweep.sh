#!/usr/bin/env bash
# BPTT unroll-length ablation for sudoku_extreme: train the loopholing+PUMA
# recipe at num_steps in {3, 4, 8} for both untied and tied vocab embeddings,
# 300k steps each, with validation completely disabled (pure training only).
#
# Job names match the requested:
#   sudoku_extreme_loopholing_puma_bptt_steps<T>_300k_<tying>
# i.e. they extend the existing `_bptt_steps2_300k_<tying>` jobs from
# submit_sudoku_300k_sweep.sh to T = 3, 4, 8. Single seed only (Hydra
# default seed=1, untagged) to match the original eight runs in
# wandb.ai/ilm-extensions/BPTT-sudoku.
#
# Submission order: outer loop over T (3 → 4 → 8), inner loop over tying
# (untied → tied), so all jobs for a given T enter the queue before the
# next T value.
#
# Validation disabled via:
#   ++trainer.limit_val_batches=0    # Lightning treats this as "no val"
#   ++trainer.num_sanity_val_steps=0 # skip pre-train sanity check
# This silences the val_threshold_sweep callback (its hooks only fire
# during validation) and prevents any val/* metric from being logged.
# checkpoint_monitor (which monitors val/lm/accumulated_loss) becomes a
# no-op; periodic snapshotting via checkpoint_every_n_steps_with_thinning
# is unaffected so checkpoints continue to roll over.
#
# Usage (from repo root):
#   ./submit_sudoku_300k_bptt_steps_sweep.sh                      # 6 jobs (T=3,4,8 × 2 tyings)
#   DO=print ./submit_sudoku_300k_bptt_steps_sweep.sh             # dry-run
#   BPTT_STEPS="3 4" ./submit_sudoku_300k_bptt_steps_sweep.sh     # subset of T
#   ONLY_TYINGS="untied" ./submit_sudoku_300k_bptt_steps_sweep.sh # subset of tyings
#
# Memory note: BPTT activation memory grows ~linearly with num_steps. T=8
# at batch_size=512 may push past 80GB; if it OOMs, drop BATCH_SIZE or
# bump SLURM_CONSTRAIN_PUMA to a larger-VRAM partition.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DO="${DO:-submit}"
BPTT_STEPS="${BPTT_STEPS:-3 4 8}"
TYINGS="${TYINGS:-untied tied}"

BATCH_SIZE="${BATCH_SIZE:-512}"
TRAIN_COMPILE="${TRAIN_COMPILE:-false}"
TRAIN_PRECISION="${TRAIN_PRECISION:-bf16-mixed}"
HARDWARE="${HARDWARE:-1_node_1_gpu}"
SLURM_TIME="${SLURM_TIME:-36:00:00}"
USE_JOB_NAME_AS_ID="${USE_JOB_NAME_AS_ID:-false}"
SLURM_RESERVATION="${SLURM_RESERVATION:-dhruvesh}"
SLURM_CONSTRAIN_PUMA="${SLURM_CONSTRAIN_PUMA:-vram80,bf16}"

TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-300000}"
TAG_SWEEP="${TAG_SWEEP:-sudoku_extreme_300k_bptt_steps_ablation}"

ONLY_TYINGS="${ONLY_TYINGS:-}"
SKIP_EXISTING_LOGDIR="${SKIP_EXISTING_LOGDIR:-0}"
LOG_DIR_ROOT="${LOG_DIR_ROOT:-logs}"

_in_filter() {
  local needle="$1"
  local haystack="$2"
  if [[ -z "${haystack// }" ]]; then
    return 0
  fi
  for tok in ${haystack}; do
    if [[ "${tok}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

_submit_one() {
  local job_name=$1
  local embed_tying=$2
  local bptt_steps=$3
  local tie_extra=$4  # either "" or "++model.tie_embeddings=true"

  if [[ "${SKIP_EXISTING_LOGDIR}" == "1" && -d "${LOG_DIR_ROOT}/${job_name}" ]]; then
    echo "[skip] log dir exists: ${LOG_DIR_ROOT}/${job_name}"
    return 0
  fi

  local extra_args=()
  if [[ -n "${tie_extra}" ]]; then
    extra_args+=("${tie_extra}")
  fi

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
    "trainer.max_steps=${TRAIN_MAX_STEPS}" \
    "++trainer.limit_val_batches=0" \
    "++trainer.num_sanity_val_steps=0" \
    "loss.with_loopholing=true" \
    "loss.stop_grad_h_s=false" \
    "loss.num_steps=${bptt_steps}" \
    "predictor.with_loopholing=true" \
    "+tags.sweep=${TAG_SWEEP}" \
    "+tags.embed_tying=${embed_tying}" \
    "+tags.objective=puma_loopholing_bptt" \
    "+tags.bptt_steps=${bptt_steps}" \
    "${extra_args[@]}"
}

submitted=0
skipped=0
for T in ${BPTT_STEPS}; do
  for tying in ${TYINGS}; do
    if ! _in_filter "${tying}" "${ONLY_TYINGS}"; then
      skipped=$((skipped + 1)); continue
    fi
    job_name="sudoku_extreme_loopholing_puma_bptt_steps${T}_300k_${tying}"
    tie_extra=""
    if [[ "${tying}" == "tied" ]]; then
      tie_extra="++model.tie_embeddings=true"
    fi
    submitted=$((submitted + 1))
    echo "[${submitted}] ${job_name}  (T=${T} tying=${tying})"
    _submit_one "${job_name}" "${tying}" "${T}" "${tie_extra}"
  done
done

echo ""
echo "Done. submitted=${submitted}  skipped=${skipped}  (DO=${DO}, BPTT_STEPS='${BPTT_STEPS}', TYINGS='${TYINGS}')"
