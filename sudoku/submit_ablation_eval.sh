#!/usr/bin/env bash
# Submit evaluation jobs for the three model types in the sudoku ablation:
#   1. MLM uniform  (mlm)
#   2. Loopholing no-BPTT PUMA  (no_bptt)
#   3. Loopholing BPTT PUMA  (bptt)
#
# Each job loads a single checkpoint (default: 100k steps) and runs
# ValidationThresholdSweepCallback across thresholds {0.05, 0.10, 0.15, 0.20, 0.25},
# logging val_sweep/t_<threshold>/{exact_match,token_accuracy,rollout_steps} to W&B.
#
# Usage:
#   bash submit_ablation_eval.sh                        # submit with _wandb suffix (default)
#   JOB_SUFFIX="" bash submit_ablation_eval.sh          # no suffix (overwrites previous job logs)
#   CHECKPOINT=6-50000.ckpt bash submit_ablation_eval.sh  # use 50k-step checkpoints
#   DO=print bash submit_ablation_eval.sh               # dry-run (print scripts without submitting)
#   LIMIT_VAL_BATCHES=null bash submit_ablation_eval.sh # use full validation set

set -euo pipefail
cd "$(dirname "$0")"

if [ -f "venv-double-backprop/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "venv-double-backprop/bin/activate"
fi

# ── Tuneable parameters ────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-40-300000.ckpt}"     # must exist in all 3 checkpoint dirs
STEPS_TAG="${STEPS_TAG:-300k}"                 # label used in job/run names
JOB_SUFFIX="${JOB_SUFFIX:-_wandb}"             # suffix appended to job name (use "" to omit)
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-500}"  # 500 × 64 ≈ 32 k examples (use null for full set)
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-08:00:00}"
EVAL_CONSTRAINT="${EVAL_CONSTRAINT:-vram40,bf16}"
WANDB_GROUP="${WANDB_GROUP:-ablation_threshold_sweep_${STEPS_TAG}}"
DO="${DO:-submit}"  # "submit" or "print"

# ── W&B API key ────────────────────────────────────────────────────────────────
# The sbatch job runs in a clean environment and won't inherit WANDB_API_KEY
# unless we bake it in explicitly. Read it from ~/.netrc (set by `wandb login`).
if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api\.wandb\.ai/{found=1} found && /password/{print $2; exit}' \
    ~/.netrc 2>/dev/null || true)
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WARNING: WANDB_API_KEY not found in environment or ~/.netrc." >&2
  echo "         W&B logging will likely be disabled. Run 'wandb login' first." >&2
fi

LOGS_DIR="logs"
THRESHOLDS_LIST="[0.05,0.10,0.15,0.20,0.25]"

# ── Model definitions ──────────────────────────────────────────────────────────
declare -a MODEL_TAGS=("mlm" "no_bptt" "bptt")

declare -A MODEL_EXPERIMENTS=(
  ["mlm"]="sudoku_extreme_mlm_uniform"
  ["no_bptt"]="sudoku_extreme_loopholing_bptt_puma"
  ["bptt"]="sudoku_extreme_loopholing_bptt_puma"
)

declare -A MODEL_CKPT_DIRS=(
  ["mlm"]="${LOGS_DIR}/puzzle_sudoku_extreme_mlm_uniform_300k_steps_ABLATION/checkpoints"
  ["no_bptt"]="${LOGS_DIR}/puzzle_sudoku_extreme_loopholing_no_bptt_puma_300k_steps_ABLATION/checkpoints"
  ["bptt"]="${LOGS_DIR}/puzzle_sudoku_extreme_loopholing_bptt_puma_300k_steps_ABLATION/checkpoints"
)

# ── Submission function ────────────────────────────────────────────────────────
submit_ablation_eval() {
  local model_tag="$1"
  local experiment="${MODEL_EXPERIMENTS[$model_tag]}"
  local ckpt_dir="${MODEL_CKPT_DIRS[$model_tag]}"
  local checkpoint_path="${ckpt_dir}/${CHECKPOINT}"
  local job_name="ablation_eval_${model_tag}_${STEPS_TAG}${JOB_SUFFIX}"

  if [ ! -f "${checkpoint_path}" ]; then
    echo "ERROR: checkpoint not found: ${checkpoint_path}" >&2
    exit 1
  fi

  echo "──────────────────────────────────────────────────────────────────────────"
  echo "Submitting eval: ${job_name}"
  echo "  experiment : ${experiment}"
  echo "  checkpoint : ${checkpoint_path}"
  echo "  val batches: ${LIMIT_VAL_BATCHES}"
  echo "  W&B group  : ${WANDB_GROUP}"

  # Inner Hydra overrides (passed after ---):
  #   • Increase validation size beyond the 100-batch training default
  #   • Tag runs in W&B with a shared group for easy filtering
  #   • For MLM: add ValidationThresholdSweepCallback (loopholing configs already include it)
  local inner_args=(
    "++trainer.limit_val_batches=${LIMIT_VAL_BATCHES}"
    "+loggers.wandb.group=${WANDB_GROUP}"
  )

  if [ "${model_tag}" = "mlm" ]; then
    inner_args+=(
      "+callbacks.val_threshold_sweep._target_=doublebackprop.callbacks.ValidationThresholdSweepCallback"
      "+callbacks.val_threshold_sweep.thresholds=${THRESHOLDS_LIST}"
      "+callbacks.val_threshold_sweep.metric_prefix=val_sweep"
      "+callbacks.val_threshold_sweep.prediction_dataloader_substring=prediction"
    )
  elif [ "${model_tag}" = "no_bptt" ]; then
    inner_args+=(
      "loss.stop_grad_h_s=true"
    )
  fi

  python slurm_scripts/submit_eval.py \
    "do=${DO}" \
    "job_name=${job_name}" \
    "experiment=${experiment}" \
    "hardware=1_node_1_gpu" \
    "slurm.constraint='${EVAL_CONSTRAINT}'" \
    "slurm.time=${EVAL_TIME_LIMIT}" \
    "eval.checkpoint_path=${checkpoint_path}" \
    "eval.split=validation" \
    "eval.dms_to_remove=[val.lm]" \
    "use_job_name_as_id=true" \
    ${WANDB_API_KEY:+"+env.WANDB_API_KEY=${WANDB_API_KEY}"} \
    "---" \
    "${inner_args[@]}"
}

# ── Submit all three models ────────────────────────────────────────────────────
for tag in "${MODEL_TAGS[@]}"; do
  submit_ablation_eval "${tag}"
done

echo ""
echo "All jobs submitted. Monitor with:  squeue --me"
echo ""
echo "W&B runs will appear in group '${WANDB_GROUP}' at:"
echo "  https://wandb.ai/ilm-extensions/BPTT-puzzles"
echo ""
echo "TensorBoard logs at:  logs/ablation_eval_{mlm,no_bptt,bptt}_${STEPS_TAG}${JOB_SUFFIX}/tensorboard/"
echo ""
echo "Once all jobs complete, run:  jupyter notebook plotting_scripts/plot_ablation_threshold_sweep.ipynb"
