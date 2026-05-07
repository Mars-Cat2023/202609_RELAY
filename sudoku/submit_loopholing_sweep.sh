#!/usr/bin/env bash
# Sweep for Loopholing + BPTT (2 steps) + PUMA:
# 1) static threshold training (mu=0.15, sigma=0.0)
# 2) dynamic threshold training (mu=0.15, sigma=0.1)
#
# Optional eval sweep (when checkpoints are available):
# predictor.threshold in {0.05, 0.10, 0.15, 0.20, 0.25} on test split.

set -euo pipefail
cd "$(dirname "$0")"

if [ -f "venv-double-backprop/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "venv-double-backprop/bin/activate"
fi

EXPERIMENT="sudoku_extreme_loopholing_bptt_puma"
BATCH_SIZE=512
PRECISION="bf16-mixed"
TRAIN_TIME_LIMIT="36:00:00"
TRAIN_CONSTRAINT="vram80,bf16"
TRAINER_MAX_STEPS=300000
MU_THRESHOLD="0.15"
STOP_GRAD_H_S="false"
NUM_STEPS=2

# Set these before running eval sweep:
#   CKPT_STATIC=/path/to/static/model.ckpt
#   CKPT_DYNAMIC=/path/to/dynamic/model.ckpt
CKPT_STATIC="${CKPT_STATIC:-}"
CKPT_DYNAMIC="${CKPT_DYNAMIC:-}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-08:00:00}"
EVAL_CONSTRAINT="${EVAL_CONSTRAINT:-vram40,bf16}"
EVAL_THRESHOLDS=("0.05" "0.10" "0.15" "0.20" "0.25")

submit_train_variant() {
  local variant="$1"
  local sigma="$2"
  local job_name="sudoku_extreme_loopholing_puma_mu_0p15_${variant}"

  echo "Submitting train: ${job_name} (sigma=${sigma})"
  python slurm_scripts/submit_train.py \
    "do=submit" \
    "job_name=${job_name}" \
    "train.experiment=${EXPERIMENT}" \
    "train.batch_size=${BATCH_SIZE}" \
    "train.compile=false" \
    "train.precision=${PRECISION}" \
    "hardware=1_node_1_gpu" \
    "slurm.constrain=\"${TRAIN_CONSTRAINT}\"" \
    "slurm.time=${TRAIN_TIME_LIMIT}" \
    "use_job_name_as_id=false" \
    "---" \
    "++loss.use_model_confidence=true" \
    "++loss.stop_grad_h_s=${STOP_GRAD_H_S}" \
    "++loss.num_steps=${NUM_STEPS}" \
    "++loss.threshold_sampling_sigma=${sigma}" \
    "++predictor.threshold=${MU_THRESHOLD}" \
    "++global_batch_size=${BATCH_SIZE}" \
    "++trainer.max_steps=${TRAINER_MAX_STEPS}"
}

submit_eval_sweep() {
  local model_tag="$1"
  local checkpoint_path="$2"

  if [ -z "${checkpoint_path}" ]; then
    echo "Skipping eval for ${model_tag}: checkpoint path not provided."
    return
  fi

  for threshold in "${EVAL_THRESHOLDS[@]}"; do
    local threshold_tag="${threshold/./p}"
    local eval_job_name="sudoku_extreme_loopholing_puma_${model_tag}_eval_t_${threshold_tag}"
    echo "Submitting eval: ${eval_job_name} (threshold=${threshold})"
    python slurm_scripts/submit_eval.py \
      "do=submit" \
      "job_name=${eval_job_name}" \
      "experiment=${EXPERIMENT}" \
      "hardware=1_node_1_gpu" \
      "slurm.constrain=\"${EVAL_CONSTRAINT}\"" \
      "slurm.time=${EVAL_TIME_LIMIT}" \
      "eval.checkpoint_path=${checkpoint_path}" \
      "eval.split=test" \
      "---" \
      "++predictor.threshold=${threshold}"
  done
}

# Training: baseline static vs dynamic threshold-noise.
submit_train_variant "static_sigma_0p0" "0.0"
submit_train_variant "dynamic_sigma_0p1" "0.1"

# Eval sweep: only runs when CKPT_STATIC / CKPT_DYNAMIC are provided.
submit_eval_sweep "static_sigma_0p0" "${CKPT_STATIC}"
submit_eval_sweep "dynamic_sigma_0p1" "${CKPT_DYNAMIC}"

echo "Done. Monitor with: squeue --me"
