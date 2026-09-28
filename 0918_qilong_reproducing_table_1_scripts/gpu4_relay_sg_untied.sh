#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID=4
OBJECTIVE=relay_sg
TYING=untied
EXPERIMENT=sudoku_extreme_relay_bptt
JOB_STEM=sudoku_extreme_relay_sg_300k_untied
EXTRA_ARGS=(loss.with_relay=true loss.stop_grad_h_s=true predictor.with_relay=true)
CURRENT_SEED=not_started

trap 's=$?; echo "[$(date --iso-8601=seconds)] ERROR: GPU ${GPU_ID} ${OBJECTIVE}/${TYING}, seed ${CURRENT_SEED}, exit ${s}." >&2; exit "$s"' ERR

source /data/qilong/miniconda3/etc/profile.d/conda.sh
conda activate relay-sudoku

export PROJECT_ROOT=/data/qilong/202609_RELAY/sudoku
export DATA_DIR=/data/qilong/202609_RELAY/data
export HF_HOME=/data/qilong/202609_RELAY/hf_home
export LOG_DIR=/data/qilong/202609_RELAY/logs
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_ENTITY=ah15300047507-university-of-michigan
export WANDB_PROJECT=RELAY-Reproducing-Table-1
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p "$DATA_DIR" "$HF_HOME" "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "[$(date --iso-8601=seconds)] START: GPU ${GPU_ID}, ${OBJECTIVE}/${TYING}, seeds 1 2 3."
echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
nvidia-smi --id="$GPU_ID" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

for SEED in 1 2 3; do
  CURRENT_SEED=$SEED
  JOB_NAME="${JOB_STEM}_seed${SEED}"
  echo "[$(date --iso-8601=seconds)] RUN START: ${JOB_NAME}."
  xlm job_type=train job_name="$JOB_NAME" experiment="$EXPERIMENT" seed="$SEED" \
    "${EXTRA_ARGS[@]}" ++trainer.precision=bf16-mixed trainer.max_steps=300000 \
    trainer.val_check_interval=5000 per_device_batch_size=512 global_batch_size=512 \
    +loggers.wandb.entity="$WANDB_ENTITY" loggers.wandb.project="$WANDB_PROJECT" \
    +tags.sweep=sudoku_extreme_300k +tags.embed_tying="$TYING" \
    +tags.objective="$OBJECTIVE" +tags.seed="$SEED"
  echo "[$(date --iso-8601=seconds)] RUN COMPLETE: ${JOB_NAME}."
done

CURRENT_SEED=complete
echo "[$(date --iso-8601=seconds)] COMPLETE: GPU ${GPU_ID}, ${OBJECTIVE}/${TYING}, all seeds finished."
