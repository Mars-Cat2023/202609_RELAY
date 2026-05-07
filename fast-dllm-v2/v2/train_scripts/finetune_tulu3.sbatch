#!/bin/bash
#
# Vanilla SFT on Tulu-3-SFT-Mixture (no PUMA / loopholing / BPTT).
# Same launcher as finetune_magicoder_oss.sbatch but sized for the
# 939k-row Tulu mixture: 8 GPUs single-node, 2-day time, no on-save eval.
#
# Submit from a login node, with cwd = Fast-dLLM/v2:
#     cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
#     sbatch train_scripts/finetune_tulu3.sbatch
#
# Prerequisites:
#     python scripts/prep_tulu3_sft.py --out_dir data/tulu3_sft
#
# Overrides via env: NUM_TRAIN_EPOCHS, DATASET_PATH, OUTPUT_DIR,
# LEARNING_RATE, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
# SAVE_STEPS, SAVE_TOTAL_LIMIT, BLOCK_SIZE, etc.
#
# Note: this script intentionally omits all `--evalplus_*` flags so
# nothing eval-related fires during training. Run EvalPlus / Math500 /
# lm-eval-harness on saved checkpoints out-of-band.
#
#SBATCH --job-name=fastdllm_v2_1p5B_tulu3_vanilla
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint=vram80,bf16,ib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# Unity rejects --mem=512GB on 512GB nodes (needs >512GB after OS
# overhead). 480GB stays safely below the cap and is still ample headroom
# for 8x1.5B under DeepSpeed ZeRO-3 (Magicoder ran fine with 256GB on 2 GPUs).
#SBATCH --mem=480GB
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x/%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=brozonoyer_umass_edu

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths + environment (same as finetune_magicoder_oss.sbatch)
# -----------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [ ! -f "${REPO_ROOT}/train_scripts/finetune.py" ] || [ ! -d "${REPO_ROOT}/src/lmflow" ]; then
    echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like Fast-dLLM/v2/." >&2
    exit 1
fi
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-lmflow}
CONDA_ROOT=${CONDA_ROOT:-/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3}
mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-fastdllm_v2_1p5B_tulu3_vanilla}"

if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load cuda/12.1 2>/dev/null \
        || module load cuda/12.4 2>/dev/null \
        || module load cuda/12.6 2>/dev/null \
        || module load cuda 2>/dev/null \
        || true
fi
if [ -z "${CUDA_HOME:-}" ]; then
    for cand in /usr/local/cuda-12.1 /usr/local/cuda-12.4 /usr/local/cuda-12.6 /usr/local/cuda; do
        if [ -x "${cand}/bin/nvcc" ]; then
            export CUDA_HOME=${cand}
            export PATH=${cand}/bin:${PATH}
            export LD_LIBRARY_PATH=${cand}/lib64:${LD_LIBRARY_PATH:-}
            break
        fi
    done
fi
if [ -z "${CUDA_HOME:-}" ] || ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: could not locate a CUDA toolkit (nvcc)." >&2
    exit 1
fi
echo "Using CUDA_HOME=${CUDA_HOME}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export TQDM_MINITERS=1000
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
_local_tmp=${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-no_slurm}}
mkdir -p "${_local_tmp}/triton" "${_local_tmp}/torch_extensions"
export TRITON_CACHE_DIR=${_local_tmp}/triton
export TORCH_EXTENSIONS_DIR=${_local_tmp}/torch_extensions

# -----------------------------------------------------------------------------
# Run config -- 1.5B + Tulu-3-SFT-Mixture (full 939k, 1 epoch on 8 GPUs)
# -----------------------------------------------------------------------------
model_name_or_path=${MODEL_NAME_OR_PATH:-Efficient-Large-Model/Fast_dLLM_v2_1.5B}
dataset_path=${DATASET_PATH:-data/tulu3_sft/train_conversation}
conversation_template=${CONVERSATION_TEMPLATE:-fast_dllm_v2}
truncation_side=${TRUNCATION_SIDE:-left}

case "${model_name_or_path}" in
    *1.5B*|*1p5B*|*1_5B*) size_tag="1p5B" ;;
    *7B*)                 size_tag="7B"   ;;
    *)                    size_tag=$(basename "${model_name_or_path}" | tr -cd 'A-Za-z0-9') ;;
esac

per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-16}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1}
learning_rate=${LEARNING_RATE:-1e-5}
warmup_ratio=${WARMUP_RATIO:-0.03}
lr_scheduler_type=${LR_SCHEDULER_TYPE:-cosine}
max_grad_norm=${MAX_GRAD_NORM:-1.0}
logging_steps=${LOGGING_STEPS:-1}
save_steps=${SAVE_STEPS:-500}
save_total_limit=${SAVE_TOTAL_LIMIT:-0}

# 1 = one padded/truncated example per row (no cross-example packing). Default ON.
disable_group_texts=${DISABLE_GROUP_TEXTS:-1}
block_size=${BLOCK_SIZE:-1024}

# 8-GPU geometry baked into the bs_tag so run names reflect effective batch.
bs_tag="bs${per_device_train_batch_size}x${gradient_accumulation_steps}x8_ep${num_train_epochs}"
run_name=${RUN_NAME:-"vanilla_${size_tag}_tulu3_sft_full_nopack${block_size}_${bs_tag}_lr${learning_rate}_v2"}
output_dir=${OUTPUT_DIR:-output_models/tulu3_sft_${size_tag}/vanilla_nopack${block_size}_${bs_tag}_lr${learning_rate}}
export WANDB_TAGS=${WANDB_TAGS:-"vanilla,${size_tag},tulu3_sft,full,nopack${block_size},${bs_tag},lr${learning_rate},v2"}

deepspeed_args="--master_port=$((10000 + SLURM_JOB_ID % 50000))"
deepspeed_config=${DEEPSPEED_CONFIG:-configs/ds_config_zero3_no_offload.json}

latest_checkpoint=""
if [ -d "${output_dir}" ]; then
    latest_checkpoint=$(find "${output_dir}" -name "checkpoint-*" -type d | sort -V | tail -1)
    [ -n "${latest_checkpoint}" ] && echo "Found latest checkpoint: ${latest_checkpoint}"
fi
resume_arg=""
[ -n "${latest_checkpoint}" ] && resume_arg="--resume_from_checkpoint ${latest_checkpoint}"

echo "Using DeepSpeed config: ${deepspeed_config}"
cmd="deepspeed ${deepspeed_args} \
  train_scripts/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code 0 \
    --truncation_side ${truncation_side} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    ${resume_arg} \
    --conversation_template ${conversation_template} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm ${max_grad_norm} \
    --disable_group_texts ${disable_group_texts} \
    --block_size ${block_size} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --deepspeed ${deepspeed_config} \
    --bf16 \
    --run_name ${run_name} \
    --validation_split_percentage 0 \
    --logging_steps ${logging_steps} \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps ${save_steps} \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --save_total_limit ${save_total_limit} \
    --gradient_checkpointing 1"

echo "===== JOB INFO ====="
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-<not in slurm>}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-<not in slurm>}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-}"
echo "model_name_or_path = ${model_name_or_path}"
echo "dataset_path       = ${dataset_path}"
echo "output_dir         = ${output_dir}"
echo "run_name           = ${run_name}"
echo "WANDB_TAGS         = ${WANDB_TAGS}"
echo "truncation_side    = ${truncation_side}"
echo "disable_group_texts= ${disable_group_texts}  (1 = no packing)"
echo "block_size         = ${block_size}"
echo "save_steps         = ${save_steps}"
echo "save_total_limit   = ${save_total_limit}"
echo "===== COMMAND ====="
echo "$cmd"
echo "==================="
eval $cmd
