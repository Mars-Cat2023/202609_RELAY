#!/bin/bash
#
# Vanilla alpaca finetuning of Fast-dLLM v2 on Unity (no PUMA, no loopholing,
# no BPTT). Sister script to train_scripts/finetune_alpaca_bptt_puma.sbatch
# and equivalent to train_scripts/finetune_alpaca.sh wrapped for slurm.
#
# Submit from a login node, with cwd = Fast-dLLM/v2:
#     cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
#     sbatch train_scripts/finetune_alpaca.sbatch
#
# To use a slurm reservation:
#     sbatch --reservation=<name> train_scripts/finetune_alpaca.sbatch
#
#SBATCH --job-name=fastdllm_v2_vanilla
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint=vram80,bf16,ib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x/%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=brozonoyer_umass_edu

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths + environment
# -----------------------------------------------------------------------------
# Anchor at the v2/ repo root. Under slurm BASH_SOURCE points at the spool
# copy of this script — useless. Prefer SLURM_SUBMIT_DIR (the dir the user
# ran `sbatch` from) and assume that dir IS Fast-dLLM/v2. For interactive
# runs without slurm, fall back to BASH_SOURCE.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [ ! -f "${REPO_ROOT}/train_scripts/finetune.py" ] || [ ! -d "${REPO_ROOT}/src/lmflow" ]; then
    echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like Fast-dLLM/v2/." >&2
    echo "       Submit from Fast-dLLM/v2 with: sbatch train_scripts/finetune_alpaca.sbatch" >&2
    echo "       Or override at submit time:    sbatch --chdir=/path/to/Fast-dLLM/v2 ..." >&2
    exit 1
fi
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-lmflow}
CONDA_ROOT=${CONDA_ROOT:-/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3}

mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-fastdllm_v2_vanilla}"

# DeepSpeed needs CUDA_HOME set at import time. On Unity the toolkit comes
# from `module load`; try the common module names then fall back to the
# standard /usr/local/cuda install paths.
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
    echo "ERROR: could not locate a CUDA toolkit (nvcc) on the compute node." >&2
    module avail cuda 2>&1 | head -50 >&2 || true
    exit 1
fi
echo "Using CUDA_HOME=${CUDA_HOME} (nvcc: $(nvcc --version | grep release))"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Make src/lmflow importable even when `pip install -e .` was never run.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export TQDM_MINITERS=1000
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
# Defrag the CUDA allocator. Mixed flex_attention + bf16 + Adam allocations
# leave large reserved-but-unallocated regions; expandable_segments lets the
# allocator return them to the pool, recovering several GB.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Move Triton + DeepSpeed JIT caches off NFS (default $HOME location hangs on
# shutdown on Unity-style NFS).
_local_tmp=${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-no_slurm}}
mkdir -p "${_local_tmp}/triton" "${_local_tmp}/torch_extensions"
export TRITON_CACHE_DIR=${_local_tmp}/triton
export TORCH_EXTENSIONS_DIR=${_local_tmp}/torch_extensions

# -----------------------------------------------------------------------------
# Run config (mirror train_scripts/finetune_alpaca.sh)
# -----------------------------------------------------------------------------
model_name_or_path=${MODEL_NAME_OR_PATH:-Efficient-Large-Model/Fast_dLLM_v2_7B}
dataset_path=${DATASET_PATH:-data/alpaca/train_conversation}
output_dir=${OUTPUT_DIR:-output_models/finetune_fast_dLLM_7B}
conversation_template=${CONVERSATION_TEMPLATE:-fast_dllm_v2}

deepspeed_args="--master_port=$((10000 + SLURM_JOB_ID % 50000))"

trust_remote_code=0   # use vendored lmflow.models.fast_dllm classes

# NOTE: the original repo default is ZeRO-2, which only shards optimizer +
# grads but keeps the bf16 model REPLICATED on every rank (14 GB / GPU for
# 7B). On Unity's A100 80GB nodes the residual headroom after activations,
# flex_attention work buffers, NCCL, and allocator fragmentation is not
# enough for AdamW's fp32 shard (~21 GB / rank) — we hit OOM at the first
# optimizer step ("torch.OutOfMemoryError: ... Tried to allocate 14.19 GiB
# ... 11.32 GiB is free" in adamw.py:_init_group).
#
# ZeRO-3 also shards params, dropping the per-GPU model footprint to ~7 GB
# and leaving plenty of room for Adam state. Override at submit time with
# DEEPSPEED_CONFIG=configs/ds_config_zero2_no_offload.json if you want to
# match the upstream repo exactly (only safe on H100 / 96GB+ A100s).
deepspeed_config=${DEEPSPEED_CONFIG:-configs/ds_config_zero3_no_offload.json}

# -----------------------------------------------------------------------------
# Resume-from-latest-checkpoint, same logic as finetune_alpaca.sh
# -----------------------------------------------------------------------------
latest_checkpoint=""
if [ -d "${output_dir}" ]; then
    latest_checkpoint=$(find "${output_dir}" -name "checkpoint-*" -type d | sort -V | tail -1)
    if [ -n "${latest_checkpoint}" ]; then
        echo "Found latest checkpoint: ${latest_checkpoint}"
    else
        echo "No checkpoint found in ${output_dir}"
    fi
else
    echo "Output directory ${output_dir} does not exist, training from scratch"
fi

resume_arg=""
if [ -n "${latest_checkpoint}" ]; then
    resume_arg="--resume_from_checkpoint ${latest_checkpoint}"
fi

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
echo "Using DeepSpeed config: ${deepspeed_config}"
cmd="deepspeed ${deepspeed_args} \
  train_scripts/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    ${resume_arg} \
    --conversation_template ${conversation_template} \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 0 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --deepspeed "${deepspeed_config}" \
    --bf16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 1000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --save_total_limit 10 \
    --gradient_checkpointing 1"

echo "===== JOB INFO ====="
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-<not in slurm>}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-<not in slurm>}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-}"
echo "REPO_ROOT          = ${REPO_ROOT}"
echo "CONDA_ENV          = ${CONDA_ENV}"
echo "===== COMMAND ====="
echo "$cmd"
echo "==================="

eval $cmd
