#!/bin/bash
#
# PUMA + 2-step BPTT + optional carry (Loopguard / CAB / none) on Magicoder-OSS.
# Data / packing / block length match finetune_magicoder_oss.sbatch; the BPTT
# stack matches finetune_numina_bptt.sbatch (loss_type bptt, PUMA buffer, carry flags).
#
# Submit from Fast-dLLM/v2:
#     MAGICODER_SPLIT=python CARRY_MODE=none  sbatch train_scripts/finetune_magicoder_oss_bptt.sbatch
#     MAGICODER_SPLIT=python CARRY_MODE=loopguard sbatch train_scripts/finetune_magicoder_oss_bptt.sbatch
#
# CARRY_MODE:
#   none       — 2-step BPTT + PUMA, no carry (bptt_disable_carry=1; forward matches vanilla SFT)
#   loopguard  — + Loopguard (loopholing) carry
#   mlp        — + MLP-bottleneck carry (then LayerNorm), like loopguard but with bottleneck MLP
#   cab        — + CAB carry
#
#SBATCH --job-name=fastdllm_v2_1p5B_magicoder_bptt
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
mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-fastdllm_v2_1p5B_magicoder_bptt}"

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
# Magicoder split (same as finetune_magicoder_oss.sbatch)
# -----------------------------------------------------------------------------
magicoder_split=${MAGICODER_SPLIT:-python}
case "${magicoder_split}" in
    python)
        default_dataset_path=data/magicoder_oss_python/train_conversation
        default_num_train_epochs=2
        split_tag=python
        output_prefix=magicoder_oss_python
        ;;
    full)
        default_dataset_path=data/magicoder_oss/train_conversation
        default_num_train_epochs=1
        split_tag=full
        output_prefix=magicoder_oss_full
        ;;
    *)
        echo "ERROR: MAGICODER_SPLIT must be 'python' or 'full' (got '${magicoder_split}')" >&2
        exit 1
        ;;
esac

model_name_or_path=${MODEL_NAME_OR_PATH:-Efficient-Large-Model/Fast_dLLM_v2_1.5B}
dataset_path=${DATASET_PATH:-$default_dataset_path}
conversation_template=${CONVERSATION_TEMPLATE:-fast_dllm_v2}
truncation_side=${TRUNCATION_SIDE:-left}
disable_group_texts=${DISABLE_GROUP_TEXTS:-1}
block_size=${BLOCK_SIZE:-1024}

case "${model_name_or_path}" in
    *1.5B*|*1p5B*|*1_5B*) size_tag="1p5B" ;;
    *7B*)                 size_tag="7B"   ;;
    *)                    size_tag=$(basename "${model_name_or_path}" | tr -cd 'A-Za-z0-9') ;;
esac

# ---- Carry (same mapping as finetune_numina_bptt.sbatch) ----------------------
carry_mode=${CARRY_MODE:-none}
bptt_use_mlp_carry=0
evalplus_use_carry=0
case "${carry_mode}" in
    cab)
        bptt_use_cab=1
        bptt_disable_carry=0
        carry_tag="cab"
        evalplus_use_carry=1
        ;;
    loopguard|loopholing|loophole)
        bptt_use_cab=0
        bptt_disable_carry=0
        carry_tag="loopguard"
        evalplus_use_carry=1
        ;;
    mlp)
        bptt_use_cab=0
        bptt_use_mlp_carry=1
        bptt_disable_carry=0
        carry_tag="mlp"
        evalplus_use_carry=1
        ;;
    none|nocarry|no-carry|disabled|off)
        bptt_use_cab=0
        bptt_disable_carry=1
        carry_tag="nocarry"
        ;;
    *)
        echo "ERROR: unknown CARRY_MODE='${carry_mode}'. Use cab|loopguard|mlp|none." >&2
        exit 1
        ;;
esac
bptt_loss_carry_mode=${carry_tag}
[ "${carry_tag}" = "loopguard" ] && bptt_loss_carry_mode="loopholing"
[ "${carry_tag}" = "nocarry" ] && bptt_loss_carry_mode="none"

per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-16}
num_train_epochs=${NUM_TRAIN_EPOCHS:-$default_num_train_epochs}
learning_rate=${LEARNING_RATE:-1e-5}
warmup_ratio=${WARMUP_RATIO:-0.03}
lr_scheduler_type=${LR_SCHEDULER_TYPE:-cosine}
max_grad_norm=${MAX_GRAD_NORM:-1.0}
logging_steps=${LOGGING_STEPS:-1}
save_steps=${SAVE_STEPS:-99999}
save_total_limit=${SAVE_TOTAL_LIMIT:-0}
evalplus_on_save=${EVALPLUS_ON_SAVE:-0}
evalplus_datasets=${EVALPLUS_DATASETS:-"humaneval mbpp"}
evalplus_seeds=${EVALPLUS_SEEDS:-"0"}
evalplus_threshold=${EVALPLUS_THRESHOLD:-0.85}
evalplus_sbatch=${EVALPLUS_SBATCH:-train_scripts/evalplus_checkpoint.sbatch}
evalplus_block_cache=${EVALPLUS_BLOCK_CACHE:-auto}
evalplus_strict=${EVALPLUS_ON_SAVE_STRICT:-0}

bptt_threshold=${BPTT_THRESHOLD:-0.85}
bptt_top_p=${BPTT_TOP_P:-0.95}
bptt_temperature=${BPTT_TEMPERATURE:-0.0}
bptt_use_streaming_buffer=${BPTT_USE_STREAMING_BUFFER:-1}

buf_tag="puma"
[ "${bptt_use_streaming_buffer}" = "0" ] && buf_tag="nobuf"
precision_flag="--bf16"
precision_tag="bf16"
use_fp32=${USE_FP32:-0}
if [ "${use_fp32}" = "1" ] || [ "${use_fp32}" = "True" ]; then
    precision_flag=""
    precision_tag="fp32"
fi

bs_tag="bs${per_device_train_batch_size}x${gradient_accumulation_steps}_ep${num_train_epochs}"
run_name=${RUN_NAME:-"bptt_${carry_tag}_${size_tag}_magicoder_${split_tag}_${buf_tag}_nopack${block_size}_${bs_tag}_lr${learning_rate}_v2"}
output_dir=${OUTPUT_DIR:-output_models/${output_prefix}_${size_tag}/bptt_${carry_tag}_puma_nopack${block_size}_${bs_tag}_lr${learning_rate}}
bptt_param_tags="carry_mode=${bptt_loss_carry_mode},threshold=${bptt_threshold},top_p=${bptt_top_p},temperature=${bptt_temperature}"
export WANDB_TAGS=${WANDB_TAGS:-"bptt,${carry_tag},${bptt_param_tags},${size_tag},magicoder_oss,${split_tag},${buf_tag},${precision_tag},${bs_tag},carryshift,v2"}

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
echo "carry_mode=${carry_mode}  -> bptt_use_cab=${bptt_use_cab}  bptt_use_mlp_carry=${bptt_use_mlp_carry}  bptt_disable_carry=${bptt_disable_carry}"
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
    ${precision_flag} \
    --run_name ${run_name} \
    --validation_split_percentage 0 \
    --logging_steps ${logging_steps} \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps ${save_steps} \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --save_total_limit ${save_total_limit} \
    --gradient_checkpointing 1 \
    --loss_type bptt \
    --bptt_threshold ${bptt_threshold} \
    --bptt_top_p ${bptt_top_p} \
    --bptt_temperature ${bptt_temperature} \
    --bptt_use_streaming_buffer ${bptt_use_streaming_buffer} \
    --bptt_use_cab ${bptt_use_cab} \
    --bptt_use_mlp_carry ${bptt_use_mlp_carry} \
    --bptt_disable_carry ${bptt_disable_carry} \
    --evalplus_on_save ${evalplus_on_save} \
    --evalplus_datasets '${evalplus_datasets}' \
    --evalplus_seeds '${evalplus_seeds}' \
    --evalplus_threshold ${evalplus_threshold} \
    --evalplus_sbatch ${evalplus_sbatch} \
    --evalplus_block_cache ${evalplus_block_cache} \
    --evalplus_use_carry ${evalplus_use_carry} \
    --evalplus_strict ${evalplus_strict}"

echo "===== JOB INFO ====="
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID:-<not in slurm>}"
echo "SLURM_JOB_NODELIST   = ${SLURM_JOB_NODELIST:-<not in slurm>}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-}"
echo "MAGICODER_SPLIT      = ${magicoder_split}"
echo "model_name_or_path   = ${model_name_or_path}"
echo "dataset_path         = ${dataset_path}"
echo "output_dir           = ${output_dir}"
echo "run_name             = ${run_name}"
echo "WANDB_TAGS           = ${WANDB_TAGS}"
echo "disable_group_texts  = ${disable_group_texts}"
echo "block_size           = ${block_size}"
echo "bptt_threshold       = ${bptt_threshold}"
echo "bptt_top_p           = ${bptt_top_p}"
echo "bptt_temperature     = ${bptt_temperature}"
echo "bptt_loss_carry_mode = ${bptt_loss_carry_mode}"
echo "save_steps           = ${save_steps}"
echo "save_total_limit     = ${save_total_limit}"
echo "evalplus_on_save     = ${evalplus_on_save}"
echo "evalplus_datasets    = ${evalplus_datasets}"
echo "evalplus_seeds       = ${evalplus_seeds}"
echo "evalplus_threshold   = ${evalplus_threshold}"
echo "evalplus_use_carry   = ${evalplus_use_carry}"
echo "evalplus_block_cache = ${evalplus_block_cache}"
echo "===== COMMAND ====="
echo "$cmd"
echo "==================="
eval $cmd
