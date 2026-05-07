#!/bin/bash
#
# PUMA + 2-step BPTT + optional carry (Loopguard / CAB / MLP / none) on
# Tulu-3-SFT-Mixture. Same data/packing as finetune_tulu3.sbatch; same
# BPTT stack as finetune_magicoder_oss_bptt.sbatch but sized for the
# 939k-row Tulu mixture: 8 GPUs single-node, 2-day time, no on-save eval.
#
# CARRY_MODE:
#   none       -- 2-step BPTT + PUMA, no carry (matches vanilla SFT
#                 architecturally; isolates the loss/buffer machinery)
#   loopguard  -- + Loopguard (loopholing) carry
#   mlp        -- + MLP-bottleneck carry (then LayerNorm)
#   cab        -- + CAB carry
#
# BPTT_STOP_GRAD_H_S (NEW):
#   0          -- default: full BPTT through the carry (gradients from
#                 forward 2 flow back into forward 1 via h_s1)
#   1          -- ablation: detach h_s1 between forwards 1 and 2;
#                 architectural carry stays on but BPTT-through-carry
#                 is disabled. Only relevant when CARRY_MODE != none.
#
# Submit from a login node, with cwd = Fast-dLLM/v2:
#     # Run 3: PUMA + Loopguard with stop-grad
#     CARRY_MODE=loopguard BPTT_STOP_GRAD_H_S=1 \
#         sbatch train_scripts/finetune_tulu3_bptt.sbatch
#
#     # Run 4: PUMA + Loopguard + BPTT (default, no stop-grad)
#     CARRY_MODE=loopguard BPTT_STOP_GRAD_H_S=0 \
#         sbatch train_scripts/finetune_tulu3_bptt.sbatch
#
# Note: this script intentionally omits all `--evalplus_*` flags so
# nothing eval-related fires during training. Run EvalPlus / Math500 /
# lm-eval-harness on saved checkpoints out-of-band.
#
#SBATCH --job-name=fastdllm_v2_1p5B_tulu3_bptt
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
mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-fastdllm_v2_1p5B_tulu3_bptt}"

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
disable_group_texts=${DISABLE_GROUP_TEXTS:-1}
block_size=${BLOCK_SIZE:-1024}

case "${model_name_or_path}" in
    *1.5B*|*1p5B*|*1_5B*) size_tag="1p5B" ;;
    *7B*)                 size_tag="7B"   ;;
    *)                    size_tag=$(basename "${model_name_or_path}" | tr -cd 'A-Za-z0-9') ;;
esac

# ---- Carry mode ------------------------------------------------------------
carry_mode=${CARRY_MODE:-none}
bptt_use_mlp_carry=0
case "${carry_mode}" in
    cab)
        bptt_use_cab=1
        bptt_disable_carry=0
        carry_tag="cab"
        ;;
    loopguard|loopholing|loophole)
        bptt_use_cab=0
        bptt_disable_carry=0
        carry_tag="loopguard"
        ;;
    mlp)
        bptt_use_cab=0
        bptt_use_mlp_carry=1
        bptt_disable_carry=0
        carry_tag="mlp"
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

# ---- Stop-grad ablation (NEW): detach h_s1 between forwards 1 and 2 -------
# No-op when carry is disabled (no carry to detach); the trainer also
# enforces this server-side, but we keep the bash side honest so the
# run name / W&B tags reflect what the model actually does.
bptt_stop_grad_h_s=${BPTT_STOP_GRAD_H_S:-0}
if [ "${bptt_disable_carry}" = "1" ] && [ "${bptt_stop_grad_h_s}" = "1" ]; then
    echo "WARNING: BPTT_STOP_GRAD_H_S=1 is a no-op when CARRY_MODE=none; forcing to 0."
    bptt_stop_grad_h_s=0
fi
stopgrad_tag=""
[ "${bptt_stop_grad_h_s}" = "1" ] && stopgrad_tag="_stopgrad"

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

bs_tag="bs${per_device_train_batch_size}x${gradient_accumulation_steps}x8_ep${num_train_epochs}"
run_name=${RUN_NAME:-"bptt_${carry_tag}${stopgrad_tag}_${size_tag}_tulu3_sft_full_${buf_tag}_nopack${block_size}_${bs_tag}_lr${learning_rate}_v2"}
output_dir=${OUTPUT_DIR:-output_models/tulu3_sft_${size_tag}/bptt_${carry_tag}${stopgrad_tag}_puma_nopack${block_size}_${bs_tag}_lr${learning_rate}}
bptt_param_tags="carry_mode=${bptt_loss_carry_mode},threshold=${bptt_threshold},top_p=${bptt_top_p},temperature=${bptt_temperature},stop_grad_h_s=${bptt_stop_grad_h_s}"
export WANDB_TAGS=${WANDB_TAGS:-"bptt,${carry_tag},${bptt_param_tags},${size_tag},tulu3_sft,full,${buf_tag},${precision_tag},${bs_tag},carryshift,v2"}

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
echo "carry_mode=${carry_mode}  -> bptt_use_cab=${bptt_use_cab}  bptt_use_mlp_carry=${bptt_use_mlp_carry}  bptt_disable_carry=${bptt_disable_carry}  bptt_stop_grad_h_s=${bptt_stop_grad_h_s}"
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
    --bptt_stop_grad_h_s ${bptt_stop_grad_h_s}"

echo "===== JOB INFO ====="
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID:-<not in slurm>}"
echo "SLURM_JOB_NODELIST   = ${SLURM_JOB_NODELIST:-<not in slurm>}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-}"
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
echo "bptt_stop_grad_h_s   = ${bptt_stop_grad_h_s}"
echo "save_steps           = ${save_steps}"
echo "save_total_limit     = ${save_total_limit}"
echo "===== COMMAND ====="
echo "$cmd"
echo "==================="
eval $cmd
