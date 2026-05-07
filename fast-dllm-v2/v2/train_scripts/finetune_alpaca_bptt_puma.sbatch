#!/bin/bash
#
# PUMA + Loopholing + 2-step BPTT finetuning of Fast-dLLM v2 on Unity.
#
# Submit from a login node (NOT from a compute-node terminal):
#     sbatch train_scripts/finetune_alpaca_bptt_puma.sbatch
#
# Key knobs vs train_scripts/finetune_alpaca.sh:
#   --loss_type bptt                  -> FastDLLMBPTTTrainer (overrides Trainer.compute_loss
#                                        with FastDLLMBlockBPTTLoss)
#   --bptt_use_streaming_buffer 1     -> per-rank persistent (x_input, h_s) buffer
#                                        across calls (PUMA, Hou et al. 2025)
#   --bptt_use_cab 1                  -> Cross Attention Bridge for hidden-state
#                                        carry (replaces simple loopholing)
#   --bptt_threshold / --bptt_top_p   -> Fast-dLLM v2 verbatim threshold rule
#                                        for on-policy reveal-position selection
#   --trust_remote_code 0             -> use vendored lmflow.models.fast_dllm classes
#                                        (CAB module + bypass_noising flag)
#
# Vendored modifications + buffer wiring live under
#   src/lmflow/models/fast_dllm/{configuration,modeling}.py
#   src/lmflow/pipeline/utils/{block_bptt_loss,bptt_trainer,streaming_batch}.py
# See scripts/_bptt_static_smoke.py for an offline sanity check.
#
#SBATCH --job-name=fastdllm_v2_bptt_puma
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
# Anchor at the v2/ repo root so paths in the deepspeed call match
# finetune_alpaca.sh. Under slurm the script is run from a copy in
# /var/spool/slurm/slurmd/..., so BASH_SOURCE points there — useless. Prefer
# SLURM_SUBMIT_DIR (the dir the user ran `sbatch` from), and assume that dir
# IS Fast-dLLM/v2 (which is the documented submit cwd). For interactive runs
# without slurm, fall back to BASH_SOURCE.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
# Sanity: refuse to run if we landed somewhere that isn't the v2/ repo root.
if [ ! -f "${REPO_ROOT}/train_scripts/finetune.py" ] || [ ! -d "${REPO_ROOT}/src/lmflow" ]; then
    echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like Fast-dLLM/v2/." >&2
    echo "       Submit from Fast-dLLM/v2 with: sbatch train_scripts/finetune_alpaca_bptt_puma.sbatch" >&2
    echo "       Or override at submit time:    sbatch --chdir=/path/to/Fast-dLLM/v2 ..." >&2
    exit 1
fi
cd "${REPO_ROOT}"

# Conda env that has torch + transformers==4.53.1 + deepspeed>=0.14.4 + evaluate
# (i.e. all of v2/requirements.txt). Override at submit time with:
#     sbatch --export=ALL,CONDA_ENV=my_env train_scripts/finetune_alpaca_bptt_puma.sbatch
CONDA_ENV=${CONDA_ENV:-lmflow}
CONDA_ROOT=${CONDA_ROOT:-/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3}

# Make sure the slurm log dir exists before sbatch redirects into it (Unity
# does NOT mkdir the parent of --output for you).
mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-fastdllm_v2_bptt_puma}"

# DeepSpeed inspects CUDA_HOME at *import time* (it walks the toolkit headers
# to decide which fused ops to JIT). The conda env only has torch's bundled
# CUDA runtime libs, not the toolkit + nvcc. On Unity the toolkit is provided
# via `module load`. Try a couple of well-known module names; if none load,
# fall back to /usr/local/cuda (Unity's default CUDA install).
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
    echo "       DeepSpeed needs CUDA_HOME set to import. Available modules:" >&2
    module avail cuda 2>&1 | head -50 >&2 || true
    exit 1
fi
echo "Using CUDA_HOME=${CUDA_HOME} (nvcc: $(nvcc --version | grep release))"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Make src/lmflow importable even when `pip install -e .` was never run.
# Prepend so an editable install (if present) still wins.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Distributed knobs that match the Unity slurm conventions used elsewhere in
# the repo group (double-backprop/slurm_scripts/README.md).
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export TQDM_MINITERS=1000
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Avoid surprise downloads of the remote modeling.py — we vendored it.
export HF_HUB_DISABLE_TELEMETRY=1
# Defrag the CUDA allocator. BPTT runs two forwards before backward, leaving
# significant transient activation memory; expandable_segments lets the
# allocator return reserved-but-unallocated regions to the pool.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Move Triton + DeepSpeed JIT caches off NFS (their default location under
# $HOME hangs on shutdown on Unity-style NFS). $TMPDIR is local-disk on each
# compute node; fall back to /tmp if slurm didn't set one.
_local_tmp=${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-no_slurm}}
mkdir -p "${_local_tmp}/triton" "${_local_tmp}/torch_extensions"
export TRITON_CACHE_DIR=${_local_tmp}/triton
export TORCH_EXTENSIONS_DIR=${_local_tmp}/torch_extensions

# -----------------------------------------------------------------------------
# Run config (mirror train_scripts/finetune_alpaca.sh)
# -----------------------------------------------------------------------------
model_name_or_path=${MODEL_NAME_OR_PATH:-Efficient-Large-Model/Fast_dLLM_v2_7B}
dataset_path=${DATASET_PATH:-data/alpaca/train_conversation}
# Effective batch size + carry-aware default landing dir. Use a per-experiment
# subdir so successive runs don't try to resume from each other's checkpoints
# (esp. between cab/loopguard variants).
output_dir=${OUTPUT_DIR:-output_models/finetune_fast_dLLM_7B_bptt_puma}
conversation_template=${CONVERSATION_TEMPLATE:-fast_dllm_v2}

# ----- Training scale knobs (overridable from the submit env) -----
# Old defaults were per_device=1, accum=1 -> effective bs = 1 * 2 GPUs * 1 = 2.
# That's far too small to average out the BPTT/PUMA per-step variance, and
# also hurts ZeRO-3 throughput (most time is in param gather/scatter, not
# math). Stateflow's working CAB run uses effective bs ~64-128. Default here
# to per_device=2, accum=16 -> effective bs = 2 * 2 GPUs * 16 = 64.
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-16}
num_train_epochs=${NUM_TRAIN_EPOCHS:-3}
learning_rate=${LEARNING_RATE:-2e-5}
warmup_ratio=${WARMUP_RATIO:-0.1}
lr_scheduler_type=${LR_SCHEDULER_TYPE:-cosine}
max_grad_norm=${MAX_GRAD_NORM:-1.0}
logging_steps=${LOGGING_STEPS:-1}

# deepspeed launches one process per visible GPU (slurm sets CUDA_VISIBLE_DEVICES
# from --gres=gpu:N). Use a port that does not collide with concurrent jobs.
deepspeed_args="--master_port=$((10000 + SLURM_JOB_ID % 50000))"

trust_remote_code=0   # use vendored lmflow.models.fast_dllm classes

# DeepSpeed config: BPTT runs forward 1 + forward 2 before a single backward.
# Each shared parameter is touched by both sub-graphs, so a single backward
# visits each param twice. ZeRO-1 and ZeRO-2 share the same backward path
# (deepspeed/runtime/zero/stage_1_and_2.py) and install per-param eager-
# reduction hooks that assert "param not already reduced" — this trips on
# the second visit ("AssertionError: ... has already been reduced. Gradient
# computed twice for this partition.").
#
# Only ZeRO-3 supports multi-forward / multi-use graphs out of the box: its
# partitioned_param_coordinator tracks per-param forward counts and defers
# reduction until the matching number of backward visits has accumulated.
# This is the same machinery that makes ZeRO-3 work for DPO/GRPO/RLHF, all
# of which also run multiple forwards before one backward.
#
# For 7B + Adam on 2x A100 80GB, ZeRO-3 is also more memory-efficient than
# ZeRO-1 (params, grads AND optimizer state sharded), so this is a strict
# upgrade for our workload.
deepspeed_config=${DEEPSPEED_CONFIG:-configs/ds_config_zero3_no_offload.json}

# ----- BPTT + PUMA buffer hyperparams -----
# threshold/top_p match Fast-dLLM v2's generation_functions.py defaults; tune
# for the unmask schedule (lower threshold -> more reveals/step -> faster
# buffer drain).
bptt_threshold=${BPTT_THRESHOLD:-0.95}
bptt_top_p=${BPTT_TOP_P:-0.95}
bptt_temperature=${BPTT_TEMPERATURE:-0.0}
bptt_use_streaming_buffer=${BPTT_USE_STREAMING_BUFFER:-1}
bptt_use_cab=${BPTT_USE_CAB:-0}

# Checkpoint saving: default to end-of-epoch only (save_steps > total steps).
save_steps=${SAVE_STEPS:-99999}
save_total_limit=${SAVE_TOTAL_LIMIT:-3}

# Precision: default bf16; set USE_FP32=1 for full-precision control experiment.
use_fp32=${USE_FP32:-0}

# ----- W&B tags & run name -----
if [ "${bptt_use_cab}" = "1" ] || [ "${bptt_use_cab}" = "True" ]; then
    carry_tag="cab"
else
    carry_tag="loopguard"
fi
buf_tag="puma"
[ "${bptt_use_streaming_buffer}" = "0" ] && buf_tag="nobuf"
precision_flag="--bf16"
precision_tag="bf16"
if [ "${use_fp32}" = "1" ] || [ "${use_fp32}" = "True" ]; then
    precision_flag=""
    precision_tag="fp32"
fi
# Effective batch size tag (per_device * num_gpus * grad_accum). We don't
# know the visible GPU count at sbatch parse time without running nvidia-smi,
# so just encode the (per_device, accum, epochs) triple — enough to
# disambiguate experiments in W&B.
bs_tag="bs${per_device_train_batch_size}x${gradient_accumulation_steps}_ep${num_train_epochs}"
run_name=${RUN_NAME:-"bptt_${carry_tag}_${buf_tag}_${precision_tag}_${bs_tag}_v2"}
export WANDB_TAGS=${WANDB_TAGS:-"bptt,${carry_tag},${buf_tag},${precision_tag},alpaca,bigbatch,carryshift,v2"}

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
    # NOTE: PUMA buffer state is per-rank Python state and is NOT serialized
    # into HF checkpoints. On resume the buffer cold-starts from the next
    # batch — fine for finetuning, but worth knowing.
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
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm ${max_grad_norm} \
    --disable_group_texts 0 \
    --block_size 512 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --deepspeed "${deepspeed_config}" \
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
    --bptt_use_cab ${bptt_use_cab}"

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
