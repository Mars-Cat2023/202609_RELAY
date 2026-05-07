#!/bin/bash
#
# Evaluation harness for an Alpaca-finetuned Fast-dLLM v2 checkpoint on Unity.
# Wraps Fast-dLLM/v2/eval_script.sh (MMLU, GPQA, GSM8K, Minerva-Math, IFEval)
# with a single-GPU, slurm-friendly setup that mirrors finetune_alpaca.sbatch.
#
# Submit from a login node, with cwd = Fast-dLLM/v2:
#
#   # vanilla checkpoint
#   cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
#   MODEL_PATH=output_models/finetune_fast_dLLM_7B \
#       sbatch --job-name=eval_vanilla train_scripts/eval_alpaca.sbatch
#
#   # BPTT+PUMA checkpoint
#   MODEL_PATH=output_models/finetune_fast_dLLM_7B_bptt_puma \
#       sbatch --job-name=eval_bptt train_scripts/eval_alpaca.sbatch
#
# Useful overrides:
#   MODEL_PATH=...                  required; absolute or repo-relative path
#   TASKS="mmlu gsm8k"              space-separated subset of the 5 tasks
#   THRESHOLD=0.9                   sampling threshold for generation tasks
#   OUTPUT_DIR=eval_results/<name>  where lm_eval writes its json reports
#
# To use a slurm reservation:
#   sbatch --reservation=<name> --job-name=... train_scripts/eval_alpaca.sbatch
#
#SBATCH --job-name=fastdllm_v2_eval
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint=vram80,bf16,ib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x/%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=brozonoyer_umass_edu

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths + environment (mirror finetune_alpaca.sbatch)
# -----------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [ ! -f "${REPO_ROOT}/eval.py" ] || [ ! -f "${REPO_ROOT}/generation_functions.py" ] \
    || [ ! -d "${REPO_ROOT}/src/lmflow/models/fast_dllm" ]; then
    echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like Fast-dLLM/v2/." >&2
    echo "       Submit from Fast-dLLM/v2 with:" >&2
    echo "         MODEL_PATH=... sbatch train_scripts/eval_alpaca.sbatch" >&2
    exit 1
fi
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-lmflow}
CONDA_ROOT=${CONDA_ROOT:-/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3}

mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-fastdllm_v2_eval}"

# CUDA toolkit (DeepSpeed isn't strictly needed for eval, but flex_attention
# and bitsandbytes both want a working nvcc on import). Same logic as the
# finetune sbatch.
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
echo "Using CUDA_HOME=${CUDA_HOME:-<unset>}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! python -c "import lm_eval" 2>/dev/null; then
    echo "ERROR: lm_eval is not installed in conda env '${CONDA_ENV}'." >&2
    echo "       Install once with:" >&2
    echo "         conda activate ${CONDA_ENV}" >&2
    echo "         pip install 'lm-eval[math,ifeval]==0.4.9'" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export TQDM_MINITERS=1000
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Required by HF Eval harness for code-execution tasks (humaneval, mbpp, etc).
# Harmless for the 5 default tasks; flips on if you add code tasks via TASKS=.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
# Move Triton + DeepSpeed JIT caches off NFS (default $HOME location hangs on
# shutdown on Unity-style NFS).
_local_tmp=${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-no_slurm}}
mkdir -p "${_local_tmp}/triton" "${_local_tmp}/torch_extensions"
export TRITON_CACHE_DIR=${_local_tmp}/triton
export TORCH_EXTENSIONS_DIR=${_local_tmp}/torch_extensions

# -----------------------------------------------------------------------------
# Resolve MODEL_PATH and inject vendored modeling.py / configuration.py
# -----------------------------------------------------------------------------
if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH is required." >&2
    echo "  e.g. MODEL_PATH=output_models/finetune_fast_dLLM_7B sbatch ..." >&2
    exit 1
fi
# Normalize to an absolute path so HF doesn't treat it as a hub repo id.
case "${MODEL_PATH}" in
    /*) ;;
    *) MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}" ;;
esac
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: ${MODEL_PATH}/config.json does not exist." >&2
    exit 1
fi

# eval.py loads with trust_remote_code=True, which makes HF look for
# modeling.py / configuration.py *inside* MODEL_PATH (the saved config.json
# has auto_map -> "modeling.Fast_dLLM_QwenForCausalLM" etc). Trainer.save_model
# does NOT copy the vendored source into the checkpoint dir, so we drop them
# in here. Idempotent; safe to re-run.
_VENDORED=${REPO_ROOT}/src/lmflow/models/fast_dllm
for f in modeling.py configuration.py; do
    if [ ! -f "${MODEL_PATH}/${f}" ] \
        || ! cmp -s "${_VENDORED}/${f}" "${MODEL_PATH}/${f}"; then
        echo "Copying ${_VENDORED}/${f} -> ${MODEL_PATH}/${f}"
        cp "${_VENDORED}/${f}" "${MODEL_PATH}/${f}"
    fi
done

# -----------------------------------------------------------------------------
# Eval config
# -----------------------------------------------------------------------------
TASKS=${TASKS:-"mmlu gpqa_main_n_shot gsm8k minerva_math ifeval"}
THRESHOLD=${THRESHOLD:-0.9}
USE_CARRY=${USE_CARRY:-0}
OUTPUT_DIR=${OUTPUT_DIR:-eval_results/${SLURM_JOB_NAME:-fastdllm_v2_eval}_${SLURM_JOB_ID:-local}}
mkdir -p "${OUTPUT_DIR}"

# Task -> (batch_size, num_fewshot, extra model_args). Mirrors eval_script.sh.
# `threshold,show_speed` are appended only to the generation tasks; mmlu and
# gpqa go through loglikelihood and ignore them.
declare -A TASK_BS=(
    [mmlu]=1
    [gpqa_main_n_shot]=1
    [gsm8k]=32
    [minerva_math]=32
    [ifeval]=32
)
declare -A TASK_FEWSHOT=(
    [mmlu]=5
    [gsm8k]=0
    [minerva_math]=0
)
declare -A TASK_EXTRA_ARGS=(
    [gsm8k]="threshold=${THRESHOLD},show_speed=True"
    [minerva_math]="threshold=${THRESHOLD},show_speed=True"
    [ifeval]="threshold=${THRESHOLD},show_speed=True"
)

# Map USE_CARRY env var to model_args.
_carry_flag=""
if [ "${USE_CARRY}" = "1" ] || [ "${USE_CARRY}" = "True" ] || [ "${USE_CARRY}" = "true" ]; then
    _carry_flag=",use_carry=True"
    echo "[eval sbatch] USE_CARRY=1 — passing use_carry=True to model_args for all tasks"
fi

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
echo "===== JOB INFO ====="
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-<not in slurm>}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-<not in slurm>}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-}"
echo "REPO_ROOT          = ${REPO_ROOT}"
echo "CONDA_ENV          = ${CONDA_ENV}"
echo "MODEL_PATH         = ${MODEL_PATH}"
echo "OUTPUT_DIR         = ${OUTPUT_DIR}"
echo "TASKS              = ${TASKS}"
echo "===================="

failed_tasks=()
for task in ${TASKS}; do
    bs=${TASK_BS[$task]:-1}
    fewshot_arg=""
    if [ -n "${TASK_FEWSHOT[$task]:-}" ]; then
        fewshot_arg="--num_fewshot ${TASK_FEWSHOT[$task]}"
    fi
    extra_args=${TASK_EXTRA_ARGS[$task]:-}
    model_args="model_path=${MODEL_PATH}${_carry_flag}"
    if [ -n "${extra_args}" ]; then
        model_args="${model_args},${extra_args}"
    fi

    task_out_dir="${OUTPUT_DIR}/${task}"
    mkdir -p "${task_out_dir}"

    echo
    echo "===== task: ${task} (bs=${bs}, fewshot=${TASK_FEWSHOT[$task]:-default}) ====="
    cmd="accelerate launch eval.py \
        --tasks ${task} \
        --batch_size ${bs} \
        ${fewshot_arg} \
        --confirm_run_unsafe_code \
        --model fast_dllm_v2 \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --output_path ${task_out_dir} \
        --model_args ${model_args}"
    echo "$cmd"
    if ! eval "$cmd"; then
        echo "WARNING: task '${task}' failed; continuing with remaining tasks." >&2
        failed_tasks+=("${task}")
    fi
done

echo
echo "===== eval summary ====="
echo "results dir: ${OUTPUT_DIR}"
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "failed tasks: ${failed_tasks[*]}"
    exit 1
fi
echo "all tasks completed successfully."
