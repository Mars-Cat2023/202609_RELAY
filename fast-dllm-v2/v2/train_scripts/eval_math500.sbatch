#!/bin/bash
#
# MATH-500 evaluation harness for Fast-dLLM v2 checkpoints (off-the-shelf or
# NuminaMath-finetuned). Wraps a single lm_eval run on the custom
# scripts/lm_eval_tasks/math500/ task — same wiring as eval_lm_eval.sbatch
# but scoped to one task and with informative naming.
#
# Submit from a login node, with cwd = Fast-dLLM/v2:
#
#     # off-the-shelf 1.5B baseline
#     MODEL_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B \
#         RUN_TAG=baseline_1p5B \
#         sbatch train_scripts/eval_math500.sbatch
#
#     # vanilla numina checkpoint -- point at the top-level output_dir
#     # (final model.safetensors lives there; checkpoint-N/ subdirs are
#     # intermediate per-step saves)
#     MODEL_PATH=output_models/numina_1p5B/vanilla_bs2x16_ep2 \
#         RUN_TAG=vanilla_1p5B_numina \
#         sbatch train_scripts/eval_math500.sbatch
#
#     # bptt+nocarry checkpoint
#     MODEL_PATH=output_models/numina_1p5B/bptt_nocarry_puma_bs2x16_ep2 \
#         RUN_TAG=bptt_nocarry_1p5B_numina \
#         sbatch train_scripts/eval_math500.sbatch
#
#     # bptt+loopguard checkpoint (set USE_CARRY=1 so the model uses h_t)
#     MODEL_PATH=output_models/numina_1p5B/bptt_loopguard_puma_bs2x16_ep2 \
#         RUN_TAG=bptt_loopguard_1p5B_numina  USE_CARRY=1 \
#         sbatch train_scripts/eval_math500.sbatch
#
#     # bptt+cab checkpoint
#     MODEL_PATH=output_models/numina_1p5B/bptt_cab_puma_bs2x16_ep2 \
#         RUN_TAG=bptt_cab_1p5B_numina  USE_CARRY=1 \
#         sbatch train_scripts/eval_math500.sbatch
#
# Useful overrides:
#     MODEL_PATH=...               required
#     RUN_TAG=<short-id>           goes into log/output dir name (default: derived
#                                  from MODEL_PATH basename)
#     THRESHOLD=0.85               generation threshold
#     BATCH_SIZE=32
#     NUM_FEWSHOT=0                Stateflow uses 0 with chat template; flip to 4
#                                  for an apples-to-apples Minerva comparison
#     APPLY_CHAT_TEMPLATE=1        match Fast-dLLM v2 training (im_start/im_end)
#     USE_CARRY=0|1                pass `use_carry=True` to the model_args
#                                  (only meaningful for CAB / Loopguard checkpoints)
#
#SBATCH --job-name=fastdllm_v2_math500
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
# Paths + environment (same scaffolding as eval_lm_eval.sbatch)
# -----------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [ ! -f "${REPO_ROOT}/eval.py" ] || [ ! -d "${REPO_ROOT}/scripts/lm_eval_tasks/math500" ]; then
    echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like Fast-dLLM/v2/" >&2
    echo "       (or the math500 lm_eval task is missing). Submit from Fast-dLLM/v2." >&2
    exit 1
fi
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-lmflow}
CONDA_ROOT=${CONDA_ROOT:-/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3}
mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-fastdllm_v2_math500}"

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
    echo "       Install with: pip install 'lm-eval[math,ifeval]==0.4.9'" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TQDM_MINITERS=1000
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
_local_tmp=${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-no_slurm}}
mkdir -p "${_local_tmp}/triton" "${_local_tmp}/torch_extensions"
export TRITON_CACHE_DIR=${_local_tmp}/triton
export TORCH_EXTENSIONS_DIR=${_local_tmp}/torch_extensions

# -----------------------------------------------------------------------------
# Resolve MODEL_PATH and inject vendored modeling.py / configuration.py
# -----------------------------------------------------------------------------
if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH is required." >&2
    echo "  e.g. MODEL_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B sbatch ..." >&2
    exit 1
fi
# Resolve MODEL_PATH:
#   - Absolute paths        -> require config.json, fail otherwise.
#   - Relative w/ >=2 slashes (e.g. output_models/foo/bar/checkpoint-X)
#       -> can ONLY be a local path (HF Hub ids have at most one slash:
#          'org/name'). Normalise to absolute and fail loudly if missing,
#          listing siblings to help spot typos like 'checkpoint-final'.
#   - Single-component or 'org/name'
#       -> ambiguous: normalise to absolute if local dir exists,
#          otherwise pass through as a Hub repo id.
case "${MODEL_PATH}" in
    /*)
        if [ ! -f "${MODEL_PATH}/config.json" ]; then
            echo "ERROR: MODEL_PATH '${MODEL_PATH}' is absolute but has no config.json." >&2
            exit 1
        fi
        ;;
    */*/*)
        if [ -f "${REPO_ROOT}/${MODEL_PATH}/config.json" ]; then
            MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}"
        else
            echo "ERROR: MODEL_PATH '${MODEL_PATH}' has >=2 slashes so it cannot be a HF Hub" >&2
            echo "       repo id, but ${REPO_ROOT}/${MODEL_PATH}/config.json does not exist." >&2
            _parent=${REPO_ROOT}/$(dirname "${MODEL_PATH}")
            if [ -d "${_parent}" ]; then
                echo "       Available siblings under ${_parent}:" >&2
                ls -d "${_parent}"/*/ 2>/dev/null | sed 's,^,           ,' >&2 || true
            fi
            exit 1
        fi
        ;;
    *)
        if [ -f "${REPO_ROOT}/${MODEL_PATH}/config.json" ]; then
            MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}"
        fi
        ;;
esac

# For local checkpoints, drop in the vendored modeling.py / configuration.py
# so eval.py's `trust_remote_code=True` lookup succeeds. Skip silently for
# hub repo ids (no local dir to write into; the hub copy is already correct).
if [ -d "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}/config.json" ]; then
    _VENDORED=${REPO_ROOT}/src/lmflow/models/fast_dllm
    for f in modeling.py configuration.py; do
        if [ ! -f "${MODEL_PATH}/${f}" ] \
            || ! cmp -s "${_VENDORED}/${f}" "${MODEL_PATH}/${f}"; then
            echo "Copying ${_VENDORED}/${f} -> ${MODEL_PATH}/${f}"
            cp "${_VENDORED}/${f}" "${MODEL_PATH}/${f}"
        fi
    done
fi

# -----------------------------------------------------------------------------
# Eval config
# -----------------------------------------------------------------------------
TASK=math500
THRESHOLD=${THRESHOLD:-0.85}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-1}
USE_CARRY=${USE_CARRY:-0}

# Run tag — goes into both the eval output dir name and (informally) the
# log file via SLURM_JOB_NAME. Default: pull a short basename of MODEL_PATH.
if [ -z "${RUN_TAG:-}" ]; then
    RUN_TAG=$(basename "${MODEL_PATH}" | tr '/.:' '___')
fi

OUTPUT_DIR=${OUTPUT_DIR:-eval_results/math500__${RUN_TAG}__job${SLURM_JOB_ID:-local}}
mkdir -p "${OUTPUT_DIR}"

_carry_flag=""
if [ "${USE_CARRY}" = "1" ] || [ "${USE_CARRY}" = "True" ] || [ "${USE_CARRY}" = "true" ]; then
    _carry_flag=",use_carry=True"
fi

_chat_flag=""
_fewshot_chat_flag=""
if [ "${APPLY_CHAT_TEMPLATE}" = "1" ] || [ "${APPLY_CHAT_TEMPLATE}" = "True" ]; then
    _chat_flag="--apply_chat_template"
    # `--fewshot_as_multiturn` requires `--apply_chat_template` and only
    # makes sense for >0 fewshot. Stateflow's default is 0-shot + chat
    # template, which is what we use too.
    if [ "${NUM_FEWSHOT}" != "0" ]; then
        _fewshot_chat_flag="--fewshot_as_multiturn"
    fi
fi

model_args="model_path=${MODEL_PATH},threshold=${THRESHOLD},show_speed=True${_carry_flag}"

echo "===== JOB INFO ====="
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-<not in slurm>}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-<not in slurm>}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL_PATH         = ${MODEL_PATH}"
echo "RUN_TAG            = ${RUN_TAG}"
echo "OUTPUT_DIR         = ${OUTPUT_DIR}"
echo "TASK               = ${TASK}"
echo "NUM_FEWSHOT        = ${NUM_FEWSHOT}"
echo "APPLY_CHAT_TEMPLATE= ${APPLY_CHAT_TEMPLATE}"
echo "USE_CARRY          = ${USE_CARRY}"
echo "BATCH_SIZE         = ${BATCH_SIZE}"
echo "model_args         = ${model_args}"
echo "===================="

cmd="accelerate launch eval.py \
    --tasks ${TASK} \
    --include_path ${REPO_ROOT}/scripts/lm_eval_tasks/math500 \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    ${_chat_flag} ${_fewshot_chat_flag} \
    --output_path ${OUTPUT_DIR} \
    --model_args ${model_args}"
echo "===== COMMAND ====="
echo "$cmd"
echo "==================="
eval $cmd
