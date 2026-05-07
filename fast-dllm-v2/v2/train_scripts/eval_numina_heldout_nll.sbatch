#!/bin/bash
#
# Held-out Numina test split: mean CE (NLL) on assistant tokens only.
# Mirrors training chat template; see scripts/eval_numina_heldout_nll.py.
#
# From Fast-dLLM/v2:
#
#     MODEL_PATH=output_models/numina_1p5B/vanilla_bs2x16_ep2 \
#         RUN_TAG=vanilla_heldout_nll \
#         sbatch --reservation=dhruvesh train_scripts/eval_numina_heldout_nll.sbatch
#
#     MODEL_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B \
#         RUN_TAG=baseline_heldout_nll \
#         sbatch train_scripts/eval_numina_heldout_nll.sbatch
#
# Optional: DATA_PATH=data/numina/test_conversation/test_2000.json
#
#SBATCH --job-name=eval_numina_heldout
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint=vram80,bf16,ib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
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
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-lmflow}
CONDA_ROOT=${CONDA_ROOT:-/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3}
mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-eval_numina_heldout}"

if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load cuda/12.1 2>/dev/null \
        || module load cuda/12.4 2>/dev/null \
        || module load cuda/12.6 2>/dev/null \
        || module load cuda 2>/dev/null \
        || true
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH is required." >&2
    exit 1
fi

DATA_PATH=${DATA_PATH:-data/numina/test_conversation/test_2000.json}
BATCH_SIZE=${BATCH_SIZE:-2}

_resolve_data_path() {
    local p="$1"
    if [[ "${p}" == /* ]]; then
        echo "${p}"
    else
        echo "${REPO_ROOT}/${p}"
    fi
}
DATA_PATH_RESOLVED=$(_resolve_data_path "${DATA_PATH}")

# Resolve local relative MODEL_PATH (Hub ids left as-is)
case "${MODEL_PATH}" in
    /*) ;;
    */*/*)
        if [ -f "${REPO_ROOT}/${MODEL_PATH}/config.json" ]; then
            MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}"
        fi
        ;;
    *)
        if [ -f "${REPO_ROOT}/${MODEL_PATH}/config.json" ]; then
            MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}"
        fi
        ;;
esac

if [ -z "${RUN_TAG:-}" ]; then
    RUN_TAG=$(basename "${MODEL_PATH}" | tr '/.:' '___')
fi

echo "===== JOB INFO ====="
echo "MODEL_PATH  = ${MODEL_PATH}"
echo "DATA_PATH   = ${DATA_PATH_RESOLVED}"
echo "RUN_TAG     = ${RUN_TAG}"
echo "BATCH_SIZE  = ${BATCH_SIZE}"
echo "===================="

if [ ! -f "${DATA_PATH_RESOLVED}" ]; then
    echo "ERROR: data file not found: ${DATA_PATH_RESOLVED}" >&2
    echo "       Run: python scripts/prep_numina.py ... (writes test_conversation/test_2000.json)" >&2
    exit 1
fi

python "${REPO_ROOT}/scripts/eval_numina_heldout_nll.py" \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH_RESOLVED}" \
    --batch_size "${BATCH_SIZE}"
