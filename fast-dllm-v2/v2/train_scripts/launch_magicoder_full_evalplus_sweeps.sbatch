#!/bin/bash
#
# Submit HumanEval+MBPP EvalPlus sweeps for full Magicoder 1.5B runs.
# Immediate targets:
#   - off-the-shelf Hub model
#   - local HF baseline clone
#   - saved vanilla SFT checkpoint-* under magicoder_oss_full_1p5B
#
# Dependent targets:
#   - BPTT PUMA nocarry / loopguard / mlp full Magicoder runs
#
# Dependency envs accept raw job IDs or full Slurm dependency specs:
#   NOCARRY_DEPENDENCY=<nocarry_slurm_job_id>
#   LOOPGUARD_DEPENDENCY=afterok:<loopguard_slurm_job_id>
#   MLP_DEPENDENCY=<mlp_slurm_job_id>
#
#SBATCH --job-name=launch_magicoder_full_evalplus
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=brozonoyer_umass_edu

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
cd "${REPO_ROOT}"
mkdir -p logs

FULL_ROOT=${FULL_ROOT:-"${REPO_ROOT}/output_models/magicoder_oss_full_1p5B"}
WORKER=${EVALPLUS_WORKER_SBATCH:-train_scripts/evalplus_model_sweep.sbatch}
SEEDS=${EVALPLUS_SEEDS:-"0 1 2"}
DATASETS=${EVALPLUS_DATASETS:-"humaneval mbpp"}
THRESHOLD=${EVALPLUS_THRESHOLD:-0.85}
BLOCK_CACHE=${EVALPLUS_BLOCK_CACHE:-auto}
DRY_RUN=${DRY_RUN:-0}

AUTO_DEPENDENCY_FROM_SQUEUE=${AUTO_DEPENDENCY_FROM_SQUEUE:-1}
SUBMIT_BPTT_WITHOUT_DEPENDENCY=${SUBMIT_BPTT_WITHOUT_DEPENDENCY:-0}

TRAIN_JOB_NOCARRY=${TRAIN_JOB_NOCARRY:-ft_magicoder_bptt_nocarry_ep1_1p5B}
TRAIN_JOB_LOOPGUARD=${TRAIN_JOB_LOOPGUARD:-ft_magicoder_bptt_loopguard_ep1_1p5B}
TRAIN_JOB_MLP=${TRAIN_JOB_MLP:-ft_magicoder_bptt_mlp_ep1_1p5B}

normalize_dependency() {
    local dep="${1:-}"
    if [ -z "$dep" ]; then
        return 0
    fi
    if [[ "$dep" == *:* ]]; then
        printf '%s\n' "$dep"
    else
        printf 'afterok:%s\n' "$dep"
    fi
}

dependency_from_squeue() {
    local job_name="$1"
    if [ "${AUTO_DEPENDENCY_FROM_SQUEUE}" != "1" ]; then
        return 0
    fi
    if ! command -v squeue >/dev/null 2>&1; then
        return 0
    fi
    timeout 10s squeue --me -h -o "%i %j" 2>/dev/null \
        | awk -v name="$job_name" '$2 == name { print $1 }' \
        | paste -sd: -
}

submit_eval() {
    local label="$1"
    local use_carry="$2"
    local dependency="${3:-}"
    shift 3
    local -a model_env=("$@")
    local -a sbatch_args=(--job-name "evalplus_${label}")

    if [ -n "$dependency" ]; then
        sbatch_args+=(--dependency "$dependency")
    fi

    local -a env_cmd=(
        env
        "MODEL_LABEL=${label}"
        "EVALPLUS_USE_CARRY=${use_carry}"
        "EVALPLUS_SEEDS=${SEEDS}"
        "EVALPLUS_DATASETS=${DATASETS}"
        "EVALPLUS_THRESHOLD=${THRESHOLD}"
        "EVALPLUS_BLOCK_CACHE=${BLOCK_CACHE}"
    )
    env_cmd+=("${model_env[@]}")

    echo "---- submit ${label} dependency=${dependency:-<none>} ----"
    printf '%q ' "${env_cmd[@]}" sbatch "${sbatch_args[@]}" "${WORKER}"
    echo
    if [ "${DRY_RUN}" != "1" ]; then
        "${env_cmd[@]}" sbatch "${sbatch_args[@]}" "${WORKER}"
    fi
}

maybe_submit_dependent_eval() {
    local label="$1"
    local use_carry="$2"
    local explicit_dep="$3"
    local train_job_name="$4"
    local model_root="$5"
    local dep discovered
    dep=$(normalize_dependency "$explicit_dep")
    if [ -z "$dep" ]; then
        discovered=$(dependency_from_squeue "$train_job_name")
        dep=$(normalize_dependency "$discovered")
    fi
    if [ -z "$dep" ] && [ "${SUBMIT_BPTT_WITHOUT_DEPENDENCY}" != "1" ]; then
        echo "[skip] ${label}: no dependency found. Set ${label^^}_DEPENDENCY or SUBMIT_BPTT_WITHOUT_DEPENDENCY=1." >&2
        return 0
    fi
    submit_eval "$label" "$use_carry" "$dep" "MODEL_ROOT=${model_root}" "ONLY_CHECKPOINTS=1"
}

echo "===== LAUNCH MAGICODER FULL EVALPLUS ====="
echo "FULL_ROOT                  = ${FULL_ROOT}"
echo "SEEDS                      = ${SEEDS}"
echo "DATASETS                   = ${DATASETS}"
echo "THRESHOLD                  = ${THRESHOLD}"
echo "BLOCK_CACHE                = ${BLOCK_CACHE}"
echo "AUTO_DEPENDENCY_FROM_SQUEUE = ${AUTO_DEPENDENCY_FROM_SQUEUE}"
echo "DRY_RUN                    = ${DRY_RUN}"
echo "=========================================="

submit_eval \
    "off_the_shelf_fast_dllm_v2_1p5b" \
    "0" \
    "" \
    "MODEL_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B"

submit_eval \
    "hf_baseline_fast_dllm_v2_1p5b" \
    "0" \
    "" \
    "MODEL_PATH=${REPO_ROOT}/output_models/hf_baseline_Fast_dLLM_v2_1.5B"

submit_eval \
    "sft_full_vanilla" \
    "0" \
    "" \
    "MODEL_ROOT=${FULL_ROOT}/vanilla_nopack1024_bs2x16_ep1_lr1e-5" \
    "ONLY_CHECKPOINTS=1"

maybe_submit_dependent_eval \
    "nocarry" \
    "0" \
    "${NOCARRY_DEPENDENCY:-${NOCARRY_JOB_ID:-}}" \
    "${TRAIN_JOB_NOCARRY}" \
    "${FULL_ROOT}/bptt_nocarry_puma_nopack1024_bs2x16_ep1_lr1e-5"

maybe_submit_dependent_eval \
    "loopguard" \
    "1" \
    "${LOOPGUARD_DEPENDENCY:-${LOOPGUARD_JOB_ID:-}}" \
    "${TRAIN_JOB_LOOPGUARD}" \
    "${FULL_ROOT}/bptt_loopguard_puma_nopack1024_bs2x16_ep1_lr1e-5"

maybe_submit_dependent_eval \
    "mlp" \
    "1" \
    "${MLP_DEPENDENCY:-${MLP_JOB_ID:-}}" \
    "${TRAIN_JOB_MLP}" \
    "${FULL_ROOT}/bptt_mlp_puma_nopack1024_bs2x16_ep1_lr1e-5"
