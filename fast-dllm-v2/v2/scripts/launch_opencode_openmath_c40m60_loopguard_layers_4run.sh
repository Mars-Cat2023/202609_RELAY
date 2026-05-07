#!/usr/bin/env bash
# Submit four 2-GPU c40m60 Loopguard layer ablations.
#
# This family keeps the default c40m60 Loopguard recipe:
#   MAX_GRAD_NORM=1.0, BPTT_STOP_GRAD_H_S=0, BPTT_UNMASK_STRATEGY=bd
# and changes only which zero-based decoder layer supplies the loophole carry.
#
# Run from Fast-dLLM/v2:
#   DRY_RUN=1 bash scripts/launch_opencode_openmath_c40m60_loopguard_layers_4run.sh
#   bash scripts/launch_opencode_openmath_c40m60_loopguard_layers_4run.sh

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

recipe_tag=${RECIPE_TAG:-opencode_openmath_60k_c40m60_loopguard_layers}
dataset_path=${DATASET_PATH:-data/opencode_openmath_60k_c40m60/train_conversation}
output_models_root=${OUTPUT_MODELS_ROOT:-/scratch4/workspace/brozonoyer_umass_edu-fastdllm_ckpts/output_models}
size_tag=1p5B
family_dir="${output_models_root}/${recipe_tag}_${size_tag}"
repo_family_link="output_models/${recipe_tag}_${size_tag}"

if [ ! -d "${dataset_path}" ]; then
    echo "ERROR: dataset path does not exist: ${dataset_path}" >&2
    echo "Run the c40m60 data prep first; this launcher reuses that dataset." >&2
    exit 1
fi

mkdir -p "${family_dir}" output_models
if [ -L "${repo_family_link}" ]; then
    current_target=$(readlink "${repo_family_link}")
    if [ "${current_target}" != "${family_dir}" ]; then
        echo "ERROR: ${repo_family_link} already points to ${current_target}, expected ${family_dir}" >&2
        exit 1
    fi
elif [ -e "${repo_family_link}" ]; then
    echo "ERROR: ${repo_family_link} already exists and is not a symlink." >&2
    exit 1
else
    ln -s "${family_dir}" "${repo_family_link}"
fi

reservation_arg=()
if [ -n "${RESERVATION:-}" ]; then
    reservation_arg=(--reservation "${RESERVATION}")
fi

common_export="ALL,DATASET_PATH=${dataset_path},RECIPE_TAG=${recipe_tag},OUTPUT_MODELS_ROOT=${output_models_root},BLOCK_SIZE=${BLOCK_SIZE:-2048},DISABLE_GROUP_TEXTS=${DISABLE_GROUP_TEXTS:-1},NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3},LEARNING_RATE=${LEARNING_RATE:-5e-6},MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0},SAVE_STEPS=${SAVE_STEPS:-200},SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0},PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2},GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16},BPTT_UNMASK_STRATEGY=bd"

submit() {
    local layer=$1
    local cmd=(
        sbatch "${reservation_arg[@]}"
        --job-name="ft_c40m60_lg_l${layer}_1p5B"
        --export="${common_export},CARRY_MODE=loopguard,BPTT_STOP_GRAD_H_S=0,BPTT_LOOPHOLE_LAYER=${layer}"
        train_scripts/finetune_opencode_openmath_bptt.sbatch
    )
    echo "[loopguard_layer${layer}] ${cmd[*]}" >&2
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    local out
    out=$("${cmd[@]}")
    echo "[loopguard_layer${layer}] ${out}" >&2
    echo "${out}" | awk '{print $NF}'
}

declare -A job_ids=()
for layer in 13 17 21 25; do
    job_ids["layer${layer}"]=$(submit "${layer}")
done

echo "Output family:"
echo "  scratch: ${family_dir}"
echo "  repo:    ${repo_family_link} -> ${family_dir}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

printf "Submitted jobs:\n"
for key in layer13 layer17 layer21 layer25; do
    printf "  %-10s %s\n" "${key}" "${job_ids[$key]}"
done

echo
echo "Eval is NOT auto-submitted. Example:"
echo "  MODEL_ROOT=${repo_family_link}/bptt_loopguard_layer13_puma_nopack2048_bs2x16x2_ep3_lr5e-6 \\"
echo "  USE_CARRY=1 CHECKPOINTS=200,400,600,800,final \\"
echo "  sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch"
