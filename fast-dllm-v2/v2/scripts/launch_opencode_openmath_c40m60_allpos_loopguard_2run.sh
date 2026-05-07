#!/usr/bin/env bash
# Submit the two all-position Loopguard ablations for the c40m60 mixture.
#
# These mirror the loopguard and loopguard_stopgrad rows in
# launch_opencode_openmath_c40m60_4run.sh, but set
# BPTT_LOOPHOLE_POSITION_GUARD=mutable so Loopholing is injected at every
# mutable noisy-half training position and every active decoding-block position.
#
# Run from Fast-dLLM/v2 after data prep:
#   bash scripts/launch_opencode_openmath_c40m60_allpos_loopguard_2run.sh
#
# Set DRY_RUN=1 to print commands without submitting.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

recipe_tag=${RECIPE_TAG:-opencode_openmath_60k_c40m60}
dataset_path=${DATASET_PATH:-data/${recipe_tag}/train_conversation}
if [ ! -d "${dataset_path}" ]; then
    echo "ERROR: dataset path does not exist: ${dataset_path}" >&2
    echo "Run:" >&2
    echo "  python scripts/prep_opencode_openmath_mix.py \\" >&2
    echo "    --out_dir data/${recipe_tag} \\" >&2
    echo "    --code_rows 24000 --math_rows 36000 --require_code_def" >&2
    exit 1
fi

reservation_arg=()
if [ -n "${RESERVATION:-}" ]; then
    reservation_arg=(--reservation "${RESERVATION}")
fi

common_export="ALL,DATASET_PATH=${dataset_path},RECIPE_TAG=${recipe_tag},BLOCK_SIZE=${BLOCK_SIZE:-2048},DISABLE_GROUP_TEXTS=${DISABLE_GROUP_TEXTS:-1},NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3},LEARNING_RATE=${LEARNING_RATE:-5e-6},SAVE_STEPS=${SAVE_STEPS:-200},SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0},PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2},GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16},BPTT_LOOPHOLE_POSITION_GUARD=mutable"

submit() {
    local label=$1
    shift
    local cmd=(sbatch "${reservation_arg[@]}" "$@")
    echo "[$label] ${cmd[*]}" >&2
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    local out
    out=$("${cmd[@]}")
    echo "[$label] ${out}" >&2
    echo "${out}" | awk '{print $NF}'
}

declare -A job_ids=()

job_ids[loopguard_allpos_stopgrad]=$(submit loopguard_allpos_stopgrad \
    --job-name=ft_c40m60_bptt_loopguard_allpos_stopgrad_1p5B \
    --export="${common_export},CARRY_MODE=loopguard,BPTT_STOP_GRAD_H_S=1" \
    train_scripts/finetune_opencode_openmath_bptt.sbatch)

job_ids[loopguard_allpos]=$(submit loopguard_allpos \
    --job-name=ft_c40m60_bptt_loopguard_allpos_1p5B \
    --export="${common_export},CARRY_MODE=loopguard,BPTT_STOP_GRAD_H_S=0" \
    train_scripts/finetune_opencode_openmath_bptt.sbatch)

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

printf "Submitted jobs:\n"
for key in loopguard_allpos_stopgrad loopguard_allpos; do
    printf "  %-30s %s\n" "${key}" "${job_ids[$key]}"
done

echo
echo "Expected output roots:"
echo "  output_models/${recipe_tag}_1p5B/bptt_loopguard_stopgrad_allpos_puma_nopack${BLOCK_SIZE:-2048}_bs${PER_DEVICE_TRAIN_BATCH_SIZE:-2}x${GRADIENT_ACCUMULATION_STEPS:-16}x2_ep${NUM_TRAIN_EPOCHS:-3}_lr${LEARNING_RATE:-5e-6}"
echo "  output_models/${recipe_tag}_1p5B/bptt_loopguard_allpos_puma_nopack${BLOCK_SIZE:-2048}_bs${PER_DEVICE_TRAIN_BATCH_SIZE:-2}x${GRADIENT_ACCUMULATION_STEPS:-16}x2_ep${NUM_TRAIN_EPOCHS:-3}_lr${LEARNING_RATE:-5e-6}"
