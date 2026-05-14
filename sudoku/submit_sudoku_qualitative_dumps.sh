#!/usr/bin/env bash
# Dump per-step Sudoku decode trajectories for the trained models in
# `wandb.ai/<entity>/BPTT-sudoku` (4 objectives x {tied, untied} x {seed1,seed2,seed3})
# at a sweep of inference confidence thresholds, then assemble a single parquet table.
#
# - Reuses ``python -m relay.dump_decode_trajectories`` (no training-time changes;
#   sets ``capture_trajectory=True`` on the loaded predictor only).
# - Writes one JSONL per (run, seed, tau) under ``$OUT_DIR/dumps/`` and a CSV
#   manifest at ``$OUT_DIR/manifest.csv`` that ``n_way_pair_trajectories``
#   consumes.
#
# Usage (from repo root):
#   DO=run    bash submit_sudoku_qualitative_dumps.sh   # foreground, sequential
#   DO=submit bash submit_sudoku_qualitative_dumps.sh   # one sbatch per (run, seed, tau)
#   DO=print  bash submit_sudoku_qualitative_dumps.sh   # dry-run; print commands only
#
# Optional env (defaults shown):
#   SEEDS="1 2 3"
#   SKIP_DIVERGED=1               # skip checkpoints that diverged at training-time `legal_rate`
#   THRESHOLDS="0.01 0.05 0.10 0.15 0.25 0.35 0.45"
#   LIMIT_VAL_BATCHES=2000
#   MAX_EXAMPLES=2000
#   SEED=1                        # shared dataloader seed (NOT model seed)
#   OUT_DIR=outputs/sudoku_analysis
#   CKPT_NAME=last.ckpt           # or best.ckpt
#   ASSEMBLE=1                    # after dumps, run n_way_pair_trajectories
#   ONLY_OBJECTIVES=""            # subset filter, e.g. "relay mlm_uniform"
#   ONLY_TYINGS=""                # subset filter, e.g. "untied"
#   ONLY_SEEDS=""                 # subset filter (overrides SEEDS for filtering only)
#   HARD_FILTER=""                # one of {"", "extreme", "deduction_only"}; biases the
#                                 # dump pool toward the long-tail difficulty buckets.
#   PARTITION=gpu
#   GPU_CONSTRAINT=vram40,bf16    # use vram80 if GPU-OOM on relay
#   SLURM_MEM=64G                 # set empty to omit `--mem` if your site forbids it
#   SLURM_CPUS_PER_TASK=2
#   SLURM_TIME=04:00:00
#   SLURM_RESERVATION=""          # cluster reservation (anonymised)
#   PYBIN=$REPO_ROOT/.venv_relay_sudoku/bin/python

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"

DO="${DO:-print}"
SEEDS="${SEEDS:-1 2 3}"
SKIP_DIVERGED="${SKIP_DIVERGED:-1}"
THRESHOLDS="${THRESHOLDS:-0.01 0.05 0.10 0.15 0.25 0.35 0.45}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-2000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-2000}"
SEED="${SEED:-1}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/sudoku_analysis}"
DUMP_DIR="$OUT_DIR/dumps"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"
ASSEMBLE="${ASSEMBLE:-0}"
ONLY_OBJECTIVES="${ONLY_OBJECTIVES:-}"
ONLY_TYINGS="${ONLY_TYINGS:-}"
ONLY_SEEDS="${ONLY_SEEDS:-}"
HARD_FILTER="${HARD_FILTER:-}"
case "${HARD_FILTER}" in
  "")
    HARD_HYDRA_OVERRIDES=""
    ;;
  extreme)
    HARD_HYDRA_OVERRIDES="++datamodule.dataset_managers.val.infill_prediction.filter_fn=xlm.tasks.sudoku_extreme.sudoku_extreme_extreme_filter_fn ++datamodule.dataset_managers.val.infill_prediction.filter_suffix=extreme"
    ;;
  deduction_only)
    HARD_HYDRA_OVERRIDES="++datamodule.dataset_managers.val.infill_prediction.filter_fn=xlm.tasks.sudoku_extreme.sudoku_extreme_deduction_only_filter_fn ++datamodule.dataset_managers.val.infill_prediction.filter_suffix=deduction_only"
    ;;
  *)
    echo "Unknown HARD_FILTER=$HARD_FILTER (expected: '' | 'extreme' | 'deduction_only')" >&2
    exit 2
    ;;
esac
PARTITION="${PARTITION:-gpu}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-vram40,bf16}"
SLURM_TIME="${SLURM_TIME:-04:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-2}"
SLURM_RESERVATION="${SLURM_RESERVATION:-}"
PYBIN="${PYBIN:-${ROOT}/.venv_relay_sudoku/bin/python}"

mkdir -p "$OUT_DIR" "$DUMP_DIR"
MANIFEST="$OUT_DIR/manifest.csv"
echo "run_id,objective,embed_tying,seed,tau,jsonl" > "$MANIFEST"

# Run table.
# Format: <key>|<base_job_name>|<experiment>|<embed_tying>|<objective>|<extra hydra overrides>
runs=(
  "mlm_uniform_untied|sudoku_extreme_mlm_uniform_300k_untied|sudoku_extreme_mlm_uniform|untied|mlm_uniform|"
  "mlm_uniform_tied|sudoku_extreme_mlm_uniform_300k_tied|sudoku_extreme_mlm_uniform|tied|mlm_uniform|++model.tie_embeddings=true"
  "rollout_untied|sudoku_extreme_rollout_300k_untied|sudoku_extreme_relay_bptt|untied|rollout|model=rotary_transformer_xtiny loss.with_relay=false loss.stop_grad_h_s=true predictor.with_relay=false"
  "rollout_tied|sudoku_extreme_rollout_300k_tied|sudoku_extreme_relay_bptt|tied|rollout|model=rotary_transformer_xtiny ++model.tie_embeddings=true loss.with_relay=false loss.stop_grad_h_s=true predictor.with_relay=false"
  "relay_sg_untied|sudoku_extreme_relay_sg_300k_untied|sudoku_extreme_relay_bptt|untied|relay_sg|loss.with_relay=true loss.stop_grad_h_s=true predictor.with_relay=true"
  "relay_sg_tied|sudoku_extreme_relay_sg_300k_tied|sudoku_extreme_relay_bptt|tied|relay_sg|++model.tie_embeddings=true loss.with_relay=true loss.stop_grad_h_s=true predictor.with_relay=true"
  "relay_untied|sudoku_extreme_relay_bptt_steps2_300k_untied|sudoku_extreme_relay_bptt|untied|relay|loss.with_relay=true loss.stop_grad_h_s=false loss.num_steps=2 predictor.with_relay=true"
  "relay_tied|sudoku_extreme_relay_bptt_steps2_300k_tied|sudoku_extreme_relay_bptt|tied|relay|++model.tie_embeddings=true loss.with_relay=true loss.stop_grad_h_s=false loss.num_steps=2 predictor.with_relay=true"
)

# (objective, tying, seed) tuples that diverged at train-time `val/prediction/legal_rate`.
DIVERGED_KEYS="mlm_uniform_tied:3 relay_sg_tied:1"

_in_filter() {
  local needle="$1" haystack="$2"
  if [[ -z "${haystack// }" ]]; then
    return 0
  fi
  for tok in ${haystack}; do
    if [[ "${tok}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

_is_diverged() {
  local key="$1" seed="$2"
  for tok in ${DIVERGED_KEYS}; do
    if [[ "${tok}" == "${key}:${seed}" ]]; then
      return 0
    fi
  done
  return 1
}

dispatch() {
  local cmd_line="$1"
  local job_name="$2"
  case "$DO" in
    print)
      echo "[print] $cmd_line"
      ;;
    run)
      echo "[run] $cmd_line"
      eval "$cmd_line"
      ;;
    submit)
      local sbatch_args=(
        --job-name="$job_name"
        --partition="$PARTITION"
        --constraint="$GPU_CONSTRAINT"
        --gres=gpu:1
        --cpus-per-task="$SLURM_CPUS_PER_TASK"
        --time="$SLURM_TIME"
        --output="$OUT_DIR/slurm-%x-%j.out"
        --error="$OUT_DIR/slurm-%x-%j.err"
      )
      if [[ -n "${SLURM_MEM:-}" ]]; then
        sbatch_args+=("--mem=$SLURM_MEM")
      fi
      if [[ -n "$SLURM_RESERVATION" ]]; then
        sbatch_args+=("--reservation=$SLURM_RESERVATION")
      fi
      sbatch "${sbatch_args[@]}" --wrap="cd $ROOT && export PROJECT_ROOT=$ROOT && $cmd_line"
      ;;
    *)
      echo "Unknown DO=$DO (expected: run | submit | print)" >&2
      exit 2
      ;;
  esac
}

dump_one() {
  local key="$1" base_job="$2" experiment="$3" tying="$4" objective="$5" extras="$6" seed="$7" tau="$8"

  local job_name run_id
  if [[ -z "${seed}" ]]; then
    job_name="${base_job}"
    run_id="${key}"
  else
    job_name="${base_job}_seed${seed}"
    run_id="${key}_seed${seed}"
  fi

  local ckpt="logs/${job_name}/checkpoints/${CKPT_NAME}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] ${run_id} tau=${tau}: missing checkpoint ${ckpt}" >&2
    return 0
  fi
  if [[ "${SKIP_DIVERGED}" == "1" ]] && _is_diverged "${key}" "${seed:-?}"; then
    echo "[skip] ${run_id} tau=${tau}: in DIVERGED_KEYS" >&2
    return 0
  fi
  local tau_tag
  tau_tag=$(printf "%.2f" "$tau" | tr '.' 'p')
  local out_jsonl="$DUMP_DIR/${run_id}__tau_${tau_tag}.jsonl"
  local sbatch_name="dump_${run_id}_tau_${tau_tag}"

  local cmd
  cmd="$PYBIN -m relay.dump_decode_trajectories"
  cmd+=" experiment=$experiment"
  cmd+=" +generation.ckpt_path=$ckpt"
  cmd+=" predictor.threshold=$tau"
  cmd+=" seed=$SEED"
  cmd+=" per_device_batch_size=1 global_batch_size=1"
  # Keep dataloader single-threaded: HF maps + multi-worker spike host RSS and Slurm
  # tends to OOM-kill the step before the first batch. PyTorch requires
  # prefetch_factor=None when num_workers==0.
  cmd+=" num_dataloader_workers=0 num_dataset_workers=0 dataloader_prefetch_factor=null"
  cmd+=" +dump.output_path=$out_jsonl"
  cmd+=" +dump.max_examples=$MAX_EXAMPLES"
  cmd+=" +dump.limit_val_batches=$LIMIT_VAL_BATCHES"
  if [[ -n "$HARD_HYDRA_OVERRIDES" ]]; then
    cmd+=" $HARD_HYDRA_OVERRIDES"
  fi
  if [[ -n "$extras" ]]; then
    cmd+=" $extras"
  fi
  dispatch "$cmd" "$sbatch_name"

  echo "${run_id},${objective},${tying},${seed:-},${tau},${out_jsonl}" >> "$MANIFEST"
}

# Iterate seeds (outer) x runs (middle) x thresholds (inner). When SEEDS is empty
# the seed loop runs once with seed="", reproducing the original 8-run behaviour
# against the un-suffixed log dirs.
seed_iter="${SEEDS}"
if [[ -z "${seed_iter// }" ]]; then
  seed_iter=""
fi

for seed in ${seed_iter:-""}; do
  if [[ -n "${seed}" ]] && ! _in_filter "${seed}" "${ONLY_SEEDS}"; then
    continue
  fi
  for row in "${runs[@]}"; do
    IFS='|' read -r key base_job experiment tying objective extras <<< "$row"
    if ! _in_filter "${objective}" "${ONLY_OBJECTIVES}"; then
      continue
    fi
    if ! _in_filter "${tying}" "${ONLY_TYINGS}"; then
      continue
    fi
    for tau in $THRESHOLDS; do
      dump_one "$key" "$base_job" "$experiment" "$tying" "$objective" "$extras" "$seed" "$tau"
    done
  done
done

echo
echo "Manifest written: $MANIFEST"
echo "JSONL dumps under: $DUMP_DIR"
echo "(run with DO=run for foreground, DO=submit for SLURM, DO=print for dry-run.)"

if [[ "$ASSEMBLE" == "1" && "$DO" == "run" ]]; then
  echo
  echo "Assembling parquet via n_way_pair_trajectories..."
  "$PYBIN" -m relay.n_way_pair_trajectories \
    --manifest "$MANIFEST" \
    --hf-split test \
    --out "$OUT_DIR/sudoku_analysis.parquet" \
    --shared-only
fi
