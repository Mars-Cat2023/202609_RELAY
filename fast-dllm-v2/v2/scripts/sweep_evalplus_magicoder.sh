#!/usr/bin/env bash
# Sweep EvalPlus (HumanEval / MBPP) over Magicoder finetune dirs: python vs full;
# SFT, nocarry, loopguard, mlp (no CAB); every existing checkpoint-* under each run dir
# (e.g. only the last N kept by SAVE_TOTAL_LIMIT=3 + SAVE_STEPS=250 is fine); 3 seeds.
# Checkpoints are evaluated in ascending step order (earliest -> latest) by parsing the
# number after "checkpoint-".
#
# Stochasticity: only `generate_evalplus_jsonl.py` is stochastic here — it sets
#   `--seed` (torch, cuDNN, and any sampling in the diffusion path). The EvalPlus
#   scorer `evalplus.evaluate` re-runs tests on a fixed JSONL and is deterministic.
#
# Usage:
#   cd /path/to/Fast-dLLM/v2
#   conda activate venv_fastdllm   # torch + evalplus + repo
#   ./scripts/sweep_evalplus_magicoder.sh
#
# Env (optional):
#   REPO_ROOT   — default: Fast-dLLM/v2 (parent of scripts/)
#   SEEDS       — space-separated, default: "0 1 2"
#   DATASETS    — e.g. "humaneval" and/or "mbpp", default: "humaneval mbpp"
#   ONLY_LATEST_CHECKPOINT=1 — only the last checkpoint-* per run dir (highest step; or
#     the `final` directory when there are no checkpoint-*)
#   RUN_SFT=0  — skip vanilla SFT (python + full). Default: 1 (run SFT first).
#   BPTT_MODES  — space-separated subset of: nocarry loopguard mlp. Default: all three.
#     Example:  BPTT_MODES="loopguard mlp"  to skip nocarry and only run BPTT carry runs.
#   THRESHOLD   — for generate_evalplus_jsonl, default: 0.85
#   DRY_RUN=1   — print commands only
#   SKIP_DONE=1 — skip if output JSONL already exists (default 1)
#   FORCE=1     — regenerate even if JSONL exists
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
if [[ ! -f "${REPO_ROOT}/eval.py" ]]; then
  echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like Fast-dLLM/v2 (missing eval.py)" >&2
  exit 1
fi
cd "${REPO_ROOT}"

SEEDS=${SEEDS:-"0 1 2"}
DATASETS=${DATASETS:-"humaneval mbpp"}
ONLY_LATEST_CHECKPOINT=${ONLY_LATEST_CHECKPOINT:-0}
RUN_SFT=${RUN_SFT:-1}
# shellcheck disable=SC2206
BPTT_MODES=(${BPTT_MODES:-"nocarry loopguard mlp"})
THRESHOLD=${THRESHOLD:-0.85}
DRY_RUN=${DRY_RUN:-0}
SKIP_DONE=${SKIP_DONE:-1}
FORCE=${FORCE:-0}

# Default training output dirs (match finetune_magicoder_oss.sbatch + finetune_magicoder_oss_bptt.sbatch)
OUT_PY="${REPO_ROOT}/output_models/magicoder_oss_python_1p5B"
OUT_FULL="${REPO_ROOT}/output_models/magicoder_oss_full_1p5B"
VAN_PY="${OUT_PY}/vanilla_nopack1024_bs2x16_ep2_lr1e-5"
VAN_FULL="${OUT_FULL}/vanilla_nopack1024_bs2x16_ep1_lr1e-5"

run_gen_eval() {
  local model_path="$1"
  local out_jsonl="$2"
  local use_carry_flag="$3"
  local seed="$4"
  local dataset="$5"

  mkdir -p "$(dirname "${out_jsonl}")"

  if [[ "${FORCE}" != "1" && "${SKIP_DONE}" == "1" && -f "${out_jsonl}" ]]; then
    echo "  [skip] exists: ${out_jsonl}"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${use_carry_flag}" == "1" ]]; then
      echo "  DRY: python scripts/generate_evalplus_jsonl.py --model_path ${model_path} --dataset ${dataset} --output_jsonl ${out_jsonl} --seed ${seed} --threshold ${THRESHOLD} --use_carry"
    else
      echo "  DRY: python scripts/generate_evalplus_jsonl.py --model_path ${model_path} --dataset ${dataset} --output_jsonl ${out_jsonl} --seed ${seed} --threshold ${THRESHOLD}"
    fi
    echo "  DRY: evalplus.evaluate --dataset ${dataset} --samples ${out_jsonl}"
    return 0
  fi

  echo "  [gen] ${dataset} seed=${seed} -> $(basename "${out_jsonl}")"
  if [[ "${use_carry_flag}" == "1" ]]; then
    python scripts/generate_evalplus_jsonl.py \
      --model_path "${model_path}" \
      --dataset "${dataset}" \
      --output_jsonl "${out_jsonl}" \
      --seed "${seed}" \
      --threshold "${THRESHOLD}" \
      --use_carry
  else
    python scripts/generate_evalplus_jsonl.py \
      --model_path "${model_path}" \
      --dataset "${dataset}" \
      --output_jsonl "${out_jsonl}" \
      --seed "${seed}" \
      --threshold "${THRESHOLD}"
  fi
  echo "  [eval] evalplus ${dataset}"
  evalplus.evaluate --dataset "${dataset}" --samples "${out_jsonl}"
}

# List checkpoint-* under run dir, sorted by numeric step (earliest first).
# If none, use the run dir if it has config.json (e.g. only a merged final save).
list_model_paths() {
  local d="$1" p bn n line
  [[ -d "$d" ]] || return
  local -a cks=()
  mapfile -t cks < <(
    find "$d" -mindepth 1 -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | while IFS= read -r p; do
      bn=$(basename "$p")
      n="${bn#checkpoint-}"
      if [[ "$n" =~ ^[0-9]+$ ]]; then
        printf '%d\t%s\n' "$n" "$p"
      else
        # Non-numeric suffix: stable fallback at end
        printf '%d\t%s\n' "9999999999" "$p"
      fi
    done | sort -n -k1,1 | cut -f2-
  )
  if ((${#cks[@]} > 0)); then
    printf '%s\n' "${cks[@]}"
  elif [[ -f "$d/config.json" ]]; then
    echo "$d"
  fi
}

sweep_dir() {
  local label_split="$1"   # python | full
  local run_name="$2"      # sft | nocarry | loopguard | mlp
  local out_base="$3"
  local use_carry="$4"     # 0 or 1

  echo "---- ${label_split} / ${run_name} (base: ${out_base}) use_carry=${use_carry} ----"
  if [[ ! -d "${out_base}" ]]; then
    echo "  [warn] missing directory, skip: ${out_base}"
    return 0
  fi

  local -a paths=()
  mapfile -t paths < <(list_model_paths "${out_base}")

  if ((${#paths[@]} == 0)); then
    echo "  [warn] no checkpoint-* and no config at: ${out_base}"
    return 0
  fi

  if [[ "${ONLY_LATEST_CHECKPOINT}" == "1" ]]; then
    local _i=$((${#paths[@]} - 1))
    local _last="${paths[_i]}"
    paths=("${_last}")
    echo "  [only latest] using $(basename "${_last}") (ONLY_LATEST_CHECKPOINT=1)"
  fi

  echo -n "  [checkpoints, earliest->latest]:"
  local _b
  for _b in "${paths[@]}"; do
    echo -n " $(basename "${_b}")"
  done
  echo

  local p ck_label jsonl
  for p in "${paths[@]}"; do
    if [[ "${p}" == "${out_base}" ]]; then
      ck_label="final"
    else
      ck_label=$(basename "$p")
    fi
    for seed in ${SEEDS}; do
      for ds in ${DATASETS}; do
        jsonl="${REPO_ROOT}/evalplus_results/magicoder_sweep/${label_split}__${run_name}__${ck_label}__${ds}__s${seed}.jsonl"
        run_gen_eval "$p" "$jsonl" "$use_carry" "$seed" "$ds"
      done
    done
  done
}

echo "REPO_ROOT=${REPO_ROOT}"
echo "SEEDS=${SEEDS}  DATASETS=${DATASETS}  ONLY_LATEST_CHECKPOINT=${ONLY_LATEST_CHECKPOINT}  RUN_SFT=${RUN_SFT}  BPTT_MODES: ${BPTT_MODES[*]}  THRESHOLD=${THRESHOLD}  DRY_RUN=${DRY_RUN}  SKIP_DONE=${SKIP_DONE}  FORCE=${FORCE}"

if [[ "${RUN_SFT}" == "1" ]]; then
  sweep_dir "python" "sft" "${VAN_PY}" "0"
  sweep_dir "full"   "sft" "${VAN_FULL}" "0"
else
  echo "Skipping SFT (RUN_SFT=0)"
fi

for c in "${BPTT_MODES[@]}"; do
  if [[ "$c" != "nocarry" && "$c" != "loopguard" && "$c" != "mlp" ]]; then
    echo "ERROR: unknown BPTT mode '${c}' (use nocarry, loopguard, mlp)" >&2
    exit 1
  fi
  u=0
  if [[ "$c" == "loopguard" || "$c" == "mlp" ]]; then u=1; fi
  sweep_dir "python" "${c}" "${OUT_PY}/bptt_${c}_puma_nopack1024_bs2x16_ep2_lr1e-5" "$u"
  sweep_dir "full" "${c}" "${OUT_FULL}/bptt_${c}_puma_nopack1024_bs2x16_ep1_lr1e-5" "$u"
done

echo "Done. JSONL and *_eval_results.json under: ${REPO_ROOT}/evalplus_results/magicoder_sweep/"
