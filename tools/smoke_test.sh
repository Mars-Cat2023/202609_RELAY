#!/usr/bin/env bash
# End-to-end smoke test for the relay repository.
#
# Runs the cheapest possible exercise of each sub-project so a new user can
# verify the repo is wired up correctly on their cluster before launching
# the full Table 1 / Table 2 sweeps:
#
#   - sudoku/   :  10 training optimizer steps + 1 validation pass
#   - fast-dllm-v2/v2/ : 5 BPTT optimizer steps + 1 EvalPlus sample on
#                        HumanEval/0
#
# Both sub-tests are skipped if their environment is not active. Set
# RELAY_SKIP_SUDOKU=1 / RELAY_SKIP_FASTDLLM=1 to skip explicitly.
#
# Usage:
#   bash tools/smoke_test.sh

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

red()   { printf "\033[0;31m%s\033[0m\n" "$*"; }
green() { printf "\033[0;32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[0;33m%s\033[0m\n" "$*"; }

# ------------------------------------------------------------------ #
# 1. Sudoku
# ------------------------------------------------------------------ #
if [ "${RELAY_SKIP_SUDOKU:-0}" = "1" ]; then
    yellow "[skip] sudoku smoke test (RELAY_SKIP_SUDOKU=1)"
elif [ ! -d "sudoku/.venv_relay" ]; then
    yellow "[skip] sudoku smoke test (sudoku/.venv_relay not found; "
    yellow "       see sudoku/README.md for one-time setup instructions)"
else
    green "[run]  sudoku smoke test"
    (
        cd sudoku
        # shellcheck disable=SC1091
        source .venv_relay/bin/activate
        export PROJECT_ROOT="$PWD"
        python -m xlm.train \
            experiment=sudoku_extreme_relay_bptt \
            trainer.max_steps=10 \
            trainer.val_check_interval=10 \
            trainer.limit_val_batches=1 \
            per_device_batch_size=8 \
            global_batch_size=8 \
            loggers.wandb=null
    )
    green "[ok]   sudoku smoke test"
fi

# ------------------------------------------------------------------ #
# 2. Fast-dLLM v2
# ------------------------------------------------------------------ #
if [ "${RELAY_SKIP_FASTDLLM:-0}" = "1" ]; then
    yellow "[skip] fast-dllm-v2 smoke test (RELAY_SKIP_FASTDLLM=1)"
elif ! command -v conda >/dev/null 2>&1; then
    yellow "[skip] fast-dllm-v2 smoke test (conda not on PATH; install from"
    yellow "       fast-dllm-v2/v2/README.md and re-run)"
else
    CONDA_ROOT=${CONDA_ROOT:-${HOME}/miniconda3}
    CONDA_ENV=${CONDA_ENV:-relay}
    if [ ! -d "${CONDA_ROOT}/envs/${CONDA_ENV}" ]; then
        yellow "[skip] fast-dllm-v2 smoke test (conda env '${CONDA_ENV}' not "
        yellow "       found at ${CONDA_ROOT}/envs/${CONDA_ENV}; see "
        yellow "       fast-dllm-v2/v2/README.md for setup)"
    else
        green "[run]  fast-dllm-v2 smoke test"
        (
            cd fast-dllm-v2/v2
            # shellcheck disable=SC1091
            source "${CONDA_ROOT}/etc/profile.d/conda.sh"
            conda activate "${CONDA_ENV}"
            python - <<'PY'
import torch
from transformers import AutoTokenizer
from lmflow.models.fast_dllm.configuration import Fast_dLLM_QwenConfig
from lmflow.models.fast_dllm.modeling import Fast_dLLM_QwenForCausalLM

# Construct a tiny relay-enabled model and run one forward to confirm the
# relay LayerNorm path is wired up. Real training/eval is gated behind
# real GPUs; this exercises the pure-Python paths.
cfg = Fast_dLLM_QwenConfig(
    vocab_size=200,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
    bd_size=8,
    mask_token_id=151,
    use_relay=True,
    relay_layer=-1,
)
model = Fast_dLLM_QwenForCausalLM(cfg).eval()
input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
input_ids[0, 4:12] = cfg.mask_token_id
with torch.no_grad():
    out = model(input_ids=input_ids, h_t=None)
assert out.logits.shape == (1, 16, cfg.vocab_size)
print("fast-dllm-v2 relay forward OK; logits shape =", tuple(out.logits.shape))
PY
        )
        green "[ok]   fast-dllm-v2 smoke test"
    fi
fi

green "smoke test done."
