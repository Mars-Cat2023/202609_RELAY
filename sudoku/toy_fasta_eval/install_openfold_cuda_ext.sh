#!/usr/bin/env bash
# Build vendored OpenFold's attn_core_inplace_cuda into the *current* Python env.
# Run this once in venv-dplm-esmfold (or any env where you run cal_plddt_dir.py).
#
# Needs: matching PyTorch+CUDA, NVIDIA driver, and nvcc (CUDA toolkit) on PATH
# when CUDA is available. On a CPU-only node with CPU torch, setup.py may fall
# back to a C++ stub (see dplm/vendor/openfold/setup.py).
#
# We use `python setup.py develop` (not `pip install -e`) because pip’s editable
# path (PEP 660 / build_editable) can still spawn an isolated env without torch
# while reading setup.py — the exact failure you see as ModuleNotFoundError: torch
# inside pip-build-env-*/overlay.
set -euo pipefail

DPLM_ROOT="${DPLM_ROOT:-/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/dplm}"
OPENFOLD_DIR="${DPLM_ROOT}/vendor/openfold"

echo "=== install_openfold_cuda_ext.sh (setuptools develop, not pip -e) ==="
echo "If you see 'Obtaining file://' or 'build_editable', you are running an OLD script — git pull / sync double-backprop."
echo "Using: $(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

python -m pip install -q "wheel>=0.40" "setuptools>=65"

cd "${OPENFOLD_DIR}"
echo "Running setuptools develop in ${OPENFOLD_DIR}"
python setup.py develop --no-deps
