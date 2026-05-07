#!/usr/bin/env bash
# Run DPLM cal_plddt_dir.py on this toy FASTA tree. Requires fair-esm + deps and
# built OpenFold CUDA kernels (use the same setup as a working DPLM install).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPLM_ROOT="${DPLM_ROOT:-/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/dplm}"

export PYTHONPATH="${DPLM_ROOT}/vendor/openfold${PYTHONPATH:+:${PYTHONPATH}}"

cd "${DPLM_ROOT}"
exec python analysis/cal_plddt_dir.py \
  -i "${SCRIPT_DIR}" \
  -o "${SCRIPT_DIR}/esmfold_pdb" \
  --max-tokens-per-batch "${MAX_TOKENS_PER_BATCH:-1024}" \
  "$@"
