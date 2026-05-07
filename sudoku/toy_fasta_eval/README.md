# Toy FASTA for DPLM-style structure evaluation

Example `.fasta` files in the layout `cal_plddt_dir.py` expects (one or more `.fasta` files anywhere under an input directory).

### If the install script still prints `Obtaining file://` / `build_editable`

That output is **`pip install -e`**, not the current **`python setup.py develop`** flow. Your cluster copy of `install_openfold_cuda_ext.sh` is **out of date** (git pull / re-copy from `double-backprop`). A current run must start with:

`=== install_openfold_cuda_ext.sh (setuptools develop, not pip -e) ===`

### End-to-end commands (GPU node, `venv-dplm-esmfold` active)

```bash
# 1) Build OpenFold extension (no pip -e)
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/dplm/vendor/openfold
python -m pip install -q "wheel>=0.40" "setuptools>=65"
python setup.py develop --no-deps

# 1b) Python 3.11+: fair-esm trunk.py uses an invalid dataclass default (upstream bug)
python /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/toy_fasta_eval/patch_fair_esm_trunk_py311.py

# 2) Run toy eval (PYTHONPATH required for vendored openfold)
export PYTHONPATH="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/dplm/vendor/openfold${PYTHONPATH:+:${PYTHONPATH}}"
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/dplm
python analysis/cal_plddt_dir.py \
  -i /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/toy_fasta_eval \
  -o /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/toy_fasta_eval/esmfold_pdb \
  --max-tokens-per-batch 1024
```

### `ValueError: mutable default ... use default_factory` (trunk or esmfold)

**Not a mixed conda/venv bug.** Your `venv-dplm-esmfold` was created from conda’s `venv_learned_noise` Python (`pyvenv.cfg` → `home = .../venv_learned_noise/bin`), so **`dataclasses.py`** correctly loads from that env’s stdlib.

**Cause:** On **Python 3.11+**, `fair-esm` ESMFold v1 uses illegal dataclass defaults in two files:

- `trunk.py`: `structure_module: StructureModuleConfig = StructureModuleConfig()`
- `esmfold.py`: `trunk: T.Any = FoldingTrunkConfig()`

**Fix** (idempotent; patches **both**; run inside `venv-dplm-esmfold`):

```bash
python /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/toy_fasta_eval/patch_fair_esm_trunk_py311.py
```

If you patched only an older version of this script (trunk only), **run it again** after `git pull` so `esmfold.py` gets fixed too. Then rerun `cal_plddt_dir.py`.

**Alternative:** use **Python 3.10** for the eval venv.

## Run ESMFold evaluation (from DPLM repo)

Requires **Meta’s [`fair-esm`](https://github.com/facebookresearch/esm)** (PyPI name `fair-esm`), which provides `esm.pretrained.esmfold_v1()`, **not** EvolutionaryScale’s PyPI package `esm` (ESM3 tooling). Both projects install a top-level module named `esm`, so they **conflict**.

### If you see `AttributeError: module 'esm' has no attribute 'pretrained'`

Your environment almost certainly has EvolutionaryScale **`esm`** instead of **`fair-esm`**. Check:

```bash
python -c "import esm; print(esm.__file__); print(getattr(esm,'__version__',None))"
```

If `pip show esm` reports *EvolutionaryScale open model repository*, use a **dedicated** eval environment:

```bash
python -m venv /path/to/dplm-esmfold-eval
source /path/to/dplm-esmfold-eval/bin/activate
pip install torch fair-esm  # plus any deps dplm/analysis needs
```

Then run `cal_plddt_dir.py` from that venv. Do **not** `pip install esm` (EvolutionaryScale) in the same env if you need DPLM’s script unchanged.

Alternatively, keep your current venv for training and only run folding in the separate env.

### If you see `ModuleNotFoundError: No module named 'omegaconf'` (or NumPy warnings)

`fair-esm` does not pull in every runtime dependency. In your eval venv:

```bash
pip install -r /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/requirements-dplm-esmfold-eval.txt
```

ESMFold then imports **OpenFold** from **`dplm/vendor/openfold`**, which needs:

```bash
export PYTHONPATH="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/dplm/vendor/openfold:${PYTHONPATH}"
```

and a **compiled** OpenFold CUDA module **`attn_core_inplace_cuda`**.

#### `ModuleNotFoundError: No module named 'attn_core_inplace_cuda'`

The extension is **not** prebuilt; DPLM’s install script builds it from **`vendor/openfold`**:

```bash
# activate venv-dplm-esmfold (or your eval env), then:
/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/toy_fasta_eval/install_openfold_cuda_ext.sh
```

The script runs **`python setup.py develop --no-deps`** inside `vendor/openfold` (after ensuring `wheel` / `setuptools`). That executes OpenFold’s `setup.py` **in your current interpreter**, so `import torch` succeeds.

**Why not `pip install -e`?** Recent pip uses a PEP 517 **editable** path (`build_editable`) that can still spawn **`pip-build-env-...`** without PyTorch, reproducing `ModuleNotFoundError: No module named 'torch'` even when you pass `--no-build-isolation` / `--no-use-pep517` for some pip/setuptools combinations.

Manual equivalent:

```bash
cd /path/to/dplm/vendor/openfold
python setup.py develop --no-deps
```

(DPLM’s `scripts/install.sh` uses `pip install -e vendor/openfold`; on older pip that was fine.)

You typically need a **GPU node** (or at least a full CUDA toolkit matching your PyTorch build) so `nvcc` can compile the kernel.

**Alternative:** run `cal_plddt_dir.py` from the **same conda env where you already installed DPLM** via `bash scripts/install.sh` (OpenFold already built there).

A minimal `torch` + `fair-esm` venv alone is **not** enough until this step succeeds.

### If you see `ModuleNotFoundError: No module named 'openfold'`

`fair-esm`’s ESMFold imports the **`openfold`** package from **DPLM’s vendored copy**. You must put it on `PYTHONPATH` **before** running the script (the `cd` into `dplm` alone is not enough):

```bash
export PYTHONPATH="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/dplm/vendor/openfold${PYTHONPATH:+:${PYTHONPATH}}"
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/dplm

python analysis/cal_plddt_dir.py \
  -i /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/toy_fasta_eval \
  -o /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/toy_fasta_eval/esmfold_pdb \
  --max-tokens-per-batch 1024
```

Or from anywhere (same `PYTHONPATH` rule), after `chmod +x`:

```bash
/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/toy_fasta_eval/run_toy_esmfold.sh
```

Override DPLM location if needed: `DPLM_ROOT=/path/to/dplm ./run_toy_esmfold.sh`.

A GPU is strongly recommended for speed.

`analysis/plddt_calculate.sh` does not set `PYTHONPATH`; export `vendor/openfold` the same way before calling it if you use that wrapper. It writes PDBs under `<input_dir>/esmfold_pdb/`.

Notes:

- Sequences containing **`X`** are skipped by `cal_plddt_dir.py`.
- Very short peptides fold but pLDDT semantics differ from full proteins; this layout is for pipeline testing only.

## Aggregate metrics from paired FASTA + PDB

If FASTAs live under `by_length/` and PDBs under `esmfold_pdb/<fasta_stem>/` (same layout `cal_plddt_dir.py` writes):

```bash
python toy_fasta_eval/aggregate_fasta_pdb_metrics.py \
  --fasta-dir /path/to/toy_fasta_eval/by_length \
  --pdb-root /path/to/toy_fasta_eval/esmfold_pdb
```

Defaults use those paths relative to the script’s directory. This reports **mean pLDDT (CA B-factors)**, **sequence diversity %**, and **pooled 20-AA entropy** (nats). **pTM / pAE / PAPL foldability** still require logging model outputs at inference time.
