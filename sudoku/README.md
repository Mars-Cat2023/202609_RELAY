# relay / sudoku

Sudoku Extreme experiments from the *Learned Relay Representations* paper
(Table 1, paper §4.1). Code paths kept here are the ones that produce the
reported numbers: Mask-uniform CE, Rollout-buffer-only, Relay (BPTT), and
Relay-sg.

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd relay/sudoku
```

Submodules:

- `xlm-core` — the XLM training harness (Hydra + PyTorch Lightning) used by
  every experiment YAML.
- `slurm_scripts` — SLURM submission helpers used by the `submit_*.sh` wrappers.

To pull updates and keep submodules in sync:

```bash
git pull --recurse-submodules
git submodule update --remote xlm-core
```

### Python environment

Create a virtual environment and install everything in editable mode:

```bash
python -m venv .venv_relay_sudoku
source .venv_relay_sudoku/bin/activate
pip install -e xlm-core
pip install -e xlm-core/xlm-models
pip install -e .
```

Then create a `.env` file at the `relay/sudoku/` root (used by Hydra +
`xlm-core`):

```bash
WANDB_ENTITY=<your-wandb-entity>
WANDB_PROJECT=BPTT-sudoku
DATA_DIR=data
HF_HOME=hf_home
HF_DATASETS_CACHE=hf_datasets_cache
LOG_DIR=logs
TOKENIZERS_PARALLELISM=false
PROJECT_ROOT=.
HYDRA_FULL_ERROR=1
OC_CAUSE=1
TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
```

`xlm_models.json` at this directory tells Hydra where to find the `relay/`
config tree — leave it as is.

## Data

The Sudoku Extreme split is downloaded from Hugging Face on first use
(`brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver`). To prebuild
the cache:

```bash
xlm "job_type=prepare_data" "job_name=sudoku_extreme_prepare_data" "experiment=sudoku_extreme_mlm_uniform"
```

## Smoke test (no SLURM)

A single-GPU sanity run to confirm the loop is wired up before launching the
full 300k training:

```bash
export PROJECT_ROOT="$PWD"
python -m xlm.train \
  experiment=sudoku_extreme_relay_bptt \
  trainer.max_steps=200 \
  trainer.val_check_interval=100 \
  trainer.limit_val_batches=2 \
  per_device_batch_size=8 \
  global_batch_size=8 \
  loggers.wandb=null
```

Substitute `experiment=sudoku_extreme_mlm_uniform` for the baseline.

## Reproducing Table 1

```bash
# Dry-run first to inspect the SLURM submission for your cluster.
DO=print ./submit_sudoku_300k_sweep.sh

# Submit the eight reported runs (single seed; ~36h on 80GB GPUs for relay).
./submit_sudoku_300k_sweep.sh

# Multi-seed sweep (paper averages over seeds 1, 2, 3 -> 24 runs).
./submit_sudoku_300k_seeds_sweep.sh
```

Cluster-specific knobs (`SLURM_RESERVATION`, `SLURM_CONSTRAIN_*`, partition,
wall-clock) are environment variables; see [`SUDOKU_COMMANDS.md`](SUDOKU_COMMANDS.md)
for the full table and the per-objective Hydra overrides.

After training:

1. `submit_sudoku_qualitative_dumps.sh` — dumps per-step decode trajectories
   at a sweep of inference confidence thresholds. Drives Figure 4 and the
   matched-NFE rows of Table 1.
2. `python -m relay.n_way_pair_trajectories ...` — joins JSONL dumps with the
   HF solver-trajectory dataset into a single parquet table.
3. `python -m relay.summarize_sudoku_table_from_manifests --tau 0.15 ...` —
   prints the matched-NFE Table 1 row from the manifest CSV.

The exact, end-to-end commands (with the parquet schema, deduction-only and
extreme-only filter sub-studies) are documented in
[`SUDOKU_ANALYSIS_COMMANDS.md`](SUDOKU_ANALYSIS_COMMANDS.md).

## Layout

| Path                                 | Contents                                                                                  |
|--------------------------------------|-------------------------------------------------------------------------------------------|
| `relay/`                             | The Sudoku training package (model, loss, predictor, datamodule, metrics, sudoku tools). |
| `relay/configs/`                     | Hydra configs (`experiment/`, `model/`, `model_type/`, `datamodule/`, `metrics/`, ...).   |
| `xlm-core/` (submodule)              | The XLM training harness used by every experiment.                                        |
| `slurm_scripts/` (submodule)         | SLURM submission helpers used by the `submit_*.sh` wrappers.                              |
| `submit_sudoku_300k_sweep.sh`        | Submits the 8 reported Sudoku runs (single seed).                                         |
| `submit_sudoku_300k_seeds_sweep.sh`  | Same 8 ablations × N seeds.                                                              |
| `submit_sudoku_qualitative_dumps.sh` | Dumps per-step decode trajectories from trained checkpoints.                              |
| `visualization/`                     | Manual decode-trajectory dump pipeline + Jupyter viewer.                                  |
| `plotting_scripts/`                  | Notebooks that produce the paper's Sudoku figures.                                        |
| `SUDOKU_COMMANDS.md`                 | Per-objective Hydra overrides + sweep environment variables.                              |
| `SUDOKU_ANALYSIS_COMMANDS.md`        | Full qualitative + quantitative analysis pipeline.                                        |

## W&B

Reported runs live under `https://wandb.ai/<entity>/BPTT-sudoku`. The relevant
filters are `+tags.sweep`, `+tags.objective`, `+tags.embed_tying`, `+tags.seed`.
