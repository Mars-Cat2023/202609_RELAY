# Decoding trajectory visualization

This directory holds **documentation and helper scripts** for dumping and exploring per-step decode trajectories (Sudoku, n-queens, graph coloring). The implementation lives in the `doublebackprop` package (`dump_decode_trajectories`, `pair_trajectory_dumps`, `viz_puzzle_state`); training configs are unchanged unless you opt in at runtime.

## Layout

| Path | Purpose |
|------|---------|
| [`README.md`](README.md) | This file: commands and workflow |
| [`requirements.txt`](requirements.txt) | Optional extras for Jupyter + plotting (install on top of the main env) |
| [`scripts/sudoku_trajectory_dump.sh`](scripts/sudoku_trajectory_dump.sh) | Example: dump Sudoku trajectories for MLM vs BPTT+loopholing+PUMA checkpoints |
| [`notebooks/decode_trajectory_viewer.ipynb`](notebooks/decode_trajectory_viewer.ipynb) | Step slider + `imshow` for a single JSONL record |
| [`outputs/`](outputs/) | Default directory for JSONL dumps (see `.gitignore`; large files stay local) |

## Prerequisites

- Repo root: `PROJECT_ROOT` should point at the repository (default `.` when you `cd` here).
- `xlm_models.json` at repo root so Hydra resolves `doublebackprop` configs (same as training).
- `.env` with `DATA_DIR`, `LOG_DIR`, etc., as in the main [README](../README.md).
- Editable install: `pip install -e doublebackprop` (from repo root).

Install visualization extras:

```bash
pip install -r visualization/requirements.txt
```

## Correctness and non-invasiveness (summary)

- **`capture_trajectory`** defaults to `False` in [`ConfidenceBasedPredictor`](../doublebackprop/predictor.py); training runs are unaffected unless you enable it.
- The dump CLI loads a checkpoint and only then sets `capture_trajectory=True` and `log_rollout_diagnostics=True` on the loaded predictor.
- For comparable **`example_index`** across two runs, use the **same** `seed`, `+dump.limit_val_batches`, `+dump.max_examples`, and batch size (use `per_device_batch_size=1`).

## Finding checkpoints

Lightning checkpoints usually live under `logs/<job_name>/checkpoints/` as `last.ckpt` or `best.ckpt`.

```bash
# From repo root
ls -la logs/sudoku_extreme_mlm_uniform/checkpoints/
ls -la logs/sudoku_extreme_loopholing_puma_mu_0p15_dynamic_sigma_0p1/checkpoints/

# Or search
find logs -path '*sudoku*' -name '*.ckpt'
```

Export paths (edit to match your tree):

```bash
export CKPT_MLM="logs/sudoku_extreme_mlm_uniform/checkpoints/last.ckpt"
export CKPT_BPTT="logs/sudoku_extreme_loopholing_puma_mu_0p15_dynamic_sigma_0p1/checkpoints/last.ckpt"
```

The run folder name (`sudoku_extreme_loopholing_puma_mu_0p15_dynamic_sigma_0p1`) is a **job name**. The Hydra experiment for BPTT+loopholing+PUMA Sudoku is **`sudoku_extreme_loopholing_bptt_puma`**; the checkpoint stores the exact hyperparameters used in that run.

## Option A: Shell script (Sudoku)

From **repository root**:

```bash
export CKPT_MLM="logs/sudoku_extreme_mlm_uniform/checkpoints/last.ckpt"
export CKPT_BPTT="logs/sudoku_extreme_loopholing_puma_mu_0p15_dynamic_sigma_0p1/checkpoints/last.ckpt"
export SEED=1
export LIMIT_VAL_BATCHES=20
export MAX_EXAMPLES=100

bash visualization/scripts/sudoku_trajectory_dump.sh
```

Outputs default to `visualization/outputs/`:

- `trajectories_sudoku_mlm.jsonl`
- `trajectories_sudoku_bptt_puma.jsonl`
- `paired_sudoku_mlm_vs_bptt.jsonl` (only rows where MLM `exact_match` is false and BPTT is true)

Override output directory:

```bash
export OUT_DIR="/path/to/my_outputs"
bash visualization/scripts/sudoku_trajectory_dump.sh
```

## Option B: Manual commands (same behavior)

Run from **repository root**:

```bash
cd /path/to/double-backprop
export PROJECT_ROOT="$PWD"
pip install -e doublebackprop

export CKPT_MLM="logs/sudoku_extreme_mlm_uniform/checkpoints/last.ckpt"
export CKPT_BPTT="logs/sudoku_extreme_loopholing_puma_mu_0p15_dynamic_sigma_0p1/checkpoints/last.ckpt"
SEED=1
LIMIT_VAL_BATCHES=20
MAX_EXAMPLES=200
OUT="visualization/outputs"
mkdir -p "$OUT"

python -m doublebackprop.dump_decode_trajectories \
  experiment=sudoku_extreme_mlm_uniform \
  +generation.ckpt_path="$CKPT_MLM" \
  seed="$SEED" \
  per_device_batch_size=1 \
  global_batch_size=1 \
  +dump.output_path="$OUT/trajectories_sudoku_mlm.jsonl" \
  +dump.max_examples="$MAX_EXAMPLES" \
  +dump.limit_val_batches="$LIMIT_VAL_BATCHES"

python -m doublebackprop.dump_decode_trajectories \
  experiment=sudoku_extreme_loopholing_bptt_puma \
  +generation.ckpt_path="$CKPT_BPTT" \
  seed="$SEED" \
  per_device_batch_size=1 \
  global_batch_size=1 \
  +dump.output_path="$OUT/trajectories_sudoku_bptt_puma.jsonl" \
  +dump.max_examples="$MAX_EXAMPLES" \
  +dump.limit_val_batches="$LIMIT_VAL_BATCHES"

python -m doublebackprop.pair_trajectory_dumps \
  --mlm "$OUT/trajectories_sudoku_mlm.jsonl" \
  --bptt "$OUT/trajectories_sudoku_bptt_puma.jsonl" \
  --out "$OUT/paired_sudoku_mlm_vs_bptt.jsonl" \
  --only-bptt-wins
```

After install, you can also use:

- `dump-decode-trajectories` (console entry point)
- `pair-trajectory-dumps`

## Jupyter viewer

```bash
jupyter notebook visualization/notebooks/decode_trajectory_viewer.ipynb
```

In the notebook, set `JSONL` to a dump file under `visualization/outputs/` (or an absolute path). For paired JSONL, open the file and copy either the `mlm` or `bptt` sub-record into a single-object JSONL line if you want to reuse the same cells unchanged, or point at the raw per-model dumps.

## Other puzzles (n-queens, graph coloring)

Use the same `dump_decode_trajectories` pattern with the matching `experiment=` and checkpoint:

- `experiment=gram_n_queens_8x8_mlm_uniform` vs `experiment=gram_n_queens_8x8_loopholing_bptt_puma`
- `experiment=gram_graph_coloring_8v_mlm_uniform` vs `experiment=gram_graph_coloring_8v_loopholing_bptt_puma`

Keep `per_device_batch_size=1` and `global_batch_size=1` for large trajectories.

## GPU

The dump script uses CUDA when available; CPU is supported but slower.
