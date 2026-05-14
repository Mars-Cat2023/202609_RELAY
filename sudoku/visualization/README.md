# Decoding trajectory visualization (Sudoku)

This directory holds **documentation and helper scripts** for dumping and exploring per-step Sudoku decode trajectories. The implementation lives in the `relay` package (`dump_decode_trajectories`, `pair_trajectory_dumps`, `viz_puzzle_state`); training configs are unchanged unless you opt in at runtime.

## Layout

| Path | Purpose |
|------|---------|
| [`README.md`](README.md) | This file: commands and workflow |
| [`requirements.txt`](requirements.txt) | Optional extras for Jupyter + plotting (install on top of the main env) |
| [`scripts/sudoku_trajectory_dump.sh`](scripts/sudoku_trajectory_dump.sh) | Example: dump Sudoku trajectories for an MLM vs Relay-BPTT checkpoint pair |
| [`notebooks/decode_trajectory_viewer.ipynb`](notebooks/decode_trajectory_viewer.ipynb) | Step slider + `imshow` for a single JSONL record |
| [`outputs/`](outputs/) | Default directory for JSONL dumps (gitignored; large files stay local) |

## Prerequisites

- Repo root: set `PROJECT_ROOT` to the `relay/sudoku/` directory.
- `xlm_models.json` at that root so Hydra resolves `relay` configs (same as training).
- Editable install: `pip install -e .` from `relay/sudoku/`.

Install visualization extras:

```bash
pip install -r visualization/requirements.txt
```

## Correctness and non-invasiveness

- **`capture_trajectory`** defaults to `False` in `relay.predictor.ConfidenceBasedPredictor`; training runs are unaffected unless you enable it.
- The dump CLI loads a checkpoint and only then sets `capture_trajectory=True` and `log_rollout_diagnostics=True` on the loaded predictor.
- For comparable **`example_index`** across two runs, use the **same** `seed`, `+dump.limit_val_batches`, `+dump.max_examples`, and batch size (use `per_device_batch_size=1`).

## Finding checkpoints

Lightning checkpoints live under `logs/<job_name>/checkpoints/` as `last.ckpt` or `best.ckpt`.

```bash
ls -la logs/sudoku_extreme_mlm_uniform_300k_untied/checkpoints/
ls -la logs/sudoku_extreme_relay_bptt_steps2_300k_untied/checkpoints/
```

Export paths (edit to match your tree):

```bash
export CKPT_MLM="logs/sudoku_extreme_mlm_uniform_300k_untied/checkpoints/last.ckpt"
export CKPT_RELAY="logs/sudoku_extreme_relay_bptt_steps2_300k_untied/checkpoints/last.ckpt"
```

The Hydra experiments referenced by the dump CLI are
`sudoku_extreme_mlm_uniform` (MLM) and `sudoku_extreme_relay_bptt` (relay BPTT).

## Option A: Shell script

From the `relay/sudoku/` root:

```bash
export CKPT_MLM="logs/sudoku_extreme_mlm_uniform_300k_untied/checkpoints/last.ckpt"
export CKPT_RELAY="logs/sudoku_extreme_relay_bptt_steps2_300k_untied/checkpoints/last.ckpt"
export SEED=1
export LIMIT_VAL_BATCHES=20
export MAX_EXAMPLES=100

bash visualization/scripts/sudoku_trajectory_dump.sh
```

Outputs default to `visualization/outputs/`:

- `trajectories_sudoku_mlm.jsonl`
- `trajectories_sudoku_relay_bptt.jsonl`
- `paired_sudoku_mlm_vs_relay.jsonl` (only rows where MLM `exact_match` is false and RELAY is true)

Override the output directory:

```bash
export OUT_DIR="/path/to/my_outputs"
bash visualization/scripts/sudoku_trajectory_dump.sh
```

## Option B: Manual commands (same behavior)

Run from `relay/sudoku/`:

```bash
cd /path/to/relay/sudoku
export PROJECT_ROOT="$PWD"
pip install -e .

export CKPT_MLM="logs/sudoku_extreme_mlm_uniform_300k_untied/checkpoints/last.ckpt"
export CKPT_RELAY="logs/sudoku_extreme_relay_bptt_steps2_300k_untied/checkpoints/last.ckpt"
SEED=1
LIMIT_VAL_BATCHES=20
MAX_EXAMPLES=200
OUT="visualization/outputs"
mkdir -p "$OUT"

python -m relay.dump_decode_trajectories \
  experiment=sudoku_extreme_mlm_uniform \
  +generation.ckpt_path="$CKPT_MLM" \
  seed="$SEED" \
  per_device_batch_size=1 \
  global_batch_size=1 \
  +dump.output_path="$OUT/trajectories_sudoku_mlm.jsonl" \
  +dump.max_examples="$MAX_EXAMPLES" \
  +dump.limit_val_batches="$LIMIT_VAL_BATCHES"

python -m relay.dump_decode_trajectories \
  experiment=sudoku_extreme_relay_bptt \
  +generation.ckpt_path="$CKPT_RELAY" \
  seed="$SEED" \
  per_device_batch_size=1 \
  global_batch_size=1 \
  +dump.output_path="$OUT/trajectories_sudoku_relay_bptt.jsonl" \
  +dump.max_examples="$MAX_EXAMPLES" \
  +dump.limit_val_batches="$LIMIT_VAL_BATCHES"

python -m relay.pair_trajectory_dumps \
  --mlm "$OUT/trajectories_sudoku_mlm.jsonl" \
  --bptt "$OUT/trajectories_sudoku_relay_bptt.jsonl" \
  --out "$OUT/paired_sudoku_mlm_vs_relay.jsonl" \
  --only-bptt-wins
```

After install, you can also use:

- `dump-decode-trajectories` (console entry point)
- `pair-trajectory-dumps`

## Jupyter viewer

```bash
jupyter notebook visualization/notebooks/decode_trajectory_viewer.ipynb
```

In the notebook, set `JSONL` to a dump file under `visualization/outputs/` (or an absolute path).

## GPU

The dump script uses CUDA when available; CPU is supported but slower.
