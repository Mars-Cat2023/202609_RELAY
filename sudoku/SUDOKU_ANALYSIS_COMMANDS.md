# Sudoku qualitative + quantitative analysis (8 ablations × 3 seeds)

End-to-end workflow for the Sudoku results section: dump per-step decoding
trajectories from each trained model, join them with the curated HF dataset
(env var `RELAY_SUDOKU_HF_DATASET`; built by combining
`sapientinc/sudoku-extreme` with `timvink/sudoku-solver`, which together
provide solver trajectories and strategy labels), compute algorithmic primitives
(naked / hidden singles, fill-time grids, solver alignment) into a single
parquet table, and produce the headline figures from a Jupyter notebook.

The pipeline is **checkpoint-only**; nothing here retrains a model. Inference
is performed by the existing `relay.dump_decode_trajectories` CLI,
which only enables `predictor.capture_trajectory=True` after the checkpoint is
loaded — so the training-time training behaviour is unaffected.

The seeded training sweep produced 24 checkpoints (4 objectives × 2 tying ×
3 seeds) under `logs/<job>_seed{1,2,3}/`. Two of those diverged at training
time (zero `legal_rate` for the entire run): `mlm_uniform_tied_seed3` and
`relay_sg_tied_seed1` — both flagged by W&B's
`val/prediction/legal_rate` metric and skipped by default in the dump driver.

---

## TL;DR (seeded, bigger eval slice)

```bash
cd $REPO_ROOT
export PROJECT_ROOT=$REPO_ROOT
source .venv_relay/bin/activate

# 1. Dump trajectories for 22 alive checkpoints × 7 thresholds × 2000 puzzles.
#    (Skips the 2 diverged seeded runs; defaults to seeds 1, 2, 3.)
DO=submit \
  SEEDS="1 2 3" \
  THRESHOLDS="0.01 0.05 0.10 0.15 0.25 0.35 0.45" \
  MAX_EXAMPLES=2000 LIMIT_VAL_BATCHES=2000 \
  SLURM_MEM=64G SLURM_TIME=04:00:00 SLURM_RESERVATION= \
  bash submit_sudoku_qualitative_dumps.sh

# 2. Once all 22*7=154 sbatch jobs finish, join JSONLs + HF metadata into one parquet.
python -m relay.n_way_pair_trajectories \
  --manifest outputs/sudoku_analysis/manifest.csv \
  --hf-split test \
  --out outputs/sudoku_analysis/sudoku_analysis.parquet \
  --shared-only

# 3. Open the notebook and Run All. Figures land in outputs/sudoku_analysis/figures/.
jupyter notebook plotting_scripts/sudoku_qualitative_study.ipynb
```

The `seed` column is propagated through the manifest → parquet, so the notebook's
existing `(objective, embed_tying)` aggregations now average over both puzzles
**and** seeds. To inspect seed variability separately, add `seed` to the groupby
key in the F1/F4 aggregations.

---

## Models in scope

The 24 seeded 300k-step Sudoku Extreme runs (4 objectives × 2 weight-tying × 3
random seeds). Each row maps to one log directory per seed:
`logs/<base>_seed{1,2,3}/`. W&B project name shown in the paper is
`<your-wandb-entity>/BPTT-sudoku`; the authors' entity is withheld during
the double-blind review period.

| key                         | objective tag                | embed_tying | base log dir                                                | seeds                    |
|-----------------------------|------------------------------|-------------|-------------------------------------------------------------|--------------------------|
| `mlm_uniform_untied`        | `mlm_uniform`                | untied      | `logs/sudoku_extreme_mlm_uniform_300k_untied`              | 1, 2, 3                  |
| `mlm_uniform_tied`          | `mlm_uniform`                | tied        | `logs/sudoku_extreme_mlm_uniform_300k_tied`                | 1, 2, **3 diverged**     |
| `rollout_untied`          | `rollout`         | untied      | `logs/sudoku_extreme_rollout_300k_untied`                | 1, 2, 3                  |
| `rollout_tied`            | `rollout`         | tied        | `logs/sudoku_extreme_rollout_300k_tied`                  | 1, 2, 3                  |
| `relay_sg_untied` | `relay_sg`  | untied      | `logs/sudoku_extreme_relay_sg_300k_untied`  | 1, 2, 3                  |
| `relay_sg_tied`   | `relay_sg`  | tied        | `logs/sudoku_extreme_relay_sg_300k_tied`    | **1 diverged**, 2, 3     |
| `relay_untied`    | `relay`       | untied      | `logs/sudoku_extreme_relay_bptt_steps2_300k_untied` | 1, 2, 3              |
| `relay_tied`      | `relay`       | tied        | `logs/sudoku_extreme_relay_bptt_steps2_300k_tied`   | 1, 2, 3              |

→ 22 alive seed-checkpoints; 2 diverged seeds skipped by default
(`SKIP_DIVERGED=1` in the dump driver). The `mlp` and `cab` sub-sweeps in W&B
remain excluded.

**Stability summary (val/prediction/exact_match @ τ=0.15, mean ± std over
non-diverged seeds, from the authors' W&B project `<entity>/BPTT-sudoku`):**

| objective                   | tied            | untied          |
|-----------------------------|-----------------|-----------------|
| `mlm_uniform`               | 0.290 ± 0.022   | 0.226 ± 0.030   |
| `rollout`        | 0.422 ± 0.011   | 0.416 ± 0.009   |
| `relay_sg` | 0.619 ± 0.011   | 0.613 ± 0.012   |
| `relay`      | 0.648 ± 0.014   | 0.649 ± 0.008   |

The objective ordering is preserved across seeds, the ablation is now clean
(MLM-uniform << Rollout-only << Relay-sg << Relay), and the
seed std is small relative to the inter-objective gap. Divergence is not
seed-stable: tied StopGrad failed at seed=1 but trains fine at seeds 2/3, and
tied MLM failed at seed=3 but trains fine at seeds 1/2 — so the original
8-run sweep (which used the Hydra default `seed=1`, hence "untagged") was
unlucky on tied StopGrad and lucky on tied MLM. The seeded sweep is the
correct source for any quantitative claim.

---

## Step 1 — Dump trajectories

`submit_sudoku_qualitative_dumps.sh` is a thin driver around
`python -m relay.dump_decode_trajectories` that loops over
**(seed × run × tau)** tuples and writes one JSONL per `(run, seed, tau)` plus
a manifest CSV consumed by the next step. It supports three modes:

- `DO=print` — dry-run; prints the commands it *would* run.
- `DO=run` — sequential foreground execution (good for one node with one GPU).
- `DO=submit` — one `sbatch` per `(run, seed, tau)` (good for cluster fan-out).

Key environment variables (defaults in `[]`):

- `SEEDS` `["1 2 3"]` — whitespace-separated training seeds; pulls checkpoints
  from `logs/<base_job>_seed${seed}/`. Set `SEEDS=""` to fall back to the
  original 8 untagged log dirs (effectively seed=1).
- `SKIP_DIVERGED` `[1]` — skips `(mlm_uniform_tied, seed=3)` and
  `(relay_sg_tied, seed=1)`, the two seeded combos that diverged at
  training time (zero `val/prediction/legal_rate`). Set to `0` to dump them
  anyway (e.g., to verify the divergence shows up in the trajectory dump too).
- `THRESHOLDS` `["0.01 0.05 0.10 0.15 0.25 0.35 0.45"]` — predictor confidence
  thresholds. The default span widens what the original sweep had on both ends:
  τ=0.01–0.05 pushes relay models up to NFE ≈ 25–35; τ=0.35–0.45 pushes
  MLM/Rollout-only down to NFE ≈ 18–25. Together they bracket a matched-NFE
  band of ≈ [18, 35] across all four objectives, which the notebook's
  `per_model_nfe_range` diagnostic verifies before running interpolation.
- `LIMIT_VAL_BATCHES` `[2000]` — caps val-loader batches (with
  `per_device_batch_size=1`); the underlying dataset is the
  `${RELAY_SUDOKU_HF_DATASET}/test` split (422,786 puzzles), so 2000 is
  a small but representative slice.
- `MAX_EXAMPLES` `[2000]` — caps written examples per `(run, seed, tau)`.
- `SEED` `[1]` — shared *dataloader* seed across all dumps so `example_index`
  aligns and `--shared-only` keeps every puzzle. **Not** the model training
  seed (that's encoded in the log dir).
- `OUT_DIR` `[outputs/sudoku_analysis]` — root directory for dumps + manifest.
- `CKPT_NAME` `[last.ckpt]` — checkpoint file under `logs/<run>/checkpoints/`.
- `PARTITION` `[gpu]`, `GPU_CONSTRAINT` `[vram40,bf16]` — forwarded to `sbatch`.
  If you see **GPU** OOM (CUDA OOM in `.out`), try `GPU_CONSTRAINT=vram80,bf16`.
- `SLURM_MEM` `[64G]` — **host** RAM (`sbatch --mem`). Many sites default to a
  few GB; without a higher limit the Slurm cgroup hits `oom_kill` (see
  `slurm-*.err`: "Detected N oom_kill events") **even for MLM**, right after
  the model loads, leaving **0-line** JSONLs. Set `SLURM_MEM=` (empty) to omit
  `--mem` if your scheduler forbids it.
- `SLURM_CPUS_PER_TASK` `[2]` — passed to `sbatch`. The dump command already
  sets `num_dataloader_workers=0` / `num_dataset_workers=0` to keep peak RSS down.
- `SLURM_TIME`, `SLURM_RESERVATION` — forwarded to `sbatch`.
- `ONLY_OBJECTIVES`, `ONLY_TYINGS`, `ONLY_SEEDS` — whitespace-separated subset
  filters (empty string = no filter), useful for partial / staged retries.

The script writes:

- `outputs/sudoku_analysis/dumps/<run_id>__tau_0pXX.jsonl` — one record per
  puzzle. `<run_id>` is `<key>_seed<N>` when `SEEDS` is non-empty (e.g.
  `relay_untied_seed2`) and bare `<key>` otherwise. Each record
  contains `trajectory` (per-step token ids), `target_ids`, `pred_ids`,
  `pred_text`, `truth_text`, `rollout_steps_per_sample`, `exact_match`.
- `outputs/sudoku_analysis/manifest.csv` — header
  `run_id,objective,embed_tying,seed,tau,jsonl` (the new `seed` column is
  empty for `SEEDS=""` runs; older 5-column manifests still parse correctly).

Notes on per-objective Hydra overrides:

- `mlm_uniform`: uses `experiment=sudoku_extreme_mlm_uniform`.
- `rollout`, `relay_sg`, `relay`:
  use `experiment=sudoku_extreme_relay_bptt` plus `loss.*`,
  `predictor.with_relay=...`, and (for `rollout`)
  `model=rotary_transformer_xtiny` to instantiate the non-relay transformer.
- Tied variants additionally pass `++model.tie_embeddings=true` so that
  `load_model_for_inference` reconstructs the architecture matching the
  checkpoint state dict.

---

## Step 2 — Join into a parquet

`python -m relay.n_way_pair_trajectories` consumes the manifest and
emits one parquet with one row per `(run_id, tau, puzzle_hash)`. The HF
metadata join is keyed by `puzzle_hash` (SHA1 of the normalized 81-char
puzzle string). The puzzle question is recovered from `trajectory[0]`:

> The first trajectory snapshot is the masked input *before* any decoding step
> (see `ConfidenceBasedPredictor.predict`). Cells whose token id is not the
> `[MASK]` id are exactly the original clues. No change to the dump format is
> required.

Important flags:

- `--shared-only` — keeps only puzzles present under every `run_id`. Required
  for fair cross-run comparisons (if you change `MAX_EXAMPLES` you might lose
  the shared subset, in which case bump it up or align seeds explicitly).
- `--no-trajectory` — drops `model_traj_digit_flat` (the full `T x 81`
  trajectory). Recommended when you only need aggregate metrics, not the
  qualitative gallery.
- `--hf-parquet` — point at a local parquet snapshot of the HF dataset to
  avoid re-downloading from the Hub on every run (recommended for compute
  nodes without unrestricted network access).

Output columns (excerpt — see
[`relay/n_way_pair_trajectories.py`](relay/n_way_pair_trajectories.py)
for the full schema):

| group | columns |
|-------|---------|
| identity | `run_id, objective, embed_tying, seed, tau, puzzle_hash, question, source, rating, num_steps_solver, strategies_used, hardest_strategy, num_clues` |
| outcome | `exact_match, rollout_steps, pred_text, truth_text, answer_grid_flat, final_grid_flat` |
| per-step trace | `model_traj_digit_flat, fill_step_flat, solver_fill_step_flat, purity_per_step, violations_total_steps, final_violations` |
| primitives at step 1 | `first_step_naked_single_rate, first_step_hidden_single_rate, first_step_forced_rate, first_step_advanced_rate, first_step_illegal_rate, first_step_correct_rate, n_first_step_fills` |
| solver alignment | `fill_order_spearman, prefix_jaccard_at_5, prefix_jaccard_at_10, prefix_jaccard_at_20, prefix_jaccard_at_50pct` |

The categorical strategy tier is computed from `strategies_used` via
[`relay.sudoku_analysis.hardest_strategy_tier`](relay/sudoku_analysis.py)
following the [timvink/sudoku-solver](https://github.com/timvink/sudoku-solver)
README's grouping (Easy → Medium → Advanced → Master → BruteForce).

---

## Step 3 — Run the notebook

[`plotting_scripts/sudoku_qualitative_study.ipynb`](plotting_scripts/sudoku_qualitative_study.ipynb)
loads the parquet and produces five figures into `outputs/sudoku_analysis/figures/`:

1. **F1 — Pareto frontier**. Exact match vs mean rollout steps (NFE), one
   curve per `(objective × embed_tying)`, threshold annotated. Reproduces the
   Pareto plot from the in-progress writeup directly from the same evaluation
   slice that drives the rest of the analyses.
2. **F2 — Strategy-conditioned gain**. Bar chart of `Δ exact_match` for
   `Relay − {MLM, Rollout-only, Relay-sg}` bucketed by `hardest_strategy`. Tests "BPTT
   helps most on harder strategies".
3. **F3 — Solver alignment**. Per-tier Spearman ρ of fill order; prefix
   Jaccard with the solver at `k ∈ {5, 10, 20}`. Process-level evidence that
   relay+BPTT decodes in a more solver-like order.
4. **F4 — Algorithmic primitives**. Naked / hidden / advanced / illegal rates
   among the cells the model commits to at step 1, plus per-step purity
   curves. Computed deterministically from the *clues* — independent of the
   answer.
5. **F5 — Qualitative gallery**. One representative puzzle per strategy tier,
   showing clues, solver fill-time heatmap, and per-model fill-time heatmaps
   with overlaid digits via `viz_puzzle_state.overlay_sudoku_digits_on_heatmap_ax`.
   Final wrong cells are highlighted in red.

The notebook also includes a **matched-NFE comparison** that interpolates the
threshold sweep to a target NFE budget `B ∈ {15, 20, 25, 30}` and reports the
per-objective exact match. This decouples "more steps" from "smarter steps".

---

## Files added by this analysis

| Path | Role |
|------|------|
| [`relay/sudoku_analysis.py`](relay/sudoku_analysis.py) | Token-id → digit grid; candidate / naked / hidden / advanced / illegal classifiers; fill-step grid; Spearman ρ; prefix Jaccard; one-shot `summarize_trajectory`. |
| [`relay/n_way_pair_trajectories.py`](relay/n_way_pair_trajectories.py) | N-way join keyed by `puzzle_hash` + HF metadata → parquet. |
| [`submit_sudoku_qualitative_dumps.sh`](submit_sudoku_qualitative_dumps.sh) | Driver over 8 runs × thresholds; emits manifest. |
| [`plotting_scripts/sudoku_qualitative_study.ipynb`](plotting_scripts/sudoku_qualitative_study.ipynb) | Interactive analysis; five figures + matched NFE table. |

Existing files reused unchanged:

- [`relay/dump_decode_trajectories.py`](relay/dump_decode_trajectories.py) — sets `capture_trajectory=True` after checkpoint load.
- [`relay/viz_puzzle_state.py`](relay/viz_puzzle_state.py) — `sudoku_first_nonmask_step_grid`, digit overlay.
- [`relay/predictor.py`](relay/predictor.py) — `ConfidenceBasedPredictor`, no edits.

---

## Reproducibility checks

1. **Shared puzzle universe**. After step 2, verify
   `df["puzzle_hash"].nunique()` is the same after `restrict_to_shared_puzzles`
   for any pair of (run_id × tau) you compare. Use `--shared-only`. With the
   seeded sweep (22 alive seed-checkpoints × 7 thresholds = 154 runs) the
   shared puzzle count should be exactly `LIMIT_VAL_BATCHES` because all dumps
   use the same dataloader seed.
2. **Seed alignment (dataloader vs training)**. The training seed is encoded
   in the log directory (`..._seedN/`), but at dump time
   `submit_sudoku_qualitative_dumps.sh` passes `seed=1` to every command —
   that's the *dataloader* seed and is intentionally constant so all 154 runs
   see the same first 2000 puzzles in the same order. Every parquet row also
   carries the training seed in its `seed` column.
3. **Diverged checkpoints**. The seeded sweep has two failure modes that the
   `legal_rate` metric on W&B catches at the very first val epoch:
   - `mlm_uniform_tied_seed3` (W&B `enc7sjpy`) — `exact_match=0`, `legal_rate=0`.
   - `relay_sg_tied_seed1` (W&B `lx6ffcgd`) — same pattern,
     reproducing the original `v6nzrrye` divergence at the same seed.
   Both are skipped by default via `SKIP_DIVERGED=1`. Note that the original
   `v6nzrrye` story flips on the seeded sweep: tied StopGrad trains fine at
   seeds 2 and 3 (EM ≈ 0.62), so the failure was a seed-1 init quirk, not a
   structural problem with tied StopGrad.
4. **Optional sanity test on the existing 5-row dump**. The existing
   `visualization/outputs/trajectories_sudoku_*.jsonl` files (from
   `visualization/scripts/sudoku_trajectory_dump.sh`) work as-is with
   `n_way_pair_trajectories` once you write a manifest CSV pointing at them
   — handy for spot-checks before a full sweep.

## Eval slice composition (why 2000 puzzles + brute-force exclusion)

The HF `test` split has 422,786 puzzles labelled by `timvink/sudoku-solver`.
The strategy tier distribution is heavily right-tailed:

| tier        |   count |   share |
|-------------|---------|---------|
| BruteForce  | 364,236 |  86.2%  |
| Easy        |  47,289 |  11.2%  |
| Medium      |  11,260 |   2.7%  |
| Master      |       1 |  0.0002%|
| Advanced    |       0 |   0%    |

Two consequences for the qualitative analysis:

- **Strategy-tier bar charts are limited to {Easy, Medium, BruteForce}**. The
  notebook's F2 figure pushes `n_clues_quartile` and `rating_quartile` to the
  top of the panel and demotes `hardest_strategy` to a secondary axis exactly
  for this reason.
- **Brute-force dominates raw averages**. Diffusion models with our objective
  do not perform recursive backtracking, so brute-force puzzles bake in a
  ~80% floor of "impossible" examples. The notebook's `DROP_BRUTE_FORCE`
  cohort toggle excludes them; expect the Relay / Relay-sg / Rollout-only / MLM gap
  to remain in the same order but with a higher absolute level on every
  model.

A 2000-puzzle slice (default) gives ~1720 BruteForce + 240 Easy + 40 Medium
puzzles — adequate for `n_clues`/`rating` quartile axes and for an Easy-only
sub-study, but tight on Medium. If you need Medium-tier statistical power,
bump `LIMIT_VAL_BATCHES=5000` (the `Medium` count rises to ~125 in expectation
and the dump time grows by 2.5×).

---

## Take-home claim, restated

Relay (BPTT) and Relay-sg (stop-grad) do not merely
dominate Uniform MLM and Rollout-only on accuracy vs NFE; on the same
puzzles and at matched NFE, they:

1. commit to algorithmic primitives (naked / hidden singles) earlier (F4);
2. preserve Sudoku legality more reliably during decoding (`first_step_illegal_rate`, `violations_total_steps`);
3. align more closely with the solver's strategy-conditioned fill order (F3);
4. show the largest exact-match gains on the hardest strategy tiers (F2);
5. push the Pareto frontier on accuracy vs NFE (F1).

Relay vs Relay-sg is the cleaner ablation for the relay-with-BPTT benefit, so the
notebook treats them as separate tracks throughout.
