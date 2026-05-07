# Sudoku Extreme 300k Training Commands

All commands train `sudoku_extreme` for 300k steps, validate every 5k steps, and use the
`dhruvesh` SLURM reservation.

**W&B tags:** each run adds Hydra `+tags.*` overrides so the logger records stable
`key=value` tags (see `xlm-core/src/xlm/configs/lightning_train/loggers/wandb.yaml` and
`dict_to_list`). Use them to filter and compare runs in the W&B UI:

- `sweep=sudoku_extreme_300k` — all jobs in this ablation
- `embed_tying=untied` or `tied` — weight tying
- `objective=…` — training recipe:
  - `mlm_uniform` — vanilla MLM
  - `puma_no_loopholing` — streaming PUMA loss, `loss.with_loopholing=false`, plain MLM model
  - `puma_loopholing_stop_grad` — loopholing + PUMA, `loss.stop_grad_h_s=true`
  - `puma_loopholing_bptt` — loopholing + PUMA with BPTT through `h` (`num_steps=2`, `stop_grad_h_s=false`)
- `seed=…` — random seed (only set by the seeded sweep below; absent on the original eight runs which used the Hydra default `seed=1`).

**Batch submission:** the same eight jobs (including these tags) are defined in
`submit_sudoku_300k_sweep.sh` (`TAG_SWEEP` is overridable; `DO=print` dry-runs). The blocks
below are the expanded equivalents for copy-paste or documentation. To launch the same
eight jobs across multiple random seeds, use `submit_sudoku_300k_seeds_sweep.sh`
(documented at the bottom of this file).

## Validation metrics

The two sudoku experiment YAMLs (`sudoku_extreme_mlm_uniform`, `sudoku_extreme_loopholing_bptt_puma`)
log the following per validation epoch:

- `val/prediction/exact_match` — fraction of boards predicted exactly correctly.
- `val/prediction/token_accuracy` — fraction of masked cells filled correctly.
- `val/prediction/rollout_steps` — mean number of forward evaluations (NFE) until the predictor terminates.
- `val/prediction/legal_rate` — fraction of boards whose final 81-cell prediction is **fully filled** AND has **no Sudoku constraint violations** (no duplicate digit in any row, column, or 3×3 box). Distinguishes "wrong but plausible" from "completely broken"; this would have flagged the original tied-StopGrad run as diverged at the very first validation epoch instead of after a full training run + offline trajectory dump. Implemented in `doublebackprop/sudoku_metrics.py`.
- `val_sweep/t_{0p05,0p10,0p15,0p20,0p25}/{exact_match, token_accuracy, rollout_steps, legal_rate}` — same metrics computed at five inference confidence thresholds. The `legal_rate` entry was added at the same time as the per-step metric; older runs do not have it.
- The same suite is also reported under `test/prediction/*` and the BPTT-specific diagnostics (`train/L0`, `train/L1`, `train/h_carry_rms_*`, `train/h_delta_rms_*`, `train/h_s_rms_*`, `train/conf_threshold_t_*`, `train/loss_share_*`, `train/masks_at_x_*`, `train/unmask_frac_*`, `train/evicted_count`) continue to be logged on the loopholing recipes.

The `legal_rate` metric is the only addition over what's already in `wandb.ai/ilm-extensions/BPTT-sudoku`. Other metrics that were considered and rejected as fluff: rollout-step distribution percentiles (already implied by `rollout_steps` mean across the threshold sweep); validation gradient norm (uninformative for sudoku); per-step rollout diagnostics (already printed via Python logger; logging them every val epoch creates many noisy series with no actionable signal); EMA vs raw weight performance (not tracked anywhere, but EMA-eval is already enabled via the existing callback).

## Untied Vocab Embeddings / Unembeddings

### MLM Uniform

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_mlm_uniform_300k_untied" \
"train.experiment=sudoku_extreme_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"+tags.sweep=sudoku_extreme_300k" \
"+tags.embed_tying=untied" \
"+tags.objective=mlm_uniform" \
"trainer.max_steps=300000" \
"trainer.val_check_interval=5000"
```

### PUMA Only, No Loopholing

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_puma_only_300k_untied" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram80,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"+tags.sweep=sudoku_extreme_300k" \
"+tags.embed_tying=untied" \
"+tags.objective=puma_no_loopholing" \
"model=rotary_transformer_xtiny" \
"loss.with_loopholing=false" \
"loss.stop_grad_h_s=true" \
"predictor.with_loopholing=false" \
"trainer.max_steps=300000" \
"trainer.val_check_interval=5000"
```

### PUMA With Loopholing, BPTT Disabled

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_loopholing_puma_no_bptt_300k_untied" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram80,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"+tags.sweep=sudoku_extreme_300k" \
"+tags.embed_tying=untied" \
"+tags.objective=puma_loopholing_stop_grad" \
"loss.with_loopholing=true" \
"loss.stop_grad_h_s=true" \
"predictor.with_loopholing=true" \
"trainer.max_steps=300000" \
"trainer.val_check_interval=5000"
```

### PUMA With Loopholing, BPTT Enabled (`num_steps=2`)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_loopholing_puma_bptt_steps2_300k_untied" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram80,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"+tags.sweep=sudoku_extreme_300k" \
"+tags.embed_tying=untied" \
"+tags.objective=puma_loopholing_bptt" \
"loss.with_loopholing=true" \
"loss.stop_grad_h_s=false" \
"loss.num_steps=2" \
"predictor.with_loopholing=true" \
"trainer.max_steps=300000" \
"trainer.val_check_interval=5000"
```

## Tied Vocab Embeddings / Unembeddings

### MLM Uniform

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_mlm_uniform_300k_tied" \
"train.experiment=sudoku_extreme_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"+tags.sweep=sudoku_extreme_300k" \
"+tags.embed_tying=tied" \
"+tags.objective=mlm_uniform" \
"++model.tie_embeddings=true" \
"trainer.max_steps=300000" \
"trainer.val_check_interval=5000"
```

### PUMA Only, No Loopholing

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_puma_only_300k_tied" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram80,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"+tags.sweep=sudoku_extreme_300k" \
"+tags.embed_tying=tied" \
"+tags.objective=puma_no_loopholing" \
"model=rotary_transformer_xtiny" \
"++model.tie_embeddings=true" \
"loss.with_loopholing=false" \
"loss.stop_grad_h_s=true" \
"predictor.with_loopholing=false" \
"trainer.max_steps=300000" \
"trainer.val_check_interval=5000"
```

### PUMA With Loopholing, BPTT Disabled

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_loopholing_puma_no_bptt_300k_tied" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram80,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"+tags.sweep=sudoku_extreme_300k" \
"+tags.embed_tying=tied" \
"+tags.objective=puma_loopholing_stop_grad" \
"++model.tie_embeddings=true" \
"loss.with_loopholing=true" \
"loss.stop_grad_h_s=true" \
"predictor.with_loopholing=true" \
"trainer.max_steps=300000" \
"trainer.val_check_interval=5000"
```

### PUMA With Loopholing, BPTT Enabled (`num_steps=2`)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_loopholing_puma_bptt_steps2_300k_tied" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram80,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"+tags.sweep=sudoku_extreme_300k" \
"+tags.embed_tying=tied" \
"+tags.objective=puma_loopholing_bptt" \
"++model.tie_embeddings=true" \
"loss.with_loopholing=true" \
"loss.stop_grad_h_s=false" \
"loss.num_steps=2" \
"predictor.with_loopholing=true" \
"trainer.max_steps=300000" \
"trainer.val_check_interval=5000"
```

## Seeded sweep (8 ablations × 3 seeds = 24 jobs)

Use `submit_sudoku_300k_seeds_sweep.sh` to relaunch all eight ablations across multiple
random seeds. Each job is named `<base_job_name>_seed${SEED}`, which yields:

- a unique **log directory** (`logs/<base>_seed<N>/`),
- a unique **W&B run id** (since `use_job_name_as_id=false` falls through to the job name),
- a `+tags.seed=<N>` W&B tag for filtering and aggregation in W&B.

The Hydra default seed is `1`, so the original 8 runs in `wandb.ai/ilm-extensions/BPTT-sudoku`
correspond to "seed=1 (untagged)". The seeded sweep uses fresh job names and adds `+tags.seed`,
so it never collides with the existing runs.

### Quick start

```bash
# Always dry-run first.
DO=print ./submit_sudoku_300k_seeds_sweep.sh

# Submit all 24 jobs (seeds 1, 2, 3).
./submit_sudoku_300k_seeds_sweep.sh

# Only submit the two new seeds (saves 8 redundant launches if you treat the
# pre-existing untagged runs as "seed=1").
SEEDS="2 3" ./submit_sudoku_300k_seeds_sweep.sh

# Filter to a subset of objectives or tyings (useful for partial retries).
ONLY_OBJECTIVES="puma_loopholing_bptt mlm_uniform" \
ONLY_TYINGS="untied" \
./submit_sudoku_300k_seeds_sweep.sh

# Skip jobs whose log dir already exists (idempotent re-runs).
SKIP_EXISTING_LOGDIR=1 ./submit_sudoku_300k_seeds_sweep.sh
```

### Common environment variable overrides

| Variable | Default | Purpose |
| --- | --- | --- |
| `DO` | `submit` | `print` for dry-run; `submit` actually `sbatch`'es. |
| `SEEDS` | `1 2 3` | Whitespace-separated seed list. |
| `ONLY_OBJECTIVES` | (all 4) | Filter to a subset (`mlm_uniform`, `puma_no_loopholing`, `puma_loopholing_stop_grad`, `puma_loopholing_bptt`). |
| `ONLY_TYINGS` | (`untied tied`) | Filter to weight-tying conditions. |
| `SKIP_EXISTING_LOGDIR` | `0` | If `1`, skip submission when `logs/<job_name>/` already exists. |
| `TAG_SWEEP` | `sudoku_extreme_300k_seeds` | W&B `sweep=` tag for filtering. |
| `TRAIN_MAX_STEPS` | `300000` | Length of training run. |
| `TRAIN_VAL_INTERVAL` | `5000` | Validation interval (for live tracking of `legal_rate`/`exact_match`). |
| `SLURM_CONSTRAIN_MLM` / `SLURM_CONSTRAIN_PUMA` | `vram40,bf16` / `vram80,bf16` | GPU constraints for the two job classes. |
| `SLURM_TIME` | `36:00:00` | Wall-clock limit. |

### What gets reported in W&B for each new run

The new runs add `val/prediction/legal_rate` (and `val_sweep/t_*/legal_rate`) to all the
metrics already in `wandb.ai/ilm-extensions/BPTT-sudoku`. In a 4-panel dashboard with
`exact_match`, `legal_rate`, `rollout_steps`, and `train/loss`, divergence shows up as
`legal_rate → 0` (and stays there) much earlier than the eventual `exact_match` flatline.
Aggregating across seeds is straightforward: group by `(objective, embed_tying)` and
plot mean ± stderr over `seed ∈ {1, 2, 3}`.

## BPTT unroll-length ablation (T ∈ {3, 4, 8}, no validation)

Use `submit_sudoku_300k_bptt_steps_sweep.sh` to extend the loopholing+PUMA+BPTT
recipe (`sudoku_extreme_loopholing_puma_bptt_steps2_*` from
`submit_sudoku_300k_sweep.sh`) to longer BPTT horizons. Six jobs total:

| `T` | untied                                                       | tied                                                       |
| --- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| 3   | `sudoku_extreme_loopholing_puma_bptt_steps3_300k_untied`     | `sudoku_extreme_loopholing_puma_bptt_steps3_300k_tied`     |
| 4   | `sudoku_extreme_loopholing_puma_bptt_steps4_300k_untied`     | `sudoku_extreme_loopholing_puma_bptt_steps4_300k_tied`     |
| 8   | `sudoku_extreme_loopholing_puma_bptt_steps8_300k_untied`     | `sudoku_extreme_loopholing_puma_bptt_steps8_300k_tied`     |

The launch loop iterates `T` outer, `tying` inner, so all jobs for `T=3`
enter the queue before `T=4`, then `T=8`. Single seed only (Hydra default
`seed=1`, untagged) to match the original eight runs in
`wandb.ai/ilm-extensions/BPTT-sudoku`.

**Validation is fully disabled.** Each job adds:

- `++trainer.limit_val_batches=0` — Lightning skips every val batch, so
  `val/*` and `val_sweep/*` series are never written to W&B.
- `++trainer.num_sanity_val_steps=0` — no pre-train sanity validation.

Side effects:
- The `val_threshold_sweep` callback becomes a no-op (its hooks only fire
  during validation).
- `checkpoint_monitor` (which monitors `val/lm/accumulated_loss`) cannot
  pick a "best" checkpoint and is effectively disabled. Periodic
  snapshotting via `checkpoint_every_n_steps_with_thinning` is unaffected.

W&B tags added per run for filtering:
- `sweep=sudoku_extreme_300k_bptt_steps_ablation`
- `embed_tying=untied | tied`
- `objective=puma_loopholing_bptt`
- `bptt_steps=3 | 4 | 8`

### Quick start

```bash
DO=print ./submit_sudoku_300k_bptt_steps_sweep.sh           # dry-run

./submit_sudoku_300k_bptt_steps_sweep.sh                    # 6 jobs

BPTT_STEPS="3 4" ./submit_sudoku_300k_bptt_steps_sweep.sh   # subset of T
ONLY_TYINGS="untied" ./submit_sudoku_300k_bptt_steps_sweep.sh
SKIP_EXISTING_LOGDIR=1 ./submit_sudoku_300k_bptt_steps_sweep.sh
```

### Common environment variable overrides

| Variable | Default | Purpose |
| --- | --- | --- |
| `DO` | `submit` | `print` for dry-run; `submit` actually `sbatch`'es. |
| `BPTT_STEPS` | `3 4 8` | Whitespace-separated list of BPTT unroll lengths. |
| `TYINGS` | `untied tied` | Whitespace-separated list of tying conditions. |
| `ONLY_TYINGS` | (all) | Filter to a subset of tyings. |
| `SKIP_EXISTING_LOGDIR` | `0` | If `1`, skip submission when `logs/<job_name>/` already exists. |
| `TAG_SWEEP` | `sudoku_extreme_300k_bptt_steps_ablation` | W&B `sweep=` tag for filtering. |
| `TRAIN_MAX_STEPS` | `300000` | Length of training run. |
| `BATCH_SIZE` | `512` | Per-device batch size. |
| `SLURM_CONSTRAIN_PUMA` | `vram80,bf16` | GPU constraint. |
| `SLURM_TIME` | `36:00:00` | Wall-clock limit. |

**Memory note:** BPTT activation memory grows ~linearly with `num_steps`,
so `T=8` at `batch_size=512` may exceed 80GB. If the job OOMs, drop
`BATCH_SIZE` or move to a larger-VRAM partition via `SLURM_CONSTRAIN_PUMA`.

## Follow-ups for the qualitative study

These are ergonomic options for the offline trajectory-dump pipeline (`SUDOKU_ANALYSIS_COMMANDS.md`),
not training-time changes.

### Bigger evaluation slice

The current dump uses `+dump.max_examples=200` and `+dump.limit_val_batches=200`, which
is bounded by the size of the HF validation split. Two ways to grow it:

1. **Run on the test split.** The HF dataset
   `brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver` exposes a `test` split
   in addition to `validation`. Pass `+dump.split=test` (or whichever flag your dump
   script accepts) and bump `+dump.max_examples` to the test split size.
2. **Use the full sapient `sudoku_extreme` test set** (~422k puzzles). For quantitative
   sweeps you don't need solver trajectories, so the join with the timvink solver
   metadata becomes optional. Just dump the ids/predictions and skip the
   `solver_trajectory`/`strategies_used` join in `n_way_pair_trajectories.py`.

For statistical power on the strategy-conditioned analysis (F2/F3 in
`plotting_scripts/sudoku_qualitative_study.ipynb`), you want at least ~50 puzzles per
strategy tier. The current 200-example slice has effectively zero "Advanced/Master"
puzzles — bumping to a few thousand from the test split should give you those tiers.

### Brute-force exclusion sub-study

The `timvink/sudoku-solver` annotates each puzzle with the strategies it used to solve it,
including a terminal "Brute Force" tier that means "the solver had to recursively backtrack".
Diffusion models trained with our objective do not perform recursive backtracking, so
including those puzzles bakes a ~80% floor of "impossible" examples into every comparison
and inflates the reported gap on the things the model can do.

In the parquet table produced by `n_way_pair_trajectories.py`, brute-force puzzles are
identifiable by either:

- `hardest_strategy == "Brute Force"` (computed by `sudoku_analysis.hardest_strategy_tier`), or
- `"Brute Force" in strategies_used` (raw HF column).

To run the no-brute-force sub-study, filter the dataframe at notebook-load time:

```python
df = pd.read_parquet(parquet_path)
df_no_bf = df[df["hardest_strategy"] != "Brute Force"].copy()
```

Then re-run the F1–F4 cells against `df_no_bf`. Expect:

- the Pareto frontier to compress (fewer "impossible" zeros for every model), making
  the LEAR vs. baseline gap *cleaner* but *smaller*;
- per-strategy gain (F2) to become populated for Easy/Medium/Advanced rather than dominated
  by the Brute-Force bucket;
- legality and solver-alignment metrics (F3/F4) to remain qualitatively similar — they
  weren't biased by the brute-force tier in the first place since they're computed over
  *whatever the model produced*, not over puzzles the model can solve.
