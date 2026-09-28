# RELAY inference: confidence temperature versus hidden Gaussian noise

Research question: How much of the effect of Gaussian hidden-state noise can be explained by softmax temperature scaling?

## Intervention

All existing actor parameters are frozen; `eval()` and `inference_mode()` are used. The original model, supervised loss and production predictor are unchanged.

- Baseline: confidence temperature T=1, hidden sigma=0.
- Group A: confidence = max_v softmax(logits / T). Position selection uses the original cumulative uncertainty threshold (strictly < 0.15), with the original highest-confidence fallback. Tokens remain top-1 over the full vocabulary.
- Group B: T=1. AFTER the forward has returned logits and z, pass h_next = z + sigma * epsilon to the NEXT forward, with independent standard Normal coordinates. Noise covers all positions and channels, including clue positions. Clue TOKENS remain fixed. h_initial is zero. The current step's logits are not perturbed.

The noise is generated and added in float32. Sigma is an absolute per-coordinate standard deviation in the raw hidden representation, before the next relay LayerNorm; it is not normalized by hidden RMS. Forward precision is bf16 autocast by default, matching training validation. No digit-only vocabulary filter, remasking or correction is added.

## Files

- `sudoku/relay/inference_perturbation.py`: subclass of the original predictor with the two interventions and NFE instrumentation.
- `run.py`: checkpoint/EMA loading, deterministic dataset slicing, smoke checks, eight actual rollouts per puzzle, JSONL logging and summaries.
- `analyze.py`: independently recompute exact-match metrics from saved predictions, audit group completeness, calculate paired puzzle-bootstrap intervals, and export a Markdown report plus PNG/PDF comparison plots.

The original `relay/model.py`, `relay/predictor.py`, `relay/loss.py`, training configs, and the modified `xlm-core` submodule are not edited by this experiment.

## Protocol

Use the same explicitly chosen checkpoint for every configuration. The initial experiment uses tied RELAY, seed 1, step 300000, EMA weights, because the existing validation callback evaluates EMA weights. This is one checkpoint, not a multi-training-seed estimate.

The unfiltered evaluation cohort is the first 2000 rows of the locally saved Hugging Face test split, in dataset order before batching, without shuffle or dropped remainder. The cached split contains 422786 rows. Cached token IDs are checked against the raw puzzle and answer. Subset identity, checkpoint hash, source hashes, precision, seeds and all settings are recorded in `manifest.json` and `puzzles.jsonl`. Local order is preserved; this does not independently establish equality to a different upstream dataset revision.

Baseline and A are deterministic. The runner nevertheless performs all eight forwards/rollouts per puzzle and checks identical final answers, rather than inventing independent samples or estimating pass@8 from a single run. B uses independent noise streams for the eight attempts. The same batch/sample seeds are reused across sigma values; randomness is indexed by batch layout, so changing batch size changes the noise streams. No reference answers are given to the decoder.

Sweep values are exploratory and all results should be reported. Do not present the best test-sweep setting as a separately validated optimum. Use a disjoint development cohort to choose settings for a subsequent confirmatory experiment.

## Metrics

Let R_ij be exact equality to the stored full solution for puzzle i, attempt j.

- avg@8 = 100 * sum_ij R_ij / (8*N).
- pass@8 = 100 * sum_i 1[any_j R_ij] / N. This is observed any-success over exactly eight attempts, not a majority vote or an estimator from a larger pool.
- NFE / rollout = average actual network forwards experienced by a batch row.
- NFE / 8 = average sum of forwards over all eight attempts for one puzzle.

The original predictor continues to forward finished rows until their batch stops and always executes one final fill forward, even if nothing remains to fill. We preserve and count both behaviors. Thus actual NFE depends on batch composition/size and is not the per-puzzle first-completion depth. `first_filled_step` records the latter separately (-1 if masks remain after the final forward). Fix batch size and puzzle order when comparing configurations. The maximum actual NFE is max_steps+1 (65 by default).

Additional diagnostics: legal-board rate with preserved clues, distinct final answers per group, proportion of groups containing both success and failure, and unfilled rate. Forward counts are hardware-independent operation counts; wall times on the currently shared GPU are not clean speed benchmarks.

## Run

From the repository root, using the installed relay-sudoku environment:

```bash
/data/qilong/miniconda3/envs/relay-sudoku/bin/python \
  sudoku/experiments/hidden_noise_temperature/run.py \
  --checkpoint logs/sudoku_extreme_relay_bptt_steps2_300k_tied_seed1/checkpoints/40-300000.ckpt \
  --output logs/inference_hidden_temperature/full_test2000 \
  --weights ema --device cuda:2 --precision bf16 \
  --n 2000 --batch-size 512 \
  --temperatures 0.5 0.75 1.25 1.5 2 \
  --sigmas 0.05 0.1 0.2 0.5 1 2 5
```

Use a new output directory for each invocation. `--smoke-only --n 32` performs baseline trajectory equivalence, deterministic repetition, seeded-noise reproducibility, unchanged first-step token update, and clue-preservation checks. It also reports first-forward hidden RMS to interpret absolute sigma values. The initial smoke check found RMS about 15.91; sigma=2 and 5 were therefore added to cover stronger perturbations, without choosing them by accuracy.

Outputs:

- `manifest.json`, `checks.json`, `puzzles.jsonl`.
- One JSONL per configuration, one row per puzzle/attempt, including predicted token IDs, success, legality, noise seed, actual NFE and first-filled step.
- `summary.json`, `summary.csv`, `summary.md`, updated after each completed configuration.

After the runner completes, generate the audited report and plots:

```bash
/data/qilong/miniconda3/envs/relay-sudoku/bin/python \
  sudoku/experiments/hidden_noise_temperature/analyze.py \
  logs/inference_hidden_temperature/full_test2000
```

Bootstrap intervals resample puzzles with all eight outcomes together. They are pointwise and conditional on this checkpoint and these observed rollouts; they do not measure training-seed variation.

A process is complete only when the manifest status is `complete`; partially written output is not a finished experiment.

## Interpretation limits

These rollouts measure task-level performance and compute. Similar accuracy does not prove Gaussian hidden noise is temperature scaling. Temperature at a fixed value remains deterministic; hidden noise may improve pass@8 through diversity. An additional fixed-state distribution-matching experiment (fit T against noise-averaged next-step probabilities, then evaluate residual KL and token-rank changes on held-out states) is needed to directly quantify distributional approximation. That experiment is not implemented by this rollout runner.

## Additional group: fixed confidence T=0

`--noise-temperature 0` uses the exact limit as T approaches zero from above, rather than dividing logits by zero or substituting an arbitrary tiny positive temperature. Confidence is 1 for a unique maximum logit, and 1/m for m exactly tied maximum logits. Top-1 token selection is unchanged. Ties are assessed in the same forward precision as the other experiments (BF16 autocast), so finite-precision ties are retained.

At T=0, uniquely confident masked positions have zero uncertainty cost and can all be revealed on the first step. Noise added to the outgoing hidden does not affect this first token update. Consequently, the hidden-noise intervention may have very little opportunity to change the result. `first_step_remaining_masks` (per rollout) and `first_step_filled_rate` (aggregate percent) measure this directly.

```bash
/data/qilong/miniconda3/envs/relay-sudoku/bin/python \
  sudoku/experiments/hidden_noise_temperature/run.py \
  --checkpoint logs/sudoku_extreme_relay_bptt_steps2_300k_tied_seed1/checkpoints/40-300000.ckpt \
  --output logs/inference_hidden_temperature/test2000_T0_tied_seed1_ema_20260928 \
  --n 2000 --batch-size 512 --device cuda:2 \
  --temperatures --noise-temperature 0 \
  --sigmas 0 0.05 0.1 0.2 0.5 1 2 5
```

This also reruns the original T=1, sigma=0 baseline as a regression control. The T=0, sigma=0 result is a separate control within the new group. Eight attempts are actually executed for each of the nine configurations. All other checkpoint, dataset, seed, precision, batching and decoding settings match the preceding experiment.

Original results remain in `logs/inference_hidden_temperature/test2000_tied_seed1_ema_20260928`. The previous source versions and a checksum inventory of all original result files were preserved under `logs/inference_hidden_temperature/archive_before_T0_20260928` before adding T=0 support. Plot generation in `analyze.py` requires matplotlib; the numerical audit/report can run without it.
