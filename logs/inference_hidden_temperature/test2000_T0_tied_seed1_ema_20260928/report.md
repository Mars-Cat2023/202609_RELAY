# RELAY: confidence temperature versus hidden Gaussian noise

## Protocol

- Checkpoint: `logs/sudoku_extreme_relay_bptt_steps2_300k_tied_seed1/checkpoints/40-300000.ckpt`; ema weights; 7,099,776 parameters.
- Dataset: 2000 unfiltered test puzzles, saved dataset indices 0–1999, no shuffle. All 8 attempts were actually evaluated for every configuration.
- Precision: bf16; batch size: 512; confidence threshold: 0.15; max ordinary steps: 64.
- Top-1 token selection and the original decoder are retained. Hidden noise affects the next forward, not the current logits. No training is performed.

- T=0, when requested, means the exact softmax T->0+ limit, with uniform probability on tied maxima.
## Results

| Configuration | T | sigma | avg@8 (%) | pass@8 (%) | NFE / rollout | NFE / 8 | Distinct answers / 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 0 | 67.00 | 67.00 | 22.51 | 180.10 | 1.00 |
| hidden_T0_sigma_0 | 0 | 0 | 0.55 | 0.55 | 3.23 | 25.86 | 1.00 |
| hidden_T0_sigma_0.05 | 0 | 0.05 | 0.55 | 0.55 | 3.15 | 25.16 | 1.00 |
| hidden_T0_sigma_0.1 | 0 | 0.1 | 0.55 | 0.55 | 3.12 | 24.95 | 1.00 |
| hidden_T0_sigma_0.2 | 0 | 0.2 | 0.55 | 0.55 | 3.09 | 24.74 | 1.01 |
| hidden_T0_sigma_0.5 | 0 | 0.5 | 0.55 | 0.55 | 3.00 | 24.00 | 1.02 |
| hidden_T0_sigma_1 | 0 | 1 | 0.55 | 0.55 | 3.06 | 24.49 | 1.05 |
| hidden_T0_sigma_2 | 0 | 2 | 0.55 | 0.55 | 3.06 | 24.49 | 1.12 |
| hidden_T0_sigma_5 | 0 | 5 | 0.55 | 0.55 | 3.12 | 25.00 | 1.28 |

NFE counts actual forwards per row, including work on already finished rows and the unconditional final forward. It depends on batch composition. `summary.csv` also reports first-filled steps. NFE / 8 counts the total budget across eight attempts.

## Differences from baseline

Pointwise 95% paired puzzle-bootstrap intervals (percentage points):

| Configuration | avg@8 change [95% CI] | pass@8 change [95% CI] |
|---|---:|---:|
| baseline | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| hidden_T0_sigma_0 | -66.45 [-68.50, -64.45] | -66.45 [-68.50, -64.45] |
| hidden_T0_sigma_0.05 | -66.45 [-68.50, -64.45] | -66.45 [-68.50, -64.45] |
| hidden_T0_sigma_0.1 | -66.45 [-68.50, -64.45] | -66.45 [-68.50, -64.45] |
| hidden_T0_sigma_0.2 | -66.45 [-68.50, -64.45] | -66.45 [-68.50, -64.45] |
| hidden_T0_sigma_0.5 | -66.45 [-68.50, -64.45] | -66.45 [-68.50, -64.45] |
| hidden_T0_sigma_1 | -66.45 [-68.50, -64.45] | -66.45 [-68.50, -64.45] |
| hidden_T0_sigma_2 | -66.45 [-68.50, -64.45] | -66.45 [-68.50, -64.45] |
| hidden_T0_sigma_5 | -66.45 [-68.50, -64.45] | -66.45 [-68.50, -64.45] |

## Interpretation limits

Fixed confidence temperature is deterministic: avg@8 equals pass@8. Gaussian noise can change the answer across attempts. Distinct outputs do not necessarily yield different rewards; mixed-reward group rates are in summary.csv.

This is an exploratory sweep on one checkpoint, with all settings reported. The intervals condition on that checkpoint and these eight rollouts, and are not corrected for multiple comparisons. A best observed setting is not a validated optimum. Shared GPU wall times are not clean speed benchmarks.

Task metrics alone do not establish distributional equivalence to temperature scaling. A fixed-state next-step probability matching experiment with held-out evaluation of residual KL and rank changes is still needed to quantify that approximation directly.

## Audit

Original baseline trajectory equivalence, repeat determinism, seeded noise reproducibility, first-step timing and clue preservation passed. All per-attempt exact-match labels and aggregate avg@8, pass@8 and NFE values were independently recomputed from saved predictions. See checks.json, manifest.json and bootstrap.json.

## First-step completion

This diagnostic measures where outgoing hidden noise can still affect future token decisions.

| Configuration | Filled after first step (%) | Mean first-filled step | Mixed-reward groups (%) |
|---|---:|---:|---:|
| baseline | 0.05 | 6.333 | 0.00 |
| hidden_T0_sigma_0 | 49.70 | 1.504 | 0.00 |
| hidden_T0_sigma_0.05 | 49.70 | 1.503 | 0.00 |
| hidden_T0_sigma_0.1 | 49.70 | 1.503 | 0.00 |
| hidden_T0_sigma_0.2 | 49.70 | 1.503 | 0.00 |
| hidden_T0_sigma_0.5 | 49.70 | 1.503 | 0.00 |
| hidden_T0_sigma_1 | 49.70 | 1.503 | 0.00 |
| hidden_T0_sigma_2 | 49.70 | 1.503 | 0.00 |
| hidden_T0_sigma_5 | 49.70 | 1.503 | 0.00 |

## Comparison with the preserved original experiment

All 23 original result files retain their recorded SHA-256 hashes. The rerun T=1, sigma=0 baseline matches all 16000 original predictions, success labels and NFE counts. Checkpoint and dataset-subset hashes also match.

At T=0, 98.705% of initially blank cells are already filled during the first step; only 0.7215 masked cells per puzzle remain on average. This first step precedes any influence from outgoing hidden noise. Existing filled tokens cannot be corrected by subsequent steps. All scanned sigmas have avg@8 = pass@8 = 0.55%, although some failed answers vary with noise.

See preservation_and_comparison_checks.json and combined_summary.csv for the preservation audit and all previous/new configurations.
