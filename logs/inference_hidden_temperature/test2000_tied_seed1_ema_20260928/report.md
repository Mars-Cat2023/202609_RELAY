# RELAY: confidence temperature versus hidden Gaussian noise

## Protocol

- Checkpoint: `logs/sudoku_extreme_relay_bptt_steps2_300k_tied_seed1/checkpoints/40-300000.ckpt`; ema weights; 7,099,776 parameters.
- Dataset: 2000 unfiltered test puzzles, saved dataset indices 0–1999, no shuffle. All 8 attempts were actually evaluated for every configuration.
- Precision: bf16; batch size: 512; confidence threshold: 0.15; max ordinary steps: 64.
- Top-1 token selection and the original decoder are retained. Hidden noise affects the next forward, not the current logits. No training is performed.

## Results

| Configuration | T | sigma | avg@8 (%) | pass@8 (%) | NFE / rollout | NFE / 8 | Distinct answers / 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 0 | 67.00 | 67.00 | 22.51 | 180.10 | 1.00 |
| temperature_0.5 | 0.5 | 0 | 22.20 | 22.20 | 11.74 | 93.95 | 1.00 |
| temperature_0.75 | 0.75 | 0 | 52.10 | 52.10 | 17.28 | 138.24 | 1.00 |
| temperature_1.25 | 1.25 | 0 | 73.70 | 73.70 | 26.51 | 212.10 | 1.00 |
| temperature_1.5 | 1.5 | 0 | 76.90 | 76.90 | 30.98 | 247.81 | 1.00 |
| temperature_2 | 2 | 0 | 82.55 | 82.55 | 44.70 | 357.57 | 1.00 |
| hidden_sigma_0.05 | 1 | 0.05 | 66.92 | 68.05 | 22.74 | 181.89 | 1.81 |
| hidden_sigma_0.1 | 1 | 0.1 | 66.94 | 68.40 | 22.28 | 178.23 | 1.97 |
| hidden_sigma_0.2 | 1 | 0.2 | 66.90 | 69.05 | 22.48 | 179.88 | 2.21 |
| hidden_sigma_0.5 | 1 | 0.5 | 66.78 | 71.35 | 22.70 | 181.58 | 2.58 |
| hidden_sigma_1 | 1 | 1 | 66.64 | 73.15 | 22.00 | 175.98 | 2.86 |
| hidden_sigma_2 | 1 | 2 | 66.32 | 75.70 | 22.40 | 179.21 | 3.12 |
| hidden_sigma_5 | 1 | 5 | 61.48 | 79.05 | 23.59 | 188.74 | 3.72 |

NFE counts actual forwards per row, including work on already finished rows and the unconditional final forward. It depends on batch composition. `summary.csv` also reports first-filled steps. NFE / 8 counts the total budget across eight attempts.

## Differences from baseline

Pointwise 95% paired puzzle-bootstrap intervals (percentage points):

| Configuration | avg@8 change [95% CI] | pass@8 change [95% CI] |
|---|---:|---:|
| baseline | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| temperature_0.5 | -44.80 [-47.15, -42.65] | -44.80 [-47.15, -42.65] |
| temperature_0.75 | -14.90 [-16.70, -13.20] | -14.90 [-16.70, -13.20] |
| temperature_1.25 | +6.70 [+5.50, +8.05] | +6.70 [+5.50, +8.05] |
| temperature_1.5 | +9.90 [+8.45, +11.45] | +9.90 [+8.45, +11.45] |
| temperature_2 | +15.55 [+13.65, +17.40] | +15.55 [+13.65, +17.40] |
| hidden_sigma_0.05 | -0.07 [-0.43, +0.27] | +1.05 [+0.60, +1.55] |
| hidden_sigma_0.1 | -0.06 [-0.41, +0.30] | +1.40 [+0.90, +1.95] |
| hidden_sigma_0.2 | -0.10 [-0.50, +0.30] | +2.05 [+1.40, +2.70] |
| hidden_sigma_0.5 | -0.22 [-0.73, +0.29] | +4.35 [+3.50, +5.25] |
| hidden_sigma_1 | -0.36 [-1.05, +0.31] | +6.15 [+5.10, +7.15] |
| hidden_sigma_2 | -0.68 [-1.52, +0.09] | +8.70 [+7.45, +9.90] |
| hidden_sigma_5 | -5.52 [-6.72, -4.44] | +12.05 [+10.60, +13.50] |

## Interpretation limits

Fixed confidence temperature is deterministic: avg@8 equals pass@8. Gaussian noise can change the answer across attempts. Distinct outputs do not necessarily yield different rewards; mixed-reward group rates are in summary.csv.

This is an exploratory sweep on one checkpoint, with all settings reported. The intervals condition on that checkpoint and these eight rollouts, and are not corrected for multiple comparisons. A best observed setting is not a validated optimum. Shared GPU wall times are not clean speed benchmarks.

Task metrics alone do not establish distributional equivalence to temperature scaling. A fixed-state next-step probability matching experiment with held-out evaluation of residual KL and rank changes is still needed to quantify that approximation directly.

## Audit

Original baseline trajectory equivalence, repeat determinism, seeded noise reproducibility, first-step timing and clue preservation passed. All per-attempt exact-match labels and aggregate avg@8, pass@8 and NFE values were independently recomputed from saved predictions. See checks.json, manifest.json and bootstrap.json.
