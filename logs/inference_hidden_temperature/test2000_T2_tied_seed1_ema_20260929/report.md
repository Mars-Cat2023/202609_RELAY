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
| hidden_T2_sigma_0 | 2 | 0 | 82.55 | 82.55 | 44.70 | 357.57 | 1.00 |
| hidden_T2_sigma_0.5 | 2 | 0.5 | 82.29 | 84.70 | 47.19 | 377.48 | 2.08 |
| hidden_T2_sigma_1 | 2 | 1 | 82.12 | 85.95 | 46.77 | 374.15 | 2.16 |
| hidden_T2_sigma_2 | 2 | 2 | 81.86 | 87.45 | 47.11 | 376.86 | 2.24 |

NFE counts actual forwards per row, including work on already finished rows and the unconditional final forward. It depends on batch composition. `summary.csv` also reports first-filled steps. NFE / 8 counts the total budget across eight attempts.

## Differences from reference: hidden_T2_sigma_0

Pointwise 95% paired puzzle-bootstrap intervals (percentage points):

| Configuration | avg@8 change [95% CI] | pass@8 change [95% CI] |
|---|---:|---:|
| baseline | -15.55 [-17.40, -13.65] | -15.55 [-17.40, -13.65] |
| hidden_T2_sigma_0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| hidden_T2_sigma_0.5 | -0.26 [-0.62, +0.09] | +2.15 [+1.55, +2.80] |
| hidden_T2_sigma_1 | -0.43 [-0.91, +0.06] | +3.40 [+2.60, +4.25] |
| hidden_T2_sigma_2 | -0.69 [-1.36, -0.04] | +4.90 [+3.90, +5.85] |

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
| hidden_T2_sigma_0 | 0.00 | 9.423 | 0.00 |
| hidden_T2_sigma_0.5 | 0.00 | 9.540 | 5.65 |
| hidden_T2_sigma_1 | 0.00 | 9.580 | 8.95 |
| hidden_T2_sigma_2 | 0.00 | 9.695 | 14.20 |
