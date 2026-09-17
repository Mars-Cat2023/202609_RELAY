# Sudoku Extreme 300k training commands

All commands train `sudoku_extreme` for 300k steps and validate every 5k steps.
The four objectives map directly to Table 1 of the paper:

| W&B `objective` tag | What it is                                     | Hydra overrides                                                                                                |
|---------------------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `mlm_uniform`       | Mask-uniform CE baseline                       | (uses `sudoku_extreme_mlm_uniform.yaml` directly)                                                              |
| `rollout`           | Rollout-buffer-only (no relay carry)           | `loss.with_relay=false  loss.stop_grad_h_s=true  predictor.with_relay=false  model=rotary_transformer_xtiny` |
| `relay_sg`          | Relay rollout with stop-gradient on `h`        | `loss.with_relay=true   loss.stop_grad_h_s=true   predictor.with_relay=true`                                   |
| `relay`             | Relay rollout with truncated BPTT (`T=2`)      | `loss.with_relay=true   loss.stop_grad_h_s=false  loss.num_steps=2  predictor.with_relay=true`                 |

Each objective is run in two weight-tying conditions (`+tags.embed_tying=tied`
adds `++model.tie_embeddings=true`; `untied` is the default), giving the eight
runs of Table 1. The paper averages seeds `1`, `2`, `3`.

Shared Hydra knobs on every Table 1 command (all launched via the `xlm`
console script with `job_type=train` and a unique `job_name`):

```
++trainer.precision=bf16-mixed
trainer.max_steps=300000
trainer.val_check_interval=5000
per_device_batch_size=512
global_batch_size=512
seed=<1|2|3>
```

Relay-family jobs used 80GB-class GPUs at batch 512; drop both batch-size
flags together if VRAM is smaller. Full copy-paste commands are in
[`README.md`](README.md).

## Validation metrics (logged every `val_check_interval`)

- `val/prediction/exact_match` — fraction of boards predicted exactly correctly.
- `val/prediction/token_accuracy` — fraction of masked cells filled correctly.
- `val/prediction/rollout_steps` — mean number of forward evaluations (NFE) until the predictor terminates.
- `val/prediction/legal_rate` — fraction of boards whose final 81-cell prediction is fully filled AND has no Sudoku constraint violation. Distinguishes "wrong but plausible" from "completely broken"; flags divergence in the very first validation epoch.
- `val_sweep/t_{0p05,0p10,0p15,0p20,0p25}/{exact_match, token_accuracy, rollout_steps, legal_rate}` — same metrics computed at five inference confidence thresholds.
- The same suite is also reported under `test/prediction/*`.

## Smoke test (single GPU)

Reduce steps to confirm the loop is wired up before launching the full 300k:

```bash
cd relay/sudoku
source .venv_relay/bin/activate
export PROJECT_ROOT="$PWD"

xlm job_type=train \
  job_name=sudoku_extreme_smoke \
  experiment=sudoku_extreme_relay_bptt \
  trainer.max_steps=200 \
  trainer.val_check_interval=100 \
  trainer.limit_val_batches=2 \
  per_device_batch_size=8 \
  global_batch_size=8 \
  loggers.wandb=null
```

Replace `experiment=sudoku_extreme_relay_bptt` with `sudoku_extreme_mlm_uniform`
to smoke-test the baseline.

## Paper Table 1 numbers

After all 24 (8 × 3 seeds) runs finish, the per-step Table 1 numbers are
the validation-set metrics logged to W&B (or to local TensorBoard logs).
The relevant columns map to the metric keys logged every
`val_check_interval`:

- **Exact-match accuracy** → `val/prediction/exact_match`
- **Token accuracy** → `val/prediction/token_accuracy`
- **Mean NFE** → `val/prediction/rollout_steps`
- **Legal-rate** (Sudoku constraint check, used to flag divergence) →
  `val/prediction/legal_rate`

Group by the `+tags.{objective, embed_tying, seed}` tags to recover the
eight rows × three seeds; average over seeds.
