# OpenCode/OpenMath Mixed SFT - Fast-dLLM v2 1.5B

## 1. Data Prep

From `Fast-dLLM/v2`:

```bash
conda activate lmflow
python scripts/prep_opencode_openmath_mix.py
```

Default output:

```text
data/opencode_openmath_60k/train_conversation/train_*.json
data/opencode_openmath_60k/prep_summary.json
```

The default recipe is 60k rows: 33k from `nvidia/OpenCodeInstruct` and 27k from
`nvidia/OpenMathInstruct-2`. Rows are shuffled, prompt-deduplicated, filtered to
tokenized `fast_dllm_v2` chat length <= `BLOCK_SIZE=2048`, and sharded.

Math-heavy follow-up recipe (40% code / 60% math):

```bash
conda activate lmflow
python scripts/prep_opencode_openmath_mix.py \
  --out_dir data/opencode_openmath_60k_c40m60 \
  --code_rows 24000 \
  --math_rows 36000 \
  --require_code_def
```

The explicit `--require_code_def` keeps the code filter aligned with the first
prepared OpenCode/OpenMath mixture.

## 2. Four Concurrent 2-GPU Runs

Dry run first:

```bash
DRY_RUN=1 bash scripts/launch_opencode_openmath_4run.sh
```

Submit the four ablations:

```bash
bash scripts/launch_opencode_openmath_4run.sh
```

Submit with a reservation:

```bash
RESERVATION=dhruvesh bash scripts/launch_opencode_openmath_4run.sh
```

Submit the math-heavy `c40m60` follow-up:

```bash
DRY_RUN=1 bash scripts/launch_opencode_openmath_c40m60_4run.sh
bash scripts/launch_opencode_openmath_c40m60_4run.sh
```

Submit the `c40m60` gradient-clipping follow-up. This reuses the same c40m60
dataset, keeps the original BD-block unmasking rule, and changes only
`MAX_GRAD_NORM` from `1.0` to `5.0`:

```bash
DRY_RUN=1 bash scripts/launch_opencode_openmath_c40m60_gradclip5_4run.sh
bash scripts/launch_opencode_openmath_c40m60_gradclip5_4run.sh
```

Submit the `c40m60` decoder-aligned rollout follow-up. This reuses the same
c40m60 dataset, keeps `MAX_GRAD_NORM=1.0`, and sets
`BPTT_UNMASK_STRATEGY=decode_aligned` with `BPTT_INNER_BLOCK_SIZE=8` for BPTT
runs:

```bash
DRY_RUN=1 bash scripts/launch_opencode_openmath_c40m60_decodealigned8_4run.sh
bash scripts/launch_opencode_openmath_c40m60_decodealigned8_4run.sh
```

Submit the `c40m60` Loopguard layer ablations. This reuses the same c40m60
dataset, keeps the default Loopguard setting (`MAX_GRAD_NORM=1.0`,
`BPTT_STOP_GRAD_H_S=0`, `BPTT_UNMASK_STRATEGY=bd`), and launches four runs
where the loophole carry comes from zero-based decoder layers 13, 17, 21, and
25 instead of the final normalized hidden state. Intermediate layers feed the
raw residual-stream output (pre-final-RMSNorm) into the existing zero-init
`h_t_layer_norm`; matches CAB's convention for intermediate `read_layers`:

```bash
DRY_RUN=1 bash scripts/launch_opencode_openmath_c40m60_loopguard_layers_4run.sh
bash scripts/launch_opencode_openmath_c40m60_loopguard_layers_4run.sh
```

By default, this launcher writes checkpoints under
`/scratch4/workspace/brozonoyer_umass_edu-fastdllm_ckpts/output_models` and
creates a symlink at
`output_models/opencode_openmath_60k_c40m60_loopguard_layers_1p5B`. That keeps
the repo paths usable for eval commands while avoiding large checkpoint writes
inside the working tree. Override with `OUTPUT_MODELS_ROOT=...` if needed.

The full-family helpers launch:

- Vanilla SFT
- BPTT + PUMA, `CARRY_MODE=none`
- BPTT + PUMA + Loopguard, `BPTT_STOP_GRAD_H_S=1`
- BPTT + PUMA + Loopguard, `BPTT_STOP_GRAD_H_S=0`

The Loopguard layer-ablation helper instead launches only the four
`BPTT_STOP_GRAD_H_S=0` Loopguard jobs, one per `BPTT_LOOPHOLE_LAYER`.

Default training settings:

```text
DATASET_PATH=data/opencode_openmath_60k/train_conversation
RECIPE_TAG=opencode_openmath_60k
BLOCK_SIZE=2048
DISABLE_GROUP_TEXTS=1
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=5e-6
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=0
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=16
```

With 2 GPUs per run, the effective batch size is `2 * 16 * 2 = 64`.

The launchers export distinct `RECIPE_TAG` values so W&B run names, W&B tags,
and output directories do not collide. The math-heavy recipe uses
`RECIPE_TAG=opencode_openmath_60k_c40m60`. The follow-ups use
`RECIPE_TAG=opencode_openmath_60k_c40m60_gradclip5` and
`RECIPE_TAG=opencode_openmath_60k_c40m60_decodealigned8`.

## 3. Eval Sweep (manual, after training)

The launcher does not auto-submit evals. Once the four training jobs have
written the checkpoints you care about (or finished), submit one eval sweep
per run. The sweep submits Math500, GSM8K, HumanEval+, and MBPP+ via
`scripts/submit_eval.py`.

```bash
# vanilla
MODEL_ROOT=output_models/opencode_openmath_60k_1p5B/vanilla_nopack2048_bs2x16x2_ep3_lr5e-6 \
USE_CARRY=0 \
sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch

# bptt nocarry
MODEL_ROOT=output_models/opencode_openmath_60k_1p5B/bptt_nocarry_puma_nopack2048_bs2x16x2_ep3_lr5e-6 \
USE_CARRY=0 \
sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch

# bptt loopguard, stop-grad
MODEL_ROOT=output_models/opencode_openmath_60k_1p5B/bptt_loopguard_stopgrad_puma_nopack2048_bs2x16x2_ep3_lr5e-6 \
USE_CARRY=1 \
sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch

# bptt loopguard, full BPTT
MODEL_ROOT=output_models/opencode_openmath_60k_1p5B/bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6 \
USE_CARRY=1 \
sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch
```

Override `CHECKPOINTS=200,400,600,...` (comma-separated step numbers, plus
optional `final`) to target a specific subset. The default is
`300,600,900,1200,1800,final`. With the new `SAVE_STEPS=200` cadence, useful
explicit lists include
`200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,final`.

For vanilla and nocarry runs use `USE_CARRY=0`; for loopguard runs use
`USE_CARRY=1`.

Manual eval for the math-heavy `c40m60` family:

```bash
# vanilla
MODEL_ROOT=output_models/opencode_openmath_60k_c40m60_1p5B/vanilla_nopack2048_bs2x16x2_ep3_lr5e-6 \
USE_CARRY=0 \
CHECKPOINTS=200,400,600,800,final \
sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch

# bptt nocarry
MODEL_ROOT=output_models/opencode_openmath_60k_c40m60_1p5B/bptt_nocarry_puma_nopack2048_bs2x16x2_ep3_lr5e-6 \
USE_CARRY=0 \
CHECKPOINTS=200,400,600,800,final \
sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch

# bptt loopguard, stop-grad
MODEL_ROOT=output_models/opencode_openmath_60k_c40m60_1p5B/bptt_loopguard_stopgrad_puma_nopack2048_bs2x16x2_ep3_lr5e-6 \
USE_CARRY=1 \
CHECKPOINTS=200,400,600,800,final \
sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch

# bptt loopguard, full BPTT
MODEL_ROOT=output_models/opencode_openmath_60k_c40m60_1p5B/bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6 \
USE_CARRY=1 \
CHECKPOINTS=200,400,600,800,final \
sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch
```
