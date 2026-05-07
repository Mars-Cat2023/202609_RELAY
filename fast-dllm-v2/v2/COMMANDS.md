# Commands

Submit everything from `Fast-dLLM/v2/`. All sbatch scripts read knobs from
the environment, so the same script handles every combination via `--export`.

Effective batch size below = `PER_DEVICE_TRAIN_BATCH_SIZE × num_GPUs × GRADIENT_ACCUMULATION_STEPS`,
where `num_GPUs = 2` from `--gres=gpu:2` in the sbatch.

---

## Train

Script: `train_scripts/finetune_alpaca_bptt_puma.sbatch`
Knobs: `BPTT_USE_CAB=0` ⇒ position-guarded LayerNorm loopholing,
       `BPTT_USE_CAB=1` ⇒ Cross Attention Bridge.

### 7B (effective BS = 64: per_device=2, accum=16)

**Loopguard:**

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,BPTT_USE_CAB=0,OUTPUT_DIR=output_models/finetune_fast_dLLM_7B_bptt_puma_loopguard_v2,SLURM_JOB_NAME=fastdllm_v2_7B_bptt_loopguard_v2 \
  --job-name=fastdllm_v2_7B_bptt_loopguard_v2 \
  train_scripts/finetune_alpaca_bptt_puma.sbatch
```

**CAB:**

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,BPTT_USE_CAB=1,OUTPUT_DIR=output_models/finetune_fast_dLLM_7B_bptt_puma_cab_v2,SLURM_JOB_NAME=fastdllm_v2_7B_bptt_cab_v2 \
  --job-name=fastdllm_v2_7B_bptt_cab_v2 \
  train_scripts/finetune_alpaca_bptt_puma.sbatch
```

### 1.5B (effective BS = 128: per_device=8, accum=8, lr=5e-5)

**Loopguard:**

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_NAME_OR_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B,BPTT_USE_CAB=0,PER_DEVICE_TRAIN_BATCH_SIZE=8,GRADIENT_ACCUMULATION_STEPS=8,LEARNING_RATE=5e-5,OUTPUT_DIR=output_models/finetune_fast_dLLM_1.5B_bptt_puma_loopguard_v2,SLURM_JOB_NAME=fastdllm_v2_1.5B_bptt_loopguard_v2 \
  --job-name=fastdllm_v2_1.5B_bptt_loopguard_v2 \
  train_scripts/finetune_alpaca_bptt_puma.sbatch
```

**CAB:**

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_NAME_OR_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B,BPTT_USE_CAB=1,PER_DEVICE_TRAIN_BATCH_SIZE=8,GRADIENT_ACCUMULATION_STEPS=8,LEARNING_RATE=5e-5,OUTPUT_DIR=output_models/finetune_fast_dLLM_1.5B_bptt_puma_cab_v2,SLURM_JOB_NAME=fastdllm_v2_1.5B_bptt_cab_v2 \
  --job-name=fastdllm_v2_1.5B_bptt_cab_v2 \
  train_scripts/finetune_alpaca_bptt_puma.sbatch
```

### Vanilla (no BPTT, no carry) — baseline

Script: `train_scripts/finetune_alpaca.sbatch`. Override the model and
output dir for the 1.5B variant; otherwise just submit as-is for 7B.

```bash
sbatch --reservation=dhruvesh train_scripts/finetune_alpaca.sbatch                # 7B
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_NAME_OR_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B,OUTPUT_DIR=output_models/finetune_fast_dLLM_1.5B_vanilla,SLURM_JOB_NAME=fastdllm_v2_1.5B_vanilla \
  --job-name=fastdllm_v2_1.5B_vanilla \
  train_scripts/finetune_alpaca.sbatch                                            # 1.5B
```

---

## Eval

Script: `train_scripts/eval_alpaca.sbatch`. Pass `MODEL_PATH` to point at the
checkpoint dir, and `USE_CARRY=1` for any BPTT-trained checkpoint so eval
runs the 2-step forward with hidden-state carry (matches training).

### Hugging Face base weights (no Alpaca finetune)

`eval_alpaca.sbatch` requires a **local** directory with `config.json` (it copies
vendored `modeling.py` / `configuration.py` into that dir). Use hub IDs only to
**download** once, then set `MODEL_PATH` to the snapshot path below.

**One-time download** (from `Fast-dLLM/v2/`):

```bash
huggingface-cli download Efficient-Large-Model/Fast_dLLM_v2_7B \
  --local-dir output_models/hf_baseline_Fast_dLLM_v2_7B
huggingface-cli download Efficient-Large-Model/Fast_dLLM_v2_1.5B \
  --local-dir output_models/hf_baseline_Fast_dLLM_v2_1.5B
```

**7B** — same default task bundle as other evals (`mmlu`, `gpqa`, `gsm8k`, etc.):

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_PATH=output_models/hf_baseline_Fast_dLLM_v2_7B,SLURM_JOB_NAME=eval_baseline_7B_hf \
  --job-name=eval_baseline_7B_hf \
  train_scripts/eval_alpaca.sbatch
```

**1.5B:**

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_PATH=output_models/hf_baseline_Fast_dLLM_v2_1.5B,SLURM_JOB_NAME=eval_baseline_1.5B_hf \
  --job-name=eval_baseline_1.5B_hf \
  train_scripts/eval_alpaca.sbatch
```

**MMLU only** (faster):

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_PATH=output_models/hf_baseline_Fast_dLLM_v2_7B,SLURM_JOB_NAME=eval_baseline_7B_mmlu,TASKS=mmlu \
  --job-name=eval_baseline_7B_mmlu \
  train_scripts/eval_alpaca.sbatch
```

### Vanilla checkpoint (no carry)

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_PATH=output_models/finetune_fast_dLLM_7B/checkpoint-XXXX,SLURM_JOB_NAME=eval_vanilla_7B \
  --job-name=eval_vanilla_7B \
  train_scripts/eval_alpaca.sbatch
```

### BPTT checkpoint (carry on)

7B Loopguard:

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_PATH=output_models/finetune_fast_dLLM_7B_bptt_puma_loopguard_v2/checkpoint-XXXX,USE_CARRY=1,SLURM_JOB_NAME=eval_7B_bptt_loopguard_v2 \
  --job-name=eval_7B_bptt_loopguard_v2 \
  train_scripts/eval_alpaca.sbatch
```

7B CAB:

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_PATH=output_models/finetune_fast_dLLM_7B_bptt_puma_cab_v2/checkpoint-XXXX,USE_CARRY=1,SLURM_JOB_NAME=eval_7B_bptt_cab_v2 \
  --job-name=eval_7B_bptt_cab_v2 \
  train_scripts/eval_alpaca.sbatch
```

1.5B Loopguard:

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_PATH=output_models/finetune_fast_dLLM_1.5B_bptt_puma_loopguard_v2/checkpoint-XXXX,USE_CARRY=1,SLURM_JOB_NAME=eval_1.5B_bptt_loopguard_v2 \
  --job-name=eval_1.5B_bptt_loopguard_v2 \
  train_scripts/eval_alpaca.sbatch
```

1.5B CAB:

```bash
sbatch --reservation=dhruvesh \
  --export=ALL,MODEL_PATH=output_models/finetune_fast_dLLM_1.5B_bptt_puma_cab_v2/checkpoint-XXXX,USE_CARRY=1,SLURM_JOB_NAME=eval_1.5B_bptt_cab_v2 \
  --job-name=eval_1.5B_bptt_cab_v2 \
  train_scripts/eval_alpaca.sbatch
```

---

## Common knob overrides

All read from the env via `--export=ALL,KEY=VAL,...`.

### Train

| env var                          | default (sbatch)              | notes                                                    |
| -------------------------------- | ----------------------------- | -------------------------------------------------------- |
| `MODEL_NAME_OR_PATH`             | `…Fast_dLLM_v2_7B`            | swap to `…1.5B` for the small model                      |
| `BPTT_USE_CAB`                   | `0` (loopguard)               | `1` ⇒ Cross Attention Bridge                             |
| `BPTT_USE_STREAMING_BUFFER`      | `1` (PUMA)                    | `0` ⇒ stateless 2-step BPTT                              |
| `BPTT_THRESHOLD` / `BPTT_TOP_P`  | `0.95` / `0.95`               | lower threshold ⇒ more reveals/step                      |
| `PER_DEVICE_TRAIN_BATCH_SIZE`    | `2`                           | 7B: 2; 1.5B: 8 fits comfortably                          |
| `GRADIENT_ACCUMULATION_STEPS`    | `16`                          | tune to keep effective BS in {64, 128}                   |
| `NUM_TRAIN_EPOCHS`               | `3`                           | 1 epoch is too short at the new batch size               |
| `LEARNING_RATE`                  | `2e-5`                        | bump to `5e-5` at BS=128                                 |
| `LR_SCHEDULER_TYPE`              | `cosine`                      |                                                          |
| `WARMUP_RATIO`                   | `0.1`                         | carry params start at 0; needs slow ramp                 |
| `MAX_GRAD_NORM`                  | `1.0`                         |                                                          |
| `USE_FP32`                       | `0` (bf16)                    | `1` ⇒ full precision (control experiment)                |
| `SAVE_STEPS` / `SAVE_TOTAL_LIMIT`| `99999` / `3`                 | default = end-only; lower `SAVE_STEPS` for mid-train ckpts |
| `OUTPUT_DIR`                     | `…/finetune_fast_dLLM_7B_bptt_puma` | always override per experiment to avoid resume collisions |

### Eval

| env var                          | default                       | notes                                                    |
| -------------------------------- | ----------------------------- | -------------------------------------------------------- |
| `MODEL_PATH`                     | (required)                    | `checkpoint-XXXX` or local Hub snapshot (e.g. `output_models/hf_baseline_…`) |
| `USE_CARRY`                      | `0`                           | `1` ⇒ 2-step forward with hidden-state carry (BPTT ckpts)|
| `TASKS`                          | `mmlu gpqa_main_n_shot gsm8k minerva_math ifeval` |                                       |
| `THRESHOLD`                      | `0.9`                         | unmasking threshold for generative tasks                 |

---

## What to watch in W&B

W&B run names encode `{cab,loopguard}_puma_{bf16,fp32}_bs{P}x{A}_ep{E}_v2`
and tags include `bptt`, `{cab,loopguard}`, `puma`, `bigbatch`, `carryshift`, `v2`.

Critical metrics (logged every step):

- `bptt/carry_grad_norm` — must be > 0; flat-zero ⇒ carry not in autograd
- `bptt/carry_to_base_grad_ratio` — should grow modestly off 0
- `bptt/cab_gamma_norm` (CAB) or `bptt/loop_ln_weight_norm` (loopguard) — gate opening?
- `bptt/L1` vs `bptt/L2` — L2 should drop below L1 once carry helps
- `bptt/h_t_inject_norm` — raw injection magnitude
- `bptt/buf_mask_ratio` — PUMA buffer occupancy
