# MATH-500 — Fast-dLLM v2 1.5B (NuminaMath)

**Scope.** NuminaMath SFT, then MATH-500 evaluation through `lm_eval`. Typical progression: (1) off-the-shelf baseline → (2) vanilla NuminaMath SFT → (3) BPTT + PUMA, carry off → (4) BPTT + PUMA + Loopguard or MLP carry → (5) optional CAB. Carry checkpoints must be evaluated with `USE_CARRY=1`.

**Training defaults.**

| | |
|--|--|
| Micro-batch × accum × GPUs | 2×16×2 → 64 examples / step |
| Vanilla SFT data | `data/numina/train_conversation` |
| Vanilla SFT block size | `BLOCK_SIZE=512`, packed chunks by default (`DISABLE_GROUP_TEXTS=0`) |
| BPTT defaults | `BPTT_THRESHOLD=0.85`, `BPTT_TOP_P=0.95`, `BPTT_TEMPERATURE=0.0`, PUMA buffer on |
| Checkpoints | Use `SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0` to keep every `checkpoint-*` |
| Evaluation | `THRESHOLD=0.85`, `APPLY_CHAT_TEMPLATE=1`, `NUM_FEWSHOT=0` |

**Why this ladder.** Vanilla SFT tests whether NuminaMath helps MATH-500 at all. BPTT + PUMA with `CARRY_MODE=none` isolates the two-step loss and streaming buffer without architectural changes. Loopguard / MLP / CAB then test whether feeding `h_s` into the next forward helps beyond that.

---

## 1. First-Time Setup

From `Fast-dLLM/v2`:

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate venv_fastdllm

# Convert HF NuminaMath -> data/numina/{train,test}_conversation/*.json
# (50k train / 2k test).
python scripts/prep_numina.py \
  --hf_dataset jacopo-minniti/NuminaMath-CoT-LLaDA-Goldilocks \
  --train_size 50000 --test_size 2000 \
  --out_dir data/numina

# Cache MATH-500 once so the first eval does not pay the download cost
# inside an sbatch slot.
python -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/MATH-500')"
```

Expected:

```text
data/numina/train_conversation/train_50000.json
data/numina/test_conversation/test_2000.json
```

---

## 2. Training

Submit from `Fast-dLLM/v2` (`cwd` = repo root). `SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0` saves every checkpoint. The sequence below mirrors the ablation order: vanilla → nocarry → loopguard → mlp → cab. Each BPTT job starts from `MODEL_NAME_OR_PATH` unless you explicitly point it at a prior output directory.

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate lmflow

# 1) Vanilla SFT
SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_numina_vanilla_1p5B \
  train_scripts/finetune_numina.sbatch

# 2) BPTT + PUMA, no carry
CARRY_MODE=none BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_numina_bptt_nocarry_1p5B \
  train_scripts/finetune_numina_bptt.sbatch

# 3) BPTT + PUMA + Loopguard
CARRY_MODE=loopguard BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_numina_bptt_loopguard_1p5B \
  train_scripts/finetune_numina_bptt.sbatch

# 4) BPTT + PUMA + MLP carry
CARRY_MODE=mlp BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_numina_bptt_mlp_1p5B \
  train_scripts/finetune_numina_bptt.sbatch

# 5) BPTT + PUMA + CAB
CARRY_MODE=cab BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_numina_bptt_cab_1p5B \
  train_scripts/finetune_numina_bptt.sbatch
```

Default output directories:

| Run | Default output dir |
|--|--|
| vanilla | `output_models/numina_1p5B/vanilla_bs2x16_ep2` |
| nocarry | `output_models/numina_1p5B/bptt_nocarry_puma_bs2x16_ep2` |
| loopguard | `output_models/numina_1p5B/bptt_loopguard_puma_bs2x16_ep2` |
| mlp | `output_models/numina_1p5B/bptt_mlp_puma_bs2x16_ep2` |
| cab | `output_models/numina_1p5B/bptt_cab_puma_bs2x16_ep2` |

Common overrides: `MODEL_NAME_OR_PATH`, `DATASET_PATH`, `OUTPUT_DIR`, `RUN_NAME`, `WANDB_TAGS`, `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, `BLOCK_SIZE`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, `BPTT_THRESHOLD`, `BPTT_TOP_P`, `BPTT_TEMPERATURE`, `CARRY_MODE`.

BPTT W&B tags include the actual loss parameters: `carry_mode=...` (`loopholing` for `CARRY_MODE=loopguard`), `threshold=...`, `top_p=...`, and `temperature=...` in addition to the dataset/model/batch tags.

---

## 3. Evaluation

`train_scripts/eval_math500.sbatch` runs the custom `scripts/lm_eval_tasks/math500` task with chat-template evaluation (`APPLY_CHAT_TEMPLATE=1`, `NUM_FEWSHOT=0`) and `THRESHOLD=0.85` by default.

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate venv_fastdllm

# 0) Off-the-shelf 1.5B baseline
MODEL_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B \
  RUN_TAG=baseline_1p5B \
  THRESHOLD=0.85 \
  sbatch --reservation dhruvesh --job-name=eval_math500_baseline_1p5B \
  train_scripts/eval_math500.sbatch

# 1) Vanilla SFT
MODEL_PATH=output_models/numina_1p5B/vanilla_bs2x16_ep2 \
  RUN_TAG=vanilla_1p5B_numina \
  THRESHOLD=0.85 \
  sbatch --reservation dhruvesh --job-name=eval_math500_vanilla_1p5B_numina \
  train_scripts/eval_math500.sbatch

# 2) BPTT + PUMA, no carry
MODEL_PATH=output_models/numina_1p5B/bptt_nocarry_puma_bs2x16_ep2 \
  RUN_TAG=bptt_nocarry_1p5B_numina \
  THRESHOLD=0.85 \
  sbatch --reservation dhruvesh --job-name=eval_math500_bptt_nocarry_1p5B_numina \
  train_scripts/eval_math500.sbatch

# 3) BPTT + PUMA + Loopguard
MODEL_PATH=output_models/numina_1p5B/bptt_loopguard_puma_bs2x16_ep2 \
  RUN_TAG=bptt_loopguard_1p5B_numina USE_CARRY=1 \
  THRESHOLD=0.85 \
  sbatch --reservation dhruvesh --job-name=eval_math500_bptt_loopguard_1p5B_numina \
  train_scripts/eval_math500.sbatch

# 4) BPTT + PUMA + MLP carry
MODEL_PATH=output_models/numina_1p5B/bptt_mlp_puma_bs2x16_ep2 \
  RUN_TAG=bptt_mlp_1p5B_numina USE_CARRY=1 \
  THRESHOLD=0.85 \
  sbatch --reservation dhruvesh --job-name=eval_math500_bptt_mlp_1p5B_numina \
  train_scripts/eval_math500.sbatch

# 5) BPTT + PUMA + CAB
MODEL_PATH=output_models/numina_1p5B/bptt_cab_puma_bs2x16_ep2 \
  RUN_TAG=bptt_cab_1p5B_numina USE_CARRY=1 \
  THRESHOLD=0.85 \
  sbatch --reservation dhruvesh --job-name=eval_math500_bptt_cab_1p5B_numina \
  train_scripts/eval_math500.sbatch
```

`MODEL_PATH` can point at the top-level output directory or a specific `checkpoint-*`. Use the top-level output directory for the final saved model, and a `checkpoint-*` path when comparing intermediate snapshots.

---

## 4. Comparing Results

After the eval jobs finish, JSON results land under:

```text
eval_results/math500__<RUN_TAG>__job<jobid>/math500/results_*.json
```

Primary comparisons:

| Comparison | What it tests |
|--|--|
| baseline vs vanilla | Does NuminaMath SFT improve MATH-500? |
| vanilla vs nocarry | Does 2-step BPTT + PUMA help without an architectural carry? |
| nocarry vs loopguard / mlp / cab | Does the carry mechanism add signal beyond BPTT + PUMA? |
| loopguard vs mlp vs cab | Which carry architecture is best at fixed data/loss/eval settings? |

The custom MATH-500 task asks the model to put the final answer in `\boxed{...}`, extracts the last boxed answer, and feeds it into the Minerva-style equivalence utilities. `math_verify` remains available as a second, more permissive metric.

---

## 5. Optional: 7B

Re-run any training or evaluation command with `MODEL_NAME_OR_PATH=Efficient-Large-Model/Fast_dLLM_v2_7B`. Output paths, run names, and tags derive a `7B` size tag automatically.

```bash
MODEL_NAME_OR_PATH=Efficient-Large-Model/Fast_dLLM_v2_7B \
  CARRY_MODE=cab BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_numina_bptt_cab_7B \
  train_scripts/finetune_numina_bptt.sbatch
```

```bash
MODEL_PATH=output_models/numina_7B/bptt_cab_puma_bs2x16_ep2 \
  RUN_TAG=bptt_cab_7B_numina USE_CARRY=1 THRESHOLD=0.85 \
  sbatch --reservation dhruvesh --job-name=eval_math500_bptt_cab_7B_numina \
  train_scripts/eval_math500.sbatch
```

---

## File Reference

* `scripts/prep_numina.py` — NuminaMath HF data → Fast-dLLM conversation JSON shards.
* `train_scripts/finetune_numina.sbatch` — vanilla NuminaMath SFT.
* `train_scripts/finetune_numina_bptt.sbatch` — BPTT + PUMA (`CARRY_MODE=none` | `loopguard` | `mlp` | `cab`).
* `train_scripts/eval_math500.sbatch` — single-task MATH-500 eval with `USE_CARRY` for carry checkpoints.
* `scripts/lm_eval_tasks/math500/{math500.yaml,utils.py}` — custom 0-shot chat MATH-500 task and boxed-answer parser.
* `src/lmflow/pipeline/utils/block_bptt_loss.py` — 2-step BPTT loss, PUMA buffer use, and carry preparation.
* `src/lmflow/pipeline/utils/bptt_trainer.py` — BPTT trainer wrapper and carry/base gradient logging.
