# HumanEval / MBPP — Fast-dLLM v2 1.5B (EvalPlus)

**Scope.** [Magicoder-OSS](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) SFT with **no sequence packing**, then **EvalPlus** pass@1. Typical progression: (1) **vanilla** SFT → (2) **BPTT + PUMA, carry off** → (3) **BPTT + PUMA + Loopguard** (plain LN on `h_t`) or **+ MLP carry** (bottleneck MLP + LN, before CAB) → (optional) **CAB**. After carry stages, run **Evaluation** with the matching `MODEL_PATH` and **`--use_carry`** when the checkpoint has carry (`loopguard`, `mlp`, or `cab`).

**Paper (§A.4, [arXiv:2509.26328](https://arxiv.org/pdf/2509.26328)).** Use **EvalPlus** for these code tasks, not lm-eval-harness. Non-code benchmarks use `eval.py` + `lm_eval` (`eval_script.sh`, `train_scripts/eval_lm_eval.sbatch`).

**Inference.** Block size **32**, sub-block **8**, threshold **0.85**; script defaults for `--bd_size` / `--small_block_size` are already 32 / 8. `--use_block_cache` is validated for vanilla / nocarry only; carry checkpoints (`loopguard`, `mlp`, `cab`) should run without block cache until the carry-aware cache path is implemented.

**Magicoder training defaults** (`train_scripts/finetune_magicoder_oss.sbatch`):

| | |
|--|--|
| Micro-batch × accum × GPUs | 2×16×2 → 64 ex / step |
| `disable_group_texts` | **1** (one conversation per row; avoid packing) |
| `block_size` | **1024**, `TRUNCATION_SIDE` **left** |
| `MAGICODER_SPLIT` | **`python`**: ~38k rows, default **2** epochs → `output_models/magicoder_oss_python_1p5B/.../vanilla_nopack1024_bs2x16_ep2_lr1e-5` |
| | **`full`**: ~75k rows, default **1** epoch → `.../magicoder_oss_full_1p5B/.../vanilla_nopack1024_bs2x16_ep1_lr1e-5` |
| `LEARNING_RATE` | **1e-5** (try 2e-5 if loss flat; **5e-6** if unstable) |

**BPTT** (`train_scripts/finetune_magicoder_oss_bptt.sbatch`, same `MAGICODER_SPLIT` / packing / `block_size=1024` as vanilla; `CARRY_MODE=none` | `loopguard` | `mlp` | `cab`; default `BPTT_THRESHOLD=0.85`, `BPTT_TOP_P=0.95`, `BPTT_TEMPERATURE=0.0`):

| Stage | `CARRY_MODE` | Default `output_models/...` ( `MAGICODER_SPLIT=python`, 2 ep, lr **1e-5** ) |
|--------|----------------|-----------------------------------|
| BPTT + PUMA, no carry | `none` | `magicoder_oss_python_1p5B/bptt_nocarry_puma_nopack1024_bs2x16_ep2_lr1e-5` |
| BPTT + PUMA + Loopguard | `loopguard` | `.../bptt_loopguard_puma_nopack1024_bs2x16_ep2_lr1e-5` |
| BPTT + PUMA + MLP carry | `mlp` | `.../bptt_mlp_puma_nopack1024_bs2x16_ep2_lr1e-5` |
| BPTT + PUMA + CAB | `cab` | `.../bptt_cab_puma_nopack1024_bs2x16_ep2_lr1e-5` |

Python-only data usually matches the Python evals; use **`full`** for maximum diversity. Override epochs, LR, `DATASET_PATH`, `OUTPUT_DIR`, etc. via the same env vars as other `finetune_*.sbatch` drivers. The Magicoder BPTT script accepts the same `BPTT_*`, `SAVE_STEPS`, `DEEPSPEED_CONFIG`, and `USE_FP32` environment overrides as `train_scripts/finetune_numina_bptt.sbatch`.

**Eval flow.** `scripts/generate_evalplus_jsonl.py` (diffusion `mdm_sample`, EvalPlus `make_raw_chat_prompt`–style chat) → `evalplus.evaluate --samples`. The scoring step only needs `pip install evalplus` (use `evalplus` without `[vllm]`). `generate_evalplus_jsonl.py` uses a fixed `max_new_tokens` window; `evalplus.sanitize` cleans before tests.

---

## 1. First-time setup

From `Fast-dLLM/v2` (adjust conda env names to yours):

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2

# Eval / generation: same env as eval.py (PyTorch, Transformers, repo generation_functions)
conda activate venv_fastdllm
pip install 'evalplus'   # not: evalplus[vllm]

# Optional: prefetch EvalPlus task payloads
python -c "from evalplus.data import get_human_eval_plus, get_mbpp_plus; get_human_eval_plus(); get_mbpp_plus('full'); print('ok')"

# Training data (HF Magicoder-OSS)
conda activate lmflow
python scripts/prep_magicoder_oss.py --out_dir data/magicoder_oss_python --lang python
# optional: python scripts/prep_magicoder_oss.py --out_dir data/magicoder_oss
```

Expected: `data/magicoder_oss_python/train_conversation/train_<N>.json` (and optionally `data/magicoder_oss/...`).

---

## 2. Training

Submit from `Fast-dLLM/v2` (`cwd` = repo root). `SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0` saves every `checkpoint-*` (no rolling deletion). **Order:** vanilla SFT → BPTT nocarry → loopguard → mlp → cab. Wait for each job to finish (or its intended checkpoint) before the next if you depend on a clean progression; the commands below are independent re-starts from the same base model unless you wire `MODEL_NAME_OR_PATH` to a previous `output_dir`.

**Python-only** (`MAGICODER_SPLIT=python`, ~38k rows, default **2** epochs, `ep2` in default paths: `output_models/magicoder_oss_python_1p5B/...`):

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate lmflow

# 1) Vanilla SFT
MAGICODER_SPLIT=python SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_oss_python_ep2_1p5B \
  train_scripts/finetune_magicoder_oss.sbatch

# 2) BPTT + PUMA, no carry
MAGICODER_SPLIT=python CARRY_MODE=none BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_nocarry_ep2_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch

# 3) BPTT + PUMA + Loopguard
MAGICODER_SPLIT=python CARRY_MODE=loopguard BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_loopguard_ep2_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch

# 4) BPTT + PUMA + MLP carry
MAGICODER_SPLIT=python CARRY_MODE=mlp BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_mlp_ep2_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch

# 5) BPTT + PUMA + CAB
MAGICODER_SPLIT=python CARRY_MODE=cab BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_cab_ep2_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch
```

**Full Magicoder** (`MAGICODER_SPLIT=full`, ~75k rows, default **1** epoch, `ep1` in default paths: `output_models/magicoder_oss_full_1p5B/...`):

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate lmflow

# 1) Vanilla SFT
MAGICODER_SPLIT=full SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_oss_full_ep1_1p5B \
  train_scripts/finetune_magicoder_oss.sbatch

# 2) BPTT + PUMA, no carry
MAGICODER_SPLIT=full CARRY_MODE=none BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_nocarry_ep1_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch

# 3) BPTT + PUMA + Loopguard
MAGICODER_SPLIT=full CARRY_MODE=loopguard BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_loopguard_ep1_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch

# 4) BPTT + PUMA + MLP carry
MAGICODER_SPLIT=full CARRY_MODE=mlp BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_mlp_ep1_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch

# 5) BPTT + PUMA + CAB
MAGICODER_SPLIT=full CARRY_MODE=cab BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_cab_ep1_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch
```

Default output dirs (unless `OUTPUT_DIR` / `RUN_NAME` override) are in the table under **BPTT** in the intro; for **`full`**, use `magicoder_oss_full_1p5B` and `..._ep1_...` instead of `python` / `ep2`. Common overrides: `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, `BLOCK_SIZE`, `OUTPUT_DIR`, `RUN_NAME`, `WANDB_TAGS`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, `DISABLE_GROUP_TEXTS` (keep **1** unless you want packed chunks), `DATASET_PATH`, `BPTT_THRESHOLD`, `BPTT_USE_STREAMING_BUFFER`, `BPTT_TOP_P`, `BPTT_TEMPERATURE`, `CARRY_MODE`. BPTT W&B tags include the actual loss parameters: `carry_mode=...` (`loopholing` for `CARRY_MODE=loopguard`), `threshold=...`, `top_p=...`, and `temperature=...`.

---

## 3. Optional On-Save EvalPlus

Set `EVALPLUS_ON_SAVE=1` on a Magicoder training job to submit a separate one-GPU Slurm EvalPlus job after each saved `checkpoint-*`. Default behavior is off. The callback writes `.evalplus_submitted` in each checkpoint for idempotency, and the eval job writes JSONL/results under `evalplus_results/on_save/<run_name>/<checkpoint-N>/`.

Default on-save settings are `EVALPLUS_DATASETS="humaneval mbpp"`, `EVALPLUS_SEEDS="0"`, `EVALPLUS_THRESHOLD=0.85`, and `EVALPLUS_BLOCK_CACHE=auto`. In `auto`, block cache is used for vanilla / nocarry and forced off for carry checkpoints.

```bash
MAGICODER_SPLIT=python EVALPLUS_ON_SAVE=1 EVALPLUS_SEEDS="0" SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_oss_python_ep2_1p5B \
  train_scripts/finetune_magicoder_oss.sbatch
```

For carry runs:

```bash
MAGICODER_SPLIT=python CARRY_MODE=loopguard EVALPLUS_ON_SAVE=1 EVALPLUS_SEEDS="0" \
  BPTT_THRESHOLD=0.85 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=0 \
  sbatch --reservation dhruvesh --job-name=ft_magicoder_bptt_loopguard_ep2_1p5B \
  train_scripts/finetune_magicoder_oss_bptt.sbatch
```

If Slurm submission is unavailable from compute nodes, the callback logs a warning and writes a pending manifest under the training output directory. Use `EVALPLUS_ON_SAVE_STRICT=1` only when failed eval submission should fail training.

---

## 4. Evaluation (EvalPlus)

Point `--model_path` at a Hub id, a training `output_dir`, or a `checkpoint-*` subdir. Trainer checkpoints often lack local `configuration.py` / `modeling.py`; `generate_evalplus_jsonl.py` defaults to the in-tree `src/lmflow/models/fast_dllm` implementation and builds from the checkpoint `config.json`. Use **`--code_model_path`** only to override that source.

**Randomness.** Only **generation** (`generate_evalplus_jsonl.py`) is stochastic: it honors **`--seed`** (torch / cuDNN and the diffusion sampling path). **`evalplus.evaluate`** re-executes tests on a fixed JSONL and is **deterministic** for a given file — it does not resample from the model. For multi-seed reports, vary **`--seed`** on generation; one `evalplus.evaluate` per JSONL is enough.

On a cluster, run generation on GPU (wrap in `#SBATCH` like other `train_scripts/` evals); `evalplus.evaluate` is CPU-only.

### 4a. Off-the-shelf 1.5B

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate venv_fastdllm

python scripts/generate_evalplus_jsonl.py \
  --model_path Efficient-Large-Model/Fast_dLLM_v2_1.5B \
  --dataset humaneval \
  --threshold 0.85 \
  --output_jsonl evalplus_results/baseline_1p5B_humaneval.jsonl
evalplus.evaluate --dataset humaneval --samples evalplus_results/baseline_1p5B_humaneval.jsonl

python scripts/generate_evalplus_jsonl.py \
  --model_path Efficient-Large-Model/Fast_dLLM_v2_1.5B \
  --dataset mbpp \
  --threshold 0.85 \
  --output_jsonl evalplus_results/baseline_1p5B_mbpp.jsonl
evalplus.evaluate --dataset mbpp --samples evalplus_results/baseline_1p5B_mbpp.jsonl
```

### 4b. Sweep: python vs full, SFT + BPTT (nocarry, loopguard, mlp), all checkpoints, 3 seeds

After the training runs in §2 (default `output_models/...` names; **CAB excluded** in this sweep), a single script walks every requested **`checkpoint-*`**, both Magicoder splits, run types **sft** / **nocarry** / **loopguard** / **mlp**, **HumanEval and MBPP**, and seeds **`0` `1` `2`**. It writes JSONL to `evalplus_results/magicoder_sweep/`, then runs `evalplus.evaluate` each time. **Loopguard** and **mlp** use **`--use_carry`**; **SFT** and **nocarry** do not. Default **`THRESHOLD=0.85`**.

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate venv_fastdllm
chmod +x scripts/sweep_evalplus_magicoder.sh
THRESHOLD=0.85 ./scripts/sweep_evalplus_magicoder.sh
```

Useful env vars (see script header): `SEEDS`, `DATASETS` (e.g. `humaneval` only), `THRESHOLD`, `DRY_RUN=1`, `SKIP_DONE=0`, `FORCE=1`.

### 4c. Full Magicoder Slurm Sweep With Training Dependencies

Use `train_scripts/launch_magicoder_full_evalplus_sweeps.sbatch` to submit one EvalPlus GPU job per full Magicoder model family. It launches the off-the-shelf 1.5B model, the local HF baseline clone, and the saved vanilla SFT checkpoints immediately, then submits nocarry / loopguard / mlp jobs with `afterok` dependencies when job IDs are supplied or discoverable from `squeue`.

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2

# Dry-run first: prints the child sbatch commands.
DRY_RUN=1 sbatch train_scripts/launch_magicoder_full_evalplus_sweeps.sbatch

# If squeue cannot discover the active training jobs, pass Slurm job IDs explicitly.
NOCARRY_DEPENDENCY=<nocarry_slurm_job_id> \
LOOPGUARD_DEPENDENCY=<loopguard_slurm_job_id> \
MLP_DEPENDENCY=<mlp_slurm_job_id> \
  sbatch train_scripts/launch_magicoder_full_evalplus_sweeps.sbatch
```

Defaults: `EVALPLUS_DATASETS="humaneval mbpp"`, `EVALPLUS_SEEDS="0 1 2"`, `EVALPLUS_THRESHOLD=0.85`, and `EVALPLUS_BLOCK_CACHE=auto`. Outputs go under `evalplus_results/magicoder_full_sweep/<model_label>/<checkpoint>/`.

**Manual one-off** (see `python scripts/generate_evalplus_jsonl.py --help`): `--batch_size`, `--max_new_tokens`, `--threshold`, `--seed`, `--code_model_path`, **`--use_block_cache`** for vanilla / nocarry, and **`--use_carry`** for checkpoints whose `config.json` has a carry mode (`use_loopholing` / `use_mlp_carry` / `use_cab`). If both are passed, the script disables block cache for carry correctness.

### 4d. lm-eval harness + EvalPlus judge (`humaneval_plus_evalplus` / `mbpp_plus_evalplus`)

Same **`eval.py`** / **`accelerate launch`** stack as math500, but tasks live under **`lm_eval_tasks/`** (EvalPlus **`make_raw_chat_prompt`**, sanitize, **`check_correctness`**). **Do not** pass lm-eval **`--apply_chat_template`** for these tasks — prompting is applied inside the task.

Push-button Slurm submission (sets **`--include_path`**, **`PYTHONPATH`**, forces chat-template off, optional tokenizer):

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2

python scripts/submit_eval.py \
  --task humaneval_plus_evalplus \
  --model_path output_models/magicoder_oss_python_1p5B/bptt_nocarry_puma_nopack1024_bs2x16_ep2_lr1e-5/checkpoint-750 \
  --evalplus_tokenizer Efficient-Large-Model/Fast_dLLM_v2_1.5B \
  --dry_run
```

Use **`mbpp_plus_evalplus`** for MBPP+. Details: **`lm_eval_tasks/README.md`**.

---

## File reference

* `scripts/prep_magicoder_oss.py` — Magicoder → `data/magicoder_oss_python/...` or `data/magicoder_oss/...`
* `train_scripts/finetune_magicoder_oss.sbatch` — vanilla Magicoder SFT, `disable_group_texts=1`, `MAGICODER_SPLIT=python` or `full`
* `train_scripts/finetune_magicoder_oss_bptt.sbatch` — BPTT + PUMA (`CARRY_MODE=none` | `loopguard` | `mlp` | `cab`), same data/packing as vanilla Magicoder
* `scripts/generate_evalplus_jsonl.py` — JSONL for `evalplus.evaluate --samples`
* `scripts/run_evalplus_checkpoint.py` — one-checkpoint EvalPlus generation, scoring, result parsing, and optional W&B logging
* `train_scripts/evalplus_checkpoint.sbatch` — one-GPU Slurm wrapper used by on-save EvalPlus
* `train_scripts/evalplus_model_sweep.sbatch` — one-GPU Slurm worker that sweeps all checkpoints for one model family
* `train_scripts/launch_magicoder_full_evalplus_sweeps.sbatch` — submits the full Magicoder EvalPlus sweep jobs with optional training dependencies
* `src/lmflow/pipeline/utils/evalplus_on_save_callback.py` — rank-zero Trainer callback that submits on-save EvalPlus jobs
* `scripts/sweep_evalplus_magicoder.sh` — batch EvalPlus over Magicoder SFT + BPTT dirs, checkpoints, seeds
* `train_scripts/eval_lm_eval.sbatch` — non-code lm_eval tasks only; not for paper HumanEval / MBPP
* `eval.py` — `Fast_dLLM_v2EvalHarness` for non-code paths
* `scripts/submit_eval.py` — Slurm wrapper for **`eval.py`** (math500, **`humaneval_plus_evalplus`**, **`mbpp_plus_evalplus`**, …); EvalPlus lm-eval tasks auto-set **`PYTHONPATH`** and forbid **`--apply_chat_template`**
* `lm_eval_tasks/` — custom EvalPlus-aligned lm-eval tasks (see **`lm_eval_tasks/README.md`**)
