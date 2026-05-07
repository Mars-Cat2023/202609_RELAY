# Setup

```bash
git clone --recurse-submodules https://github.com/brozonoyer/double-backprop.git
```

To use the latest xlm-core main branch (recommended for compatibility):

```bash
git submodule update --remote xlm-core
```

Note: Since the project contains submodules, you need to make sure that all the pulls also update the submodules. To do this, instead of using the usual `git pull`, use `git pull --recurse-submodules`.

```bash
conda create -p .venv_bptt python=3.11.10 pip ipykernel -y
conda activate .venv_bptt
pip install -e xlm-core
pip install -e xlm-core/xlm-models
pip install -r requirements.txt
```
Create a `.env` file in the root directory from where you plan to run the experiments. Edit and add the following environment variables:

```bash
# wandb
WANDB_ENTITY=ilm-extensions
WANDB_PROJECT=BPTT
DATA_DIR=data
HF_HOME=hf_home
HF_DATASETS_CACHE=hf_datasets_cache
# output
LOG_DIR=logs
# misc
TOKENIZERS_PARALLELISM=false
PROJECT_ROOT=.
# hydra
HYDRA_FULL_ERROR=1
OC_CAUSE=1
# emails from SLURM scheduler if applicable
EMAIL=???
# torch compile logs
TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
```

# Download pretrained weights

```bash
pip install gdown
gdown --folder -O <output_directory> <gdrive_id>
```

# Eval

# Decode trajectory visualization (offline checkpoints)

Documentation, a Sudoku dump script, and a Jupyter notebook live under [`visualization/`](visualization/README.md) (`dump_decode_trajectories`, pairing, and step-by-step plots).

# Train

## Prepare data
```bash
xlm "job_type=prepare_data" "job_name=sudoku_extreme_prepare_data" "experiment=sudoku_extreme_mlm"
xlm "job_type=prepare_data" "job_name=lm1b_prepare_data" "experiment=lm1b_mlm"
```

## Debug runs
```bash
xlm "job_type=train" "job_name=sudoku_extreme_mlm" "experiment=sudoku_extreme_mlm" "debug=overfit"
```


## Submit full run to SLURM

**Note:**

1. Use the login node to submit the job, not the terminal of your editor (which may be running on a compute node).
2. Activate the virtual environment in the terminal from which you are submitting before you submit.

```bash
python lib/slurm_scripts/submit_train.py "do=submit" "job_name=sudoku_extreme_mlm" "train.experiment=sudoku_extreme_mlm" "train.batch_size=512" "train.compile=true" "train.precision=bf16-mixed" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" "++slurm.exclude=gpu016" "++use_job_name_as_id=false"

```

## OWT packed Loopholing + PUMA + BPTT (2 steps)

Same OWT + packing setup as `owt_packed_mlm` from the **`xlm-models`** package (MLM configs and `mlm.datamodule_mlm.*`), with double-backprop enabled via `experiment=owt_packed_loopholing_bptt_puma` (Loopholing + PUMA + BPTT, `num_steps=2`).

**Setup:** `pip install xlm-models` (or `pip install -e xlm-core/xlm-models` from a submodule checkout) so `import mlm` and bundled Hydra configs resolve without path hacks. Repo-root `xlm_models.json` only needs **`doublebackprop`** so this repo’s experiments and `setup_external_models()` pick up `doublebackprop/configs`.


Debug 2 gpu run:
```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=owt_packed_loopholing_bptt_puma_1_node_2_gpu_debug" \
"train.experiment=owt_packed_loopholing_bptt_puma" \
"train.batch_size=32" \
"train.compile=true" \
"train.precision=bf16-mixed" \
"hardware=ddp_1_node_2_gpu" \
"slurm.time=7-00:00:00" \
"use_job_name_as_id=false" \
"++slurm.reservation=dhruvesh" \
"---" \
"debug=multinode"
```

Real run:
```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=owt_packed_loopholing_bptt_puma_v2" \
"train.experiment=owt_packed_loopholing_bptt_puma" \
"train.batch_size=32" \
"train.compile=true" \
"train.precision=bf16-mixed" \
"hardware=ddp_1_node_8_gpu" \
"slurm.time=7-00:00:00" \
"use_job_name_as_id=false" \
"++slurm.mem=200G" \
"++slurm.reservation=dhruvesh"
```


python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=owt_packed_loopholing_bptt_puma_v4_tied_embeddings_fix_init" \
"train.experiment=owt_packed_loopholing_bptt_puma" \
"train.batch_size=32" \
"train.compile=true" \
"train.precision=bf16-mixed" \
"hardware=ddp_1_node_8_gpu" \
"slurm.time=7-00:00:00" \
"use_job_name_as_id=false" \
"++slurm.mem=200G" \
"++slurm.reservation=dhruvesh" \
"---" \
"++model.tie_embeddings=true" \
"++loggers.wandb.id=jpg9rsjm" \
"++resume_checkpoint_path=/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/logs/owt_packed_loopholing_bptt_puma_v4_tied_embeddings_fix_init/checkpoints/last.ckpt"



**Debug (local, batch size 2 for packing checks):** `debug=overfit` defaults to batch size 1; override it so each step sees 2 packed sequences. Use `compile=false` for faster startup.

**Remote debug (debugpy):** `.vscode/launch.json` defines **Debug XLM (attach)** on `localhost:5678` (and **5679** for a second attach). Run the command below, then start that attach configuration in the editor.

```bash
python -m debugpy --listen 5678 --wait-for-client -m xlm \
  job_type=train \
  job_name=owt_packed_loopholing_bptt_puma_pack_debug \
  experiment=owt_packed_loopholing_bptt_puma \
  debug=overfit \
  per_device_batch_size=2 \
  global_batch_size=2 \
  compile=false \
  num_dataloader_workers=0 \
  dataloader_prefetch_factor=null
```

**Manual cache / `DATA_DIR`:** `xlm` loads `.env` with `override=True`, so `DATA_DIR` in `.env` **replaces** shell `export DATA_DIR=...`. Default `DATA_DIR=data` expects prepared shards under `data/<hub_path>/` (e.g. `data/Skylion007/openwebtext/train`). If you previously ran `prepare_data` under `xlm-core/` with `DATA_DIR=data`, move that tree to repo-root `data/` (same relative paths) or point `DATA_DIR` at the directory that contains `Skylion007/`.

## GRAM n-queens & graph coloring (HF)

Reasoning datasets from [GRAM](https://openreview.net/pdf?id=Vxu6kcIjwV) (`brozonoyer/gram-n-queens`, `brozonoyer/gram-graph-coloring`). **Per-size dataset configs** (filters + `filter_suffix` for cache paths) live in **xlm-core** under `xlm-core/src/xlm/configs/lightning_train/datasets/`, for example:

- N-queens: `gram_n_queens_train_8x8`, `gram_n_queens_train_10x10`, matching `val` / `val_pred` / `test` variants.
- Graph coloring: `gram_graph_coloring_train_8v`, `gram_graph_coloring_train_10v`, matching `val` / `val_pred` / `test` variants.

Unsuffixed names such as `gram_n_queens_train` default to the **8×8** / **8-vertex** split for backward compatibility.

**Double-backprop experiments** mirror the Sudoku progression (MLM uniform → loopholing BPTT PUMA). For **8×8** n-queens use `block_size=64` and `vocab_size=3`; for **8-vertex** graph coloring use `block_size=36` and `vocab_size=6` (already set in the experiment YAMLs).

**Debug (local):**

```bash
xlm job_type=train experiment=gram_n_queens_8x8_mlm_uniform job_name=gram_nq_8x8_mlm debug=overfit
xlm job_type=train experiment=gram_n_queens_8x8_loopholing_bptt_puma job_name=gram_nq_8x8_loopholing debug=overfit

xlm job_type=train experiment=gram_graph_coloring_8v_mlm_uniform job_name=gram_gc_8v_mlm debug=overfit
xlm job_type=train experiment=gram_graph_coloring_8v_loopholing_bptt_puma job_name=gram_gc_8v_loopholing debug=overfit
```

**Submit to SLURM** (same pattern as Sudoku; activate your env and submit from the login node):

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=gram_n_queens_8x8_mlm_uniform" \
"train.experiment=gram_n_queens_8x8_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false"
```

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=gram_graph_coloring_8v_mlm_uniform" \
"train.experiment=gram_graph_coloring_8v_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false"
```

For **10×10** or **10-vertex**, add matching datamodules and experiments (or override `datamodule` + `block_size` / `predictor.max_steps` / tokenizer `vocab_size`) using the `*_10x10` / `*_10v` dataset configs from xlm-core.

## MLM uniform

**Debug command:**

```
xlm job_type=train experiment=sudoku_extreme_mlm_uniform job_name=mlm_uniform debug=overfit
```

**Full training command:**

```
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_mlm_uniform" \
"train.experiment=sudoku_extreme_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false"
```

## MLM from solver

**Debug command:**

```
xlm job_type=train experiment=sudoku_extreme_mlm_from_solver job_name=mlm_from_solver debug=overfit
```

**Full training command:**

```
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_mlm_from_solver" \
"train.experiment=sudoku_extreme_mlm_from_solver" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false"
```

## Loopholing BPTT (LDDM-style)

Uses `LoopholingBPTTLoss`: per-position loopholing where the model accepts `h_t` (B,L,d_model) and returns `(logits, h_s)`. Unrolls for T steps (default T=2). With `stop_grad_h_s=False`, gradients flow through `h_s` into prior steps (full BPTT); with `stop_grad_h_s=True`, stop-grad after every forward pass (no BPTT). T=2 with `stop_grad_h_s=False` matches LDDM-style two-pass double backprop. Use `num_steps>2` to explore longer trajectories.

```
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_loopholing_bptt_threshold_0.15_rollout_5" \
"train.experiment=sudoku_extreme_loopholing_bptt" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++loss.stop_grad_h_s=false" \
"++loss.num_steps=5" \
"++predictor.threshold=0.15"
```

**Loopholing without BPTT** (`LoopholingBPTTLoss` with `stop_grad_h_s=true`): same architecture but `h_s` detached after every forward pass, so later losses cannot backprop into prior steps.



**Loopholing BPTT + PUMA (streaming)** — `LoopholingBPTTPumaLoss` combines a PUMA-style streaming buffer with multi-step loopholing BPTT. The buffer stores persistent `h_s` per slot; newly evicted/filled slots get zero `h_s`. Use `experiment=sudoku_extreme_loopholing_bptt_puma` and `model_type=loopholing_bptt_puma`. Debug: `xlm job_type=train experiment=sudoku_extreme_loopholing_bptt_puma job_name=loopholing_puma debug=overfit`.

`RotaryTransformerLoopholingModel` uses a single loopholing path: inject `LayerNorm(h_t)` at the model input (added to token embeddings), and read `h_s` from the final encoder hidden state.
- Predictor diagnostics for validation rollout can be enabled with `predictor.log_rollout_diagnostics=true`; this logs per-step `h_t/h_s` norms, masked-token count, and confidence.

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=sudoku_extreme_loopholing_final_layer_bptt_puma_0.15_3_weighted_ce" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram80,bf16\"" \
"slurm.time=36:00:00" "use_job_name_as_id=false" "---" "++loss.use_model_confidence=true" "global_batch_size=512" "trainer.max_steps=300000" loss.num_steps=3 predictor.threshold=0.15 loss.stop_grad_h_s=false ++loss.weighted_ce=true
```

# OWT evals

```bash
for CKPT in \
  "owt_last_layer_untied|/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/logs/owt_packed_loopholing_bptt_puma_v2/checkpoints/0-100000.ckpt" \
  "owt_middle_layer_untied|/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/logs/owt_packed_loopholing_bptt_puma_v2_loophole_L6/checkpoints/0-100000.ckpt" \
  "owt_last_layer_tied|/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/logs/owt_packed_loopholing_bptt_puma_v4_tied_embeddings_fix_init/checkpoints/0-100000.ckpt"; do
CHECKPOINT_TAG="${CKPT%%|*}"
CHECKPOINT_PATH="${CKPT#*|}"
CHECKPOINT_STEP="$(basename "${CHECKPOINT_PATH}" .ckpt)"
for CONFIDENCE in top_prob; do
for MAX_STEPS in 2048; do
for TOP_P in 1.0 0.9 0.8; do
for threshold in 0.001 0.01 0.1 0.15 0.20; do
    python slurm_scripts/submit_eval.py "do=submit" "job_name=owt_packed_loopholing_bptt_puma_eval_${CHECKPOINT_TAG}_${CHECKPOINT_STEP}_${CONFIDENCE}_${threshold}_${TOP_P}_${MAX_STEPS}" "experiment=[owt_packed_loopholing_bptt_puma,gpt2_generative_perplexity]" "eval.checkpoint_path=${CHECKPOINT_PATH}" "eval.split=validation" "eval.dms_to_remove=[val.lm]" "paths.log_dir=logs/eval" "++slurm.reservation=dhruvesh" "---" "++predictor.confidence=${CONFIDENCE}" "++predictor.threshold=${threshold}" "paths.log_dir=logs/eval" loggers=wandb "+tags.checkpoint=${CHECKPOINT_TAG}_${CHECKPOINT_STEP}" "datamodule.dataset_managers.val.unconditional_prediction.num_examples=1000" "++predictor.top_p=${TOP_P}" "++predictor.top_k=null" ++generative_perplexity.evaluators.gpt2.max_length=1024 "++predictor.max_steps=${MAX_STEPS}"
done
done
done
done
done
```

**OWT evals — 400k checkpoint** (`owt_packed_loopholing_bptt_puma_v2` only, same grid as above):

```bash
CHECKPOINT_TAG="owt_last_layer_untied"
CHECKPOINT_PATH="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/logs/owt_packed_loopholing_bptt_puma_v2/checkpoints/0-400000.ckpt"
CHECKPOINT_STEP="$(basename "${CHECKPOINT_PATH}" .ckpt)"
for CONFIDENCE in top_prob; do
for MAX_STEPS in 2048; do
for TOP_P in 1.0 0.9 0.8; do
for threshold in 0.001 0.01 0.1 0.15 0.20; do
    python slurm_scripts/submit_eval.py "do=submit" "job_name=owt_packed_loopholing_bptt_puma_eval_${CHECKPOINT_TAG}_${CHECKPOINT_STEP}_${CONFIDENCE}_${threshold}_${TOP_P}_${MAX_STEPS}" "experiment=[owt_packed_loopholing_bptt_puma,gpt2_generative_perplexity]" "eval.checkpoint_path=${CHECKPOINT_PATH}" "eval.split=validation" "eval.dms_to_remove=[val.lm]" "paths.log_dir=logs/eval" "++slurm.reservation=dhruvesh" "---" "++predictor.confidence=${CONFIDENCE}" "++predictor.threshold=${threshold}" "paths.log_dir=logs/eval" loggers=wandb "+tags.checkpoint=${CHECKPOINT_TAG}_${CHECKPOINT_STEP}" "datamodule.dataset_managers.val.unconditional_prediction.num_examples=1000" "++predictor.top_p=${TOP_P}" "++predictor.top_k=null" ++generative_perplexity.evaluators.gpt2.max_length=1024 "++predictor.max_steps=${MAX_STEPS}"
done
done
done
done
```

**OWT evals — tied embeddings** (`owt_packed_loopholing_bptt_puma_v4_tied_embeddings_fix_init` only; use `++model.tie_embeddings=true` so weights/EMA match the checkpoint):

```bash
CHECKPOINT_TAG="owt_last_layer_tied"
CHECKPOINT_PATH="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/double-backprop/logs/owt_packed_loopholing_bptt_puma_v4_tied_embeddings_fix_init/checkpoints/0-100000.ckpt"
CHECKPOINT_STEP="$(basename "${CHECKPOINT_PATH}" .ckpt)"
for CONFIDENCE in top_prob; do
for MAX_STEPS in 2048; do
for TOP_P in 1.0 0.9 0.8; do
for threshold in 0.001 0.01 0.1 0.15 0.20; do
    python slurm_scripts/submit_eval.py "do=submit" "job_name=owt_packed_loopholing_bptt_puma_eval_${CHECKPOINT_TAG}_${CHECKPOINT_STEP}_${CONFIDENCE}_${threshold}_${TOP_P}_${MAX_STEPS}" "experiment=[owt_packed_loopholing_bptt_puma,gpt2_generative_perplexity]" "eval.checkpoint_path=${CHECKPOINT_PATH}" "eval.split=validation" "eval.dms_to_remove=[val.lm]" "paths.log_dir=logs/eval" "++slurm.reservation=dhruvesh" "---" "++model.tie_embeddings=true" "++predictor.confidence=${CONFIDENCE}" "++predictor.threshold=${threshold}" "paths.log_dir=logs/eval" loggers=wandb "+tags.checkpoint=${CHECKPOINT_TAG}_${CHECKPOINT_STEP}" "datamodule.dataset_managers.val.unconditional_prediction.num_examples=1000" "++predictor.top_p=${TOP_P}" "++predictor.top_k=null" ++generative_perplexity.evaluators.gpt2.max_length=1024 "++predictor.max_steps=${MAX_STEPS}"
done
done
done
done
```