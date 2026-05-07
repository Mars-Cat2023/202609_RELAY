# Puzzle training (SLURM): MLM uniform → Loopholing BPTT + PUMA

Non-debug **submit** commands for the puzzle tracks: **Sudoku extreme**; **GRAM n-queens** (**8×8** or **10×10**); **GRAM graph coloring** (**8-vertex** or **10-vertex**). For each GRAM size, run **MLM uniform** first, then **Loopholing BPTT + PUMA** (streaming buffer, multi-step unroll). Dataset filters and sequence lengths are fixed per size (`10×10` → `block_size=100`, `vocab_size=3`; `10v` → `block_size=55` = 45 upper-triangle edge bits + 10 node colors, `vocab_size=6`).

Activate your environment and run from the repo root (same pattern as `README.md`). Adjust `hardware`, `slurm.*`, or `train.batch_size` for your cluster. Each `job_name` is prefixed with `puzzle_` so runs log under `logs/puzzle_<…>/` (distinct from un-prefixed jobs); `train.experiment` stays the YAML name unchanged.

**Two-step BPTT:** `model_type=loopholing_bptt_puma` sets `loss.num_steps: 2` by default (`doublebackprop/configs/model_type/loopholing_bptt_puma.yaml`). No override is required unless you want a different unroll length.

### Resume to 300k steps (completed first runs)

The commands below **resume** from `checkpoints/last.ckpt` under the matching `logs/<job_name>/runs/<timestamp>/` tree, extend training to **`trainer.max_steps=300000`**, and set **`+loggers.wandb.id`** so W&B continues the **same** run (verified against `ilm-extensions/BPTT-puzzles`).

Before submitting, set **`RESUME_CKPT`** to your actual checkpoint file (only the `runs/<timestamp>/` folder differs per machine):

```bash
# Example — replace with your run’s path:
export RESUME_CKPT="$PWD/logs/puzzle_sudoku_extreme_mlm_uniform/runs/2026-04-04_02-52-27/checkpoints/last.ckpt"
```

Hydra overrides after `"---"` are passed through to the inner `xlm` command (see `slurm_scripts/submit_train.py`). `lr_scheduler.num_training_steps` is `${trainer.max_steps}` in the experiment YAMLs, so it follows `++trainer.max_steps=300000`.

| Job name (`job_name=`) | W&B run id (`+loggers.wandb.id=`) | Notes |
|------------------------|----------------------------------|--------|
| `puzzle_sudoku_extreme_mlm_uniform` | `queh30wq` | prior run ended at 120k steps |
| `puzzle_sudoku_extreme_loopholing_bptt_puma` | `1x0zmn4h` | prior run ended at 80k steps |
| `puzzle_gram_n_queens_8x8_mlm_uniform` | `vq4ny1xq` | prior run ended at 120k steps |
| `puzzle_gram_n_queens_8x8_loopholing_bptt_puma` | `13ype7f6` | prior run ended at 80k steps |
| `puzzle_gram_graph_coloring_8v_mlm_uniform` | `90wr2kq5` | prior run ended at 120k steps |
| `puzzle_gram_graph_coloring_8v_loopholing_bptt_puma` | `am3etgh1` | prior run ended at 80k steps |

---

## Sudoku extreme

### MLM uniform

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_sudoku_extreme_mlm_uniform_300k_steps_ABLATION" \
"train.experiment=sudoku_extreme_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

### Loopholing BPTT + PUMA (2 steps)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_sudoku_extreme_loopholing_bptt_puma_300k_steps_ABLATION" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

### Loopholing BPTT + PUMA (2 steps, ablation: tied token embedding/unembedding)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_sudoku_extreme_loopholing_bptt_puma_tied_emb_300k_steps_ABLATION" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000" \
"++model.tie_embeddings=true"
```

### Loopholing no-BPTT + PUMA (2 steps, ablation: stop_grad_h_s=True)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_sudoku_extreme_loopholing_no_bptt_puma_300k_steps_ABLATION" \
"train.experiment=sudoku_extreme_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000" \
"loss.stop_grad_h_s=true"
```

---

## GRAM n-queens 8×8

### MLM uniform

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_gram_n_queens_8x8_mlm_uniform_300k_steps" \
"train.experiment=gram_n_queens_8x8_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

### Loopholing BPTT + PUMA (2 steps)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_gram_n_queens_8x8_loopholing_bptt_puma_300k_steps" \
"train.experiment=gram_n_queens_8x8_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

---

## GRAM n-queens 10×10

### MLM uniform

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_gram_n_queens_10x10_mlm_uniform_300k_steps" \
"train.experiment=gram_n_queens_10x10_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

### Loopholing BPTT + PUMA (2 steps)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_gram_n_queens_10x10_loopholing_bptt_puma_300k_steps" \
"train.experiment=gram_n_queens_10x10_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

---

## GRAM graph coloring 8-vertex

### MLM uniform

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_gram_graph_coloring_8v_mlm_uniform_300k_steps" \
"train.experiment=gram_graph_coloring_8v_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

### Loopholing BPTT + PUMA (2 steps)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_gram_graph_coloring_8v_loopholing_bptt_puma_300k_steps" \
"train.experiment=gram_graph_coloring_8v_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

---

## GRAM graph coloring 10-vertex

### MLM uniform

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_gram_graph_coloring_10v_mlm_uniform_300k_steps" \
"train.experiment=gram_graph_coloring_10v_mlm_uniform" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

### Loopholing BPTT + PUMA (2 steps)

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=puzzle_gram_graph_coloring_10v_loopholing_bptt_puma_300k_steps" \
"train.experiment=gram_graph_coloring_10v_loopholing_bptt_puma" \
"train.batch_size=512" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=\"vram40,bf16\"" \
"slurm.time=36:00:00" \
"use_job_name_as_id=false" \
"---" \
"++trainer.max_steps=300000"
```

---

## Experiment config paths

| Track | MLM uniform | Loopholing BPTT + PUMA | Loopholing no-BPTT + PUMA |
|-------|-------------|-------------------------|---------------------------|
| Sudoku | `doublebackprop/configs/experiment/sudoku_extreme_mlm_uniform.yaml` | `doublebackprop/configs/experiment/sudoku_extreme_loopholing_bptt_puma.yaml` | `doublebackprop/configs/experiment/sudoku_extreme_loopholing_bptt_puma.yaml` + `loss.stop_grad_h_s=true` |
| N-queens 8×8 | `.../gram_n_queens_8x8_mlm_uniform.yaml` | `.../gram_n_queens_8x8_loopholing_bptt_puma.yaml` | — |
| N-queens 10×10 | `.../gram_n_queens_10x10_mlm_uniform.yaml` | `.../gram_n_queens_10x10_loopholing_bptt_puma.yaml` | — |
| Graph coloring 8v | `.../gram_graph_coloring_8v_mlm_uniform.yaml` | `.../gram_graph_coloring_8v_loopholing_bptt_puma.yaml` | — |
| Graph coloring 10v | `.../gram_graph_coloring_10v_mlm_uniform.yaml` | `.../gram_graph_coloring_10v_loopholing_bptt_puma.yaml` | — |

Datamodules (GRAM only): `doublebackprop/configs/datamodule/gram_n_queens_{8x8,10x10}_mlm_uniform.yaml`, `gram_graph_coloring_{8v,10v}_mlm_uniform.yaml`. Per-size **dataset** YAMLs live in **xlm-core** under `xlm-core/src/xlm/configs/lightning_train/datasets/` (e.g. `gram_n_queens_train_10x10`, `gram_graph_coloring_train_10v`).

Local **debug** one-liners (`debug=overfit`, small batches) remain in `README.md` under *GRAM n-queens & graph coloring* and *MLM uniform* / *Loopholing BPTT + PUMA*.
