# Protein training (SLURM): UniRef50 MLM uniform → Loopholing BPTT + PUMA

Commands for training protein language models on **UniRef50** with sequence packing and
FlexAttention. The MLM-uniform baseline lives in `xlm-core`; the BPTT loopholing experiment
lives in `doublebackprop`.

Hardware: **2 nodes × 4 A100-80GB GPUs** = 8 GPUs total (`hardware=ddp_2_node_4_gpu`).

**Batch arithmetic (both experiments)**:
- `block_size=512`, `per_device_batch_size` × GPUs × `accumulate_grad_batches` = `global_batch_size`
- MLM: 32 × 8 × 2 = 512 ✓
- BPTT: 16 × 8 × 4 = 512 ✓  (halved per-device batch for BPTT's 2-step unroll memory)

**FlexAttention**: `model.use_flex_attn: true` is set in both experiment YAMLs.  This flag
flows to `PackedMLMCollator` via `${model.use_flex_attn}` in `collator/packed_mlm.yaml`,
so both the collator and model are wired together from a single source of truth.

**`compile` flag**:
- MLM uniform: `compile=true` — `BlockMask` is built in the DataLoader worker (outside the
  compiled region), so `loss_fn` is graph-break-free.
- BPTT loopholing: `compile=false` — `create_block_mask` is called inside
  `RotaryTransformerLoopholingModel.forward` (correct for the uncompiled BPTT loop;
  `StreamingBatch` cannot store a `BlockMask` since it is not a plain tensor).

---

## UniRef50 MLM uniform (baseline)

Experiment config: `xlm-core/xlm-models/mlm/configs/experiment/uniref50_packed_mlm.yaml`
(base defaults: `block_size=1024`, `max_steps=1_000_000`; overridden below to 512 / 500k).

### Dry-run

```bash
python slurm_scripts/submit_train.py \
"job_name=uniref50_packed_mlm_uniform" \
"train.experiment=uniref50_packed_mlm" \
"train.batch_size=32" \
"train.compile=true" \
"train.precision=bf16-mixed" \
"hardware=ddp_2_node_4_gpu" \
"slurm.time=5-00:00:00" \
"++slurm.mem=200G" \
"use_job_name_as_id=false" \
"---" \
"++block_size=512" \
"++trainer.max_steps=500000" \
"++trainer.val_check_interval=25000"
```

### Submit

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=uniref50_packed_mlm_uniform" \
"train.experiment=uniref50_packed_mlm" \
"train.batch_size=32" \
"train.compile=true" \
"train.precision=bf16-mixed" \
"hardware=ddp_2_node_4_gpu" \
"slurm.time=5-00:00:00" \
"++slurm.mem=200G" \
"use_job_name_as_id=false" \
"---" \
"++block_size=512" \
"++trainer.max_steps=500000" \
"++trainer.val_check_interval=25000"
```

`train.experiment=uniref50_packed_mlm` resolves via the xlm-core search path to
`xlm-core/xlm-models/mlm/configs/experiment/uniref50_packed_mlm.yaml`.
The base YAML defaults (`block_size=1024`, `max_steps=1_000_000`) are overridden via
inner args after `"---"`.
`trainer.num_nodes=2` and `trainer.devices=4` come from `hardware=ddp_2_node_4_gpu`.

---

## UniRef50 Loopholing BPTT + PUMA (2 steps)

Experiment config: `doublebackprop/configs/experiment/uniref50_packed_loopholing_bptt_puma.yaml`
(already sets `block_size=512`, `max_steps=500_000`, `model.use_flex_attn: true`).

### Dry-run

```bash
python slurm_scripts/submit_train.py \
"job_name=uniref50_packed_loopholing_bptt_puma" \
"train.experiment=uniref50_packed_loopholing_bptt_puma" \
"train.batch_size=16" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=ddp_2_node_4_gpu" \
"slurm.time=5-00:00:00" \
"++slurm.mem=200G" \
"use_job_name_as_id=false"
```

### Submit

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=uniref50_packed_loopholing_bptt_puma" \
"train.experiment=uniref50_packed_loopholing_bptt_puma" \
"train.batch_size=16" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=ddp_2_node_4_gpu" \
"slurm.time=5-00:00:00" \
"++slurm.mem=200G" \
"use_job_name_as_id=false"
```

`accumulate_grad_batches` is derived by the harness from `global_batch_size / (per_device_batch_size × gpus)` = 512 / (16 × 8) = 4.

---

## No-BPTT ablation (stop_grad_h_s=true)

Same as BPTT but blocks the gradient through `h_s`. Set `++loss.stop_grad_h_s=true` as an inner arg:

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=uniref50_packed_loopholing_no_bptt_puma" \
"train.experiment=uniref50_packed_loopholing_bptt_puma" \
"train.batch_size=16" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=ddp_2_node_4_gpu" \
"slurm.time=5-00:00:00" \
"++slurm.mem=200G" \
"use_job_name_as_id=false" \
"---" \
"++loss.stop_grad_h_s=true"
```

---

## Resume (from last.ckpt)

Set `RESUME_CKPT` to the actual checkpoint path, then add `ckpt_path` and `+loggers.wandb.id`
as inner args:

```bash
export RESUME_CKPT="$PWD/logs/uniref50_packed_loopholing_bptt_puma/runs/<timestamp>/checkpoints/last.ckpt"

python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=uniref50_packed_loopholing_bptt_puma" \
"train.experiment=uniref50_packed_loopholing_bptt_puma" \
"train.batch_size=16" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=ddp_2_node_4_gpu" \
"slurm.time=5-00:00:00" \
"++slurm.mem=200G" \
"use_job_name_as_id=false" \
"---" \
"ckpt_path=${RESUME_CKPT}" \
"+loggers.wandb.id=<wandb_run_id>"
```

---

## Single-GPU debug (1 node, 1 GPU)

Useful for catching shape errors before submitting a full multi-node job.
Overrides `global_batch_size` so accumulation stays at 1 (no batch size assertion error).

```bash
python slurm_scripts/submit_train.py \
"do=submit" \
"job_name=uniref50_packed_loopholing_bptt_puma_debug" \
"train.experiment=uniref50_packed_loopholing_bptt_puma" \
"train.batch_size=4" \
"train.compile=false" \
"train.precision=bf16-mixed" \
"hardware=1_node_1_gpu" \
"slurm.constrain=vram80,bf16" \
"slurm.time=1:00:00" \
"use_job_name_as_id=false" \
"---" \
"++global_batch_size=4" \
"++trainer.max_steps=100" \
"++trainer.val_check_interval=50"
```
