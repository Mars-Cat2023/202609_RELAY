# Tulu-3-SFT-Mixture &mdash; Fast-dLLM v2 1.5B

**Scope.** [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) (~939k rows) SFT with **no sequence packing**, 1 epoch, 8 GPUs single-node. Four runs cover the BPTT/carry ablation grid:

| # | Run | Sbatch | `CARRY_MODE` | `BPTT_STOP_GRAD_H_S` | Notes |
|---|-----|--------|--------------|----------------------|-------|
| **3** | **PUMA + Loopguard with stop-grad on `h_s`** | `finetune_tulu3_bptt.sbatch` | `loopguard` | `1` | Architectural carry on, gradient through carry off |
| **4** | **PUMA + Loopguard + BPTT** | `finetune_tulu3_bptt.sbatch` | `loopguard` | `0` | Default behavior of `block_bptt_loss.py` |
| 1 | Vanilla SFT (optional baseline) | `finetune_tulu3.sbatch` | -- | -- | No PUMA / loopholing / BPTT |
| 2 | BPTT + PUMA, no carry (optional baseline) | `finetune_tulu3_bptt.sbatch` | `none` | -- | 2-step BPTT loss + PUMA buffer, no carry tensor |

**Runs 3 and 4 are the priority; runs 1 and 2 are optional baselines** to be queued only if compute time permits.

**No on-save eval.** Both Tulu sbatch scripts intentionally omit all `--evalplus_*` flags so nothing eval-related fires during training. Run EvalPlus / Math500 / lm-eval-harness on saved checkpoints out-of-band.

---

## 1. First-time setup

From `Fast-dLLM/v2`:

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate lmflow

# Full mixture (default), pre-shuffled with seed=42, no source filter.
# Default --shard_size=50000 -> ~19 files; required for 939k rows.
python scripts/prep_tulu3_sft.py --out_dir data/tulu3_sft
```

Expected: `data/tulu3_sft/train_conversation/train_shardNNNofMMM_<rows>.json` (default ~19 shards x ~50k rows for the full mixture). HF Datasets globs `train_*.json` in the directory and concatenates the shards into one logical train split.

**Why sharded?** PyArrow's string-column offsets are int32, so any single Arrow chunk whose `messages` column exceeds 2 GB blows up `combine_chunks()` with `ArrowInvalid: offset overflow while concatenating arrays` during `Generating train split`. The full mixture serialises to ~2.8 GB as a single file (the Magicoder-shape one-file layout in `prep_magicoder_oss.py` only works because Magicoder is ~270 MB). Sharding into ~50k-row files (~150-200 MB each) makes every Arrow chunk safe regardless of the row-length distribution. Override with `--shard_size 0` to emit a single file (only do this for small subsets, e.g. `--max_rows <= ~75000`).

The prep script logs per-source counts before / after each filter / cap, removes any pre-existing `train_*.json` shards in the output dir before writing, and prints the per-shard row counts so the final mixture composition is obvious from stdout.

**Optional variants** (not needed for the default 4-run sweep):

```bash
# Math + code only (~427k rows; closer to math500 / EvalPlus evals).
python scripts/prep_tulu3_sft.py \
    --out_dir data/tulu3_sft_mathcode \
    --include_sources \
ai2-adapt-dev/personahub_math_v5_regen_149960,\
allenai/tulu-3-sft-personas-math-grade,\
ai2-adapt-dev/personahub_code_v2_34999,\
ai2-adapt-dev/numinamath_tir_math_decontaminated,\
ai2-adapt-dev/evol_codealpaca_heval_decontaminated

# Random 100k subsample (matches the Magicoder-full scale).
python scripts/prep_tulu3_sft.py --out_dir data/tulu3_sft_100k --max_rows 100000
```

---

## 2. Compute budget

Re-deriving from the W&B run [`bptt_loopguard_..._magicoder_full_..._ep1_lr1e-5_v2`](https://wandb.ai/ilm-extensions/huggingface) (75k rows, 1175 optimizer steps, 11.6h on 2 GPUs &asymp; 35.5 sec/step at `eff_bs=64`):

| Geometry | Effective batch | Optimizer steps (1 epoch) | Per-step | Wall time / run |
|----------|-----------------|---------------------------|----------|-----------------|
| 8 GPUs, `per_device_bs=2`, `grad_accum=16` | `2*16*8 = 256` | `939343 / 256 ~= 3669` | ~35 sec | **~36 hours** |

- Per-step wall time on 8 GPUs is approximately the same as on 2 GPUs because each GPU still does the same `2*16=32` examples per step.
- That exceeds the 1-day SBATCH cap, so the sbatch defaults are **`--time=2-00:00:00 --requeue`** (already wired).
- **Priority pair (runs 3 + 4): ~3 days sequential, ~24 GPU-days.**
- **Full 4-run sweep: ~6 days sequential, ~48 GPU-days.**

Checkpoint cadence at `SAVE_STEPS=500`: `floor(3669/500) = 7` periodic checkpoints at steps 500, 1000, 1500, 2000, 2500, 3000, 3500 plus the final one at ~3669 &mdash; so step 1500 is reached in ~14.5h ("does this look good enough?" suspend point).

**Representativeness at any prefix.** Two layers of shuffling guarantee that early checkpoints see a uniform mix across all Tulu sources, not just CoCoNot / FLAN (which come first in the natural HF order):

1. `prep_tulu3_sft.py` pre-shuffles the on-disk JSON with seed 42.
2. HF `Trainer` shuffles every epoch via `SeedableRandomSampler` (`accelerator_config.use_seedable_sampler = true`).

At step 1500 (`eff_bs=256`), the model has seen `1500 * 256 ~= 384k` examples drawn uniformly from the 939k mixture, i.e. ~41% of every source.

---

## 3. Training (priority pair: runs 3, 4)

Submit from `Fast-dLLM/v2` (`cwd` = repo root). `SAVE_STEPS=500 SAVE_TOTAL_LIMIT=0` are baked in as the sbatch defaults &mdash; every `checkpoint-*` is kept on disk so you can suspend the run at any periodic checkpoint without losing earlier ones. Order runs sequentially yourself (e.g. submit run 4 only after `squeue -u $USER` shows run 3 is gone).

```bash
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2
conda activate lmflow

# 3) PUMA + Loopguard with STOP-GRAD on h_s (NEW ablation; the missing piece)
sbatch --reservation dhruvesh \
  --job-name=ft_tulu3_bptt_loopguard_stopgrad_ep1_1p5B \
  --export=ALL,CARRY_MODE=loopguard,BPTT_STOP_GRAD_H_S=1 \
  train_scripts/finetune_tulu3_bptt.sbatch

# 4) PUMA + Loopguard + BPTT (no stop-grad, the current default behavior)
#    Submit after run 3 has finished (or in parallel if you have two free 8-GPU reservations).
sbatch --reservation dhruvesh \
  --job-name=ft_tulu3_bptt_loopguard_bptt_ep1_1p5B \
  --export=ALL,CARRY_MODE=loopguard,BPTT_STOP_GRAD_H_S=0 \
  train_scripts/finetune_tulu3_bptt.sbatch
```

### Optional baselines (runs 1, 2)

Only queue these if you have time after runs 3 + 4 finish.

```bash
# 1) Vanilla SFT
sbatch --reservation dhruvesh \
  --job-name=ft_tulu3_full_ep1_1p5B \
  train_scripts/finetune_tulu3.sbatch

# 2) BPTT + PUMA, no carry
sbatch --reservation dhruvesh \
  --job-name=ft_tulu3_bptt_nocarry_ep1_1p5B \
  --export=ALL,CARRY_MODE=none \
  train_scripts/finetune_tulu3_bptt.sbatch
```

### How the 4 runs map to defaults

| Run | `output_dir` | W&B `run_name` |
|-----|--------------|----------------|
| 3 | `output_models/tulu3_sft_1p5B/bptt_loopguard_stopgrad_puma_nopack1024_bs2x16x8_ep1_lr1e-5/` | `bptt_loopguard_stopgrad_1p5B_tulu3_sft_full_puma_nopack1024_bs2x16x8_ep1_lr1e-5_v2` |
| 4 | `output_models/tulu3_sft_1p5B/bptt_loopguard_puma_nopack1024_bs2x16x8_ep1_lr1e-5/` | `bptt_loopguard_1p5B_tulu3_sft_full_puma_nopack1024_bs2x16x8_ep1_lr1e-5_v2` |
| 1 | `output_models/tulu3_sft_1p5B/vanilla_nopack1024_bs2x16x8_ep1_lr1e-5/` | `vanilla_1p5B_tulu3_sft_full_nopack1024_bs2x16x8_ep1_lr1e-5_v2` |
| 2 | `output_models/tulu3_sft_1p5B/bptt_nocarry_puma_nopack1024_bs2x16x8_ep1_lr1e-5/` | `bptt_nocarry_1p5B_tulu3_sft_full_puma_nopack1024_bs2x16x8_ep1_lr1e-5_v2` |

The `stop_grad_h_s` setting is also visible in W&B per-step metrics as `bptt/stop_grad_h_s` (0.0 vs 1.0) and in the `WANDB_TAGS` (`stop_grad_h_s=0` vs `stop_grad_h_s=1`), so runs 3 and 4 are easy to disambiguate after the fact.

### Common overrides (all four sbatch invocations)

`NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, `BLOCK_SIZE`, `OUTPUT_DIR`, `RUN_NAME`, `WANDB_TAGS`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, `DISABLE_GROUP_TEXTS` (keep `1` unless you want packed chunks), `DATASET_PATH`, `DEEPSPEED_CONFIG`, `USE_FP32`. BPTT-only: `BPTT_THRESHOLD`, `BPTT_TOP_P`, `BPTT_TEMPERATURE`, `BPTT_USE_STREAMING_BUFFER`, `BPTT_STOP_GRAD_H_S`, `CARRY_MODE`. Pass via `--export=ALL,KEY=VAL,...` to `sbatch` so they propagate to the compute node.

---

## 4. Resuming, suspending, evaluating

**Resuming.** All four sbatch scripts auto-discover the latest `checkpoint-*` under `output_dir` and pass `--resume_from_checkpoint`. So if a job is preempted / requeued / scancelled, just resubmit the same command and it picks up at the latest periodic checkpoint.

**Suspending early.** To stop a run at e.g. step 1500 (the "does this look good enough?" point), wait for `checkpoint-1500/` to appear on disk, then:

```bash
scancel <JID>
```

The full checkpoint is preserved (`SAVE_TOTAL_LIMIT=0` keeps every `checkpoint-*`).

**Evaluation.** Out-of-band, after the runs finish or you suspend them:

- **Math500 / lm-eval-harness**: `train_scripts/eval_lm_eval.sbatch`, `train_scripts/eval_math500.sbatch`, `python scripts/submit_eval.py --task math500 --model_path <ckpt>` (set `--use_carry` for carry checkpoints, i.e. runs 3 / 4).
- **EvalPlus (HumanEval / MBPP)**: `python scripts/generate_evalplus_jsonl.py ... --use_carry` (carry checkpoints) then `evalplus.evaluate --samples <jsonl>`. See [`CODE_COMMANDS.md`](CODE_COMMANDS.md) sections 4a-4d.

For carry checkpoints (runs 3 and 4), pass **`--use_carry`** to the eval scripts and skip block cache (`--use_block_cache` is validated only for vanilla / nocarry).

---

## 5. File reference

* `scripts/prep_tulu3_sft.py` &mdash; Tulu-3-SFT-Mixture &rarr; `data/tulu3_sft/train_conversation/`. Pre-shuffles on disk with seed 42 and shards output (`--shard_size 50000` default) to stay under PyArrow's 2 GB string-column cap. Optional `--max_rows` / `--include_sources` / `--exclude_sources`.
* `train_scripts/finetune_tulu3.sbatch` &mdash; vanilla SFT, 8 GPUs, no on-save eval.
* `train_scripts/finetune_tulu3_bptt.sbatch` &mdash; BPTT + PUMA + (`CARRY_MODE=none|loopguard|mlp|cab`) + (`BPTT_STOP_GRAD_H_S=0|1`), 8 GPUs, no on-save eval.
* `src/lmflow/pipeline/utils/block_bptt_loss.py` &mdash; `FastDLLMBlockBPTTLoss` with the new `stop_grad_h_s` constructor knob and `bptt/stop_grad_h_s` metric.
* `src/lmflow/pipeline/utils/bptt_trainer.py` &mdash; passes `stop_grad_h_s` from training args to the loss.
* `src/lmflow/args.py` &mdash; `bptt_stop_grad_h_s: bool = False` field on `FinetunerArguments`.
