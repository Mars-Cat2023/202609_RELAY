# Fast-dLLM v2 — codebase notes (working)

## 1. Directory structure (v2)

```
v2/
├── app.py                        # Gradio UI; uses generation_functions + HF model
├── run_chatbot.py                # CLI chat; model.generate(...) with block-diffusion kwargs
├── generation_functions.py       # Custom inference: Fast_dLLM_QwenForCausalLM (batch_sample, mdm_sample_with_*)
├── eval.py                       # Eval harness (lm-eval); optional 2-step carry in get_logits / sampling
├── eval_script.sh
├── COMMANDS.md                   # Copy-paste sbatch / env patterns (BPTT-PUMA, eval)
├── MATH_COMMANDS.md              # Math-500 / related eval recipes
├── pyproject.toml / setup.py / requirements.txt
├── configs/                      # DeepSpeed ZeRO json (+ zero1 no-offload variant)
├── data/                         # download.sh for datasets
├── scripts/                      # Utilities: Numina prep, BPTT smoke tests, lm-eval task stubs
├── train_scripts/
│   ├── finetune.py               # Entry: AutoPipeline "finetuner" + AutoModel
│   ├── finetune_alpaca.sh
│   ├── *.sbatch                  # SLURM: finetune (alpaca, Numina, BPTT-PUMA), lm-eval, math500, Numina NLL
│   └── …
├── asset/                        # figures, demo media
└── src/lmflow/                   # LMFlow fork: training, data, HF wrappers
    ├── args.py                   # + loss_type, BPTT / carry flags (see §5)
    ├── models/
    │   ├── auto_model.py
    │   ├── hf_decoder_model.py
    │   ├── hf_model_mixin.py
    │   ├── fast_dllm/            # Vendored Fast_dLLM_Qwen* (config + modeling); registers with HF Auto*
    │   │   ├── __init__.py
    │   │   ├── configuration.py
    │   │   └── modeling.py       # Block-diffusion attention, forward, h_s / h_t, CAB hooks, generate
    │   ├── base_model.py, decoder_model.py, …
    │   ├── interfaces/
    │   └── vision_encoder/
    ├── pipeline/
    │   ├── finetuner.py          # group_text, Trainer; BPTT branch → FastDLLMBPTTTrainer
    │   ├── inferencer.py, vllm_inferencer.py, …
    │   └── utils/
    │       ├── peft_trainer, raft_trainer, …
    │       ├── block_bptt_loss.py   # FastDLLMBlockBPTTLoss (2-step L1+L2, CAB / loopholing / none)
    │       ├── bptt_trainer.py      # Trainer.compute_loss → block BPTT loss
    │       └── streaming_batch.py    # PUMA-style buffer (always on for BPTT)
    ├── datasets/
    ├── tokenization/
    ├── optim/
    └── utils/
```

`finetuner.py` imports `lmflow.models.fast_dllm` early so `AutoModelForCausalLM` resolves the vendored classes for local training and checkpoints that use `model_type: Fast_dLLM_Qwen`.

## 2. Where training loss is computed

- **Default (`loss_type` mlm / vanilla SFT):** unchanged — `transformers.Trainer` (or `PeftTrainer` if LoRA) in `src/lmflow/pipeline/finetuner.py` calls the backend model; scalar loss is `outputs.loss` from the Fast-dLLM forward (mask diffusion / block attention implemented in `models/fast_dllm/modeling.py`).
- **BPTT (`loss_type=bptt`):** `FinetuningTrainer` becomes `FastDLLMBPTTTrainer` (`pipeline/utils/bptt_trainer.py`), which overrides `compute_loss` to run `FastDLLMBlockBPTTLoss` (`pipeline/utils/block_bptt_loss.py`): two masked LM steps with **per-block** teacher-forced reveals after step 1 (mirrors the inference loop's per-block threshold + argmax rule, using `config.bd_size` as the block grid), total loss `L1 + L2`, always backed by a per-rank **PUMA** `StreamingBatch` so each call advances the buffered slice's `(x_input, h_s)` rather than restarting from all-mask. Hidden-state carry between steps uses **CAB** (`bptt_use_cab`) or **position-guarded LayerNorm loopholing** (default when CAB off), or **`carry_mode=none`** (`bptt_disable_carry`) for ablation.
- **Data:** `group_text` still builds `input_ids` / `labels` with block padding and `data_args.mask_id` / `bd_size`.

**Entry point:** `train_scripts/finetune.py` → `Finetuner.tune(model, dataset)`.

## 3. Where the inference loop is

- **`generation_functions.py`**
  - `Fast_dLLM_QwenForCausalLM.batch_sample`: block / sub-block denoising; supports **`use_carry`** — when the loaded config has `use_cab` or `use_loopholing`, forwards can pass **`h_t`** from a right-shifted previous **`h_s`** (same convention as training BPTT / eval).
  - `mdm_sample_with_visualization` / `setup_model_with_custom_generation`: Gradio path.
- **`eval.py` (`Fast_dLLM_v2EvalHarness`):** **`use_carry`** — optional second forward with `h_t` aligned to logits for NLL / sampling when the checkpoint has carry modules.
- **`app.py`:** Gradio + `mdm_sample_with_visualization`.
- **`run_chatbot.py`:** `model.generate(...)` with block-diffusion kwargs; `generate` lives on the vendored causal LM in `fast_dllm/modeling.py` (and matches hub checkpoints that declare the same `model_type`).

## 4. Model architecture files in this repo

- **In-repo definition:** `src/lmflow/models/fast_dllm/` — `Fast_dLLM_QwenConfig`, `Fast_dLLM_QwenForCausalLM`, block-diffusion flex-attention masks, training noising bypass flags, **`h_s` outputs**, **`h_t` input**, optional **CrossAttentionBridge** and loopholing **LayerNorm**, `GenerationMixin` sampling.
- **Training / IO wrappers:** `hf_decoder_model.py`, `hf_model_mixin.py`, `auto_model.py`.
- **Inference-only helpers:** `generation_functions.py` (Python methods bound or mixed into loaded models for batch sampling and UI).

Hub checkpoints may still load with `trust_remote_code`; the vendored package is the canonical copy used when finetuning from this tree (see `finetuner` import of `lmflow.models.fast_dllm`).

## 5. BPTT training flags (quick ref)

Configured on `TrainingArguments` / CLI via `args.py`: `loss_type`, `bptt_threshold`, `bptt_top_p`, `bptt_temperature`, `bptt_use_cab`, `bptt_disable_carry`. (`bptt_use_streaming_buffer` is kept on the dataclass as a deprecated no-op for backwards compat with existing sbatch scripts; the PUMA buffer is now the only BPTT path.) SLURM examples and env wiring: **`COMMANDS.md`**, **`MATH_COMMANDS.md`**, `train_scripts/finetune_*_bptt*.sbatch`.

## 6. Scripts and eval extras

- **`scripts/`:** e.g. `prep_numina.py`, `eval_numina_heldout_nll.py`, `test_bptt_shift_alignment.py`, `_bptt_static_smoke.py`, lm-eval **math500** task YAML.
- **`eval.py`:** carry-aware logits and generation kwargs passed through to `mdm_sample` / `mdm_sample_with_visualization`.

## 7. Reference: discrete-diffusion eval debugging

Reference repo: `discrete-diffusion`, mainly `src/dd/commands/main.py`, `src/dd/debug.py`, `src/dd/diffusion/base/prediction.py`, `src/dd/diffusion/base/evaluator.py`, `src/dd/diffusion/base/evaluation_tracker.py`, and `src/dd/diffusion/base/history.py`.

### 7.1 Selecting examples to evaluate

The reference exposes three ways to restrict evaluation:

- `args.limit`: the usual lm-eval style prefix limit. The debug configs set this to `1`, for example `configs/evals/debug/limit.yaml`.
- `args.subsample`: inline Hydra/OmegaConf content.
- `args.subsample_file`: path to a JSON file, used in the README command as `args.subsample_file=subsample.json`.

`src/dd/commands/main.py` loads exactly one of these selected-example sources:

- If `subsample_file` is set, it reads JSON from disk with shape like `{"task_name": [doc_id, ...]}`.
- Else if `subsample` is set, it converts the OmegaConf object to a plain container.
- Else if `subsample_json` is set, it parses a JSON string.
- Otherwise it evaluates the normal task stream.

The selected sample map is passed into the evaluator as `samples=subsample`. The custom evaluator then calls `task.build_all_requests(..., samples=...)` and later `task.doc_iterator(..., samples=indices)`. When logging results, it remaps the local enumerated `doc_id` back to the true dataset id with `doc_id_true = indices[doc_id]`, so saved samples preserve the original example ids.

Important constraint: the reference forbids combining `limit` with `samples` inside the custom evaluator, and also checks that `subsample` and `subsample_file` are not both provided. The README examples either use `args.limit=1` for quick first-example debugging or `args.subsample_file=subsample.json` with `args.limit=null` for named example selection.

### 7.2 Global debug flags and console trajectory printing

Debug controls are not threaded through every call. They are module-level globals in `src/dd/debug.py`:

- `PRINT_PREDS`
- `VIZ_LOGITS`
- `DUMP_LOGITS`
- `WINDOW_ANALYSIS`
- `DUMP_UNMASK`

Hydra debug configs set `global_flags`. For example, `configs/evals/debug/limit.yaml` sets `global_flags.PRINT_PREDS: true`, and `configs/evals/debug/top_k_history.yaml` sets `global_flags.VIZ_LOGITS: true`.

At process startup, `src/dd/commands/main.py` calls `debug.set_flags(cfg)`. That function reads `cfg.global_flags` and mutates the `dd.debug` module with `setattr(this_module, flag, value)`. Because sampling code imports `from dd import debug`, any generation path can check `debug.PRINT_PREDS` or `debug.VIZ_LOGITS` without receiving a new function argument.

Console trajectory printing is implemented inside the generation loop in `src/dd/diffusion/base/prediction.py`. During each `predict_single_step`, if `debug.PRINT_PREDS` is true, it prints a step header like `x_<step>` and calls `print_text(step_results["x"][0], self.tokenizer)`. `print_text` converts token ids to tokens, replaces mask tokens with `[m]`, trims trailing mask runs, and prints the current partially denoised sequence. There are additional `PRINT_PREDS` probes in remasking / selection code that print diagnostic values such as current token probability means and remask pressure.

This is the key pattern to copy conceptually for `Fast-dLLM/v2`: define one small debug module or process-global config object, initialize it once from CLI arguments, and let `generation_functions.py` / `eval.py` consult it directly when deciding whether to print per-step states.

### 7.3 Saving final outputs and structured histories

The reference always routes eval output through an evaluation tracker. In `configs/evals/main.yaml`, `eval_tracker._target_` is `dd.diffusion.base.evaluation_tracker.EvaluationTracker` and `output_path` is `${paths.run_dir}/results`.

`src/dd/commands/main.py` calls:

- `eval_tracker.save_results_aggregated(results, samples=samples)`, which writes `results.json`.
- `eval_tracker.save_results_samples(task_name=..., samples=samples[task_name])`, which writes one `samples_<task_name>.jsonl` file per task.

Each logged sample includes the original document id, document payload, target, generation arguments, raw responses, filtered responses, metrics, and hashes. This is close to lm-eval's native sample logging, but the repo carries its own tracker so it can handle diffusion-specific response structures.

For full trajectory/history capture, the reference uses a custom evaluator path. `model_harness.predictor.output_history` can be set to a Hydra-instantiated plugin such as `HistoryTopKPlugin`. The predictor attaches `history`, `time_taken`, and `steps_taken` to each generated response. The custom evaluator stores response extras in sample records. `HistoryTopKPlugin` writes per-step tensor files under `results/history/<uuid>_<batch_index>/`, including `*_x.pt`, `*_p.pt`, and `*_pids.pt`, and returns a compact `{"fid": ...}` handle so the sample JSON can point back to the heavier trajectory artifacts. `PickleBasedEvaluationTracker` is an alternate tracker that moves large `extra` payloads into pickle files and stores the path in the JSONL sample record.

For `Fast-dLLM/v2/eval.py`, lm-eval likely already provides a native version of the final-output piece via `--output_path` plus `--log_samples` / `--predict_only` when using `cli_evaluate()`. The part not covered by stock lm-eval is step-by-step diffusion trajectory data; that would need instrumentation in `generation_functions.batch_sample` and a lightweight way to surface or persist the per-step state.

## 8. Fast-dLLM v2 eval debug (implemented)

- **Working directory:** run `accelerate launch eval.py` from `v2/` (or `accelerate launch v2/eval.py` from the repo root). Launching `eval.py` from `Fast-dLLM/` fails with “can't open file …/eval.py”.
- **Dependency:** `pip install -e '.[eval]'` from `v2/` adds `lm-eval[math,ifeval]==0.4.9` (see `setup.py` extras).
- **Subset of examples:** use lm-eval `--limit N` (no `subsample.json` in this path).
- **Structured outputs on disk:** `--output_path` and `--log_samples` (lm-eval harness).
- **Global debug flags:** [`fast_dllm_eval_debug.py`](fast_dllm_eval_debug.py) — set from `--model_args` as `debug_print=True` (trajectory lines to stderr during `batch_sample`) and `debug_print_prompt=True` (question/answer block in `eval.py` `generate_until`). Wired in [`eval.py`](eval.py) and [`generation_functions.py`](generation_functions.py).

See **README → Debug evaluation** for a copy-paste command.

---

*Last updated: 2026-04-24 — synced with main through `4de744e0` (vendored `fast_dllm`, BPTT/PUMA trainer + loss, carry inference, sbatch/docs/scripts).*
