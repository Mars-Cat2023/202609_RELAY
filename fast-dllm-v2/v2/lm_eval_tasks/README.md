# Custom lm-eval tasks (EvalPlus-aligned)

These tasks mirror the three stages of [`scripts/generate_evalplus_jsonl.py`](../scripts/generate_evalplus_jsonl.py):

1. **Prompting**: `evalplus.provider.utility.make_raw_chat_prompt` with the same `INSTRUCTION_PREFIX` / `RESPONSE_PREFIX`.
2. **Post-processing**: `evalplus.sanitize` plus the same fallbacks as `to_evalplus_solution` (see `_evalplus_common/postprocess.py`).
3. **Scoring**: Official EvalPlus judge `evalplus.evaluate.check_correctness` (`pass_at_1_base` / `pass_at_1_plus`).

## Tasks

| YAML | Class | EvalPlus dataset |
|------|--------|------------------|
| `humaneval_plus_evalplus/humaneval_plus_evalplus.yaml` | `HumanEvalPlusEvalPlus` | HumanEval+ |
| `mbpp_plus_evalplus/mbpp_plus_evalplus.yaml` | `MbppPlusEvalPlus` | MBPP+ |

## Requirements

- `evalplus` (datasets + sanitize + evaluator), `datasets`, Hugging Face `lm-evaluation-harness` (you already use Fast-dLLM `v2/lib/lm-evalulation-harness` or an install).
- A tokenizer **`chat_template`**: `make_raw_chat_prompt` falls back silently to bare stubs if absent; the tasks **raise** in that case.
- **`EVALPLUS_TOKENIZER`**: HF id pointing at the tokenizer that matches your model’s chat formatting (often the same pretrained id). Alternatively set **`metadata.tokenizer_name`** in the YAML (or `--metadata`' JSON override).

## How to run

From **`Fast-dLLM/v2`**, prepend both the bundled harness source and `v2` to `PYTHONPATH` so imports resolve (`lm_eval` + `lm_eval_tasks`):

```bash
cd Fast-dLLM/v2

export PYTHONPATH="$(pwd)/lib/lm-evalulation-harness:$(pwd)"
export EVALPLUS_TOKENIZER="YOUR_HF_TOKENIZER_MATCHING_THE_MODEL"

# Do NOT pass lm-eval --apply_chat_template — EvalPlus prompting is baked into doc_to_text.
# EvalPlus prompting is strictly 0-shot — do not bump --num_fewshot above 0.

python -m lm_eval \
  --include_path lm_eval_tasks \
  --tasks humaneval_plus_evalplus \
  --model hf ... \
```

Swap `humaneval_plus_evalplus` for `mbpp_plus_evalplus` as needed.

**Slurm:** From repo root, `python scripts/submit_eval.py --task humaneval_plus_evalplus --model_path … [--evalplus_tokenizer …]` generates an sbatch that sets `PYTHONPATH`, omits lm-eval `--apply_chat_template`, and exports `EVALPLUS_TOKENIZER` when you pass `--evalplus_tokenizer`. See **`CODE_COMMANDS.md`** §4d.

## Notes

- **Unsafe execution**: Tasks set `unsafe_code: true`; EvalPlus runs untrusted completions in sandboxed subprocesses (same semantics as EvalPlus leaderboard tooling).
- **Optional deps**: Metrics registration avoids importing `lm_eval.api.metrics` up front so a minimal env without optional harness deps (e.g. sacrebleu) still loads tasks; aggregation `mean` is registered lazily when needed.
- **Ground-truth cache**: EvalPlus computes oracle outputs once per dataset version under `~/.cache/evalplus` (`get_groundtruth`).
