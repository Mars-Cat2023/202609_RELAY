# SPDX-License-Identifier: Apache-2.0
"""
ConfigurableTask subclass that matches EvalPlus codegen prompting + lm-eval scaffolding.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import lru_cache
from typing import ClassVar

from evalplus.provider.utility import make_raw_chat_prompt
from transformers import AutoTokenizer

from lm_eval.api.task import ConfigurableTask, Task

from .judge import judge_one

# Side effect: registers pass_at_1_* metrics with lm-eval.
from . import registry_metrics as _registry_metrics  # noqa: F401

eval_logger = logging.getLogger(__name__)

# Mirrors scripts/generate_evalplus_jsonl.py
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem "
    "and passes corresponding tests:"
)


def _tokenizer_name(cfg) -> str:
    meta = getattr(cfg, "metadata", None)
    md: dict | None = None
    if isinstance(meta, dict):
        md = meta
    elif meta is None:
        md = None
    if md and md.get("tokenizer_name"):
        return str(md["tokenizer_name"]).strip()

    env = os.environ.get("EVALPLUS_TOKENIZER")
    if not env or not env.strip():
        raise RuntimeError(
            "EVALPLUS tokenizer is unspecified. Export EVALPLUS_TOKENIZER=<HF_pretrained_id>"
            ' or put {"tokenizer_name": "..."} in task metadata (--metadata).\n'
            "This tokenizer must match the model's chat_template for EvalPlus-style prompts."
        )
    return env.strip()


@lru_cache(maxsize=8)
def load_tokenizer(name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    ct = getattr(tok, "chat_template", None)
    if ct is None:
        raise RuntimeError(
            f"Tokenizer {name!r} has chat_template=None. EvalPlus `make_raw_chat_prompt` "
            "would silently fall back to the bare stub without instruction formatting."
        )
    return tok


def _resolve_config_field(doc: dict, field) -> str:
    if field is None:
        return ""
    if callable(field):
        return field(doc)
    if isinstance(field, str):
        try:
            return field.format(**doc)
        except (KeyError, IndexError, ValueError):
            return field
    return str(field)


class EvalPlusTask(ConfigurableTask):
    """Base for HumanEval+ / MBPP+ EvalPlus-aligned tasks."""

    # Subclasses MUST override with "humaneval" | "mbpp"
    DATASET_KIND: ClassVar[str] = "__unset__"

    def __init__(self, config: dict | None = None) -> None:
        if config is not None:
            nf = config.get("num_fewshot", 0) if hasattr(config, "get") else 0
            try:
                nf_i = int(nf or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "EvalPlus tasks expect num_fewshot=0; could not coerce num_fewshot"
                ) from exc
            if nf_i != 0:
                raise ValueError(
                    f"{self.__class__.__name__}: num_fewshot must be 0 (EvalPlus is 0-shot). "
                    f"Got num_fewshot={nf!r}."
                )

        dk = getattr(self.__class__, "DATASET_KIND", None)
        if dk in ("__unset__", "", None) or dk not in ("humaneval", "mbpp"):
            raise RuntimeError(
                f"{self.__class__.__name__}: class attribute DATASET_KIND must be "
                "'humaneval' or 'mbpp'."
            )

        # TaskFactory passes the full YAML dict including `class:` for PY_TASK routing;
        # lm_eval TaskConfig does not accept that key — strip without mutating `config`.
        cfg_for_super = config
        if isinstance(config, dict) and "class" in config:
            cfg_for_super = {k: v for k, v in config.items() if k != "class"}

        super().__init__(config=cfg_for_super)

    # --- lm-eval safeguards matching EvalPlus reference behavior ---

    def fewshot_context(
        self,
        doc: dict,
        num_fewshot: int,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Callable[..., str] | None = None,
        gen_prefix: str | None = None,
    ) -> str | list[str]:
        if apply_chat_template:
            raise RuntimeError(
                f"{type(self).__name__}: do not pass --apply_chat_template. "
                "Prompt is already formatted via EvalPlus make_raw_chat_prompt + tokenizer.chat_template."
            )
        if num_fewshot:
            raise RuntimeError(
                f"{type(self).__name__}: EvalPlus prompting is 0-shot; got num_fewshot={num_fewshot}."
            )
        if gen_prefix:
            raise RuntimeError(
                f"{type(self).__name__}: gen_prefix is incompatible (assistant prefix lives in doc_to_text). "
                f"Got {gen_prefix!r}"
            )
        si = system_instruction.strip() if isinstance(system_instruction, str) else ""
        if si:
            raise RuntimeError(
                f"{type(self).__name__}: refuse non-empty system_instruction={system_instruction!r} "
                "(would prepend text outside EvalPlus-codegen layout)."
            )
        description = (_resolve_config_field(doc, self.config.description) or "").strip()
        if description:
            raise RuntimeError(
                f"{type(self).__name__}: task `description` would perturb the EvalPlus prompt ({description[:80]}...)."
            )

        return super().fewshot_context(
            doc,
            num_fewshot=num_fewshot,
            system_instruction=None,
            apply_chat_template=False,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=None,
            gen_prefix=None,
        )

    def doc_to_text(self, doc, doc_to_text=None):  # noqa: ARG002
        name = _tokenizer_name(self.config)
        tok = load_tokenizer(name)
        return make_raw_chat_prompt(
            doc["prompt"].strip(),
            INSTRUCTION_PREFIX,
            RESPONSE_PREFIX,
            tok,
        )

    def process_results(self, doc: dict, results: list[list[str]]) -> dict[str, float]:
        """Judge sanitized solutions with EvalPlus check_correctness (base + plus)."""
        if not results or not results[0]:
            eval_logger.warning("Empty filtered results — scoring as failures.")
            return {"pass_at_1_base": 0.0, "pass_at_1_plus": 0.0}

        preds = results[0]
        if len(preds) != 1:
            eval_logger.warning(
                "Got %d repeats; EvalPlus leaderboard is pass@1 per task — using preds[0] only.",
                len(preds),
            )
        solution = preds[0]

        base_ok, plus_ok = judge_one(
            self.DATASET_KIND,
            doc,
            solution,
            fast_check=True,
            base_only=False,
        )
        return {
            "pass_at_1_base": float(base_ok),
            "pass_at_1_plus": float(plus_ok),
        }


__all__ = ["EvalPlusTask", "load_tokenizer", "INSTRUCTION_PREFIX", "RESPONSE_PREFIX"]
