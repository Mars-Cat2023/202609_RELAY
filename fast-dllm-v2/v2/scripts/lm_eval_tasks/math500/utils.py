"""Custom MATH-500 task for lm-evaluation-harness.

Uses ``minerva_math``'s answer-equivalence utilities (``is_equiv`` /
``normalize_final_answer``) and the ``math_verify`` sympy backend, but
wires up a *boxed-answer* prompt + parser instead of the original
4-shot Minerva tail (``"Final Answer: The final answer is X. I hope it
is correct."``).

Why we diverge from ``minerva_math.utils.process_results``:
  The Minerva parser regexes the few-shot tail above. With our default
  ``num_fewshot=0`` and a chat template, the model never emits that
  sentence, so the regex returns ``[invalidanswer]`` and ``exact_match``
  is identically 0 — even when ``math_verify`` (which can recover
  ``\\boxed{…}``) reports a non-trivial score. Modern math evaluations
  (and stateflow's ``math500_reasoning`` setup) all key off the
  ``\\boxed{...}`` final-answer convention, which is also what every
  Fast-dLLM v2 / LLaDA prompt asks the model to produce.

So this module:
  * tells the model — in the user turn — to put its final answer in
    ``\\boxed{…}``;
  * extracts the last ``\\boxed{…}`` from the generation for
    ``exact_match``;
  * still falls back to the Minerva regex if the model emits the
    ``Final Answer: …`` tail (covers ``--num_fewshot 4`` ablations);
  * keeps the sympy-based ``math_verify`` metric as-is.
"""

from typing import Dict, List, Optional

import datasets

from lm_eval.tasks.minerva_math.utils import (  # type: ignore
    get_unnormalized_answer,
    is_equiv,
    last_boxed_only_string,
    list_fewshot_samples,
    normalize_final_answer,
    remove_boxed,
)
from math_verify import parse, verify  # type: ignore

__all__ = [
    "doc_to_text",
    "process_docs",
    "process_results",
    "list_fewshot_samples",
]


# Same instruction Stateflow / Fast-dLLM v2 use elsewhere (see
# ``v2/eval.py`` minerva_math / gsm8k branches). We inject it directly
# into ``doc_to_text`` because the math500 task name doesn't trigger
# eval.py's per-task prompt rewrite.
INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def doc_to_text(doc: dict) -> str:
    return f"{doc['problem']}\n\n{INSTRUCTION}"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        if doc.get("answer") is not None:
            answer = normalize_final_answer(str(doc["answer"]))
        else:
            answer = normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            )
        out = {
            "problem": doc["problem"],
            "solution": doc.get("solution", ""),
            "answer": answer,
        }
        if doc.get("few_shot") is not None:
            out["few_shot"] = True
        return out

    return dataset.map(_process_doc)


def _extract_boxed(text: str) -> Optional[str]:
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except Exception:
        return None


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    boxed = _extract_boxed(candidates)
    if boxed is not None:
        answer = normalize_final_answer(boxed)
    else:
        answer = normalize_final_answer(get_unnormalized_answer(candidates))

    em = 1 if is_equiv(answer, doc["answer"]) else 0
    try:
        mv = 1 if verify(parse(doc["answer"]), parse(candidates)) else 0
    except Exception:
        mv = 0
    return {"exact_match": em, "math_verify": mv}
