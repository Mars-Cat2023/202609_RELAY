# SPDX-License-Identifier: Apache-2.0
"""
Post-processing matching scripts/generate_evalplus_jsonl.py (EvalPlus sanitize + stubs).
"""

from evalplus.sanitize import sanitize


def postprocess_raw(raw: str) -> str:
    t = raw.strip()
    if "```" in t:
        t = t.split("```")[0].rstrip()
    return t


def to_evalplus_solution(raw: str, task: dict) -> str:
    entry = task["entry_point"]
    raw = postprocess_raw(raw)
    sol = sanitize(raw, entrypoint=entry)
    if not sol.strip():
        sol = sanitize(raw, entrypoint=None)
    if f"def {entry}" not in sol:
        stub = task["prompt"].rstrip()
        sol = sanitize(stub + "\n" + raw, entrypoint=entry)
    if not sol.strip() or f"def {entry}" not in sol:
        sol = (task["prompt"].rstrip() + "\n" + raw.strip()).strip()
    return sol


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """lm_eval Filter 'custom': map raw model completions to sanitized solutions."""
    return [
        [to_evalplus_solution(r, doc) for r in resp] for resp, doc in zip(resps, docs)
]
