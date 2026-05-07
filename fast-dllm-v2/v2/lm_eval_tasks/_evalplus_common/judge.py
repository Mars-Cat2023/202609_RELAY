# SPDX-License-Identifier: Apache-2.0
"""EvalPlus in-process judging (official check_correctness + ground-truth cache)."""

from __future__ import annotations

import threading
from typing import Literal

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.eval import PASS
from evalplus.evaluate import check_correctness, get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

_bundle_lock = threading.Lock()
_problem_cache: dict[str, dict[str, object]] = {}
_expected_cache: dict[str, dict[str, object]] = {}


DatasetKind = Literal["humaneval", "mbpp"]


def _load_problems(kind: DatasetKind) -> dict[str, object]:
    if kind == "humaneval":
        return get_human_eval_plus()
    if kind == "mbpp":
        return get_mbpp_plus()
    raise ValueError(kind)


def _dataset_hash(kind: DatasetKind) -> str:
    if kind == "humaneval":
        return get_human_eval_plus_hash(version="default")
    if kind == "mbpp":
        return get_mbpp_plus_hash(version="default")
    raise ValueError(kind)


def _oracle_entrypoints(kind: DatasetKind) -> list:
    if kind == "mbpp":
        return MBPP_OUTPUT_NOT_NONE_TASKS
    return []


def get_expected_outputs(
    kind: DatasetKind,
) -> tuple[dict[str, object], dict[str, object]]:
    """Problems dict keyed by task_id plus expected oracle outputs (cached)."""
    with _bundle_lock:
        if kind not in _problem_cache:
            probs = _load_problems(kind)
            h = _dataset_hash(kind)
            exp = get_groundtruth(probs, h, _oracle_entrypoints(kind))
            _problem_cache[kind] = probs
            _expected_cache[kind] = exp
        return _problem_cache[kind], _expected_cache[kind]


def judge_one(
    kind: DatasetKind,
    doc: dict,
    solution: str,
    *,
    fast_check: bool = True,
    min_time_limit: float | None = None,
    gt_time_limit_factor: float | None = None,
    base_only: bool = False,
) -> tuple[bool, bool]:
    """
    Run EvalPlus correctness on one sample. Returns (pass_base, pass_plus).
    When base_only=True, plus is reported as False.
    """
    _problems, expected = get_expected_outputs(kind)
    tid = doc["task_id"]
    problem = _problems[tid]

    kw: dict[str, object] = {
        "base_only": base_only,
        "fast_check": fast_check,
        "identifier": f"{tid}-lm_eval",
    }
    if min_time_limit is not None:
        kw["min_time_limit"] = min_time_limit
    if gt_time_limit_factor is not None:
        kw["gt_time_limit_factor"] = gt_time_limit_factor

    ret = check_correctness(
        kind,
        0,
        problem,
        solution,
        expected[tid],
        **kw,
    )
    base_pass = ret["base"][0] == PASS
    if base_only:
        return base_pass, False
    plus_pass = ret["plus"][0] == PASS  # noqa: SLF001 — plus always present unless base_only
    return base_pass, plus_pass


__all__ = ["get_expected_outputs", "judge_one", "DatasetKind"]
