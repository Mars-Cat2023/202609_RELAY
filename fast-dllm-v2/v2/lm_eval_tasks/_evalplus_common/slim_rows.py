# SPDX-License-Identifier: Apache-2.0
"""lm-eval–friendly doc lists for EvalPlus (plain dict rows, no HF ``datasets``)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Mapping


class _FeatureKeys:
    """Minimal stand-in for ``datasets.Features``: only ``keys()`` is required by lm-eval."""

    __slots__ = ("_cols",)

    def __init__(self, cols: tuple[str, ...]) -> None:
        self._cols = cols

    def keys(self) -> tuple[str, ...]:
        return self._cols


class LmEvalDocList(Sequence[dict[str, str]]):
    """
    ``list[dict]`` that exposes ``.features.keys()`` so ``ConfigurableTask`` can build
    ``self.features`` without converting rows to Arrow / ``datasets.Dataset``.
    """

    __slots__ = ("_rows", "features")

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows
        cols = tuple(rows[0].keys()) if rows else ()
        self.features = _FeatureKeys(cols)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self._rows)


def evalplus_lm_eval_rows(problems: Mapping[str, dict]) -> list[dict[str, str]]:
    """
    Slim string-keyed rows for lm-eval (prompt / entry_point / task_id).

    Full EvalPlus dicts contain nested ``base_input`` / ``plus_input``; judging uses
    canonical problems from ``judge.get_expected_outputs``.
    """
    rows: list[dict[str, str]] = []
    for task in problems.values():
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "prompt": str(task["prompt"]),
                "entry_point": str(task["entry_point"]),
            }
        )
    return rows


def evalplus_test_split(problems: Mapping[str, dict]) -> dict[str, LmEvalDocList]:
    """Return ``{\"test\": LmEvalDocList(...)}`` for ``custom_dataset`` loaders."""
    return {"test": LmEvalDocList(evalplus_lm_eval_rows(problems))}


__all__ = ["LmEvalDocList", "evalplus_lm_eval_rows", "evalplus_test_split"]
