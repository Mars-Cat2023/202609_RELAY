# SPDX-License-Identifier: Apache-2.0
"""MBPP+ evaluated with EvalPlus prompting + sanitize + harness judge."""

from __future__ import annotations

import sys
from pathlib import Path

from evalplus.data import get_mbpp_plus

_V2_ROOT = Path(__file__).resolve().parents[2]
if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))

from lm_eval_tasks._evalplus_common.slim_rows import evalplus_test_split  # noqa: E402
from lm_eval_tasks._evalplus_common.task import EvalPlusTask  # noqa: E402


class MbppPlusEvalPlus(EvalPlusTask):
    """EvalPlus-aligned MBPP+ task."""

    DATASET_KIND = "mbpp"


def load_mbpp_plus_dataset(**_: object) -> dict[str, object]:
    return evalplus_test_split(get_mbpp_plus())


__all__ = ["MbppPlusEvalPlus", "load_mbpp_plus_dataset"]
