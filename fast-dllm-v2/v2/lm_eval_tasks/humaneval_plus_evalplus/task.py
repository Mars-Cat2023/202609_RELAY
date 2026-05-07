# SPDX-License-Identifier: Apache-2.0
"""HumanEval+ evaluated with EvalPlus prompting + sanitize + harness judge."""

from __future__ import annotations

import sys
from pathlib import Path

from evalplus.data import get_human_eval_plus

# Ensure imports work when YAML loads this module by path (--include_path v2/lm_eval_tasks).
_V2_ROOT = Path(__file__).resolve().parents[2]
if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))

from lm_eval_tasks._evalplus_common.slim_rows import evalplus_test_split  # noqa: E402
from lm_eval_tasks._evalplus_common.task import EvalPlusTask  # noqa: E402


class HumanEvalPlusEvalPlus(EvalPlusTask):
    """EvalPlus-aligned HumanEval+ task."""

    DATASET_KIND = "humaneval"


def load_humaneval_plus_dataset(**_: object) -> dict[str, object]:
    """Feed lm_eval ``ConfigurableTask.download(custom_dataset=...)`` (plain dict rows)."""
    return evalplus_test_split(get_human_eval_plus())


__all__ = ["HumanEvalPlusEvalPlus", "load_humaneval_plus_dataset"]
