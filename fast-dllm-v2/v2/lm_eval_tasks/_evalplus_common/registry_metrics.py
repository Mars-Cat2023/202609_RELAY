# SPDX-License-Identifier: Apache-2.0
"""Register metric names used by humaneval_plus_evalplus / mbpp_plus_evalplus YAMLs.

The Harness still calls `ConfigurableTask(process_results=...)` for aggregation metadata;
actual scores come from ``EvalPlusTask.process_results``.
"""

from __future__ import annotations

#
# Ensure the "mean" aggregation exists without importing lm_eval.api.metrics,
# which pulls optional deps (e.g. sacrebleu) users may omit.
#
from lm_eval.api.registry import (
    AGGREGATION_REGISTRY,
    register_aggregation,
    register_metric,
)

if "mean" not in AGGREGATION_REGISTRY:
    @register_aggregation("mean")
    def _mean_fallback(arr):  # type: ignore[misc]
        return sum(arr) / max(1, len(arr))


@register_metric(
    metric="pass_at_1_base",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def pass_at_1_base(
    references: list | None = None,
    predictions: list | None = None,
    **_: object,
):
    """Placeholder metric (scores are emitted by EvalPlusTask.process_results)."""
    _ = references, predictions
    return 0.0


@register_metric(
    metric="pass_at_1_plus",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def pass_at_1_plus(
    references: list | None = None,
    predictions: list | None = None,
    **_: object,
):
    _ = references, predictions
    return 0.0

