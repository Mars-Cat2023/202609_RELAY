#!/usr/bin/env python3
"""Plot val/prediction metrics vs threshold for sudoku_extreme bptt vs no_bptt runs."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
BPTT_PREFIX = "sudoku_extreme_loopholing_bptt_threshold_"
NO_BPTT_PREFIX = "sudoku_extreme_loopholing_no_bptt_threshold_"

METRICS = {
    "exact_match": ("val/prediction/exact_match", "Exact Match"),
    "token_accuracy": ("val/prediction/token_accuracy", "Token Accuracy"),
    "rollout_steps": ("val/prediction/rollout_steps", "Rollout Steps"),
}


def parse_threshold_from_name(name: str) -> float | None:
    """Extract threshold from run name like sudoku_extreme_bptt_threshold_0.15 or ..._0.15_rollout_2."""
    for prefix in (BPTT_PREFIX, NO_BPTT_PREFIX):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            # Handle suffix like "0.15" or "0.15_rollout_2"
            match = re.match(r"([\d.]+)", suffix)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
    return None


def is_bptt_run(name: str) -> bool:
    """Check if run is BPTT variant by name."""
    return name.startswith(BPTT_PREFIX)


def _get_metric(run, metric_key: str, search_substr: str) -> float | None:
    """Get a single metric from run summary or history."""
    val = None
    if run.summary:
        val = run.summary.get(metric_key)
    if val is None and run.summary:
        for k in run.summary:
            if search_substr in k.lower() and "val" in k.lower():
                val = run.summary[k]
                break
    if val is None:
        try:
            hist = run.history(keys=[metric_key])
            if not hist.empty:
                val = hist[metric_key].dropna().iloc[-1]
        except Exception:
            pass
    return float(val) if val is not None else None


def fetch_runs(
    api: wandb.Api, entity: str, project: str, metric_key: str
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Fetch runs and return (bptt_data, no_bptt_data) as [(threshold, value), ...]."""
    runs = api.runs(f"{entity}/{project}")
    bptt_data: list[tuple[float, float]] = []
    no_bptt_data: list[tuple[float, float]] = []

    search_substr = metric_key.split("/")[-1]

    for run in runs:
        name = run.name or ""
        threshold = parse_threshold_from_name(name)
        if threshold is None:
            continue

        val = _get_metric(run, metric_key, search_substr)
        if val is None:
            print(f"Warning: No {metric_key} for run {name} (id={run.id})")
            continue

        row = (threshold, val)
        if is_bptt_run(name):
            bptt_data.append(row)
        else:
            no_bptt_data.append(row)

    return bptt_data, no_bptt_data


def fetch_runs_scatter(
    api: wandb.Api, entity: str, project: str
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Fetch runs with (rollout_steps, exact_match, threshold) for scatter plot."""
    runs = api.runs(f"{entity}/{project}")
    bptt_data: list[tuple[float, float, float]] = []
    no_bptt_data: list[tuple[float, float, float]] = []

    x_key = "val/prediction/rollout_steps"
    y_key = "val/prediction/exact_match"

    for run in runs:
        name = run.name or ""
        threshold = parse_threshold_from_name(name)
        if threshold is None:
            continue

        x_val = _get_metric(run, x_key, "rollout_steps")
        y_val = _get_metric(run, y_key, "exact_match")
        if x_val is None or y_val is None:
            print(f"Warning: Missing metric for run {name} (id={run.id})")
            continue

        row = (x_val, y_val, threshold)
        if is_bptt_run(name):
            bptt_data.append(row)
        else:
            no_bptt_data.append(row)

    return bptt_data, no_bptt_data


def plot(
    bptt_data: list[tuple[float, float]],
    no_bptt_data: list[tuple[float, float]],
    out_path: Path,
    y_label: str,
    title_suffix: str,
) -> None:
    """Draw two curves and save."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if bptt_data:
        bptt_data.sort(key=lambda x: x[0])
        xs, ys = zip(*bptt_data)
        ax.plot(xs, ys, "o-", label="BPTT", color="C0", linewidth=2, markersize=8)
    if no_bptt_data:
        no_bptt_data.sort(key=lambda x: x[0])
        xs, ys = zip(*no_bptt_data)
        ax.plot(xs, ys, "s-", label="No BPTT", color="C1", linewidth=2, markersize=8)

    ax.set_xlabel("Threshold")
    ax.set_ylabel(y_label)
    ax.set_title(f"Sudoku Extreme: {title_suffix} vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(THRESHOLDS)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_scatter(
    bptt_data: list[tuple[float, float, float]],
    no_bptt_data: list[tuple[float, float, float]],
    out_path: Path,
) -> None:
    """Scatter plot: rollout_steps (x) vs exact_match (y), annotated with threshold."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if bptt_data:
        xs, ys, threshs = zip(*bptt_data)
        ax.scatter(xs, ys, label="BPTT", color="C0", s=80, zorder=3)
        for x, y, t in bptt_data:
            ax.annotate(str(t), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9, color="C0")
    if no_bptt_data:
        xs, ys, threshs = zip(*no_bptt_data)
        ax.scatter(xs, ys, label="No BPTT", color="C1", s=80, zorder=3)
        for x, y, t in no_bptt_data:
            ax.annotate(str(t), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9, color="C1")

    ax.set_xlabel("Rollout Steps")
    ax.set_ylabel("Exact Match")
    ax.set_title("Sudoku Extreme: Exact Match vs Rollout Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BPTT vs threshold from wandb")
    parser.add_argument("--entity", default=None, help="Wandb entity (default: from env)")
    parser.add_argument("--project", default="BPTT", help="Wandb project")
    parser.add_argument(
        "--prefix",
        default="sudoku_extreme_loopholing",
        help="Run name prefix before _bptt_threshold_ / _no_bptt_threshold_ (default: sudoku_extreme_loopholing)",
    )
    parser.add_argument(
        "--metric",
        choices=list(METRICS),
        default="exact_match",
        help="Metric to plot (default: exact_match)",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Scatter plot: rollout_steps (x) vs exact_match (y)",
    )
    parser.add_argument("-o", "--output", default=None, help="Output path (default: bptt_threshold_{metric}.png)")
    args = parser.parse_args()

    global BPTT_PREFIX, NO_BPTT_PREFIX
    BPTT_PREFIX = f"{args.prefix}_bptt_threshold_"
    NO_BPTT_PREFIX = f"{args.prefix}_no_bptt_threshold_"

    api = wandb.Api()
    entity = args.entity
    if not entity:
        try:
            entity = wandb.api.viewer()["entity"]
        except Exception:
            entity = "ilm-extensions"  # fallback

    if args.scatter:
        out_path = Path(args.output or "bptt_exact_match_vs_rollout_steps.png")
        bptt_data, no_bptt_data = fetch_runs_scatter(api, entity, args.project)
        print(f"Found {len(bptt_data)} BPTT runs, {len(no_bptt_data)} No-BPTT runs")
        if bptt_data:
            print("BPTT (rollout_steps, exact_match):", bptt_data)
        if no_bptt_data:
            print("No-BPTT (rollout_steps, exact_match):", no_bptt_data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_scatter(bptt_data, no_bptt_data, out_path)
    else:
        metric_key, y_label = METRICS[args.metric]
        out_path = Path(args.output or f"bptt_threshold_{args.metric}.png")
        bptt_data, no_bptt_data = fetch_runs(api, entity, args.project, metric_key)
        print(f"Found {len(bptt_data)} BPTT runs, {len(no_bptt_data)} No-BPTT runs")
        if bptt_data:
            print("BPTT:", bptt_data)
        if no_bptt_data:
            print("No-BPTT:", no_bptt_data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot(bptt_data, no_bptt_data, out_path, y_label, y_label)


if __name__ == "__main__":
    main()
