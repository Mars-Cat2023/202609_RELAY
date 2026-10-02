#!/usr/bin/env python
"""Summarize Gaussian Streaming LoRA-SFT across training and evaluation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


METHOD = "gaussian_streaming_lora_sft"
METRICS = (
    "avg_at_8",
    "pass_at_8",
    "mixed_reward_group_rate",
    "all_fail_group_rate",
    "all_success_group_rate",
    "mean_distinct_answers",
    "effective_decision_steps",
    "actual_nfe_per_rollout",
)


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def load_result(
    output_prefix: str,
    evaluation_directory: str,
    train_seed: int,
    method: str,
    sigma: float,
    eval_seed: int,
) -> dict:
    path = (
        Path(f"{output_prefix}{train_seed}")
        / evaluation_directory
        / f"{method}_sigma{sigma:g}_seed{eval_seed}"
        / "evaluation.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation result: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--evaluation-directory", default="matched_evaluation")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"{args.output_prefix}summary_3trainseeds.json")

    report = {
        "method": METHOD,
        "train_seeds": args.train_seeds,
        "eval_seeds": args.eval_seeds,
        "aggregation": (
            "Average evaluation seeds within each trained adapter, then report "
            "mean and sample standard deviation across training seeds. The baseline "
            "from the first training-seed directory is used once as the canonical baseline."
        ),
        "configurations": {},
    }

    for sigma in args.sigmas:
        canonical_baseline = [
            load_result(
                args.output_prefix,
                args.evaluation_directory,
                args.train_seeds[0],
                "pretrained",
                sigma,
                eval_seed,
            )
            for eval_seed in args.eval_seeds
        ]
        adapter_by_train_seed = {
            train_seed: [
                load_result(
                    args.output_prefix,
                    args.evaluation_directory,
                    train_seed,
                    METHOD,
                    sigma,
                    eval_seed,
                )
                for eval_seed in args.eval_seeds
            ]
            for train_seed in args.train_seeds
        }

        configuration: dict = {
            "canonical_baseline_training_seed": args.train_seeds[0],
            "pretrained_across_evaluation_seeds": {},
            "gaussian_streaming_sft_across_training_seeds": {},
            "paired_delta_across_training_seeds": {},
            "per_training_seed": {},
        }
        for metric in METRICS:
            baseline_values = [float(row[metric]) for row in canonical_baseline]
            adapter_train_means = []
            delta_train_means = []
            for train_seed in args.train_seeds:
                adapter_values = [
                    float(row[metric]) for row in adapter_by_train_seed[train_seed]
                ]
                deltas = [
                    adapter - baseline
                    for adapter, baseline in zip(
                        adapter_values, baseline_values, strict=True
                    )
                ]
                adapter_train_means.append(statistics.mean(adapter_values))
                delta_train_means.append(statistics.mean(deltas))
                configuration["per_training_seed"].setdefault(str(train_seed), {})[
                    metric
                ] = {
                    "adapter_across_evaluation_seeds": mean_std(adapter_values),
                    "paired_delta_across_evaluation_seeds": mean_std(deltas),
                }

            configuration["pretrained_across_evaluation_seeds"][metric] = mean_std(
                baseline_values
            )
            configuration["gaussian_streaming_sft_across_training_seeds"][metric] = (
                mean_std(adapter_train_means)
            )
            configuration["paired_delta_across_training_seeds"][metric] = mean_std(
                delta_train_means
            )

        report["configurations"][f"sigma={sigma:g}"] = configuration

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    for sigma_key, configuration in report["configurations"].items():
        print(f"\n{sigma_key}")
        print("Per-training-seed means across evaluation seeds")
        print(
            f"{'Metric':<31} {'Train seed':>10} {'Pretrained':>13} "
            f"{'Gaussian SFT':>15} {'Delta':>11}"
        )
        print("-" * 86)
        for metric in METRICS:
            baseline = configuration["pretrained_across_evaluation_seeds"][metric]["mean"]
            for train_seed in args.train_seeds:
                seed_row = configuration["per_training_seed"][str(train_seed)][metric]
                adapter = seed_row["adapter_across_evaluation_seeds"]["mean"]
                delta = seed_row["paired_delta_across_evaluation_seeds"]["mean"]
                print(
                    f"{metric:<31} {train_seed:>10d} {baseline:>13.4f} "
                    f"{adapter:>15.4f} {delta:>+11.4f}"
                )

        print("\nPrimary aggregate")
        print(
            f"{'Metric':<31} {'Pretrained eval mean +/- std':>28} "
            f"{'Gaussian SFT mean +/- std':>28} {'Delta mean +/- std':>24}"
        )
        print("-" * 117)
        for metric in METRICS:
            baseline = configuration["pretrained_across_evaluation_seeds"][metric]
            adapter = configuration[
                "gaussian_streaming_sft_across_training_seeds"
            ][metric]
            delta = configuration["paired_delta_across_training_seeds"][metric]
            print(
                f"{metric:<31} "
                f"{baseline['mean']:.4f} +/- {baseline['std']:.4f}".rjust(28)
                + " "
                + f"{adapter['mean']:.4f} +/- {adapter['std']:.4f}".rjust(28)
                + " "
                + f"{delta['mean']:+.4f} +/- {delta['std']:.4f}".rjust(24)
            )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
