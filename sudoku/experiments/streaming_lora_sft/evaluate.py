#!/usr/bin/env python
"""Matched evaluation of pretrained RELAY and one-step Streaming LoRA-SFT."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sudoku"))
sys.path.insert(0, str(ROOT / "sudoku/xlm-core/src"))

import torch
from datasets import load_from_disk

from experiments.gaussian_hidden_grpo.train import evaluate, load_pretrained, sha256, tensor_hash
from relay.lora import adapters_disabled, inject_lora, load_lora_state_dict


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


@contextmanager
def adapter_mode(model, enabled: bool):
    if enabled:
        yield
    else:
        with adapters_disabled(model):
            yield


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        output[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument(
        "--dev-data", type=Path,
        default=ROOT / "data/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver/test_first2000",
    )
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--confidence-temperature", type=float, default=2.0)
    parser.add_argument("--token-temperature", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-rollout-steps", type=int, default=64)
    parser.add_argument("--eval-prompt-batch", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    if args.group_size != 8:
        parser.error("This report uses avg@8/pass@8 and therefore requires group-size=8")
    if args.lora_rank != 32 or args.lora_alpha != 64:
        parser.error("This baseline is locked to LoRA rank=32 and alpha=64")
    if any(sigma < 0 for sigma in args.sigmas):
        parser.error("sigmas must be nonnegative")
    if args.dev_size < 1:
        parser.error("dev-size must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": sha256(args.adapter),
        "weights": args.weights,
        "dev_data": str(args.dev_data.resolve()),
        "dev_size": args.dev_size,
        "eval_seeds": args.eval_seeds,
        "sigmas": args.sigmas,
        "confidence_temperature": args.confidence_temperature,
        "token_temperature": args.token_temperature,
        "threshold": args.threshold,
        "group_size": args.group_size,
        "max_rollout_steps": args.max_rollout_steps,
        "precision": args.precision,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }
    evaluation_manifest_path = args.output / "evaluation_manifest.json"
    if evaluation_manifest_path.exists():
        previous_manifest = json.loads(evaluation_manifest_path.read_text())
        if previous_manifest != evaluation_manifest:
            raise ValueError(
                "Evaluation output contains results from a different configuration"
            )
    else:
        evaluation_manifest_path.write_text(
            json.dumps(evaluation_manifest, indent=2) + "\n"
        )

    device = torch.device(args.device)
    model, tokenizer, _, _ = load_pretrained(args.checkpoint, args.weights, device)
    inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    adapter_args = adapter.get("arguments", {})
    if int(adapter_args.get("lora_rank", args.lora_rank)) != args.lora_rank:
        raise ValueError("Adapter rank does not match evaluation rank")
    if float(adapter_args.get("lora_alpha", args.lora_alpha)) != args.lora_alpha:
        raise ValueError("Adapter alpha does not match evaluation alpha")
    frozen_hash = tensor_hash(
        (name, parameter) for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    if adapter.get("frozen_base_sha256") != frozen_hash:
        raise ValueError("Adapter was not trained from this frozen pretrained model")
    load_lora_state_dict(model, adapter["lora_state_dict"])
    model.eval()

    dataset = load_from_disk(str(args.dev_data))
    if args.dev_size > len(dataset):
        raise ValueError("Requested development slice is out of bounds")
    dataset = dataset.select(range(args.dev_size))

    all_rows: list[dict] = []
    for sigma in args.sigmas:
        for eval_seed in args.eval_seeds:
            # The two methods use the same seed and decoding configuration.
            for method, enabled in (("pretrained", False), ("streaming_lora_sft", True)):
                result_dir = args.output / f"{method}_sigma{sigma:g}_seed{eval_seed}"
                result_path = result_dir / "evaluation.json"
                if result_path.exists():
                    row = json.loads(result_path.read_text())
                else:
                    result_dir.mkdir(parents=True, exist_ok=True)
                    args.sigma = sigma
                    with adapter_mode(model, enabled):
                        metrics = evaluate(
                            model, dataset, args, tokenizer, seed=20_000 + eval_seed
                        )
                    row = {
                        "method": method,
                        "sigma": sigma,
                        "evaluation_seed": eval_seed,
                        "dev_size": args.dev_size,
                        "confidence_temperature": args.confidence_temperature,
                        "token_temperature": args.token_temperature,
                        **metrics,
                    }
                    result_path.write_text(json.dumps(row, indent=2) + "\n")
                all_rows.append(row)
                print(json.dumps(row), flush=True)

    summary: dict = {"configurations": {}}
    for sigma in args.sigmas:
        sigma_key = f"sigma={sigma:g}"
        baseline_rows = [
            row for row in all_rows
            if row["method"] == "pretrained" and float(row["sigma"]) == sigma
        ]
        sft_rows = [
            row for row in all_rows
            if row["method"] == "streaming_lora_sft" and float(row["sigma"]) == sigma
        ]
        baseline = aggregate(baseline_rows)
        sft = aggregate(sft_rows)
        paired_delta = {}
        base_by_seed = {row["evaluation_seed"]: row for row in baseline_rows}
        sft_by_seed = {row["evaluation_seed"]: row for row in sft_rows}
        for metric in METRICS:
            deltas = [
                float(sft_by_seed[seed][metric]) - float(base_by_seed[seed][metric])
                for seed in args.eval_seeds
            ]
            paired_delta[metric] = {
                "mean": statistics.mean(deltas),
                "std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "values": deltas,
            }
        summary["configurations"][sigma_key] = {
            "pretrained": baseline,
            "streaming_lora_sft": sft,
            "paired_delta_sft_minus_pretrained": paired_delta,
        }

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    for sigma_key, values in summary["configurations"].items():
        print(f"\n{sigma_key}")
        print(f"{'Metric':<30} {'Pretrained':>12} {'Streaming SFT':>14} {'Delta':>12}")
        for metric in METRICS:
            base = values["pretrained"][metric]["mean"]
            sft = values["streaming_lora_sft"][metric]["mean"]
            delta = values["paired_delta_sft_minus_pretrained"][metric]["mean"]
            print(f"{metric:<30} {base:>12.4f} {sft:>14.4f} {delta:>+12.4f}")


if __name__ == "__main__":
    main()
