#!/usr/bin/env python
"""LoRA post-training for RELAY: SFT, Gaussian SFT, or Gaussian-hidden GRPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sudoku"))
sys.path.insert(0, str(ROOT / "sudoku/xlm-core/src"))

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from relay.gaussian_grpo import collect_rollouts, grpo_update_epoch, supervised_lora_loss
from relay.lora import (
    assert_only_lora_trainable,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
)
from relay.model import RotaryTransformerRelayModel
from xlm.datamodule import SimpleSpaceTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_pretrained(path: Path, weights: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]["model"]
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=False)
    config = dict(config)
    target = config.pop("_target_")
    if target != "relay.model.RotaryTransformerRelayModel":
        raise ValueError(f"Expected RELAY checkpoint, found {target}")
    tokenizer = SimpleSpaceTokenizer.for_numbers(vocab_size=10)
    config.update(
        num_embeddings=tokenizer.full_vocab_size,
        padding_idx=tokenizer.pad_token_id,
        mask_idx=tokenizer.mask_token_id,
    )
    model = RotaryTransformerRelayModel(**config)
    state = {
        name.removeprefix("model."): tensor
        for name, tensor in checkpoint["state_dict"].items()
        if name.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    if weights == "ema":
        shadows = checkpoint["ema"]["shadow_params"]
        parameters = list(model.parameters())
        if len(shadows) != len(parameters):
            raise ValueError("EMA parameter count does not match actor")
        with torch.no_grad():
            for parameter, shadow in zip(parameters, shadows):
                if parameter.shape != shadow.shape:
                    raise ValueError("EMA parameter shape mismatch")
                parameter.copy_(shadow)
    model.eval().to(device)
    source_global_step = int(checkpoint.get("global_step", -1))
    del checkpoint
    return model, tokenizer, source_global_step, config


def collate(rows: List[dict], *, mask_token_id: int) -> Dict[str, torch.Tensor]:
    prompt = torch.tensor([row["prompt_token_ids"] for row in rows], dtype=torch.long)
    target = torch.tensor([row["input_token_ids"] for row in rows], dtype=torch.long)
    return {
        "prompt_ids": prompt,
        "target_ids": target,
        "fixed": prompt != mask_token_id,
    }


def sample_train_batch(dataset, *, batch_size: int, mask_token_id: int, seed: int):
    """Sample a step-local batch so a resumed run reproduces every future batch."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size].tolist()
    return collate(
        [dataset[index] for index in indices], mask_token_id=mask_token_id
    )


def prepare_data(args, tokenizer):
    train = load_from_disk(str(args.train_data))
    dev = load_from_disk(str(args.dev_data))
    if args.train_start < 0 or args.train_size < 1 or args.train_start + args.train_size > len(train):
        raise ValueError("Requested training slice is out of bounds")
    if args.dev_size < 1 or args.dev_size > len(dev):
        raise ValueError("Requested development slice is out of bounds")
    train = train.select(range(args.train_start, args.train_start + args.train_size))
    dev = dev.select(range(args.dev_size))
    dev_questions = {row["question"].replace(".", "0") for row in dev}
    overlap = [
        index for index, question in enumerate(train["question"])
        if question.replace(".", "0") in dev_questions
    ]
    if overlap:
        raise ValueError(f"Training/development puzzle overlap at {len(overlap)} rows")
    mask_id = tokenizer.mask_token_id
    for name, dataset in (("train", train), ("dev", dev)):
        if any(len(ids) != 81 for ids in dataset["prompt_token_ids"]):
            raise ValueError(f"Malformed {name} prompt")
        if any(len(ids) != 81 for ids in dataset["input_token_ids"]):
            raise ValueError(f"Malformed {name} target")
        if not any(mask_id in ids for ids in dataset["prompt_token_ids"]):
            raise ValueError(f"No masked puzzles in {name}")
    return train, dev


def dataset_manifest(dataset, *, path: Path, start: int, size: int) -> dict:
    identity = hashlib.sha256()
    for row in dataset:
        identity.update(row["question"].encode())
        identity.update(b"\0")
        identity.update(row["answer"].encode())
        identity.update(b"\n")
    return {
        "path": str(path.resolve()), "start": start, "size": size,
        "fingerprint": dataset._fingerprint, "identity_sha256": identity.hexdigest(),
    }


@torch.inference_mode()
def evaluate(model, dataset, args, tokenizer, seed: int) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=args.eval_prompt_batch,
        shuffle=False,
        collate_fn=partial(collate, mask_token_id=tokenizer.mask_token_id),
    )
    generator = torch.Generator(device=args.device).manual_seed(seed)
    rewards, distinct = [], []
    effective_steps = 0
    actual_nfe = 0
    for batch in loader:
        rollout = collect_rollouts(
            model, prompt_ids=batch["prompt_ids"], target_ids=batch["target_ids"],
            fixed=batch["fixed"], mask_token_id=tokenizer.mask_token_id,
            group_size=args.group_size, sigma=args.sigma,
            confidence_temperature=args.confidence_temperature,
            token_temperature=args.token_temperature, threshold=args.threshold,
            max_steps=args.max_rollout_steps, generator=generator,
            autocast_dtype=torch.bfloat16 if args.precision == "bf16" else None,
            pin_memory=False,
        )
        group_rewards = rollout.rewards.float()
        rewards.append(group_rewards)
        final_ids = rollout.final_ids.view(len(group_rewards), args.group_size, -1)
        distinct.extend(len({tuple(row.tolist()) for row in group}) for group in final_ids)
        effective_steps += sum(step.valid.sum().item() for step in rollout.steps)
        actual_nfe += len(rollout.steps) * group_rewards.numel()
    result = torch.cat(rewards)
    successes_per_puzzle = result.sum(1).to(torch.int64)
    binary = (result == 0) | (result == 1)
    if not torch.all(binary):
        invalid = result[~binary]
        raise RuntimeError(
            "Evaluation rewards must be binary; "
            f"found {invalid.numel()} invalid values, "
            f"sample={invalid[:16].tolist()}"
        )
    success_histogram = torch.bincount(
        successes_per_puzzle, minlength=args.group_size + 1
    )
    if int(success_histogram.sum()) != len(result):
        raise RuntimeError("Success-count histogram does not cover every puzzle")
    pass_rate = (successes_per_puzzle > 0).float().mean()
    mixed_rate = (
        (successes_per_puzzle > 0) & (successes_per_puzzle < args.group_size)
    ).float().mean()
    all_success_rate = (successes_per_puzzle == args.group_size).float().mean()
    if not torch.isclose(pass_rate, mixed_rate + all_success_rate, atol=1e-7):
        raise RuntimeError("Evaluation pass/mixed/all-success rates are inconsistent")
    return {
        "avg_at_8": 100 * result.mean().item(),
        "pass_at_8": 100 * pass_rate.item(),
        "mixed_reward_group_rate": 100 * mixed_rate.item(),
        "all_fail_group_rate": 100 * (successes_per_puzzle == 0).float().mean().item(),
        "all_success_group_rate": 100 * all_success_rate.item(),
        "success_count_histogram": success_histogram.tolist(),
        "mean_distinct_answers": sum(distinct) / len(distinct),
        "effective_decision_steps": effective_steps / result.numel(),
        "actual_nfe_per_rollout": actual_nfe / result.numel(),
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, args, step: int, base_hash: str):
    torch.save(
        {
            "step": step, "lora_state_dict": lora_state_dict(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "arguments": vars(args), "frozen_base_sha256": base_hash,
        }, path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=["grpo", "sft", "gaussian_sft"], required=True)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver/train")
    parser.add_argument("--dev-data", type=Path, default=ROOT / "data/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver/test_first2000")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prompt-batch", type=int, default=4)
    parser.add_argument("--eval-prompt-batch", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--max-rollout-steps", type=int, default=64)
    parser.add_argument("--sft-num-steps", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--confidence-temperature", type=float, default=1.0)
    parser.add_argument("--token-temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--resume-adapter", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.sigma <= 0:
        parser.error("This Gaussian-hidden design requires sigma > 0")
    if args.token_temperature <= 0 or args.group_size != 8:
        parser.error("token_temperature must be positive and group_size must be 8")
    if args.confidence_temperature < 0:
        parser.error("confidence_temperature must be nonnegative")
    if args.ppo_epochs < 1 or args.max_rollout_steps < 1:
        parser.error("ppo_epochs and max_rollout_steps must be positive")
    if not 0 < args.clip_epsilon < 1:
        parser.error("clip_epsilon must be in (0, 1)")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, source_global_step, model_config = load_pretrained(
        args.checkpoint, args.weights, device
    )
    lora_report = inject_lora(
        model, rank=args.lora_rank, alpha=args.lora_alpha
    )
    assert_only_lora_trainable(model)
    frozen_base_hash = tensor_hash(
        (name, parameter) for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    train_data, dev_data = prepare_data(args, tokenizer)
    train_manifest = dataset_manifest(
        train_data, path=args.train_data, start=args.train_start, size=args.train_size
    )
    dev_manifest = dataset_manifest(dev_data, path=args.dev_data, start=0, size=args.dev_size)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    def lr_factor(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return max(step + 1, 1) / args.warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_step = 0
    if args.resume_adapter:
        resume = torch.load(args.resume_adapter, map_location="cpu", weights_only=False)
        if resume["frozen_base_sha256"] != frozen_base_hash:
            raise ValueError("Resume adapter was trained from a different frozen base")
        load_lora_state_dict(model, resume["lora_state_dict"])
        if not args.eval_only:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            start_step = int(resume["step"])

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "pretrained_checkpoint_sha256": sha256(args.checkpoint),
        "pretrained_global_step": source_global_step,
        "frozen_base_sha256": frozen_base_hash,
        "model_config": model_config, "lora": asdict(lora_report),
        "train_data": train_manifest, "development_data": dev_manifest,
        "position_treatment": "EGSPO-style: old-rollout U_t is fixed in the surrogate ratio",
        "ratio": "one clip on token_ratio * Gaussian_hidden_ratio",
        "advantage": "binary exact-match R_j minus group mean; terminal advantage broadcast to valid steps",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metrics_path = args.output / "metrics.jsonl"
    if args.eval_only:
        metrics = evaluate(model, dev_data, args, tokenizer, args.seed + 20_000)
        (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
        manifest["status"] = "complete"
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
        return

    initial_frozen_hash = frozen_base_hash
    for step in range(start_step + 1, args.steps + 1):
        batch = sample_train_batch(
            train_data,
            batch_size=args.prompt_batch,
            mask_token_id=tokenizer.mask_token_id,
            seed=args.seed + step,
        )
        step_generator = torch.Generator(device=device).manual_seed(
            args.seed + 10_000 + step
        )
        if args.objective == "grpo":
            rollout = collect_rollouts(
                model, prompt_ids=batch["prompt_ids"], target_ids=batch["target_ids"],
                fixed=batch["fixed"], mask_token_id=tokenizer.mask_token_id,
                group_size=args.group_size, sigma=args.sigma,
                confidence_temperature=args.confidence_temperature,
                token_temperature=args.token_temperature, threshold=args.threshold,
                max_steps=args.max_rollout_steps, generator=step_generator,
                autocast_dtype=torch.bfloat16 if args.precision == "bf16" else None,
            )
            epoch_metrics = []
            for epoch in range(args.ppo_epochs):
                update_metrics = grpo_update_epoch(
                    model, rollout, optimizer, sigma=args.sigma,
                    token_temperature=args.token_temperature,
                    confidence_temperature=args.confidence_temperature,
                    threshold=args.threshold, clip_epsilon=args.clip_epsilon,
                    kl_beta=args.kl_beta, max_grad_norm=args.max_grad_norm,
                    autocast_dtype=torch.bfloat16 if args.precision == "bf16" else None,
                )
                update_metrics["ppo_epoch"] = epoch + 1
                epoch_metrics.append(update_metrics)
            rewards = rollout.rewards.float()
            train_metrics = {
                **epoch_metrics[-1],
                **{
                    f"ppo_epoch_{epoch_index}/{name}": value
                    for epoch_index, values in enumerate(epoch_metrics, start=1)
                    for name, value in values.items()
                    if name != "ppo_epoch"
                },
                "reward_mean": rewards.mean().item(),
                "train_avg_at_8": 100 * rewards.mean().item(),
                "train_pass_at_8": 100 * rewards.amax(1).mean().item(),
                "train_mixed_reward_group_rate": 100 * (
                    (rewards.sum(1) > 0) & (rewards.sum(1) < args.group_size)
                ).float().mean().item(),
                "rollout_steps": len(rollout.steps),
            }
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_lora_loss(
                model, prompt_ids=batch["prompt_ids"], target_ids=batch["target_ids"],
                fixed=batch["fixed"], mask_token_id=tokenizer.mask_token_id,
                num_steps=args.sft_num_steps,
                confidence_temperature=args.confidence_temperature,
                threshold=args.threshold, sigma=args.sigma,
                gaussian_hidden=args.objective == "gaussian_sft",
                generator=step_generator,
                autocast_dtype=torch.bfloat16 if args.precision == "bf16" else None,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            train_metrics = {"loss": loss.item(), "grad_norm": float(grad_norm)}
        scheduler.step()
        row = {"step": step, "objective": args.objective,
               "learning_rate": scheduler.get_last_lr()[0], **train_metrics}
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if step % args.log_every == 0:
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            current_frozen_hash = tensor_hash(
                (name, parameter) for name, parameter in model.named_parameters()
                if not parameter.requires_grad
            )
            if current_frozen_hash != initial_frozen_hash:
                raise RuntimeError(
                    "A frozen pretrained parameter changed during post-training"
                )
            save_checkpoint(
                args.output / f"adapter_step_{step}.pt", model, optimizer,
                scheduler, args, step, initial_frozen_hash,
            )
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, dev_data, args, tokenizer, args.seed + 20_000 + step)
            result["step"] = step
            with (args.output / "evaluation.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            print(json.dumps({"development": result}), flush=True)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
