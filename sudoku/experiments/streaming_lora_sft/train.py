#!/usr/bin/env python
"""One-step full-trajectory Streaming LoRA-SFT for pretrained RELAY."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sudoku"))
sys.path.insert(0, str(ROOT / "sudoku/xlm-core/src"))

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from experiments.gaussian_hidden_grpo.train import (
    collate,
    dataset_manifest,
    load_pretrained,
    sha256,
    tensor_hash,
)
from relay.gaussian_grpo import select_positions
from relay.lora import assert_only_lora_trainable, inject_lora, lora_state_dict
from relay.streaming_batch import StreamingBatch


class PermutationCursor:
    """Deterministic without-replacement stream, reshuffled between passes."""

    def __init__(self, size: int, seed: int) -> None:
        if size < 1:
            raise ValueError("Dataset stream must be nonempty")
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.position = 0
        self.passes_started = 1

    def take(self, count: int) -> list[int]:
        result: list[int] = []
        while len(result) < count:
            available = self.size - self.position
            take = min(count - len(result), available)
            result.extend(self.order[self.position : self.position + take].tolist())
            self.position += take
            if self.position == self.size and len(result) < count:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.position = 0
                self.passes_started += 1
        return result

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "order": self.order,
            "position": self.position,
            "passes_started": self.passes_started,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("Dataset size changed across streaming-SFT resume")
        self.order = state["order"].clone()
        self.position = int(state["position"])
        self.passes_started = int(state["passes_started"])
        self.generator.set_state(state["generator_state"])


def rows_to_batch(dataset, indices: Iterable[int], mask_token_id: int) -> dict:
    indices = list(indices)
    batch = collate([dataset[index] for index in indices], mask_token_id=mask_token_id)
    batch["input_ids"] = batch.pop("prompt_ids")
    batch["dataset_index"] = torch.tensor(indices, dtype=torch.long)
    return batch


def initialize_streaming_batch(
    dataset,
    cursor: PermutationCursor,
    *,
    capacity: int,
    mask_token_id: int,
    d_model: int,
    device: torch.device,
) -> StreamingBatch:
    indices = cursor.take(capacity)
    batch = rows_to_batch(dataset, indices, mask_token_id)
    batch["h_s"] = torch.zeros(capacity, 81, d_model, dtype=torch.float32)
    batch = {key: value.to(device) for key, value in batch.items()}
    streaming = StreamingBatch(capacity=capacity, seq_len=81, device=device)
    streaming.initialize_from_batch(batch, device=device, mask_token_id=mask_token_id)
    return streaming


def refill_completed_slots(
    streaming: StreamingBatch,
    dataset,
    cursor: PermutationCursor,
    *,
    mask_token_id: int,
    d_model: int,
    device: torch.device,
) -> int:
    assert streaming.storage is not None and streaming.ready_to_evict is not None
    slots = streaming.ready_to_evict.nonzero(as_tuple=True)[0]
    count = int(slots.numel())
    if count == 0:
        return 0
    indices = cursor.take(count)
    fresh = rows_to_batch(dataset, indices, mask_token_id)
    fresh["h_s"] = torch.zeros(count, 81, d_model, dtype=torch.float32)
    for key, value in fresh.items():
        streaming.storage[key][slots] = value.to(device)
    streaming.ready_to_evict[slots] = False
    return count


def one_step_supervised_ce(
    model,
    streaming: StreamingBatch,
    *,
    mask_token_id: int,
    confidence_temperature: float,
    threshold: float,
    precision: str,
) -> tuple[torch.Tensor, dict]:
    """One forward, CE on the current state, then a detached teacher-forced transition."""
    assert streaming.storage is not None and streaming.ready_to_evict is not None
    ready_before = streaming.ready_to_evict.clone()
    x = streaming.storage["input_ids"].clone()
    targets = streaming.storage["target_ids"]
    fixed = streaming.storage["fixed"]
    h = streaming.storage["h_s"].clone().detach()
    masked = (x == mask_token_id) & ~fixed
    active = masked.any(-1)
    forward_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if x.device.type == "cuda" and precision == "bf16"
        else nullcontext()
    )
    with forward_context:
        logits, mean = model(x, h)
    per_token = F.cross_entropy(
        logits.float().transpose(1, 2), targets, reduction="none"
    )
    count = masked.sum()
    if not count.item():
        raise RuntimeError("Streaming batch contains no active Sudoku states")
    loss = (per_token * masked).sum() / count

    selected = select_positions(
        logits.detach().float(), masked,
        temperature=confidence_temperature, threshold=threshold, active=active,
    )
    x_next = torch.where(selected, targets, x)
    # Numerical relay state continues; the graph is cut at every optimizer boundary.
    h_next = torch.where(active[:, None, None], mean.detach().float(), h)
    streaming.storage["input_ids"] = x_next.detach()
    streaming.storage["h_s"] = h_next.detach()
    streaming.update_after_unmask(selected, targets, mask_token_id)
    assert streaming.ready_to_evict is not None
    return loss, {
        "active_states": int(active.sum().item()),
        "supervised_tokens": int(count.item()),
        "selected_positions": int(selected.sum().item()),
        "newly_completed": int((~ready_before & streaming.ready_to_evict).sum().item()),
        "mean_masks_before": float(masked.sum(-1).float().mean().item()),
        "mean_masks_after": float(
            (((x_next == mask_token_id) & ~fixed).sum(-1).float().mean()).item()
        ),
    }


def streaming_state_dict(streaming: StreamingBatch) -> dict:
    assert streaming.storage is not None and streaming.ready_to_evict is not None
    return {
        "storage": {
            key: value.detach().cpu().clone() for key, value in streaming.storage.items()
        },
        "ready_to_evict": streaming.ready_to_evict.detach().cpu().clone(),
        "capacity": streaming.capacity,
        "seq_len": streaming.seq_len,
    }


def load_streaming_state(state: dict, device: torch.device) -> StreamingBatch:
    streaming = StreamingBatch(
        capacity=int(state["capacity"]), seq_len=int(state["seq_len"]), device=device
    )
    streaming.storage = {
        key: value.to(device) for key, value in state["storage"].items()
    }
    streaming.ready_to_evict = state["ready_to_evict"].to(device)
    return streaming


def save_training_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    streaming: StreamingBatch,
    cursor: PermutationCursor,
    step: int,
    completed: int,
    started: int,
    args,
    frozen_base_hash: str,
) -> None:
    torch.save(
        {
            "step": step,
            "completed_trajectories": completed,
            "started_trajectories": started,
            "lora_state_dict": lora_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "streaming": streaming_state_dict(streaming),
            "cursor": cursor.state_dict(),
            "arguments": vars(args),
            "frozen_base_sha256": frozen_base_hash,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument(
        "--train-data", type=Path,
        default=ROOT / "data/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver/train",
    )
    parser.add_argument(
        "--dev-data", type=Path,
        default=ROOT / "data/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver/test_first2000",
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--completed-trajectories", type=int, default=5000)
    parser.add_argument("--streaming-batch", type=int, default=32)
    parser.add_argument("--max-optimizer-steps", type=int, default=100000)
    parser.add_argument("--confidence-temperature", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-completed", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.completed_trajectories < 1 or args.streaming_batch < 1:
        parser.error("completed-trajectories and streaming-batch must be positive")
    if args.lora_rank != 32 or args.lora_alpha != 64:
        parser.error("This baseline is locked to LoRA rank=32 and alpha=64")
    if args.confidence_temperature < 0 or args.learning_rate <= 0:
        parser.error("temperatures must be nonnegative and learning-rate positive")
    if args.resume is None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume is not None)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, source_step, model_config = load_pretrained(
        args.checkpoint, args.weights, device
    )
    lora_report = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    assert_only_lora_trainable(model)
    model.eval()
    frozen_hash = tensor_hash(
        (name, parameter) for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )

    train = load_from_disk(str(args.train_data)).select(range(args.train_size))
    dev = load_from_disk(str(args.dev_data)).select(range(args.dev_size))
    train_questions = {row["question"].replace(".", "0") for row in train}
    dev_questions = {row["question"].replace(".", "0") for row in dev}
    if train_questions & dev_questions:
        raise ValueError("Training and development puzzles overlap")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / args.warmup_steps, 1.0)
        if args.warmup_steps else 1.0,
    )
    cursor = PermutationCursor(len(train), args.seed)
    streaming = initialize_streaming_batch(
        train, cursor, capacity=args.streaming_batch,
        mask_token_id=tokenizer.mask_token_id, d_model=model.d_model, device=device,
    )
    step = 0
    completed = 0
    started = args.streaming_batch

    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume["frozen_base_sha256"] != frozen_hash:
            raise ValueError("Resume checkpoint has a different frozen base")
        model_state = model.state_dict()
        with torch.no_grad():
            for name, value in resume["lora_state_dict"].items():
                model_state[name].copy_(value)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        streaming = load_streaming_state(resume["streaming"], device)
        cursor.load_state_dict(resume["cursor"])
        step = int(resume["step"])
        completed = int(resume["completed_trajectories"])
        started = int(resume["started_trajectories"])

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "method": "one-step streaming full-trajectory LoRA-SFT",
        "state_transition": "teacher-forced x; h_next=stop_gradient(mu); one forward per optimizer update",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pretrained_checkpoint_sha256": sha256(args.checkpoint),
        "pretrained_global_step": source_step,
        "frozen_base_sha256": frozen_hash,
        "model_config": model_config,
        "lora": asdict(lora_report),
        "train_data": dataset_manifest(train, path=args.train_data, start=0, size=args.train_size),
        "development_data": dataset_manifest(dev, path=args.dev_data, start=0, size=args.dev_size),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metrics_path = args.output / "metrics.jsonl"
    next_save = ((completed // args.save_every_completed) + 1) * args.save_every_completed
    initial_frozen_hash = frozen_hash

    while completed < args.completed_trajectories:
        if step >= args.max_optimizer_steps:
            raise RuntimeError("Reached max-optimizer-steps before the completion target")
        refilled = refill_completed_slots(
            streaming, train, cursor, mask_token_id=tokenizer.mask_token_id,
            d_model=model.d_model, device=device,
        )
        started += refilled
        step += 1
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = one_step_supervised_ce(
            model, streaming, mask_token_id=tokenizer.mask_token_id,
            confidence_temperature=args.confidence_temperature,
            threshold=args.threshold, precision=args.precision,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        completed += diagnostics["newly_completed"]
        row = {
            "optimizer_step": step,
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "learning_rate": scheduler.get_last_lr()[0],
            "completed_trajectories": completed,
            "started_trajectories": started,
            "dataset_passes_started": cursor.passes_started,
            **diagnostics,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if step % args.log_every == 0 or diagnostics["newly_completed"]:
            print(json.dumps(row), flush=True)

        if completed >= next_save or completed >= args.completed_trajectories:
            current_frozen_hash = tensor_hash(
                (name, parameter) for name, parameter in model.named_parameters()
                if not parameter.requires_grad
            )
            if current_frozen_hash != initial_frozen_hash:
                raise RuntimeError("A frozen pretrained parameter changed")
            checkpoint_path = args.output / (
                f"adapter_completed_{completed}_step_{step}.pt"
            )
            save_training_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler,
                streaming=streaming, cursor=cursor, step=step, completed=completed,
                started=started, args=args, frozen_base_hash=initial_frozen_hash,
            )
            (args.output / "latest_checkpoint.txt").write_text(str(checkpoint_path) + "\n")
            while next_save <= completed:
                next_save += args.save_every_completed

    final_path = args.output / "adapter_final.pt"
    save_training_checkpoint(
        final_path, model=model, optimizer=optimizer, scheduler=scheduler,
        streaming=streaming, cursor=cursor, step=step, completed=completed,
        started=started, args=args, frozen_base_hash=initial_frozen_hash,
    )
    manifest.update(
        status="complete",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        optimizer_steps=step,
        completed_trajectories=completed,
        started_trajectories=started,
        dataset_passes_started=cursor.passes_started,
        final_adapter=str(final_path),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"complete": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
