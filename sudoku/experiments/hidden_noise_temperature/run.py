#!/usr/bin/env python
"""Fixed-checkpoint, inference-only RELAY temperature/hidden-noise sweep."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sudoku"))
sys.path.insert(0, str(ROOT / "sudoku/xlm-core/src"))

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from xlm.datamodule import SimpleSpaceTokenizer
from relay.model import RotaryTransformerRelayModel
from relay.predictor import ConfidenceBasedPredictor
from relay.inference_perturbation import PerturbedRelayPredictor
from relay.sudoku_metrics import sudoku_legal_mask


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_actor(path, weights, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=False)
    cfg = dict(cfg["model"])
    if cfg.pop("_target_") != "relay.model.RotaryTransformerRelayModel":
        raise ValueError("Expected a RELAY checkpoint")
    tokenizer = SimpleSpaceTokenizer.for_numbers(vocab_size=10)
    cfg.update(num_embeddings=tokenizer.full_vocab_size,
               padding_idx=tokenizer.pad_token_id, mask_idx=tokenizer.mask_token_id)
    model = RotaryTransformerRelayModel(**cfg)
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    if weights == "ema":
        shadows = ckpt["ema"]["shadow_params"]
        params = list(model.parameters())  # deduplicates tied embeddings
        if len(shadows) != len(params):
            raise ValueError("EMA parameter count does not match actor")
        with torch.no_grad():
            for p, shadow in zip(params, shadows):
                if p.shape != shadow.shape:
                    raise ValueError("EMA parameter shape mismatch")
                p.copy_(shadow)
    model.eval().requires_grad_(False).to(device)
    return model, tokenizer, {"global_step": ckpt["global_step"], "model": cfg,
                              "weights": weights, "parameters": sum(p.numel() for p in model.parameters())}


def prepare_data(path, start, n, tokenizer):
    ds = load_from_disk(str(path))
    if start < 0 or n < 1 or start + n > len(ds):
        raise ValueError("Requested dataset slice is out of bounds")
    subset = ds.select(range(start, start + n))
    prompts, targets, identities = [], [], []
    for idx, row in enumerate(subset, start):
        q, a = row["question"], row["answer"]
        if len(q) != 81 or len(a) != 81 or not set(a) <= set("123456789"):
            raise ValueError(f"Malformed Sudoku row {idx}")
        p = [tokenizer.mask_token_id if c in ".0" else tokenizer.convert_tokens_to_ids(c) for c in q]
        y = [tokenizer.convert_tokens_to_ids(c) for c in a]
        if "prompt_token_ids" in row and row["prompt_token_ids"] != p:
            raise ValueError(f"Cached prompt encoding mismatch at row {idx}")
        if "input_token_ids" in row and row["input_token_ids"] != y:
            raise ValueError(f"Cached solution encoding mismatch at row {idx}")
        if any(c not in ".0" and c != a[i] for i, c in enumerate(q)):
            raise ValueError(f"Clue mismatch at row {idx}")
        prompts.append(p); targets.append(y)
        identities.append({"puzzle_id": idx, "question": q, "answer": a})
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    return torch.tensor(prompts), torch.tensor(targets), identities, {
        "path": str(path.resolve()), "available_rows": len(ds),
        "fingerprint": ds._fingerprint, "start": start, "n": n,
        "cohort": "unfiltered", "order": "saved Hugging Face dataset order; no shuffle",
        "subset_sha256": digest,
    }


def base_kwargs(model, tokenizer, args):
    return dict(model=model, tokenizer=tokenizer, with_relay=True,
                confidence="top_prob", threshold=args.threshold,
                top_k=1, top_p=None, max_steps=args.max_steps,
                log_rollout_diagnostics=False)


def autocast(args):
    return (torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16)
            if args.precision == "bf16" else contextlib.nullcontext())


@torch.inference_mode()
def smoke(model, tokenizer, prompts, args):
    x = prompts[:min(16, len(prompts))].to(args.device)
    batch = dict(input_ids=x, fixed=x != tokenizer.mask_token_id)
    old = ConfidenceBasedPredictor(**base_kwargs(model, tokenizer, args), capture_trajectory=True)
    new = PerturbedRelayPredictor(**base_kwargs(model, tokenizer, args), capture_trajectory=True)
    with autocast(args):
        a = old.predict(batch)
        b = new.predict(batch)
        c = new.predict(batch)
        noisy = PerturbedRelayPredictor(**base_kwargs(model, tokenizer, args),
            hidden_sigma=0.5, capture_trajectory=True,
            noise_generator=torch.Generator(device=args.device).manual_seed(123))
        d = noisy.predict(batch)
        noisy.noise_generator.manual_seed(123)
        e = noisy.predict(batch)
    assert a["trajectory"] == b["trajectory"] == c["trajectory"], "Baseline regression"
    assert a["rollout_steps"] == b["actual_forward_calls"], "NFE mismatch"
    assert d["trajectory"] == e["trajectory"], "Noise seed is not reproducible"
    assert all(u[:2] == v[:2] for u, v in zip(b["trajectory"], d["trajectory"])), "Hidden noise changed the first token update"
    assert torch.equal(d["ids"][batch["fixed"]], x[batch["fixed"]]), "Clues changed"
    with autocast(args):
        _, h = model(x, torch.zeros(*x.shape, model.d_model, device=args.device))
    return {"baseline_trajectory_matches_original": True,
            "deterministic_repeat_matches": True, "noise_seed_reproducible": True,
            "first_step_unaffected_by_hidden_noise": True, "clues_preserved": True,
            "first_step_hidden_rms": h.float().square().mean().sqrt().item(),
            "first_step_hidden_centered_rms": (h.float()-h.float().mean(-1,keepdim=True)).square().mean().sqrt().item()}


def summarize(rows, label, temperature, sigma, args, seconds):
    by_puzzle = {}
    for row in rows:
        by_puzzle.setdefault(row["puzzle_id"], []).append(row)
    if len(by_puzzle) != args.n or any(len(r) != args.samples for r in by_puzzle.values()):
        raise ValueError("Incomplete sample groups")
    deterministic = sigma == 0
    if deterministic and any(len({tuple(r["prediction_ids"]) for r in group}) != 1 for group in by_puzzle.values()):
        raise RuntimeError("Deterministic repeats disagree; inspect numerical reproducibility")
    return dict(configuration=label, temperature=temperature, sigma=sigma,
        n_puzzles=args.n, samples_per_puzzle=args.samples,
        avg_at_8=100*sum(r["exact_match"] for r in rows)/len(rows),
        pass_at_8=100*sum(any(r["exact_match"] for r in group) for group in by_puzzle.values())/args.n,
        mean_nfe=sum(r["nfe"] for r in rows)/len(rows),
        mean_nfe_per_8=sum(r["nfe"] for r in rows)/args.n,
        mean_first_filled_step=sum(max(0,r["first_filled_step"]) for r in rows if r["first_filled_step"] >= 0)/max(1,sum(r["first_filled_step"] >= 0 for r in rows)),
        unfilled_rate=100*sum(r["first_filled_step"] < 0 for r in rows)/len(rows),
        legal_rate=100*sum(r["legal"] and r["clues_preserved"] for r in rows)/len(rows),
        mean_distinct_answers=sum(len({tuple(r["prediction_ids"]) for r in group}) for group in by_puzzle.values())/args.n,
        mixed_reward_group_rate=100*sum(0 < sum(r["exact_match"] for r in group) < args.samples for group in by_puzzle.values())/args.n,
        elapsed_seconds=seconds)


def write_summaries(out, summaries):
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    lines = ["| Configuration | T | sigma | avg@8 (%) | pass@8 (%) | NFE / rollout | NFE / 8 | Distinct answers / 8 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['configuration']} | {s['temperature']:g} | {s['sigma']:g} | {s['avg_at_8']:.2f} | {s['pass_at_8']:.2f} | {s['mean_nfe']:.2f} | {s['mean_nfe_per_8']:.2f} | {s['mean_distinct_answers']:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver/test")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--samples", type=int, choices=[8], default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--temperatures", type=float, nargs="*", default=[0.5,0.75,1.25,1.5,2.0])
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.05,0.1,0.2,0.5,1.0])
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    if args.batch_size < 1 or args.max_steps < 1 or args.threshold < 0:
        ap.error("Invalid batch size, step limit or threshold")
    if any(not math.isfinite(t) or t <= 0 for t in args.temperatures) or any(not math.isfinite(s) or s < 0 for s in args.sigmas):
        ap.error("Temperatures must be positive; sigmas must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a new output directory to avoid mixing experiments")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model, tokenizer, model_meta = load_actor(args.checkpoint, args.weights, args.device)
    prompts, targets, identities, data_meta = prepare_data(args.data, args.start, args.n, tokenizer)
    configurations = [("baseline", 1., 0.)]
    configurations += [(f"temperature_{t:g}", t, 0.) for t in dict.fromkeys(args.temperatures) if t != 1]
    configurations += [(f"hidden_sigma_{s:g}", 1., s) for s in dict.fromkeys(args.sigmas) if s != 0]
    manifest = {"arguments": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "checkpoint_sha256": sha256(args.checkpoint), "model": model_meta, "data": data_meta,
        "configurations": configurations, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else None,
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),ROOT / "sudoku/relay/inference_perturbation.py",ROOT / "sudoku/relay/predictor.py",ROOT / "sudoku/relay/model.py"]},
        "nfe_definition": "Actual model forwards per batch row, including forwards on finished rows and the original unconditional final forward. Eight actual runs per puzzle, without caching deterministic repeats.",
        "noise_seed_definition": "seed + sample_id*1000003 + batch_start; same streams across sigma values; changing batching changes streams",
        "selection": "Original cumulative uncertainty < 0.15 with highest-confidence fallback; top-1 over full vocabulary; no digit-only masking", "status":"running"}
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.output / "puzzles.jsonl").write_text("".join(json.dumps(r)+"\n" for r in identities))
    checks = smoke(model, tokenizer, prompts, args)
    (args.output / "checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    print("CHECKS", json.dumps(checks), flush=True)
    if args.smoke_only:
        manifest["status"]="smoke_complete"
        (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        return
    summaries = []
    for label, temperature, sigma in configurations:
        started = time.monotonic()
        rows = []
        predictor = PerturbedRelayPredictor(**base_kwargs(model,tokenizer,args),
            confidence_temperature=temperature, hidden_sigma=sigma)
        with (args.output / f"{label}.jsonl").open("w") as f:
            for start in range(0,args.n,args.batch_size):
                end = min(args.n,start+args.batch_size)
                x = prompts[start:end].to(args.device)
                y = targets[start:end].to(args.device)
                fixed = x != tokenizer.mask_token_id
                for sample_id in range(args.samples):
                    noise_seed = args.seed + sample_id*1000003 + start
                    predictor.noise_generator = torch.Generator(device=args.device).manual_seed(noise_seed)
                    with autocast(args):
                        result = predictor.predict(dict(input_ids=x, fixed=fixed))
                    ids = result["ids"]
                    exact = (ids == y).all(-1).cpu().tolist()
                    legal = sudoku_legal_mask(ids).cpu().tolist()
                    clues = ((ids == x) | ~fixed).all(-1).cpu().tolist()
                    if not all(clues):
                        raise RuntimeError("Decoder modified puzzle clues")
                    cpu_ids = ids.cpu().tolist()
                    for j in range(end-start):
                        row = dict(puzzle_id=args.start+start+j, sample_id=sample_id,
                            configuration=label, temperature=temperature, sigma=sigma,
                            noise_seed=noise_seed, batch_start=start, batch_row=j,
                            prediction_ids=cpu_ids[j], exact_match=exact[j], legal=legal[j],
                            clues_preserved=clues[j], nfe=result["actual_forward_calls"],
                            first_filled_step=result["first_filled_step"][j])
                        f.write(json.dumps(row)+"\n"); rows.append(row)
                f.flush()
                print(f"{label}: {end}/{args.n} puzzles x {args.samples}; elapsed {time.monotonic()-started:.1f}s",flush=True)
        summary = summarize(rows,label,temperature,sigma,args,time.monotonic()-started)
        summaries.append(summary); write_summaries(args.output,summaries)
        print("RESULT",json.dumps(summary),flush=True)
    manifest["status"]="complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
