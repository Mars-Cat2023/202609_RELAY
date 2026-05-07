#!/usr/bin/env python3
"""
Aggregate EvalPlus `*.eval_results.json` files from the Magicoder sweep
(`sweep_evalplus_magicoder.sh` → `evalplus_results/magicoder_sweep/`).

EvalPlus writes `samples.jsonl` → `samples.eval_results.json` (or legacy
`samples_eval_results.json`). Each JSON has `pass_at_k.base` and
`pass_at_k.plus` (HumanEval+/MBPP+ "base + extra" tests).

Usage (from Fast-dLLM/v2):
  python scripts/aggregate_evalplus_magicoder.py
  python scripts/aggregate_evalplus_magicoder.py --results_dir evalplus_results/magicoder_sweep --out_dir evalplus_results/magicoder_sweep_summary
  python scripts/aggregate_evalplus_magicoder.py --no-plot

Optional: `pip install matplotlib` for figures.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterator


RUN_ORDER = ("sft", "nocarry", "loopguard", "mlp")
SPLIT_ORDER = ("python", "full")


@dataclass(frozen=True)
class SweepKey:
    split: str
    run: str
    ck_label: str
    dataset: str
    seed: int


def result_paths(results_dir: Path) -> list[Path]:
    """
    New format: `stem.eval_results.json` (from `stem.jsonl`).
    Legacy: `stem_eval_results.json` (underscore before `eval` — not matched by
    `*.eval_results.json`).
    """
    by_stem: dict[str, Path] = {}
    for p in results_dir.glob("*.eval_results.json"):
        by_stem[path_to_jsonl_stem(p)] = p
    for p in results_dir.glob("*_eval_results.json"):
        stem = path_to_jsonl_stem(p)
        if stem not in by_stem:
            by_stem[stem] = p
    return sorted(by_stem.values(), key=lambda x: x.name)


def path_to_jsonl_stem(p: Path) -> str:
    n = p.name
    if n.endswith(".eval_results.json"):
        return n[: -len(".eval_results.json")]
    if n.endswith("_eval_results.json"):
        return n[: -len("_eval_results.json")]
    raise ValueError(f"Unrecognized result filename: {p}")


def parse_sweep_name(stem: str) -> SweepKey | None:
    """Parse `{split}__{run}__{ck}__{dataset}__s{seed}` (basename without .jsonl)."""
    parts = stem.split("__")
    if len(parts) != 5:
        return None
    split, run, ck, ds, sx = parts
    m = re.fullmatch(r"s(\d+)", sx)
    if not m or ds not in ("humaneval", "mbpp"):
        return None
    if split not in SPLIT_ORDER or run not in RUN_ORDER:
        return None
    if not re.fullmatch(r"(final|checkpoint-\d+)", ck):
        return None
    return SweepKey(
        split=split, run=run, ck_label=ck, dataset=ds, seed=int(m.group(1))
    )


def ck_step(ck: str) -> int:
    if ck == "final":
        return 10**9
    m = re.fullmatch(r"checkpoint-(\d+)", ck)
    if not m:
        return -1
    return int(m.group(1))


def _pass_rates_from_per_task_eval(
    ev: Any,
) -> tuple[float | None, float | None, int]:
    """
    Newer evalplus `*_eval_results.json` often omit top-level ``pass_at_k`` and
    only list per-task outcomes under ``eval`` (``base_status`` / ``plus_status``).
    """
    if not isinstance(ev, dict) or not ev:
        return None, None, 0
    n_ok = 0
    n_base = 0
    n_plus = 0
    for _tid, rows in ev.items():
        if not rows:
            continue
        r = rows[0] if isinstance(rows, (list, tuple)) else rows
        if not isinstance(r, dict):
            continue
        n_ok += 1
        if r.get("base_status") == "pass":
            n_base += 1
        if r.get("plus_status") == "pass":
            n_plus += 1
    if n_ok == 0:
        return None, None, 0
    return n_base / n_ok, n_plus / n_ok, n_ok


def load_pass_at_k(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [skip] {path.name}: {e}", file=sys.stderr)
        return None
    p = data.get("pass_at_k") or {}
    base = p.get("base") or {}
    plus = p.get("plus") or {}
    p1b: float | None = base.get("pass@1")
    p1p: float | None = plus.get("pass@1") if plus else None
    ev = data.get("eval")
    n_tasks = len(ev) if isinstance(ev, dict) else 0
    if p1b is None and isinstance(ev, dict) and ev:
        p1b, p1p, n_tasks = _pass_rates_from_per_task_eval(ev)
    return {
        "path": path,
        "pass@1_base": p1b,
        "pass@1_plus": p1p,
        "n_tasks": n_tasks,
    }


def iter_rows(
    results_dir: Path,
) -> Iterator[tuple[SweepKey, dict[str, Any]]]:
    for path in result_paths(results_dir):
        stem = path_to_jsonl_stem(path)
        key = parse_sweep_name(stem)
        if key is None:
            print(
                f"  [skip] could not parse sweep key from: {path.name}",
                file=sys.stderr,
            )
            continue
        m = load_pass_at_k(path)
        if m is None:
            continue
        if m["pass@1_base"] is None:
            print(
                f"  [warn] no pass@1 in base: {path.name}",
                file=sys.stderr,
            )
        yield key, m


def agg_by_group(
    rows: list[tuple[SweepKey, dict[str, Any]]],
) -> dict[tuple, dict[str, Any]]:
    """Group by (split, run, ck, dataset)."""
    g: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for k, m in rows:
        g[(k.split, k.run, k.ck_label, k.dataset)].append(
            {**m, "seed": k.seed}
        )
    out: dict[tuple, dict[str, Any]] = {}
    for key, items in g.items():
        b = [x["pass@1_base"] for x in items if x["pass@1_base"] is not None]
        pl = [x["pass@1_plus"] for x in items if x["pass@1_plus"] is not None]
        out[key] = {
            "n": len(items),
            "seeds": sorted(x["seed"] for x in items),
            "base_mean": mean(b) if b else float("nan"),
            "base_std": pstdev(b) if len(b) > 1 else 0.0,
            "plus_mean": mean(pl) if pl else float("nan"),
            "plus_std": pstdev(pl) if len(pl) > 1 else 0.0,
        }
    return out


def pick_latest(
    groups: dict[tuple, dict[str, Any]],
) -> dict[tuple[str, str, str], tuple[str, dict[str, Any]]]:
    """For each (split, run, dataset) choose the checkpoint with largest step (or `final`)."""
    by_srd: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
    for (split, run, ck, ds), st in groups.items():
        by_srd[(split, run, ds)].append((ck, st))
    best: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
    for srd, lst in by_srd.items():
        ck_sel, st_sel = max(lst, key=lambda t: ck_step(t[0]))
        best[srd] = (ck_sel, st_sel)
    return best


def print_tables(
    groups: dict[tuple, dict[str, Any]], latest: dict[tuple, tuple[str, dict]]
) -> None:
    print("\n=== Latest checkpoint per (split × run × dataset) (mean over seeds) ===\n")
    header = f"{'split':6} {'run':10} {'dataset':8} {'ck':12} {'n':2}  {'base@1':>8}  {'+extra@1':>10}"
    print(header)
    print("-" * len(header))
    for split in SPLIT_ORDER:
        for run in RUN_ORDER:
            for ds in ("humaneval", "mbpp"):
                key = (split, run, ds)
                if key not in latest:
                    print(f"{split:6} {run:10} {ds:8} {'(missing)':12}  {'':2}  {'':>8}  {'':>10}")
                    continue
                ck, st = latest[key]
                b = st["base_mean"]
                p = st["plus_mean"]
                b_s = f"{b:.3f}±{st['base_std']:.3f}" if not math.isnan(b) else "—"
                p_s = f"{p:.3f}±{st['plus_std']:.3f}" if not math.isnan(p) else "—"
                print(
                    f"{split:6} {run:10} {ds:8} {ck:12} {st['n']:2d}  {b_s:>8}  {p_s:>10}"
                )
        print()

    print("\n=== All (split, run, ck, dataset) with ≥1 result file ===\n")
    keys_sorted = sorted(
        groups.keys(),
        key=lambda k: (SPLIT_ORDER.index(k[0]) if k[0] in SPLIT_ORDER else 9, RUN_ORDER.index(k[1]) if k[1] in RUN_ORDER else 9, ck_step(k[2]), k[3]),
    )
    h2 = f"{'split':6} {'run':10} {'ck':14} {'dataset':8} {'n':2}  {'base@1':>14}  {'+extra@1':>16}"
    print(h2)
    print("-" * len(h2))
    for (split, run, ck, ds) in keys_sorted:
        st = groups[(split, run, ck, ds)]
        b = st["base_mean"]
        p = st["plus_mean"]
        b_s = f"{b:.3f}±{st['base_std']:.3f}" if not math.isnan(b) else "—"
        p_s = f"{p:.3f}±{st['plus_std']:.3f}" if not math.isnan(p) else "—"
        print(
            f"{split:6} {run:10} {ck:14} {ds:8} {st['n']:2d}  {b_s:>14}  {p_s:>16}"
        )


def write_csv(
    out_dir: Path,
    groups: dict[tuple, dict[str, Any]],
    latest: dict[tuple, tuple[str, dict]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    longp = out_dir / "sweep_by_checkpoint.csv"
    with longp.open("w", encoding="utf-8") as f:
        f.write("split,run,ck_label,ck_step,dataset,n_seeds,base_mean,base_std,plus_mean,plus_std\n")
        for (split, run, ck, ds), st in sorted(
            groups.items(),
            key=lambda kv: (kv[0][0], kv[0][1], ck_step(kv[0][2]), kv[0][3]),
        ):
            f.write(
                f"{split},{run},{ck},{ck_step(ck)},{ds},{st['n']},"
                f"{st['base_mean']:.6f},{st['base_std']:.6f},"
                f"{st['plus_mean'] if not math.isnan(st['plus_mean']) else ''},"
                f"{st['plus_std'] if not math.isnan(st['plus_std']) else ''}\n"
            )
    print(f"Wrote {longp}")

    latp = out_dir / "sweep_latest_per_run.csv"
    with latp.open("w", encoding="utf-8") as f:
        f.write("split,run,dataset,ck_label,ck_step,n_seeds,base_mean,base_std,plus_mean,plus_std\n")
        for split in SPLIT_ORDER:
            for run in RUN_ORDER:
                for ds in ("humaneval", "mbpp"):
                    if (split, run, ds) not in latest:
                        continue
                    ck, st = latest[(split, run, ds)]
                    f.write(
                        f"{split},{run},{ds},{ck},{ck_step(ck)},{st['n']},"
                        f"{st['base_mean']:.6f},{st['base_std']:.6f},"
                        f"{st['plus_mean'] if not math.isnan(st['plus_mean']) else ''},"
                        f"{st['plus_std'] if not math.isnan(st['plus_std']) else ''}\n"
                    )
    print(f"Wrote {latp}")


def maybe_plots(
    out_dir: Path,
    groups: dict[tuple, dict[str, Any]],
    latest: dict[tuple, tuple[str, dict]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed; skip plots (`pip install matplotlib`).",
            file=sys.stderr,
        )
        return

    # (1) Grouped bar: latest checkpoint, pass@1 on +extra (plus), compare splits within run
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, ds in zip(axes, ("humaneval", "mbpp")):
        x = list(range(len(RUN_ORDER)))
        w = 0.35
        for i, split in enumerate(SPLIT_ORDER):
            heights = []
            yerr = []
            for run in RUN_ORDER:
                st = latest.get((split, run, ds))
                if st is None:
                    heights.append(0.0)
                    yerr.append(0.0)
                    continue
                _, s = st
                p = s["plus_mean"]
                h = 0.0 if math.isnan(p) else p
                heights.append(h)
                yerr.append(s["plus_std"] if s["n"] > 1 else 0.0)
            off = (i - 0.5) * w
            ax.bar(
                [xx + off for xx in x],
                heights,
                w,
                yerr=yerr,
                capsize=2,
                label=split,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(RUN_ORDER, rotation=15, ha="right")
        ax.set_ylabel("pass@1 (+ extra tests, EvalPlus+)")
        ax.set_title(f"{ds} (latest ck per run)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Magicoder EvalPlus sweep — mean ± std over seeds")
    fig.tight_layout()
    p1 = out_dir / "plot_latest_bars_plus.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p1}")

    # (2) Learning curves: step vs plus_mean, one subplot per (split, dataset)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=True)
    for i, split in enumerate(SPLIT_ORDER):
        for j, ds in enumerate(("humaneval", "mbpp")):
            ax = axes[i][j]
            for run in RUN_ORDER:
                xs: list[int] = []
                ys: list[float] = []
                es: list[float] = []
                for (s, r, ck, d), st in groups.items():
                    if s != split or r != run or d != ds:
                        continue
                    stp = ck_step(ck)
                    if stp < 0:
                        continue
                    p = st["plus_mean"]
                    if math.isnan(p):
                        continue
                    xs.append(stp)
                    ys.append(p)
                    es.append(st["plus_std"] if st["n"] > 1 else 0.0)
                pairs = sorted(zip(xs, ys, es))
                if not pairs:
                    continue
                xs, ys, es = zip(*pairs)
                ax.errorbar(
                    xs, ys, yerr=es, marker="o", capsize=3, label=run, alpha=0.85
                )
            ax.set_title(f"{split} / {ds}")
            ax.set_xlabel("step (checkpoint-N)")
            ax.set_ylabel("pass@1 +extra")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
    h, lab = [], []
    for ax in axes.flat:
        hx, lx = ax.get_legend_handles_labels()
        if hx:
            h, lab = hx, lx
            break
    if h:
        fig.legend(
            h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8
        )
    fig.suptitle("Checkpoints: pass@1 (+extra) vs training step (mean over seeds)")
    fig.tight_layout()
    p2 = out_dir / "plot_curves_by_step.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p2}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate EvalPlus results from magicoder_sweep"
    )
    ap.add_argument(
        "--results_dir",
        type=Path,
        default=Path("evalplus_results/magicoder_sweep"),
        help="Directory with *.eval_results.json (default: evalplus_results/magicoder_sweep)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("evalplus_results/magicoder_sweep_summary"),
        help="Output CSV/PNG directory",
    )
    ap.add_argument(
        "--no-plot", action="store_true", help="Do not write PNG figures"
    )
    ap.add_argument(
        "--no-csv", action="store_true", help="Do not write CSV files"
    )
    ap.add_argument(
        "--list-only", action="store_true", help="Only list parsed result files"
    )
    args = ap.parse_args()
    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = list(iter_rows(results_dir))
    if args.list_only:
        for k, m in rows:
            print(
                f"{k.split}  {k.run:10}  {k.ck_label:14}  {k.dataset:8}  s{k.seed}  base={m['pass@1_base']}  plus={m['pass@1_plus']}"
            )
        return

    if not rows:
        print(
            f"No .eval_results.json files parsed under {results_dir}. "
            f"Run the sweep and evalplus.evaluate first.",
            file=sys.stderr,
        )
        sys.exit(2)

    groups = agg_by_group(rows)
    latest = pick_latest(groups)
    print_tables(groups, latest)
    if not args.no_csv:
        write_csv(args.out_dir, groups, latest)
    if not args.no_plot:
        maybe_plots(args.out_dir, groups, latest)


if __name__ == "__main__":
    main()
