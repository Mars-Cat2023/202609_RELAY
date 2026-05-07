#!/usr/bin/env python3
"""Compare two lm-eval sample dumps (samples_*.jsonl) pairwise by ``doc_id``.


    python scripts/compare_eval_samples.py --left PATH --right PATH

Writes ``eval_results_pairwise/<slug>/`` and optionally logs to W&B
``Fast-dLLM-v2-evals-analysis``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import manage_evals as me  # noqa: E402
import submit_eval as se  # noqa: E402

DEFAULT_WANDB_PROJECT = "Fast-dLLM-v2-evals-analysis"

try:
    from scipy.stats import chi2
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "compare_eval_samples requires scipy (e.g. pip install scipy). " + str(e)
    ) from e


@dataclass
class PairwiseStats:
    """Paired correctness comparison on doc_id intersection."""

    n_both_correct: int = 0
    n_only_left: int = 0
    n_only_right: int = 0
    n_both_wrong: int = 0
    n_metric_missing_left: int = 0
    n_metric_missing_right: int = 0
    doc_ids_only_left_file: list[int] = field(default_factory=list)
    doc_ids_only_right_file: list[int] = field(default_factory=list)
    disagreement_left_wins_doc_ids: list[int] = field(default_factory=list)
    disagreement_right_wins_doc_ids: list[int] = field(default_factory=list)


@dataclass
class PairComparison:
    stats: PairwiseStats
    mcnemar_statistic: float
    mcnemar_pvalue: float
    metric: str
    correct_threshold: float
    left_samples_path: str
    right_samples_path: str
    left_run_tag: str
    right_run_tag: str
    left_task: str
    right_task: str
    left_label: str
    right_label: str


def _parse_jsonl(path: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Load doc_id -> record. Last line wins on duplicate doc_id (with warning)."""
    by_id: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.append(f"{path}:{line_no}: JSON decode error: {e}")
                continue
            if "doc_id" not in obj:
                warnings.append(f"{path}:{line_no}: missing doc_id, skipped")
                continue
            did = int(obj["doc_id"])
            if did in by_id:
                warnings.append(
                    f"{path}: duplicate doc_id {did} (line {line_no} overrides earlier)"
                )
            by_id[did] = obj
    return by_id, warnings


def _metric_value(rec: dict[str, Any], metric: str) -> float | None:
    if metric not in rec:
        return None
    v = rec[metric]
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_correct(val: float | None, threshold: float) -> bool | None:
    if val is None:
        return None
    return val >= threshold


def mcnemar_asymptotic(
    n_only_left: int, n_only_right: int, *, correction: bool = True
) -> tuple[float, float]:
    """McNemar on discordant pairs. Returns (chi-square statistic, two-sided p-value)."""
    b = n_only_left
    c = n_only_right
    n = b + c
    if n == 0:
        return float("nan"), float("nan")
    if correction:
        stat = (abs(b - c) - 1.0) ** 2 / n
    else:
        stat = (b - c) ** 2 / n
    p = float(chi2.sf(stat, df=1))
    return float(stat), p


def compare_loaded(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    *,
    metric: str,
    correct_threshold: float,
    max_disagreement_ids: int | None,
) -> PairwiseStats:
    st = PairwiseStats()
    left_ids = set(left)
    right_ids = set(right)
    st.doc_ids_only_left_file = sorted(left_ids - right_ids)
    st.doc_ids_only_right_file = sorted(right_ids - left_ids)
    common = left_ids & right_ids

    for did in sorted(common):
        vl = _metric_value(left[did], metric)
        vr = _metric_value(right[did], metric)
        cl = _is_correct(vl, correct_threshold)
        cr = _is_correct(vr, correct_threshold)
        if cl is None:
            st.n_metric_missing_left += 1
        if cr is None:
            st.n_metric_missing_right += 1
        if cl is None or cr is None:
            continue
        if cl and cr:
            st.n_both_correct += 1
        elif cl and not cr:
            st.n_only_left += 1
            if (
                max_disagreement_ids is None
                or len(st.disagreement_left_wins_doc_ids) < max_disagreement_ids
            ):
                st.disagreement_left_wins_doc_ids.append(did)
        elif not cl and cr:
            st.n_only_right += 1
            if (
                max_disagreement_ids is None
                or len(st.disagreement_right_wins_doc_ids) < max_disagreement_ids
            ):
                st.disagreement_right_wins_doc_ids.append(did)
        else:
            st.n_both_wrong += 1
    return st


def find_samples_under_output_dir(output_dir: Path, task: str) -> Path:
    pat = f"samples_{task}_*.jsonl"
    matches = list(output_dir.rglob(pat))
    if not matches:
        raise FileNotFoundError(
            f"No {pat!r} under {output_dir} (log_samples output missing?)"
        )
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].resolve()


def resolve_run_to_samples_path(
    run: dict[str, Any], repo_root: Path
) -> tuple[Path, se.SubmitEvalConfig]:
    argv = me.run_dict_to_parse_argv(run)
    ns = se.parse_args(argv)
    cfg = se.make_cfg(ns, repo_root)
    path = find_samples_under_output_dir(cfg.output_dir, cfg.task)
    return path, cfg


def infer_task_from_samples_filename(name: str) -> str:
    """Parse ``samples_{task}_{date_id}.jsonl``."""
    base = Path(name).name
    m = re.match(r"^samples_(.+)\.jsonl$", base)
    if not m:
        return "unknown"
    rest = m.group(1)
    parts = rest.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return rest


def resolve_user_path(path: Path, task_hint: str | None, repo_root: Path) -> Path:
    """File -> as-is; directory -> newest matching samples."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")
    if task_hint:
        return find_samples_under_output_dir(path, task_hint)
    matches = list(path.rglob("samples_*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No samples_*.jsonl under {path}")
    if len(matches) == 1:
        return matches[0].resolve()
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].resolve()


def stub_cfg_for_display(
    repo_root: Path, task: str, run_tag: str
) -> se.SubmitEvalConfig:
    """Hub model + explicit run_tag for labels (no filesystem access needed)."""
    argv = [
        "--task",
        task,
        "--model_path",
        "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
        "--run_tag",
        run_tag,
    ]
    ns = se.parse_args(argv)
    return se.make_cfg(ns, repo_root)


def build_pair_comparison(
    st: PairwiseStats,
    *,
    metric: str,
    correct_threshold: float,
    left_path: Path,
    right_path: Path,
    left_cfg: se.SubmitEvalConfig,
    right_cfg: se.SubmitEvalConfig,
    left_run: dict[str, Any] | None,
    right_run: dict[str, Any] | None,
) -> PairComparison:
    stat, p = mcnemar_asymptotic(st.n_only_left, st.n_only_right)
    le = (
        left_run.get("experiment", "") if left_run else ""
    )
    re = (
        right_run.get("experiment", "") if right_run else ""
    )
    return PairComparison(
        stats=st,
        mcnemar_statistic=stat,
        mcnemar_pvalue=p,
        metric=metric,
        correct_threshold=correct_threshold,
        left_samples_path=str(left_path),
        right_samples_path=str(right_path),
        left_run_tag=left_cfg.run_tag,
        right_run_tag=right_cfg.run_tag,
        left_task=left_cfg.task,
        right_task=right_cfg.task,
        left_label=f"{le or 'manual'} | run_tag={left_cfg.run_tag} | task={left_cfg.task}",
        right_label=f"{re or 'manual'} | run_tag={right_cfg.run_tag} | task={right_cfg.task}",
    )


def print_report(pc: PairComparison) -> None:
    st = pc.stats
    print("--- Pairwise comparison ---")
    print(f"left:  {pc.left_label}")
    print(f"right: {pc.right_label}")
    print(f"files: {pc.left_samples_path}")
    print(f"       {pc.right_samples_path}")
    print(f"metric={pc.metric!r} correct_if value >= {pc.correct_threshold}")
    print()
    print("Contingency (doc_id present in BOTH files, metric defined for both):")
    print(f"  both_correct:    {st.n_both_correct}")
    print(f"  only_left:       {st.n_only_left}  (left correct, right wrong)")
    print(f"  only_right:      {st.n_only_right}  (left wrong, right correct)")
    print(f"  both_wrong:      {st.n_both_wrong}")
    print(f"  net_wins (left-right): {st.n_only_left - st.n_only_right}")
    print()
    print("Coverage:")
    print(f"  doc_id only in left file:  {len(st.doc_ids_only_left_file)}")
    print(f"  doc_id only in right file: {len(st.doc_ids_only_right_file)}")
    print(
        f"  metric missing (skipped):  left={st.n_metric_missing_left} right={st.n_metric_missing_right}"
    )
    print()
    print(
        f"McNemar (discordant pairs): statistic={pc.mcnemar_statistic:.6g} p={pc.mcnemar_pvalue:.6g}"
    )


def write_local_artifacts(
    out_dir: Path,
    comparisons: list[PairComparison],
    *,
    wandb_url: str | None,
    extra_meta: dict[str, Any],
    write_pair_disagreements: bool,
    max_examples: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wandb_url": wandb_url,
        "meta": extra_meta,
        "pairs": [],
    }
    for i, pc in enumerate(comparisons):
        st = pc.stats
        pair_entry: dict[str, Any] = {
            "left_label": pc.left_label,
            "right_label": pc.right_label,
            "left_run_tag": pc.left_run_tag,
            "right_run_tag": pc.right_run_tag,
            "left_task": pc.left_task,
            "right_task": pc.right_task,
            "left_samples_path": pc.left_samples_path,
            "right_samples_path": pc.right_samples_path,
            "metric": pc.metric,
            "correct_threshold": pc.correct_threshold,
            "n_both_correct": st.n_both_correct,
            "n_only_left": st.n_only_left,
            "n_only_right": st.n_only_right,
            "n_both_wrong": st.n_both_wrong,
            "net_wins_left_minus_right": st.n_only_left - st.n_only_right,
            "doc_ids_only_left_file": st.doc_ids_only_left_file,
            "doc_ids_only_right_file": st.doc_ids_only_right_file,
            "mcnemar_statistic": pc.mcnemar_statistic,
            "mcnemar_pvalue": pc.mcnemar_pvalue,
        }
        if write_pair_disagreements:
            pair_entry["disagreement_doc_ids_sample_left_wins"] = (
                st.disagreement_left_wins_doc_ids
            )
            pair_entry["disagreement_doc_ids_sample_right_wins"] = (
                st.disagreement_right_wins_doc_ids
            )
        payload["pairs"].append(pair_entry)

        if write_pair_disagreements:
            dj = (
                out_dir / f"disagreements_pair{i}.jsonl"
                if len(comparisons) > 1
                else out_dir / "disagreements.jsonl"
            )
            with dj.open("w", encoding="utf-8") as f:
                cap = max_examples if max_examples > 0 else None
                for kind, ids in (
                    (
                        "only_left_correct",
                        st.disagreement_left_wins_doc_ids[:cap]
                        if cap is not None
                        else st.disagreement_left_wins_doc_ids,
                    ),
                    (
                        "only_right_correct",
                        st.disagreement_right_wins_doc_ids[:cap]
                        if cap is not None
                        else st.disagreement_right_wins_doc_ids,
                    ),
                ):
                    for did in ids:
                        f.write(
                            json.dumps({"doc_id": did, "kind": kind}) + "\n"
                        )

    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_lines = [
        "# compare_eval_samples",
        f"wandb_url: {wandb_url or '(none)'}",
        "",
    ]
    for i, pc in enumerate(comparisons):
        manifest_lines.append(f"## pair {i}")
        manifest_lines.append(f"- left:  {pc.left_samples_path}")
        manifest_lines.append(f"- right: {pc.right_samples_path}")
        manifest_lines.append("")
    (out_dir / "MANIFEST.md").write_text("\n".join(manifest_lines), encoding="utf-8")


def wandb_log(
    *,
    comparisons: list[PairComparison],
    wandb_entity: str,
    wandb_project: str,
    run_name: str,
    wandb_tags: list[str],
) -> str | None:
    import wandb

    run = wandb.init(
        entity=wandb_entity if wandb_entity else None,
        project=wandb_project,
        name=run_name,
        tags=wandb_tags or None,
    )
    url: str | None = None
    try:
        if len(comparisons) == 1:
            pc = comparisons[0]
            st = pc.stats
            wandb.log(
                {
                    "n_both_correct": st.n_both_correct,
                    "n_only_left": st.n_only_left,
                    "n_only_right": st.n_only_right,
                    "n_both_wrong": st.n_both_wrong,
                    "net_wins_left_minus_right": st.n_only_left - st.n_only_right,
                    "mcnemar_statistic": pc.mcnemar_statistic,
                    "mcnemar_pvalue": pc.mcnemar_pvalue,
                    "left_run_tag": pc.left_run_tag,
                    "right_run_tag": pc.right_run_tag,
                    "metric": pc.metric,
                    "correct_threshold": pc.correct_threshold,
                }
            )
        else:
            table = wandb.Table(
                columns=[
                    "left_run_tag",
                    "right_run_tag",
                    "left_task",
                    "right_task",
                    "n_both_correct",
                    "n_only_left",
                    "n_only_right",
                    "n_both_wrong",
                    "net_wins",
                    "mcnemar_p",
                    "left_samples",
                    "right_samples",
                ]
            )
            for pc in comparisons:
                st = pc.stats
                table.add_data(
                    pc.left_run_tag,
                    pc.right_run_tag,
                    pc.left_task,
                    pc.right_task,
                    st.n_both_correct,
                    st.n_only_left,
                    st.n_only_right,
                    st.n_both_wrong,
                    st.n_only_left - st.n_only_right,
                    pc.mcnemar_pvalue,
                    pc.left_samples_path,
                    pc.right_samples_path,
                )
            wandb.log({"pairwise_summary": table})
        if run is not None:
            url = run.get_url()
    finally:
        wandb.finish()
    return url


def _default_metric_for_task(task: str) -> str:
    return "exact_match"


def _slug(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{h}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pairwise compare two lm-eval samples_*.jsonl dumps by doc_id."
    )
    ap.add_argument("--left", default="", help="Path to samples JSONL or directory")
    ap.add_argument("--right", default="", help="Path to samples JSONL or directory")
    ap.add_argument("--left-experiment", default="", help="RUNS experiment name (left)")
    ap.add_argument("--right-experiment", default="", help="RUNS experiment name (right)")
    ap.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="k=v",
        help="Filter RUNS rows (repeatable), same as manage_evals",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=500,
        help="Max M×N comparisons in run-grid mode; 0 = no limit",
    )
    ap.add_argument(
        "--metric",
        default="",
        help="Metric key in each JSONL row (default: exact_match)",
    )
    ap.add_argument(
        "--correct-threshold",
        type=float,
        default=1.0,
        help="Correct if metric >= this (default 1.0)",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Max disagreement doc_ids to keep per side; 0 means no limit (--write-disagreements)",
    )
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Fast-dLLM/v2 root")
    ap.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    ap.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", "ilm-extensions"),
    )
    ap.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    ap.add_argument("--wandb-name", default="", help="W&B run name (default auto)")
    ap.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags")
    ap.add_argument(
        "--write-disagreements",
        action="store_true",
        help="Write disagreements.jsonl under output dir (capped by --max-examples)",
    )
    args = ap.parse_args(argv)

    repo_root = args.repo_root.resolve()
    comparisons: list[PairComparison] = []
    all_warnings: list[str] = []

    use_grid = bool(args.left_experiment or args.right_experiment)
    if use_grid:
        if not args.left_experiment or not args.right_experiment:
            print(
                "Run-grid mode needs both --left-experiment and --right-experiment.",
                file=sys.stderr,
            )
            return 2
        if args.left or args.right:
            print("Do not mix --left/--right with --*-experiment.", file=sys.stderr)
            return 2
        filters = me.parse_filter_tokens(args.filter)
        left_rows = me.filter_runs(
            me.load_runs(),
            experiments=[args.left_experiment],
            filters=filters,
        )
        right_rows = me.filter_runs(
            me.load_runs(),
            experiments=[args.right_experiment],
            filters=filters,
        )
        if not left_rows:
            print("No RUNS rows matched left side.", file=sys.stderr)
            return 2
        if not right_rows:
            print("No RUNS rows matched right side.", file=sys.stderr)
            return 2
        m, n = len(left_rows), len(right_rows)
        total = m * n
        mp = args.max_pairs
        if mp and total > mp:
            print(
                f"Refusing {m}×{n}={total} pairs (max {mp}). "
                "Narrow --filter or use --max-pairs 0.",
                file=sys.stderr,
            )
            return 2
        pair_specs = [(lr, rr) for lr in left_rows for rr in right_rows]
    else:
        if not args.left or not args.right:
            print("Need --left and --right paths, or run-grid mode.", file=sys.stderr)
            return 2
        pair_specs = [(None, None)]

    max_d = None if args.max_examples == 0 else args.max_examples

    for pair_idx, spec in enumerate(pair_specs):
        if use_grid:
            lr, rr = spec  # type: ignore[misc]
            left_path, left_cfg = resolve_run_to_samples_path(lr, repo_root)
            right_path, right_cfg = resolve_run_to_samples_path(rr, repo_root)
            metric = args.metric or _default_metric_for_task(left_cfg.task)
            if left_cfg.task != right_cfg.task:
                print(
                    f"Warning: pair {pair_idx}: task mismatch "
                    f"{left_cfg.task!r} vs {right_cfg.task!r}",
                    file=sys.stderr,
                )
        else:
            lr = rr = None
            left_path = resolve_user_path(Path(args.left), None, repo_root)
            right_path = resolve_user_path(Path(args.right), None, repo_root)
            lt = infer_task_from_samples_filename(left_path.name)
            rt = infer_task_from_samples_filename(right_path.name)
            if lt != rt and lt != "unknown" and rt != "unknown":
                print(
                    f"Warning: inferred task from filename differs: {lt!r} vs {rt!r}",
                    file=sys.stderr,
                )
            resolved_task = (
                lt
                if lt != "unknown"
                else (rt if rt != "unknown" else "math500")
            )
            metric = args.metric or _default_metric_for_task(resolved_task)
            tag_l = f"cmp_l_{hashlib.sha1(str(left_path).encode()).hexdigest()[:8]}"
            tag_r = f"cmp_r_{hashlib.sha1(str(right_path).encode()).hexdigest()[:8]}"
            left_cfg = stub_cfg_for_display(repo_root, resolved_task, tag_l)
            right_cfg = stub_cfg_for_display(repo_root, resolved_task, tag_r)

        left_by, w1 = _parse_jsonl(left_path)
        right_by, w2 = _parse_jsonl(right_path)
        all_warnings.extend(w1)
        all_warnings.extend(w2)
        if left_by:
            rk0 = next(iter(left_by.values()))
            if metric not in rk0:
                print(
                    f"Warning: metric {metric!r} not in first left row keys: "
                    f"{sorted(rk0.keys())[:25]}…",
                    file=sys.stderr,
                )

        st = compare_loaded(
            left_by,
            right_by,
            metric=metric,
            correct_threshold=args.correct_threshold,
            max_disagreement_ids=max_d,
        )
        pc = build_pair_comparison(
            st,
            metric=metric,
            correct_threshold=args.correct_threshold,
            left_path=left_path,
            right_path=right_path,
            left_cfg=left_cfg,
            right_cfg=right_cfg,
            left_run=lr,
            right_run=rr,
        )
        comparisons.append(pc)
        print_report(pc)
        print()

    for w in all_warnings[:20]:
        print(f"[warn] {w}", file=sys.stderr)
    if len(all_warnings) > 20:
        print(
            f"[warn] … {len(all_warnings) - 20} more warnings omitted",
            file=sys.stderr,
        )

    slug_src = comparisons[0].left_samples_path + "|" + comparisons[0].right_samples_path
    if len(comparisons) > 1:
        slug_src += f"|pairs={len(comparisons)}"
    out_slug = _slug(slug_src)
    out_dir = repo_root / "eval_results_pairwise" / out_slug

    meta = {
        "argv": argv if argv is not None else sys.argv[1:],
        "n_pairs": len(comparisons),
    }
    wandb_url = None
    if not args.no_wandb:
        try:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            wname = args.wandb_name or out_slug[:120]
            wandb_url = wandb_log(
                comparisons=comparisons,
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                run_name=wname,
                wandb_tags=tags + ["compare_eval_samples"],
            )
        except Exception as e:  # pragma: no cover
            print(f"[warn] W&B logging failed ({e}); continuing with local artifacts.", file=sys.stderr)

    write_local_artifacts(
        out_dir,
        comparisons,
        wandb_url=wandb_url,
        extra_meta=meta,
        write_pair_disagreements=args.write_disagreements,
        max_examples=args.max_examples,
    )
    print(f"Wrote: {out_dir}")
    if wandb_url:
        print(f"wandb: {wandb_url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
