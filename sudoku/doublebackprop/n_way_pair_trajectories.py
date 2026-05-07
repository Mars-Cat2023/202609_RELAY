"""N-way join of decode-trajectory dumps + HF solver metadata into one parquet.

Replaces the 2-way ``pair_trajectory_dumps`` for the qualitative Sudoku study.

Each input JSONL is produced by ``python -m doublebackprop.dump_decode_trajectories``
(see :mod:`doublebackprop.dump_decode_trajectories`) and is keyed at *load*
time by the SHA1 hash of the recovered puzzle question (taken from
``trajectory[0]``). All inputs are joined on ``puzzle_hash`` and merged with
the Hugging Face dataset
``brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver`` (configurable)
to attach solver metadata: ``num_steps``, ``strategies_used``, ``rating``,
``source``, the per-step solver ``trajectory`` (textual board states), and
``answer``.

Output schema (one row per (run, threshold τ, puzzle)):

- Identity: ``run_id``, ``objective``, ``embed_tying``, ``seed`` (optional;
  ``None`` for manifests written before the seeded sweep), ``tau``,
  ``puzzle_hash``, ``question`` (81-char string with ``.``), ``source``,
  ``rating``, ``num_steps_solver``, ``strategies_used`` (list of strings),
  ``hardest_strategy``, ``num_clues``.
- Outcome: ``exact_match``, ``rollout_steps``, ``pred_text``, ``truth_text``,
  ``answer_grid_flat``, ``final_grid_flat``.
- Per-step trace (compact): ``model_traj_digit_flat`` (T x 81 ints in 0..9),
  ``model_fill_step_flat`` (length-81 floats; nan for unfilled),
  ``solver_fill_step_flat``.
- Algorithmic primitives (from :mod:`doublebackprop.sudoku_analysis`):
  ``first_step_naked_single_rate``, ``first_step_hidden_single_rate``,
  ``first_step_forced_rate``, ``first_step_advanced_rate``,
  ``first_step_illegal_rate``, ``first_step_correct_rate``,
  ``violations_total_steps``, ``final_violations``, ``purity_per_step``.
- Alignment: ``fill_order_spearman``, ``prefix_jaccard_at_5``,
  ``prefix_jaccard_at_10``, ``prefix_jaccard_at_20``, ``prefix_jaccard_at_50pct``.

Usage::

    python -m doublebackprop.n_way_pair_trajectories \\
      --manifest manifest.csv \\
      --hf-dataset brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver \\
      --hf-split test \\
      --out outputs/sudoku_analysis.parquet

Where ``manifest.csv`` has header ``run_id,objective,embed_tying[,seed],tau,jsonl``.
The ``seed`` column is optional: backwards-compatible with the original 8-run
manifest format, and populated by the seeded sweep produced by
``submit_sudoku_qualitative_dumps.sh`` with ``SEEDS="1 2 3"``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .sudoku_analysis import (
    digit_grid_to_string,
    hardest_strategy_tier,
    ids_to_digit_grid,
    normalize_question,
    puzzle_hash,
    summarize_trajectory,
    trajectory_to_digit_grids,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifestEntry:
    run_id: str
    objective: str
    embed_tying: str
    tau: float
    jsonl: Path
    seed: Optional[int] = None


def load_manifest(path: Path) -> List[ManifestEntry]:
    """Read a CSV with header ``run_id,objective,embed_tying[,seed],tau,jsonl``.

    The ``seed`` column is optional (older manifests written before the seeded
    sweep do not have it). Empty / missing values become ``None``.
    Lines starting with ``#`` are ignored.
    """
    entries: List[ManifestEntry] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#"))
        )
        for row in reader:
            seed_raw = (row.get("seed") or "").strip()
            seed_val: Optional[int] = None
            if seed_raw not in ("", "None", "none"):
                try:
                    seed_val = int(seed_raw)
                except ValueError:
                    seed_val = None
            entries.append(
                ManifestEntry(
                    run_id=row["run_id"].strip(),
                    objective=row["objective"].strip(),
                    embed_tying=row["embed_tying"].strip(),
                    tau=float(row["tau"]),
                    jsonl=Path(row["jsonl"]).expanduser().resolve(),
                    seed=seed_val,
                )
            )
    return entries


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def question_from_trajectory(record: Dict[str, Any]) -> str:
    """Recover the 81-char puzzle question from the captured trajectory.

    The first trajectory snapshot is the masked input *before* any decoding
    step (see ``ConfidenceBasedPredictor.predict``), so non-mask cells are
    exactly the original clues.
    """
    traj = record.get("trajectory")
    if not traj:
        raise ValueError("Record missing trajectory.")
    grid = ids_to_digit_grid(traj[0])
    return digit_grid_to_string(grid)


def load_hf_metadata(
    name: str,
    split: str,
) -> pd.DataFrame:
    """Load and index the HF Sudoku-solver dataset by ``puzzle_hash``.

    Columns retained: ``question``, ``answer``, ``rating``, ``num_steps``,
    ``trajectory`` (list of 81-char strings), ``strategies_used`` (list of strings),
    ``source``, plus the derived ``puzzle_hash`` and ``hardest_strategy``.

    Falls back to ``datasets.load_dataset`` so any HF id is supported. When
    the dataset is unavailable (e.g. offline), the user can pre-download a
    parquet snapshot and pass ``--hf-parquet`` instead.
    """
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    df = ds.to_pandas()
    df["puzzle_hash"] = df["question"].apply(puzzle_hash)
    df["question_norm"] = df["question"].apply(normalize_question)
    df["hardest_strategy"] = df["strategies_used"].apply(
        lambda lst: hardest_strategy_tier(lst if lst is not None else [])
    )
    keep = [
        "puzzle_hash",
        "question_norm",
        "answer",
        "rating",
        "num_steps",
        "trajectory",
        "strategies_used",
        "source",
        "hardest_strategy",
    ]
    df = df[keep].rename(
        columns={
            "num_steps": "num_steps_solver",
            "trajectory": "solver_trajectory",
            "question_norm": "question",
        }
    )
    df = df.drop_duplicates(subset=["puzzle_hash"], keep="first").reset_index(drop=True)
    return df


def load_hf_metadata_from_parquet(path: Path) -> pd.DataFrame:
    """Same schema as :func:`load_hf_metadata`, but loaded from a local parquet."""
    df = pd.read_parquet(path)
    if "puzzle_hash" not in df.columns:
        if "question" not in df.columns:
            raise ValueError(
                f"{path} lacks 'puzzle_hash' / 'question'; cannot derive join key."
            )
        df["puzzle_hash"] = df["question"].apply(puzzle_hash)
    if "hardest_strategy" not in df.columns and "strategies_used" in df.columns:
        df["hardest_strategy"] = df["strategies_used"].apply(
            lambda lst: hardest_strategy_tier(lst if lst is not None else [])
        )
    return df


def build_run_dataframe(
    entry: ManifestEntry,
    hf_index: pd.DataFrame,
    *,
    keep_full_trajectory: bool,
    max_examples: Optional[int] = None,
) -> pd.DataFrame:
    """Convert one JSONL dump into a wide row table joined with HF metadata."""
    hf_lookup = hf_index.set_index("puzzle_hash")
    rows: List[Dict[str, Any]] = []
    for k, rec in enumerate(iter_jsonl(entry.jsonl)):
        if max_examples is not None and k >= max_examples:
            break
        try:
            question_str = question_from_trajectory(rec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s row %d (no trajectory): %s", entry.jsonl, k, exc)
            continue
        ph = puzzle_hash(question_str)
        if ph not in hf_lookup.index:
            logger.warning(
                "puzzle_hash %s (run=%s, idx=%d) not found in HF metadata; "
                "this is expected if the run used a different split / preprocessing.",
                ph,
                entry.run_id,
                k,
            )
            solver_traj = None
            hf_row = None
        else:
            hf_row = hf_lookup.loc[ph]
            solver_traj = list(hf_row["solver_trajectory"])
        summary = summarize_trajectory(
            rec["trajectory"],
            rec["target_ids"],
            solver_trajectory=solver_traj,
        )
        out: Dict[str, Any] = {
            "run_id": entry.run_id,
            "objective": entry.objective,
            "embed_tying": entry.embed_tying,
            "seed": entry.seed,
            "tau": entry.tau,
            "jsonl": str(entry.jsonl),
            "example_index": int(rec.get("example_index", k)),
            "puzzle_hash": ph,
            "question": question_str,
            "exact_match": bool(rec.get("exact_match", False)),
            "rollout_steps": int(rec.get("rollout_steps_per_sample", 0)),
            "pred_text": rec.get("pred_text"),
            "truth_text": rec.get("truth_text"),
        }
        if hf_row is not None:
            out["answer"] = hf_row.get("answer")
            out["rating"] = (
                int(hf_row["rating"]) if pd.notna(hf_row.get("rating")) else None
            )
            out["num_steps_solver"] = (
                int(hf_row["num_steps_solver"])
                if pd.notna(hf_row.get("num_steps_solver"))
                else None
            )
            out["strategies_used"] = (
                list(hf_row["strategies_used"])
                if hf_row.get("strategies_used") is not None
                else []
            )
            out["hardest_strategy"] = hf_row.get("hardest_strategy", "Easy")
            out["source"] = hf_row.get("source")
        out.update(summary)
        if keep_full_trajectory:
            digit_traj = trajectory_to_digit_grids(rec["trajectory"])
            out["model_traj_digit_flat"] = digit_traj.reshape(digit_traj.shape[0], -1).tolist()
        rows.append(out)
    df = pd.DataFrame(rows)
    return df


def join_runs(
    manifest: List[ManifestEntry],
    hf_index: pd.DataFrame,
    *,
    keep_full_trajectory: bool = True,
    max_examples_per_run: Optional[int] = None,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for entry in manifest:
        logger.info(
            "Loading %s (run=%s, objective=%s, tying=%s, tau=%.2f)",
            entry.jsonl,
            entry.run_id,
            entry.objective,
            entry.embed_tying,
            entry.tau,
        )
        df = build_run_dataframe(
            entry,
            hf_index,
            keep_full_trajectory=keep_full_trajectory,
            max_examples=max_examples_per_run,
        )
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def restrict_to_shared_puzzles(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only puzzles that appear under every (run_id, tau) combination.

    Ensures fair comparisons; drops puzzles unique to a subset of runs.
    """
    counts = df.groupby("puzzle_hash")["run_id"].nunique()
    expected = df["run_id"].nunique()
    keep = set(counts[counts == expected].index.tolist())
    n_before = len(df)
    df2 = df[df["puzzle_hash"].isin(keep)].reset_index(drop=True)
    logger.info(
        "Restricted shared puzzles: %d / %d unique hashes; %d / %d rows kept.",
        len(keep),
        df["puzzle_hash"].nunique(),
        len(df2),
        n_before,
    )
    return df2


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="CSV manifest.")
    p.add_argument(
        "--hf-dataset",
        type=str,
        default="brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver",
    )
    p.add_argument("--hf-split", type=str, default="test")
    p.add_argument(
        "--hf-parquet",
        type=Path,
        default=None,
        help="Use a local pre-downloaded parquet of the HF dataset instead of `datasets`.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output parquet path.")
    p.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Drop the per-step digit trajectory to shrink the parquet.",
    )
    p.add_argument(
        "--max-examples-per-run",
        type=int,
        default=None,
        help="Optional cap on rows per (run, τ) for quick iterations.",
    )
    p.add_argument(
        "--shared-only",
        action="store_true",
        help="Keep only puzzles present in every run (for matched comparisons).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    manifest = load_manifest(args.manifest)
    if not manifest:
        raise SystemExit(f"Manifest is empty: {args.manifest}")
    if args.hf_parquet is not None:
        hf_index = load_hf_metadata_from_parquet(args.hf_parquet)
    else:
        hf_index = load_hf_metadata(args.hf_dataset, args.hf_split)
    df = join_runs(
        manifest,
        hf_index,
        keep_full_trajectory=not args.no_trajectory,
        max_examples_per_run=args.max_examples_per_run,
    )
    if args.shared_only and not df.empty:
        df = restrict_to_shared_puzzles(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("Wrote %d rows to %s", len(df), args.out)


if __name__ == "__main__":
    main()
