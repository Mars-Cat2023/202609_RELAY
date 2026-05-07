"""Join two JSONL trajectory dumps by ``example_index`` (e.g. MLM vs BPTT checkpoints).

Example::

    python -m doublebackprop.pair_trajectory_dumps \\
      --mlm mlm.jsonl --bptt bptt.jsonl --out paired.jsonl --only-bptt-wins
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_by_example(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        k = int(r["example_index"])
        if k in out:
            raise ValueError(f"Duplicate example_index {k} in file")
        out[k] = r
    return out


def iter_paired_rows(
    mlm_rows: List[Dict[str, Any]],
    bptt_rows: List[Dict[str, Any]],
    *,
    only_bptt_wins: bool,
) -> Iterator[Dict[str, Any]]:
    mlm_i = index_by_example(mlm_rows)
    bptt_i = index_by_example(bptt_rows)
    keys = sorted(set(mlm_i.keys()) & set(bptt_i.keys()))
    for k in keys:
        a, b = mlm_i[k], bptt_i[k]
        em_m = bool(a.get("exact_match", False))
        em_b = bool(b.get("exact_match", False))
        if only_bptt_wins and not (em_m is False and em_b is True):
            continue
        yield {
            "example_index": k,
            "mlm_exact_match": em_m,
            "bptt_exact_match": em_b,
            "mlm": {kk: a[kk] for kk in a if kk != "example_index"},
            "bptt": {kk: b[kk] for kk in b if kk != "example_index"},
        }


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mlm", type=Path, required=True, help="JSONL from MLM / baseline run")
    p.add_argument("--bptt", type=Path, required=True, help="JSONL from BPTT+loopholing+PUMA run")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL")
    p.add_argument(
        "--only-bptt-wins",
        action="store_true",
        help="Keep only rows where MLM exact_match is False and BPTT exact_match is True",
    )
    args = p.parse_args(argv)

    mlm_rows = load_jsonl(args.mlm)
    bptt_rows = load_jsonl(args.bptt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in iter_paired_rows(
            mlm_rows, bptt_rows, only_bptt_wins=args.only_bptt_wins
        ):
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"Wrote {n} paired rows to {args.out}")


if __name__ == "__main__":
    main()
