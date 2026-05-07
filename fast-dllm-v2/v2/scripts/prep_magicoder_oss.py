"""Convert ``ise-uiuc/Magicoder-OSS-Instruct-75K`` into Fast-dLLM v2's
conversation JSON format under ``data/magicoder_oss/train_conversation/``.

Each row has ``problem`` and ``solution`` (plus metadata such as ``lang``).
We emit the same structure as ``scripts/prep_numina.py`` so
``train_scripts/finetune.py`` + ``--conversation_template fast_dllm_v2`` work
unchanged.

Usage (from ``Fast-dLLM/v2/``):

    python scripts/prep_magicoder_oss.py --out_dir data/magicoder_oss

    # Python-only slice (closer to HumanEval / MBPP + EvalPlus):
    python scripts/prep_magicoder_oss.py --out_dir data/magicoder_oss --lang python
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

from datasets import load_dataset


def _row_to_instance(ex: dict) -> dict | None:
    problem = ex.get("problem")
    solution = ex.get("solution")
    if problem is None or solution is None:
        return None
    return {
        "messages": [
            {"role": "user", "content": str(problem).strip()},
            {"role": "assistant", "content": str(solution).strip()},
        ]
    }


def _write(out_path: str, instances: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"type": "conversation", "instances": list(instances)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return len(payload["instances"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf_dataset",
        default="ise-uiuc/Magicoder-OSS-Instruct-75K",
        help="HF dataset id.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split name (this dataset is train-only).",
    )
    p.add_argument(
        "--lang",
        default=None,
        help="If set, keep only rows where column `lang` equals this (e.g. `python`).",
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Cap rows after optional lang filter (debug / quick runs).",
    )
    p.add_argument(
        "--out_dir",
        default="data/magicoder_oss",
        help="Output root. Writes {out_dir}/train_conversation/train_<N>.json",
    )
    args = p.parse_args()

    print(f"Loading {args.hf_dataset} split={args.split!r} ...")
    ds = load_dataset(args.hf_dataset, split=args.split)

    if args.lang is not None:
        before = len(ds)
        lang_lower = args.lang.lower()

        def _lang_ok(ex: dict) -> bool:
            return str(ex.get("lang", "")).lower() == lang_lower

        ds = ds.filter(_lang_ok, desc=f"lang == {args.lang!r}")
        print(f"Lang filter {args.lang!r}: {before} -> {len(ds)} rows")

    if args.max_rows is not None:
        ds = ds.select(range(min(len(ds), args.max_rows)))

    instances: list[dict] = []
    for ex in ds:
        inst = _row_to_instance(ex)
        if inst is not None:
            instances.append(inst)
    train_path = os.path.join(
        args.out_dir, "train_conversation", f"train_{len(instances)}.json"
    )
    n = _write(train_path, instances)
    print(f"Wrote {n} train instances -> {train_path}")


if __name__ == "__main__":
    main()
