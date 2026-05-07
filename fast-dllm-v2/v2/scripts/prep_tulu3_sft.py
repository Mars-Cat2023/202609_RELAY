"""Convert ``allenai/tulu-3-sft-mixture`` into Fast-dLLM v2's conversation
JSON format under ``data/tulu3_sft/train_conversation/``.

Each Tulu row carries a Qwen-shape ``messages`` field (system / user /
assistant, possibly multi-turn) which the ``fast_dllm_v2`` Jinja chat
template handles natively (see ``src/lmflow/utils/conversation_template/qwen.py``
``Fast_dLLM_v2_TEMPLATE``). We therefore emit ``{"messages": ex["messages"]}``
as-is.

The HF dataset is concatenated source-by-source (CoCoNot, then FLAN, then
No Robots, ...). To guarantee that early checkpoints see a representative
mix across all sources -- not just the first few in HF order -- this
script **always pre-shuffles on disk** with ``--shuffle_seed=42`` (default).
HF Trainer's per-epoch ``SeedableRandomSampler`` shuffles on top of that;
this is belt-and-suspenders insurance.

Usage (from ``Fast-dLLM/v2/``):

    # Default: full mixture, pre-shuffled, no source filter.
    python scripts/prep_tulu3_sft.py --out_dir data/tulu3_sft

    # Optional cap (subsample after shuffle):
    python scripts/prep_tulu3_sft.py --out_dir data/tulu3_sft_100k --max_rows 100000

    # Filter to math + code subsets only:
    python scripts/prep_tulu3_sft.py --out_dir data/tulu3_sft_mathcode \
        --include_sources \
        ai2-adapt-dev/personahub_math_v5_regen_149960,\
allenai/tulu-3-sft-personas-math-grade,\
ai2-adapt-dev/personahub_code_v2_34999,\
ai2-adapt-dev/numinamath_tir_math_decontaminated,\
ai2-adapt-dev/evol_codealpaca_heval_decontaminated
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Iterable

from datasets import load_dataset


def _row_to_instance(ex: dict) -> dict | None:
    """Pass the row's ``messages`` field through unchanged.

    Tulu rows are already in the Qwen ``[{role, content}, ...]`` shape
    that ``Fast_dLLM_v2_TEMPLATE`` (Jinja chat template) consumes via
    ``tokenizer.apply_chat_template``. Skip rows missing or with an
    empty messages list -- defensive only; the official mixture has no
    such rows.
    """
    messages = ex.get("messages")
    if not messages:
        return None
    return {"messages": messages}


def _write(out_path: str, instances: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"type": "conversation", "instances": list(instances)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return len(payload["instances"])


def _parse_csv(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    items = [s.strip() for s in arg.split(",") if s.strip()]
    return items or None


def _log_source_counts(label: str, ds) -> None:
    """Print per-source counts so the final mixture is obvious from stdout.

    ``ds`` is an HF Dataset with a ``source`` column; we materialise that
    one column only (cheap) and tally with ``Counter``.
    """
    if "source" not in ds.column_names:
        print(f"[{label}] {len(ds)} rows (no `source` column to tally)")
        return
    counts = Counter(ds["source"])
    total = sum(counts.values())
    print(f"[{label}] {total} rows across {len(counts)} sources:")
    for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>8d}  {src}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf_dataset",
        default="allenai/tulu-3-sft-mixture",
        help="HF dataset id.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split name (Tulu-3-SFT-Mixture is train-only).",
    )
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help=(
            "Seed used to shuffle on disk before writing. Use -1 to "
            "disable on-disk shuffle (not recommended -- the HF order is "
            "source-clumped)."
        ),
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help=(
            "Optional cap (applied AFTER source filter and shuffle). "
            "Default: no cap (~939k rows)."
        ),
    )
    p.add_argument(
        "--include_sources",
        default=None,
        help=(
            "Comma-separated `source` values to KEEP. Default: keep all. "
            "Use the exact `source` strings from the dataset (e.g. "
            "`ai2-adapt-dev/personahub_math_v5_regen_149960`)."
        ),
    )
    p.add_argument(
        "--exclude_sources",
        default=None,
        help=(
            "Comma-separated `source` values to DROP. Applied after "
            "include_sources. Default: drop nothing."
        ),
    )
    p.add_argument(
        "--out_dir",
        default="data/tulu3_sft",
        help=(
            "Output root. Writes {out_dir}/train_conversation/train_*.json "
            "(sharded; see --shard_size)."
        ),
    )
    p.add_argument(
        "--shard_size",
        type=int,
        default=50000,
        help=(
            "Rows per output shard. The default 50000 keeps each shard's "
            "`messages` column safely under PyArrow's 2 GB int32-offset "
            "string-column cap (Magicoder-full at 75k -> ~270 MB; Tulu "
            "rows are similar, so 50k -> ~180 MB worst-case). HF Datasets "
            "globs `train_*.json` in the directory and treats each file "
            "as a separate Arrow chunk, so multiple shards avoid the "
            "single-column overflow that hits 939k-row mixtures. Pass 0 "
            "or a value >= total_rows to disable sharding (single file, "
            "matches `prep_magicoder_oss.py` shape -- only safe for small "
            "datasets like Magicoder)."
        ),
    )
    args = p.parse_args()

    print(f"Loading {args.hf_dataset} split={args.split!r} ...")
    ds = load_dataset(args.hf_dataset, split=args.split)
    _log_source_counts("loaded", ds)

    include = _parse_csv(args.include_sources)
    exclude = _parse_csv(args.exclude_sources)

    if include is not None:
        include_set = set(include)

        def _in_include(ex: dict) -> bool:
            return ex.get("source") in include_set

        before = len(ds)
        ds = ds.filter(_in_include, desc=f"include_sources ({len(include_set)})")
        print(f"include_sources filter: {before} -> {len(ds)} rows")
        _log_source_counts("after include_sources", ds)

    if exclude is not None:
        exclude_set = set(exclude)

        def _not_in_exclude(ex: dict) -> bool:
            return ex.get("source") not in exclude_set

        before = len(ds)
        ds = ds.filter(_not_in_exclude, desc=f"exclude_sources ({len(exclude_set)})")
        print(f"exclude_sources filter: {before} -> {len(ds)} rows")
        _log_source_counts("after exclude_sources", ds)

    if args.shuffle_seed is not None and args.shuffle_seed >= 0:
        print(f"Shuffling on disk with seed={args.shuffle_seed} ...")
        ds = ds.shuffle(seed=args.shuffle_seed)
    else:
        print(
            "WARNING: skipping on-disk shuffle (--shuffle_seed=-1). "
            "HF order is source-clumped; early checkpoints will be "
            "non-representative unless HF Trainer shuffling is enabled."
        )

    if args.max_rows is not None:
        before = len(ds)
        ds = ds.select(range(min(len(ds), args.max_rows)))
        print(f"max_rows cap: {before} -> {len(ds)} rows")
        _log_source_counts("after max_rows", ds)

    instances: list[dict] = []
    for ex in ds:
        inst = _row_to_instance(ex)
        if inst is not None:
            instances.append(inst)

    out_dir = os.path.join(args.out_dir, "train_conversation")
    os.makedirs(out_dir, exist_ok=True)

    # Wipe any prior `train_*.json` shards in the output dir so a re-prep
    # with a different shard size / row count doesn't leave a mix of
    # stale shards behind.
    for stale in sorted(os.listdir(out_dir)):
        if stale.startswith("train_") and stale.endswith(".json"):
            stale_path = os.path.join(out_dir, stale)
            os.remove(stale_path)
            print(f"Removed stale shard: {stale_path}")

    n_total = len(instances)
    if args.shard_size and args.shard_size > 0 and args.shard_size < n_total:
        shard_size = args.shard_size
        n_shards = (n_total + shard_size - 1) // shard_size
        for i in range(n_shards):
            chunk = instances[i * shard_size : (i + 1) * shard_size]
            shard_path = os.path.join(
                out_dir, f"train_shard{i:03d}of{n_shards:03d}_{len(chunk)}.json"
            )
            n = _write(shard_path, chunk)
            print(f"Wrote shard {i + 1}/{n_shards}: {n} rows -> {shard_path}")
        print(f"Total: {n_total} train instances across {n_shards} shards.")
    else:
        # Single-file path (matches prep_magicoder_oss.py shape). Only
        # safe for datasets small enough to keep the `messages` Arrow
        # string column under 2 GB.
        single_path = os.path.join(out_dir, f"train_{n_total}.json")
        n = _write(single_path, instances)
        print(f"Wrote {n} train instances -> {single_path}")


if __name__ == "__main__":
    main()
