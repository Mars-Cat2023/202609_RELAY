"""Convert ``jacopo-minniti/NuminaMath-CoT-LLaDA-Goldilocks`` (or any
``{problem, solution}`` HF dataset) into Fast-dLLM v2's conversation JSON
format under ``data/numina/{train,test}_conversation/``.

Output layout (matches ``data/alpaca/train_conversation/``):

    data/numina/train_conversation/train_<N>.json
    data/numina/test_conversation/test_<N>.json

Each file is one JSON object:

    {
      "type": "conversation",
      "instances": [
        {"messages": [
          {"role": "user",      "content": "..."},
          {"role": "assistant", "content": "..."}
        ]},
        ...
      ]
    }

The ``fast_dllm_v2`` conversation template (``<|im_start|>...<|im_end|>``)
is applied at *training* time by ``Finetuner`` via
``--conversation_template fast_dllm_v2`` — same as for Alpaca. We only
emit the structured messages here.

Usage (from ``Fast-dLLM/v2/``):

    python scripts/prep_numina.py \
        --hf_dataset jacopo-minniti/NuminaMath-CoT-LLaDA-Goldilocks \
        --train_size 50000 --test_size 2000 \
        --out_dir data/numina

The defaults match the slice used by stateflow's
``configs/keywords.yaml`` (``numina = ...[train:50000,test:2000]``) so
the two repos can compare apples to apples.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

from datasets import load_dataset


def _instances_from_split(split, n: int | None) -> list[dict]:
    n = len(split) if n is None else min(n, len(split))
    out = []
    for i in range(n):
        ex = split[i]
        problem = ex.get("problem")
        solution = ex.get("solution")
        if problem is None or solution is None:
            # Some NuminaMath variants use ``messages`` directly. If so,
            # forward them; else skip the row.
            msgs = ex.get("messages")
            if msgs is None:
                continue
            out.append({"messages": msgs})
            continue
        out.append({
            "messages": [
                {"role": "user", "content": str(problem).strip()},
                {"role": "assistant", "content": str(solution).strip()},
            ]
        })
    return out


def _write(out_path: str, instances: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"type": "conversation", "instances": list(instances)}
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    return len(payload["instances"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf_dataset",
        default="jacopo-minniti/NuminaMath-CoT-LLaDA-Goldilocks",
        help="HF dataset id (default mirrors the stateflow `numina` keyword).",
    )
    p.add_argument(
        "--train_size", type=int, default=50000,
        help="Take first N rows of the train split (None = all).",
    )
    p.add_argument(
        "--test_size", type=int, default=2000,
        help="Take first N rows of the test split (None = all). 0 = skip.",
    )
    p.add_argument(
        "--out_dir", default="data/numina",
        help="Output root. Will write {out_dir}/{train,test}_conversation/.",
    )
    args = p.parse_args()

    print(f"Loading {args.hf_dataset} ...")
    ds = load_dataset(args.hf_dataset)

    train_inst = _instances_from_split(ds["train"], args.train_size)
    train_path = os.path.join(args.out_dir, "train_conversation", f"train_{len(train_inst)}.json")
    n_train = _write(train_path, train_inst)
    print(f"Wrote {n_train} train instances -> {train_path}")

    if args.test_size > 0 and "test" in ds:
        test_inst = _instances_from_split(ds["test"], args.test_size)
        test_path = os.path.join(args.out_dir, "test_conversation", f"test_{len(test_inst)}.json")
        n_test = _write(test_path, test_inst)
        print(f"Wrote {n_test} test instances  -> {test_path}")


if __name__ == "__main__":
    main()
