#!/usr/bin/env python3
"""Mean CE loss (NLL) on Numina held-out JSON from ``prep_numina.py``.

Uses the same chat template as training (``fast_dllm_v2``) and masks
non-assistant tokens to ``-100``, matching SFT. In ``eval`` mode the
Fast-dLLM v2 head does not apply training-time random noising / BD3
doubling, so this is a standard next-token CE on the assistant span.

Example::

    export PYTHONPATH=/path/to/Fast-dLLM/v2/src:$PYTHONPATH
    python scripts/eval_numina_heldout_nll.py \\
        --model_path output_models/numina_1p5B/vanilla_bs2x16_ep2 \\
        --data_path data/numina/test_conversation/test_2000.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmflow.utils.conversation_template import Fast_dLLM_v2_TEMPLATE


def _load_instances(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "instances" in payload:
        return payload["instances"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected JSON shape in {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HF repo id or local checkpoint dir (with config + weights).",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="data/numina/test_conversation/test_2000.json",
        help="Held-out file from prep_numina (conversation JSON).",
    )
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise SystemExit(f"Missing data file: {data_path.resolve()}")

    instances = _load_instances(data_path)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.to(args.device)
    model.eval()

    rows: list[dict] = []
    for inst in instances:
        msgs = inst["messages"]
        enc = tok.apply_chat_template(
            msgs,
            chat_template=Fast_dLLM_v2_TEMPLATE,
            return_assistant_tokens_mask=True,
            return_dict=True,
            add_generation_prompt=False,
        )
        ids = enc["input_ids"]
        mask = enc["assistant_masks"]
        if len(ids) > args.max_length:
            ids = ids[: args.max_length]
            mask = mask[: args.max_length]
        labels = [
            tid if m == 1 else -100 for tid, m in zip(ids, mask, strict=True)
        ]
        if all(x == -100 for x in labels):
            continue
        rows.append({"input_ids": ids, "labels": labels})

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id

    total_loss = 0.0
    total_tok = 0
    n_batches = 0

    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            max_len = max(len(r["input_ids"]) for r in batch)
            in_b = []
            lab_b = []
            attn = []
            for r in batch:
                ids = r["input_ids"]
                lab = r["labels"]
                pad_n = max_len - len(ids)
                in_b.append(ids + [pad_id] * pad_n)
                lab_b.append(lab + [-100] * pad_n)
                attn.append([1] * len(ids) + [0] * pad_n)

            input_ids = torch.tensor(in_b, device=args.device, dtype=torch.long)
            labels = torch.tensor(lab_b, device=args.device, dtype=torch.long)
            attention_mask = torch.tensor(attn, device=args.device, dtype=torch.long)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if out.loss is None:
                raise RuntimeError("Model returned no loss — check labels / forward path.")

            # HF loss is mean over non-ignored positions in the batch; recover token count.
            n_lab = (labels != -100).sum().item()
            total_loss += out.loss.item() * n_lab
            total_tok += n_lab
            n_batches += 1

    mean_nll = total_loss / max(total_tok, 1)
    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "data_path": str(data_path),
                "n_examples": len(rows),
                "n_batches": n_batches,
                "labeled_tokens": int(total_tok),
                "mean_nll_natural": mean_nll,
                "mean_ce": mean_nll,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
