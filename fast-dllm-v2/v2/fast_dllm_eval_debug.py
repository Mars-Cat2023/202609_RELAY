# SPDX-License-Identifier: Apache-2.0
"""Process-global debug flags for Fast-dLLM v2 lm-eval runs.

Set once from ``Fast_dLLM_v2EvalHarness`` via ``--model_args`` so
``generation_functions.batch_sample`` can print trajectories without threading
most flags through every call. ``small_block_size`` is included so debug lines can
name the active scheduling grain.
"""

from __future__ import annotations

import sys
DEBUG_PRINT: bool = False
"""If True, print partial denoising trajectories from ``batch_sample``."""

DEBUG_PRINT_PROMPT: bool = False
"""If True, print question / final answer blocks in ``generate_until``."""

_RANK: int = 0
_WORLD_SIZE: int = 1
_SMALL_BLOCK_SIZE: int = 8


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() in ("true", "1", "yes")


def configure(
    *,
    debug_print: bool = False,
    debug_print_prompt: bool = False,
    rank: int = 0,
    world_size: int = 1,
    small_block_size: int = 8,
) -> None:
    global DEBUG_PRINT, DEBUG_PRINT_PROMPT, _RANK, _WORLD_SIZE, _SMALL_BLOCK_SIZE
    DEBUG_PRINT = bool(debug_print)
    DEBUG_PRINT_PROMPT = bool(debug_print_prompt)
    _RANK = int(rank)
    _WORLD_SIZE = int(world_size)
    _SMALL_BLOCK_SIZE = int(small_block_size)


def is_leader() -> bool:
    return _RANK == 0


def print_trajectory_step(
    *,
    x_t: "object",  # torch.Tensor
    seq_len: "object",  # torch.Tensor 1D long, prompt length per row
    tokenizer,
    mask_id: int,
    block_idx: int,
    inner_step: int,
    flush: bool = True,
) -> None:
    """Decode batch index 0 from ``seq_len[0]:`` and print (leader rank only)."""
    if not DEBUG_PRINT or not is_leader():
        return
    import torch

    if not isinstance(x_t, torch.Tensor) or x_t.shape[0] == 0:
        return
    b = 0
    sl = int(seq_len[b].item()) if seq_len.numel() > b else 0
    row = x_t[b, sl:]
    text = tokenizer.decode(row.tolist(), skip_special_tokens=False)
    mask_tok = "[M]"
    # Surface mask id as a visible placeholder in debug output
    try:
        m_str = tokenizer.convert_ids_to_tokens([mask_id])[0]
    except Exception:
        m_str = str(mask_id)
    preview = text.replace(m_str, str(mask_tok))
    print(
        f"[blk={block_idx} step={inner_step} sbs={_SMALL_BLOCK_SIZE}] {preview!r}",
        file=sys.stderr,
        flush=flush,
    )
