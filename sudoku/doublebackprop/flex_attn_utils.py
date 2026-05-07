"""Shared FlexAttention helpers for loopholing training (packed batches)."""

from typing import Any

import torch


def block_mask_from_segment_ids(
    segment_ids: torch.Tensor,
    device: torch.device,
) -> Any:
    """Build a FlexAttention ``BlockMask`` from per-row ``segment_ids`` (*, seq_len).

    Use this when the attention layout must match a **tensor buffer** (e.g. ``StreamingBatch``
    storage) after **partial eviction**: row ``b`` of ``segment_ids`` must describe row ``b``
    of ``input_ids``.  Do **not** reuse the collator's batched ``BlockMask`` from the incoming
    microbatch after eviction unless every storage row was replaced with the matching batch row
    (generally false when only some slots evict).
    """
    from torch.nn.attention.flex_attention import create_block_mask

    seg = segment_ids.to(device)
    bsz, seq_len = seg.shape
    seg_flat = seg.reshape(-1)
    seq_len_int: int = seq_len

    def _doc_mask_mod(b, h, q_idx, kv_idx, _sf=seg_flat, _sl=seq_len_int):
        return _sf[b * _sl + q_idx] == _sf[b * _sl + kv_idx]

    return create_block_mask(
        _doc_mask_mod,
        B=bsz,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=False,
    )
