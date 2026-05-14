"""Map token-id sequences to small numpy grids for Sudoku / n-queens / graph coloring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from matplotlib import patheffects
except ImportError:  # pragma: no cover
    patheffects = None  # type: ignore


def infer_dataset(tags: Optional[Dict[str, Any]]) -> str:
    """Return ``tags.dataset`` when present, else ``\"unknown\"``."""
    if not tags:
        return "unknown"
    ds = tags.get("dataset")
    if ds is None:
        return "unknown"
    return str(ds)


def sudoku_token_id_to_cell_char(
    token_id: int,
    *,
    mask_token_id: int = 2,
    first_digit_token_id: int = 7,
    num_digit_tokens: int = 10,
) -> str:
    """Map a single token id to one Sudoku cell character for ``SimpleSpaceTokenizer.for_numbers``.

    Layout matches ``xlm.datamodule.SimpleSpaceTokenizer``: ids 0–6 are special tokens,
    then ``"0"`` … ``str(num_digit_tokens-1)`` occupy consecutive ids starting at
    ``first_digit_token_id``. Masked cells use ``mask_token_id`` (``[MASK]``). Dataset
    preprocessing replaces blank ``"0"`` with the mask id, so both mask and ``"0"`` render
    as an empty cell; ``"1"``–``"9"`` are shown.

    Returns:
        ``\"\"`` for mask / empty, or a single digit string ``\"1\"``–``\"9\"`` (or ``\"0\"``
        if it appears in content).
    """
    if token_id < 0 or token_id == mask_token_id:
        return ""
    lo = first_digit_token_id
    hi = first_digit_token_id + num_digit_tokens - 1
    if lo <= token_id <= hi:
        ch = str(token_id - lo)
        if ch == "0":
            return ""
        return ch
    return ""


def sudoku_ids_to_digit_grid(
    token_ids: List[int],
    *,
    grid_side: int = 9,
    content_len: int = 81,
    **kwargs: Any,
) -> np.ndarray:
    """First ``content_len`` token ids → ``grid_side``×``grid_side`` array of digit strings (empty = masked)."""
    flat = np.array(token_ids[:content_len], dtype=np.int64)
    if flat.size < content_len:
        flat = np.pad(flat, (0, content_len - flat.size), constant_values=-1)
    g = int(grid_side)
    chars = [sudoku_token_id_to_cell_char(int(tid), **kwargs) for tid in flat.tolist()]
    return np.array(chars, dtype=object).reshape(g, g)


def sudoku_first_nonmask_step_grid(
    trajectory: List[List[int]],
    *,
    mask_token_id: int = 2,
    content_len: int = 81,
    grid_side: int = 9,
) -> np.ndarray:
    """Per-cell **first decode step** where the cell is not ``[MASK]``.

    Used for paper-style “fill time” heatmaps: **0** means the cell was already a
    clue at step 0; larger values mean the model committed a digit later in the
    trajectory. ``nan`` if the cell is still masked at the last recorded step.

    Args:
        trajectory: ``trajectory[t]`` is the length-``content_len`` token-id vector
            at decode step ``t`` (same convention as dumps).
    """
    g = int(grid_side)
    if not trajectory:
        return np.full((g, g), np.nan, dtype=np.float64)
    flat = np.full((content_len,), np.nan, dtype=np.float64)
    for t, step_ids in enumerate(trajectory):
        ids = list(step_ids[:content_len])
        if len(ids) < content_len:
            ids.extend([mask_token_id] * (content_len - len(ids)))
        for i in range(content_len):
            if not np.isnan(flat[i]):
                continue
            if int(ids[i]) != mask_token_id:
                flat[i] = float(t)
    return flat.reshape(g, g)


def sudoku_truth_text_to_digit_grid(truth_text: str) -> Optional[np.ndarray]:
    """Parse ``truth_text`` (space-separated digits, 81 cells) → 9×9 array of digit strings.

    Returns ``None`` if the string cannot be parsed as 81 single-digit tokens.
    """
    parts = truth_text.strip().split()
    if len(parts) < 81:
        return None
    chars: List[str] = []
    for p in parts[:81]:
        if len(p) != 1 or not p.isdigit():
            return None
        chars.append(p)
    return np.array(chars, dtype=object).reshape(9, 9)


def overlay_sudoku_digits_on_heatmap_ax(
    ax,
    digit_grid: np.ndarray,
    *,
    fontsize: float = 12,
    color: str = "white",
) -> None:
    """Draw digit labels at cell centers over an ``imshow`` Sudoku heatmap (origin ``upper``).

    Uses a dark outline so digits stay readable on arbitrary colormap values (paper-style).

    Coordinates match ``imshow(..., origin="upper")`` default extent: cell centers lie at
    integer ``(x, y) = (col, row)`` in data space, **not** at half-integers.
    """
    if patheffects is None:
        raise RuntimeError("matplotlib is required for overlay_sudoku_digits_on_heatmap_ax")
    pe = [patheffects.withStroke(linewidth=2.2, foreground="black")]
    g = int(digit_grid.shape[0])
    for r in range(g):
        for c in range(g):
            ch = digit_grid[r, c]
            if ch is None or str(ch).strip() == "":
                continue
            ax.text(
                float(c),
                float(r),
                str(ch),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                path_effects=pe,
            )


def sudoku_ids_to_grid(
    token_ids: List[int],
    *,
    grid_side: int = 9,
    content_len: int = 81,
) -> np.ndarray:
    """First ``content_len`` tokens → ``grid_side``×``grid_side`` int array (raw token ids)."""
    flat = np.array(token_ids[:content_len], dtype=np.int64)
    g = int(grid_side)
    if flat.size < content_len:
        pad = np.full((content_len - flat.size,), -1, dtype=np.int64)
        flat = np.concatenate([flat, pad])
    return flat.reshape(g, g)


def n_queens_ids_to_grid(
    token_ids: List[int],
    *,
    board_side: int = 8,
) -> np.ndarray:
    """8×8 n-queens board from first ``board_side**2`` tokens."""
    n = board_side * board_side
    flat = np.array(token_ids[:n], dtype=np.int64)
    if flat.size < n:
        pad = np.full((n - flat.size,), -1, dtype=np.int64)
        flat = np.concatenate([flat, pad])
    return flat.reshape(board_side, board_side)


def graph_coloring_split(
    token_ids: List[int],
    *,
    n_vertices: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split [graph upper triangle | node colors] for 8-vertex GRAM instances."""
    n = n_vertices
    tri_len = n * (n - 1) // 2
    full = np.array(token_ids, dtype=np.int64)
    graph = full[:tri_len].copy()
    colors = full[tri_len : tri_len + n].copy()
    return graph, colors


def trajectory_to_grids(
    trajectory: List[List[int]],
    *,
    puzzle_type: str,
    **kwargs: Any,
) -> List[np.ndarray]:
    """One grid (or tuple for graph coloring) per decoding step."""
    pt = puzzle_type.lower()
    out: List[np.ndarray] = []
    for step_ids in trajectory:
        if "sudoku" in pt:
            out.append(sudoku_ids_to_grid(step_ids, **kwargs))
        elif "queens" in pt or "n_queens" in pt:
            out.append(n_queens_ids_to_grid(step_ids, **kwargs))
        elif "graph" in pt or "coloring" in pt:
            g, c = graph_coloring_split(step_ids, **kwargs)
            out.append(np.concatenate([g, np.array([-1]), c]))
        else:
            out.append(np.array(step_ids, dtype=np.int64))
    return out
