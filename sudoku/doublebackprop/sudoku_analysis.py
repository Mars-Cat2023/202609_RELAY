"""Sudoku-specific primitives for analyzing model decoding trajectories.

Pure functions: every input is a small numpy array or list of ints, every
output is a numpy array, scalar, or list. No torch / model dependencies; safe
to run inside a Jupyter notebook without GPU.

Convention (matches ``xlm.datamodule.SimpleSpaceTokenizer.for_numbers(10)``
plus the puzzle preprocessor in ``xlm.tasks.sudoku_extreme``):

- Each Sudoku puzzle is a flat length-81 sequence of token ids.
- Special tokens occupy ids 0..6; ``mask_token_id == 2``.
- Digit tokens ``"0"``..``"9"`` occupy ids ``7``..``16`` (``first_digit_token_id == 7``).
- Blank cells in the prompt and target are stored as ``mask_token_id``; the
  literal token ``"0"`` (id ``7``) should not appear in any reference solution.
- Boards are returned as 9x9 ``int8`` arrays where ``0`` is blank and digits
  are ``1..9``. ``int8`` keeps memory tiny for parquet storage.

The trajectory captured by ``ConfidenceBasedPredictor`` (with
``capture_trajectory=True``) is a ``T x 81`` list of token ids; ``trajectory[0]``
is the *initial* masked board (only clues are non-mask), so the **clue mask**
can be recovered without needing the original ``prompt_ids``.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

MASK_TOKEN_ID: int = 2
FIRST_DIGIT_TOKEN_ID: int = 7  # token "0"
NUM_DIGIT_TOKENS: int = 10  # "0".."9"
GRID_SIDE: int = 9
CONTENT_LEN: int = 81

STRATEGY_TIERS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "Easy",
        ("Single Candidate", "Single Position"),
    ),
    (
        "Medium",
        ("Candidate Lines", "Double Pairs", "Multiple Lines"),
    ),
    (
        "Advanced",
        (
            # timvink-solver emits the plural form
            # ("Naked Pairs", "Hidden Triples", ...); accept both for safety.
            "Naked Pair", "Naked Pairs",
            "Naked Triple", "Naked Triples",
            "Naked Quad", "Naked Quads",
            "Hidden Pair", "Hidden Pairs",
            "Hidden Triple", "Hidden Triples",
            "Hidden Quad", "Hidden Quads",
        ),
    ),
    (
        "Master",
        (
            "X-Wing", "X-Wings",
            "Swordfish",
            "Jellyfish", "Jellyfishes",
            "Forcing Chain", "Forcing Chains",
        ),
    ),
    (
        "BruteForce",
        ("Brute Force",),
    ),
]
TIER_RANK = {name: i for i, (name, _) in enumerate(STRATEGY_TIERS)}


def hardest_strategy_tier(strategies_used: Iterable[str]) -> str:
    """Return the hardest tier label for a list of strategy names.

    Unknown strategies are treated as ``BruteForce`` (most conservative).
    Empty / ``None`` inputs return ``"Easy"``. Accepts numpy arrays whose
    truthiness is ambiguous in the standard ``not`` form.

    Strategy names are normalized to handle the plural / singular drift in
    timvink-solver output (which emits ``"Naked Pairs"`` while older docs
    sometimes use ``"Naked Pair"``); see ``STRATEGY_TIERS`` for the full set.
    """
    if strategies_used is None:
        return "Easy"
    seq = list(strategies_used)
    if len(seq) == 0:
        return "Easy"
    best = -1
    for s in seq:
        s = str(s).strip()
        for tier_name, members in STRATEGY_TIERS:
            if s in members:
                best = max(best, TIER_RANK[tier_name])
                break
        else:
            best = max(best, TIER_RANK["BruteForce"])
    return STRATEGY_TIERS[max(best, 0)][0]


# ---------------------------------------------------------------------------
# Token-id <-> digit-grid conversion.
# ---------------------------------------------------------------------------

def token_id_to_digit(
    token_id: int,
    *,
    mask_token_id: int = MASK_TOKEN_ID,
    first_digit_token_id: int = FIRST_DIGIT_TOKEN_ID,
) -> int:
    """Single token id -> digit in ``0..9``. Returns ``0`` for masks / unknown."""
    if token_id == mask_token_id or token_id < 0:
        return 0
    if first_digit_token_id <= token_id < first_digit_token_id + NUM_DIGIT_TOKENS:
        return int(token_id - first_digit_token_id)
    return 0


def ids_to_digit_grid(
    token_ids: Sequence[int],
    *,
    mask_token_id: int = MASK_TOKEN_ID,
    first_digit_token_id: int = FIRST_DIGIT_TOKEN_ID,
    grid_side: int = GRID_SIDE,
    content_len: int = CONTENT_LEN,
) -> np.ndarray:
    """Length-81 token-id sequence -> 9x9 ``int8`` grid (``0`` = blank)."""
    n = grid_side * grid_side
    if len(token_ids) < content_len:
        flat = list(token_ids) + [mask_token_id] * (content_len - len(token_ids))
    else:
        flat = list(token_ids[:content_len])
    out = np.zeros(n, dtype=np.int8)
    for i, t in enumerate(flat):
        out[i] = token_id_to_digit(
            int(t),
            mask_token_id=mask_token_id,
            first_digit_token_id=first_digit_token_id,
        )
    return out.reshape(grid_side, grid_side)


def trajectory_to_digit_grids(
    trajectory: Sequence[Sequence[int]],
    **kwargs,
) -> np.ndarray:
    """``T x 81`` token-id trajectory -> ``T x 9 x 9`` ``int8`` array."""
    grids = [ids_to_digit_grid(step, **kwargs) for step in trajectory]
    if not grids:
        return np.zeros((0, GRID_SIDE, GRID_SIDE), dtype=np.int8)
    return np.stack(grids, axis=0)


def digit_grid_to_string(grid: np.ndarray, *, blank_char: str = ".") -> str:
    """9x9 grid -> 81-char string. ``0`` cells render as ``blank_char``."""
    flat = np.asarray(grid, dtype=np.int8).reshape(-1)
    chars = [
        blank_char if int(v) == 0 else str(int(v))
        for v in flat.tolist()
    ]
    return "".join(chars)


def string_to_digit_grid(s: str, *, blank_char: str = ".") -> np.ndarray:
    """81-char string -> 9x9 ``int8`` grid. ``blank_char`` and ``"0"`` -> ``0``."""
    if len(s) != CONTENT_LEN:
        raise ValueError(f"Expected length {CONTENT_LEN}, got {len(s)}")
    out = np.zeros(CONTENT_LEN, dtype=np.int8)
    for i, ch in enumerate(s):
        if ch == blank_char or ch == "0":
            out[i] = 0
        elif ch.isdigit():
            out[i] = int(ch)
        else:
            out[i] = 0
    return out.reshape(GRID_SIDE, GRID_SIDE)


def truth_text_to_digit_grid(text: str) -> np.ndarray:
    """``"5 9 8 4 ..."`` (81 space-separated digits) -> 9x9 ``int8`` grid.

    Tolerates blanks expressed as ``"0"`` or ``"."``.
    """
    parts = str(text).strip().split()
    if len(parts) < CONTENT_LEN:
        raise ValueError(
            f"truth_text has {len(parts)} tokens, expected {CONTENT_LEN}"
        )
    out = np.zeros(CONTENT_LEN, dtype=np.int8)
    for i, p in enumerate(parts[:CONTENT_LEN]):
        if p == "." or p == "0":
            out[i] = 0
        elif p.isdigit():
            out[i] = int(p)
        else:
            out[i] = 0
    return out.reshape(GRID_SIDE, GRID_SIDE)


# ---------------------------------------------------------------------------
# Identity / hashing.
# ---------------------------------------------------------------------------

def normalize_question(question: str, *, blank_char: str = ".") -> str:
    """Normalize the puzzle question to an 81-char string with ``blank_char`` for blanks."""
    cleaned = "".join(question.split())
    if len(cleaned) != CONTENT_LEN:
        raise ValueError(
            f"Question must collapse to {CONTENT_LEN} chars; got {len(cleaned)}"
        )
    out = []
    for ch in cleaned:
        if ch == "." or ch == "0":
            out.append(blank_char)
        elif ch.isdigit():
            out.append(ch)
        else:
            out.append(blank_char)
    return "".join(out)


def puzzle_hash(question: str, *, blank_char: str = ".") -> str:
    """SHA1 of the normalized 81-char puzzle string. Stable join key."""
    norm = normalize_question(question, blank_char=blank_char)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def recover_clues_from_trajectory_step0(
    step0_token_ids: Sequence[int],
    **kwargs,
) -> np.ndarray:
    """Initial trajectory snapshot is the masked input -> 9x9 clue grid (0 elsewhere)."""
    return ids_to_digit_grid(step0_token_ids, **kwargs)


# ---------------------------------------------------------------------------
# Sudoku constraint logic: candidates, naked / hidden singles, legality.
# ---------------------------------------------------------------------------

def _box_index(row: int, col: int) -> int:
    return (row // 3) * 3 + (col // 3)


def used_digits_per_unit(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three boolean arrays of shape (9, 10) indicating, per unit, which
    digits ``1..9`` are already placed in each row, col, and 3x3 box.

    Index 0 is unused (kept for direct ``digit`` indexing without ``-1``).
    """
    rows = np.zeros((9, 10), dtype=bool)
    cols = np.zeros((9, 10), dtype=bool)
    boxes = np.zeros((9, 10), dtype=bool)
    for r in range(9):
        for c in range(9):
            d = int(grid[r, c])
            if d != 0:
                rows[r, d] = True
                cols[c, d] = True
                boxes[_box_index(r, c), d] = True
    return rows, cols, boxes


def candidate_mask(grid: np.ndarray) -> np.ndarray:
    """Compute per-cell digit candidate masks for an in-progress board.

    Returns a ``(9, 9, 10)`` boolean array. ``mask[r, c, d]`` is ``True`` iff
    digit ``d`` is a *legal* placement at ``(r, c)`` given current row/col/box
    constraints. Index 0 is always ``False``. Cells already filled with digit
    ``d`` have only ``mask[r, c, d] == True``.
    """
    rows, cols, boxes = used_digits_per_unit(grid)
    mask = np.zeros((9, 9, 10), dtype=bool)
    for r in range(9):
        for c in range(9):
            d = int(grid[r, c])
            if d != 0:
                mask[r, c, d] = True
                continue
            for digit in range(1, 10):
                if rows[r, digit] or cols[c, digit] or boxes[_box_index(r, c), digit]:
                    continue
                mask[r, c, digit] = True
    return mask


def candidate_count(grid: np.ndarray) -> np.ndarray:
    """``9 x 9`` ``int8`` array: number of candidates at each empty cell.

    Filled cells are reported as ``0`` (already determined, no choice).
    """
    cm = candidate_mask(grid)
    counts = cm[:, :, 1:].sum(axis=2).astype(np.int8)
    out = np.where(grid != 0, np.int8(0), counts)
    return out


def naked_single_mask(grid: np.ndarray) -> np.ndarray:
    """Boolean ``9 x 9``: empty cells whose only candidate is one digit."""
    counts = candidate_count(grid)
    return (grid == 0) & (counts == 1)


def hidden_single_mask(grid: np.ndarray) -> np.ndarray:
    """Boolean ``9 x 9``: empty cells where some digit ``d`` is the only
    candidate for ``d`` in at least one of (row, col, box) the cell belongs to.

    A cell can be both a naked and hidden single -- the rate metrics treat
    them as independent indicators of "easy" cells.
    """
    cm = candidate_mask(grid)
    hidden = np.zeros((9, 9), dtype=bool)
    for d in range(1, 10):
        digit_candidates = cm[:, :, d] & (grid == 0)
        # Row-only single
        row_counts = digit_candidates.sum(axis=1)
        for r in range(9):
            if row_counts[r] == 1:
                c = int(np.argmax(digit_candidates[r]))
                hidden[r, c] = True
        # Col-only single
        col_counts = digit_candidates.sum(axis=0)
        for c in range(9):
            if col_counts[c] == 1:
                r = int(np.argmax(digit_candidates[:, c]))
                hidden[r, c] = True
        # Box-only single
        for br in range(3):
            for bc in range(3):
                box = digit_candidates[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3]
                if int(box.sum()) == 1:
                    idx = int(np.argmax(box))
                    hidden[br * 3 + idx // 3, bc * 3 + idx % 3] = True
    return hidden


def is_legal_partial(grid: np.ndarray) -> bool:
    """Sudoku constraint: no row/col/box has a duplicated non-zero digit."""
    for unit_iter in (
        (grid[r, :] for r in range(9)),
        (grid[:, c] for c in range(9)),
        (
            grid[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3].reshape(-1)
            for br in range(3)
            for bc in range(3)
        ),
    ):
        for unit in unit_iter:
            vals = unit[unit != 0]
            if len(set(vals.tolist())) != int(len(vals)):
                return False
    return True


def count_violations(grid: np.ndarray) -> int:
    """Number of unique constraint violations (duplicated digits across all units)."""
    total = 0
    for unit_iter in (
        (grid[r, :] for r in range(9)),
        (grid[:, c] for c in range(9)),
        (
            grid[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3].reshape(-1)
            for br in range(3)
            for bc in range(3)
        ),
    ):
        for unit in unit_iter:
            vals = unit[unit != 0].tolist()
            total += len(vals) - len(set(vals))
    return int(total)


# ---------------------------------------------------------------------------
# Per-step trajectory analysis: fill-time grids, classification of new cells.
# ---------------------------------------------------------------------------

def fill_step_grid(
    digit_traj: np.ndarray,
    *,
    clue_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the first decoding step at which each cell becomes non-zero.

    Args:
        digit_traj: ``T x 9 x 9`` digit trajectory (already converted from token ids).
        clue_grid: optional 9x9 of clues; clue cells are reported as fill step 0.

    Returns ``9 x 9`` ``float32``, ``np.nan`` for cells still empty at step T-1.
    """
    if digit_traj.ndim != 3 or digit_traj.shape[1:] != (9, 9):
        raise ValueError(f"Expected T x 9 x 9 digit trajectory, got {digit_traj.shape}")
    T = digit_traj.shape[0]
    out = np.full((9, 9), np.nan, dtype=np.float32)
    for t in range(T):
        nonblank = (digit_traj[t] != 0) & np.isnan(out)
        out[nonblank] = float(t)
    if clue_grid is not None:
        out = np.where(clue_grid != 0, np.float32(0.0), out)
    return out


def step_new_cells(digit_traj: np.ndarray) -> List[List[Tuple[int, int, int]]]:
    """For each step ``t > 0``, list ``(row, col, digit)`` triples of cells that
    became non-zero between steps ``t-1`` and ``t``.

    The digit is the value placed *at step t*. Cells whose digit changes (rare
    if the predictor is monotone) are still included as new placements.
    """
    if digit_traj.shape[0] == 0:
        return []
    out: List[List[Tuple[int, int, int]]] = []
    prev = digit_traj[0]
    for t in range(1, digit_traj.shape[0]):
        cur = digit_traj[t]
        diffs: List[Tuple[int, int, int]] = []
        new_mask = (cur != 0) & ((prev == 0) | (cur != prev))
        rs, cs = np.where(new_mask)
        for r, c in zip(rs.tolist(), cs.tolist()):
            diffs.append((int(r), int(c), int(cur[r, c])))
        out.append(diffs)
        prev = cur
    return out


def step_purity(
    digit_traj: np.ndarray,
    answer_grid: np.ndarray,
) -> List[float]:
    """Per-step purity: fraction of newly committed cells that match the answer.

    Steps with no new placements are reported as ``np.nan``.
    """
    new_cells = step_new_cells(digit_traj)
    out: List[float] = []
    for diffs in new_cells:
        if not diffs:
            out.append(float("nan"))
            continue
        correct = sum(
            1 for (r, c, d) in diffs if int(answer_grid[r, c]) == int(d)
        )
        out.append(correct / len(diffs))
    return out


def trajectory_violation_steps(
    digit_traj: np.ndarray,
) -> int:
    """Number of steps in which the partial board contained a constraint violation."""
    return int(sum(1 for t in range(digit_traj.shape[0]) if not is_legal_partial(digit_traj[t])))


# ---------------------------------------------------------------------------
# Algorithmic primitives: classify a *single fill* against the pre-step board.
# ---------------------------------------------------------------------------

def classify_fills_at_step(
    grid_before: np.ndarray,
    fills: List[Tuple[int, int, int]],
    answer_grid: Optional[np.ndarray] = None,
) -> List[dict]:
    """Tag each ``(r, c, digit)`` fill with which primitive solved it.

    Categories (mutually non-exclusive booleans + dominant ``label``):

    - ``naked_single``: ``cand_count(grid_before)[r, c] == 1``.
    - ``hidden_single``: ``hidden_single_mask(grid_before)[r, c]``.
    - ``forced``: ``naked_single or hidden_single`` -- a deterministic step.
    - ``advanced``: empty cell with >1 candidate (model used a higher-tier
      strategy or guessed).
    - ``illegal``: digit is not in the candidate set for the cell (constraint
      violation introduced by the model).
    - ``correct``: matches ``answer_grid`` if provided.
    """
    counts = candidate_count(grid_before)
    cm = candidate_mask(grid_before)
    hidden = hidden_single_mask(grid_before)
    out: List[dict] = []
    for (r, c, d) in fills:
        nc = int(counts[r, c])
        is_naked = bool((grid_before[r, c] == 0) and (nc == 1))
        is_hidden = bool(hidden[r, c]) and bool(grid_before[r, c] == 0)
        legal = bool(cm[r, c, d]) if 1 <= d <= 9 else False
        is_advanced = bool(grid_before[r, c] == 0) and (not is_naked) and (not is_hidden)
        if not legal:
            label = "illegal"
        elif is_naked:
            label = "naked_single"
        elif is_hidden:
            label = "hidden_single"
        elif is_advanced:
            label = "advanced"
        else:
            label = "rewrite"  # the cell was already filled before this step
        item: dict = {
            "row": int(r),
            "col": int(c),
            "digit": int(d),
            "candidate_count_before": nc,
            "naked_single": is_naked,
            "hidden_single": is_hidden,
            "forced": is_naked or is_hidden,
            "advanced": is_advanced,
            "legal": legal,
            "illegal": not legal,
            "label": label,
        }
        if answer_grid is not None:
            item["correct"] = bool(int(answer_grid[r, c]) == int(d))
        out.append(item)
    return out


def classify_first_commit_step(
    digit_traj: np.ndarray,
    answer_grid: Optional[np.ndarray] = None,
) -> List[dict]:
    """Classify every cell committed at step 1 (first model action).

    Step 1 is the first decoding step *after* the initial masked board, so the
    "before" state is ``digit_traj[0]`` (clues only).
    """
    if digit_traj.shape[0] < 2:
        return []
    new_cells_t1 = step_new_cells(digit_traj[:2])[0]
    return classify_fills_at_step(digit_traj[0], new_cells_t1, answer_grid)


# ---------------------------------------------------------------------------
# Solver alignment: derive a fill-step grid from the solver's textual trajectory.
# ---------------------------------------------------------------------------

def solver_fill_step_grid(
    solver_trajectory: Sequence[str],
) -> np.ndarray:
    """``T`` 81-char board strings -> 9x9 fill-step grid.

    The first element is the initial puzzle (clues -> step 0); subsequent
    elements record progressively more cells filled. Cells unfilled at the
    last step return ``np.nan``.
    """
    grids = np.stack(
        [string_to_digit_grid(s) for s in solver_trajectory],
        axis=0,
    )
    return fill_step_grid(grids)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two flat arrays (NaNs dropped pairwise).

    Returns ``np.nan`` if fewer than 3 valid pairs.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = ~(np.isnan(a) | np.isnan(b))
    if int(mask.sum()) < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    ar = _rankdata(a)
    br = _rankdata(b)
    am = ar - ar.mean()
    bm = br - br.mean()
    denom = float(np.sqrt((am ** 2).sum() * (bm ** 2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((am * bm).sum() / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) -- minimal implementation to avoid scipy dep."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def fill_order_alignment(
    model_fill_step: np.ndarray,
    solver_fill_step: np.ndarray,
) -> float:
    """Spearman ρ between two fill-step grids restricted to non-clue cells.

    Cells that are clues in either grid (step 0) are excluded so the metric
    only scores the *order in which the remaining 64 cells get filled*.
    """
    a = np.asarray(model_fill_step, dtype=np.float64).reshape(-1)
    b = np.asarray(solver_fill_step, dtype=np.float64).reshape(-1)
    nonclue = (a != 0) & (b != 0)
    return spearman_corr(np.where(nonclue, a, np.nan), np.where(nonclue, b, np.nan))


def prefix_jaccard(
    digit_traj: np.ndarray,
    solver_traj: Sequence[str],
    *,
    k_steps: int,
    fraction: bool = False,
) -> float:
    """Jaccard between the cells filled in the first ``k`` model steps and the
    first ``k`` solver steps. If ``fraction`` is ``True``, ``k_steps`` is
    interpreted as a fraction of total fills for each side (``0..1``).

    Cells already present at step 0 (clues) are excluded.
    """
    if digit_traj.shape[0] == 0 or len(solver_traj) == 0:
        return float("nan")
    clue_grid = digit_traj[0]
    if fraction:
        total_model = int((digit_traj[-1] != 0).sum() - (clue_grid != 0).sum())
        total_solver = int(
            (string_to_digit_grid(solver_traj[-1]) != 0).sum()
            - (clue_grid != 0).sum()
        )
        if total_model <= 0 or total_solver <= 0:
            return float("nan")
        k_model = max(1, int(round(k_steps * digit_traj.shape[0])))
        k_solver = max(1, int(round(k_steps * len(solver_traj))))
    else:
        k_model = min(k_steps, digit_traj.shape[0] - 1)
        k_solver = min(k_steps, len(solver_traj) - 1)
    model_set = set()
    for t in range(1, k_model + 1):
        diffs = digit_traj[t] != 0
        clues_only = clue_grid != 0
        for r, c in zip(*np.where(diffs & ~clues_only)):
            model_set.add((int(r), int(c)))
    solver_grid_k = string_to_digit_grid(solver_traj[k_solver])
    clues_only = clue_grid != 0
    solver_set = set()
    for r, c in zip(*np.where((solver_grid_k != 0) & ~clues_only)):
        solver_set.add((int(r), int(c)))
    if not model_set and not solver_set:
        return float("nan")
    inter = len(model_set & solver_set)
    union = len(model_set | solver_set)
    return float(inter / union) if union > 0 else float("nan")


# ---------------------------------------------------------------------------
# Top-level convenience: one-shot per-puzzle feature extraction.
# ---------------------------------------------------------------------------

def summarize_trajectory(
    trajectory: Sequence[Sequence[int]],
    target_ids: Sequence[int],
    *,
    solver_trajectory: Optional[Sequence[str]] = None,
    mask_token_id: int = MASK_TOKEN_ID,
    first_digit_token_id: int = FIRST_DIGIT_TOKEN_ID,
) -> dict:
    """End-to-end summary of one model trajectory.

    Returns a flat dict with primitives consumable by parquet / pandas:
    fill-step grid (flattened), per-step purity, first-step naked / hidden /
    advanced rates, illegal-step count, total violations, and (if a solver
    trajectory is given) Spearman ρ of fill order plus prefix-Jaccard at a
    handful of cut-offs.

    All grids are returned as 81-element lists for ergonomic parquet writing.
    """
    digit_traj = trajectory_to_digit_grids(
        trajectory,
        mask_token_id=mask_token_id,
        first_digit_token_id=first_digit_token_id,
    )
    answer = ids_to_digit_grid(
        target_ids,
        mask_token_id=mask_token_id,
        first_digit_token_id=first_digit_token_id,
    )
    fsg = fill_step_grid(digit_traj)
    purity = step_purity(digit_traj, answer)
    first_step = classify_first_commit_step(digit_traj, answer_grid=answer)
    n_first = max(len(first_step), 1)
    naked_rate = sum(1 for x in first_step if x["naked_single"]) / n_first
    hidden_rate = sum(1 for x in first_step if x["hidden_single"]) / n_first
    forced_rate = sum(1 for x in first_step if x["forced"]) / n_first
    advanced_rate = sum(1 for x in first_step if x["advanced"]) / n_first
    illegal_rate = sum(1 for x in first_step if x["illegal"]) / n_first
    correct_first = sum(1 for x in first_step if x.get("correct", False)) / n_first
    final = digit_traj[-1] if digit_traj.shape[0] > 0 else answer * 0
    exact_match = bool(np.array_equal(final, answer))
    out = {
        "exact_match": exact_match,
        "n_decode_steps": int(digit_traj.shape[0] - 1) if digit_traj.shape[0] > 0 else 0,
        "n_clues": int((digit_traj[0] != 0).sum()) if digit_traj.shape[0] > 0 else 0,
        "fill_step_flat": fsg.reshape(-1).tolist(),
        "purity_per_step": purity,
        "n_first_step_fills": len(first_step),
        "first_step_naked_single_rate": float(naked_rate),
        "first_step_hidden_single_rate": float(hidden_rate),
        "first_step_forced_rate": float(forced_rate),
        "first_step_advanced_rate": float(advanced_rate),
        "first_step_illegal_rate": float(illegal_rate),
        "first_step_correct_rate": float(correct_first),
        "violations_total_steps": trajectory_violation_steps(digit_traj),
        "final_violations": count_violations(final),
        "clue_grid_flat": digit_traj[0].reshape(-1).tolist() if digit_traj.shape[0] else [],
        "answer_grid_flat": answer.reshape(-1).tolist(),
        "final_grid_flat": final.reshape(-1).tolist(),
    }
    if solver_trajectory is not None and len(solver_trajectory) > 0:
        try:
            ssg = solver_fill_step_grid(solver_trajectory)
            out["solver_fill_step_flat"] = ssg.reshape(-1).tolist()
            out["fill_order_spearman"] = float(fill_order_alignment(fsg, ssg))
            out["prefix_jaccard_at_5"] = float(
                prefix_jaccard(digit_traj, solver_trajectory, k_steps=5)
            )
            out["prefix_jaccard_at_10"] = float(
                prefix_jaccard(digit_traj, solver_trajectory, k_steps=10)
            )
            out["prefix_jaccard_at_20"] = float(
                prefix_jaccard(digit_traj, solver_trajectory, k_steps=20)
            )
            out["prefix_jaccard_at_50pct"] = float(
                prefix_jaccard(
                    digit_traj, solver_trajectory, k_steps=0.5, fraction=True
                )
            )
        except Exception as exc:  # noqa: BLE001 -- defensive: missing solver data
            out["solver_alignment_error"] = repr(exc)
    return out
