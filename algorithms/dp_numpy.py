"""The same dynamic programme as ``algorithms/dp.py``, with each row filled by NumPy.

This is the speed improvement, not a different algorithm: the recurrence, the
tie rule (smaller index wins) and therefore the seams are identical to the
pure-Python finder. The row loop stays in Python because row ``i`` depends on
row ``i - 1``; only the work across one row is vectorised.

Unlike the other finders, these accept a NumPy array as well as a list of
lists, so a caller that already holds the energy array does not pay for
``tolist()``.
"""

from __future__ import annotations

import numpy as np


def find_dp_vertical_seam_numpy(energy) -> list[int]:
    """Return one column index per row for a minimum-energy top-to-bottom seam."""
    return _solve(_as_grid(energy))


def find_dp_horizontal_seam_numpy(energy) -> list[int]:
    """Return one row index per column for a minimum-energy left-to-right seam."""
    return _solve(_as_grid(energy).T)


def _as_grid(energy) -> np.ndarray:
    message = "energy map must be a non-empty 2D array of finite real numbers"
    try:
        grid = np.asarray(energy)
    except ValueError as error:  # ragged rows
        raise ValueError(message) from error
    if grid.ndim != 2 or grid.size == 0 or grid.dtype.kind not in "iuf":
        raise ValueError(message)
    grid = grid.astype(np.float64, copy=False)
    if not np.isfinite(grid).all():
        raise ValueError(message)
    return grid


def _solve(grid: np.ndarray) -> list[int]:
    height, width = grid.shape
    columns = np.arange(width)
    parents = np.empty((height - 1, width), dtype=np.int64)
    # Rows of ``candidates`` are the left, upper and right neighbour totals.
    # argmin returns the first minimum, so a tie keeps the smaller column.
    candidates = np.empty((3, width), dtype=np.float64)
    totals = grid[0].copy()
    for row in range(1, height):
        candidates[0, 0] = np.inf
        candidates[0, 1:] = totals[:-1]
        candidates[1] = totals
        candidates[2, -1] = np.inf
        candidates[2, :-1] = totals[1:]
        choice = np.argmin(candidates, axis=0)
        parents[row - 1] = columns + choice - 1
        totals = grid[row] + candidates[choice, columns]

    column = int(np.argmin(totals))
    seam = [column]
    for row in range(height - 2, -1, -1):
        column = int(parents[row, column])
        seam.append(column)
    seam.reverse()
    return seam
