"""Local greedy seam selection.

Each step takes the lowest-energy pixel among the pixels that touch the
previous choice. The sum along the seam is not used to choose the next pixel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def find_greedy_vertical_seam(energy: Sequence) -> list[int]:
    """Return one column index per row for a top-to-bottom greedy seam.

    Row 0 uses the lowest energy, and a tie keeps the smallest column. Each
    later row looks only at the in-bounds columns ``previous - 1``,
    ``previous``, and ``previous + 1``, and applies the same tie rule.
    ``energy`` is not modified.
    """
    rows = _copy_finite_grid(energy)
    width = len(rows[0])
    column = _leftmost_argmin(rows[0])
    seam = [column]
    for row in rows[1:]:
        best_column = None
        best_energy = None
        for candidate in (column - 1, column, column + 1):
            if candidate < 0 or candidate >= width:
                continue
            value = row[candidate]
            # Strict < keeps the smaller column when energies are equal,
            # because candidates are visited from left to right.
            if best_column is None or value < best_energy:
                best_energy = value
                best_column = candidate
        column = best_column
        seam.append(column)
    return seam


def find_greedy_horizontal_seam(energy: Sequence) -> list[int]:
    """Return one row index per column by running the vertical finder on the transpose.

    A leftmost tie on the transposed map is a topmost tie on ``energy``.
    ``energy`` is not modified.
    """
    rows = _copy_finite_grid(energy)
    return find_greedy_vertical_seam(_transpose(rows))


def vertical_seam_cost(energy: Sequence, seam: Sequence[int]) -> float:
    """Sum ``energy[row][seam[row]]`` after a vertical seam has been chosen."""
    rows = _copy_finite_grid(energy)
    if len(seam) != len(rows):
        raise ValueError("vertical seam must contain one column index per row")
    width = len(rows[0])
    total = 0.0
    for row_index, column in enumerate(seam):
        if isinstance(column, bool) or not isinstance(column, int):
            raise ValueError("seam indexes must be integers")
        if column < 0 or column >= width:
            raise ValueError("seam index out of range")
        total += rows[row_index][column]
    return total


def _leftmost_argmin(row: Sequence[float]) -> int:
    best_column = 0
    best_energy = row[0]
    for column in range(1, len(row)):
        if row[column] < best_energy:
            best_energy = row[column]
            best_column = column
    return best_column


def _transpose(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    height = len(rows)
    width = len(rows[0])
    return [[rows[row][column] for row in range(height)] for column in range(width)]


def _copy_finite_grid(energy: object) -> list[list[float]]:
    """Return a new finite 2D grid without modifying the caller's object."""
    if isinstance(energy, (str, bytes)) or not isinstance(energy, Sequence):
        raise ValueError("energy map must be a non-empty 2D array of finite real numbers")
    if len(energy) == 0:
        raise ValueError("energy map must be a non-empty 2D array of finite real numbers")

    rows: list[list[float]] = []
    width: int | None = None
    for row in energy:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError("energy map must be a non-empty 2D array of finite real numbers")
        if len(row) == 0:
            raise ValueError("energy map must be a non-empty 2D array of finite real numbers")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("energy map rows must have equal length")
        rows.append([_finite_real(value) for value in row])
    return rows


def _finite_real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("energy map must contain finite real numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("energy map must contain finite real numbers")
    return number
