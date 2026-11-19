"""Minimum-energy seam selection by dynamic programming.

For a vertical seam, ``M[i][j]`` is the smallest total energy of any connected
seam that starts in row 0 and ends at pixel ``(i, j)``:

    M[0][j] = e[0][j]
    M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1])

with out-of-range columns skipped. The smallest value in the last row is the
cost of the best seam, and following the stored choices back to row 0 gives
the seam itself. Every table entry is filled once, so one seam is O(H * W)
time and O(H * W) space.

Ties keep the smaller index, the same rule the greedy finder uses, so the
result is deterministic. The input contract is also the greedy finder's: a
list-of-lists energy map, validated by the same function, never modified.
"""

from __future__ import annotations

from collections.abc import Sequence

from algorithms.greedy import _copy_finite_grid, _leftmost_argmin, _transpose


def find_dp_vertical_seam(energy: Sequence) -> list[int]:
    """Return one column index per row for a minimum-energy top-to-bottom seam."""
    rows = _copy_finite_grid(energy)
    last_row, parents = _fill(rows)
    return _backtrack(last_row, parents)


def find_dp_horizontal_seam(energy: Sequence) -> list[int]:
    """Return one row index per column by running the vertical finder on the transpose.

    A leftmost tie on the transposed map is a topmost tie on ``energy``.
    """
    rows = _copy_finite_grid(energy)
    last_row, parents = _fill(_transpose(rows))
    return _backtrack(last_row, parents)


def solve_dp_vertical(energy: Sequence) -> tuple[list[int], float]:
    """Return ``(seam, cost)``. ``cost`` is the minimum of the table's last row."""
    rows = _copy_finite_grid(energy)
    last_row, parents = _fill(rows)
    seam = _backtrack(last_row, parents)
    return seam, last_row[seam[-1]]


def cumulative_energy_vertical(energy: Sequence) -> list[list[float]]:
    """Return the whole table ``M`` for a vertical seam, one row per image row.

    The seam finders keep only the newest row of ``M``. This keeps all of it,
    for inspection and for the tests.
    """
    rows = _copy_finite_grid(energy)
    table = [rows[0]]
    for row in rows[1:]:
        table.append(_next_row(table[-1], row)[0])
    return table


def _fill(rows: list[list[float]]) -> tuple[list[float], list[list[int]]]:
    """Fill the table row by row. Returns its last row and every row's parent columns.

    ``parents[i - 1][j]`` is the column in row ``i - 1`` that the best seam
    ending at ``(i, j)`` came from.
    """
    current = rows[0]
    parents: list[list[int]] = []
    for row in rows[1:]:
        current, came_from = _next_row(current, row)
        parents.append(came_from)
    return current, parents


def _next_row(previous: list[float], row: list[float]) -> tuple[list[float], list[int]]:
    width = len(row)
    totals = [0.0] * width
    came_from = [0] * width
    for column in range(width):
        best_column = column
        best_total = previous[column]
        # Equal totals keep the smaller column: the left neighbour wins a tie
        # (<=), the right neighbour has to be strictly smaller (<).
        if column > 0 and previous[column - 1] <= best_total:
            best_column = column - 1
            best_total = previous[column - 1]
        if column + 1 < width and previous[column + 1] < best_total:
            best_column = column + 1
            best_total = previous[column + 1]
        totals[column] = row[column] + best_total
        came_from[column] = best_column
    return totals, came_from


def _backtrack(last_row: list[float], parents: list[list[int]]) -> list[int]:
    column = _leftmost_argmin(last_row)
    seam = [column]
    for came_from in reversed(parents):
        column = came_from[column]
        seam.append(column)
    seam.reverse()
    return seam
