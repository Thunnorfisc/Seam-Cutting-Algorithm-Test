"""Tests for the local greedy seam finder. No image or DP code is involved."""

from __future__ import annotations

import copy
import unittest

from algorithms.greedy import (
    find_greedy_horizontal_seam,
    find_greedy_vertical_seam,
    vertical_seam_cost,
)

COUNTEREXAMPLE = [
    [1, 2, 100],
    [4, 50, 0],
    [5, 50, 0],
]


class GreedySeamTests(unittest.TestCase):
    def test_counterexample_is_local_seam_not_optimum(self) -> None:
        seam = find_greedy_vertical_seam(COUNTEREXAMPLE)
        self.assertEqual(seam, [0, 0, 0])
        self.assertEqual(vertical_seam_cost(COUNTEREXAMPLE, seam), 10)
        self.assert_vertical_seam(seam, height=3, width=3)

    def test_one_pixel_map(self) -> None:
        seam = find_greedy_vertical_seam([[7]])
        self.assertEqual(seam, [0])
        self.assert_vertical_seam(seam, height=1, width=1)

    def test_single_row(self) -> None:
        seam = find_greedy_vertical_seam([[3, 0, 2, 1]])
        self.assertEqual(seam, [1])
        self.assert_vertical_seam(seam, height=1, width=4)

    def test_single_column(self) -> None:
        seam = find_greedy_vertical_seam([[5], [1], [8]])
        self.assertEqual(seam, [0, 0, 0])
        self.assert_vertical_seam(seam, height=3, width=1)

    def test_ties_are_leftmost_and_repeatable(self) -> None:
        energy = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        first = find_greedy_vertical_seam(energy)
        second = find_greedy_vertical_seam(energy)
        self.assertEqual(first, [0, 0, 0])
        self.assertEqual(second, [0, 0, 0])
        self.assert_vertical_seam(first, height=3, width=3)

    def test_diagonal_steps(self) -> None:
        seam = find_greedy_vertical_seam([[8, 1, 8], [0, 8, 8], [8, 0, 8]])
        self.assertEqual(seam, [1, 0, 1])
        self.assert_vertical_seam(seam, height=3, width=3)

    def test_left_and_right_boundaries(self) -> None:
        left = find_greedy_vertical_seam([[0, 5], [1, 4]])
        right = find_greedy_vertical_seam([[5, 0], [4, 1]])
        self.assertEqual(left, [0, 0])
        self.assertEqual(right, [1, 1])
        self.assert_vertical_seam(left, height=2, width=2)
        self.assert_vertical_seam(right, height=2, width=2)

    def test_horizontal_counterexample_and_rectangle(self) -> None:
        horizontal = find_greedy_horizontal_seam(COUNTEREXAMPLE)
        self.assertEqual(horizontal, [0, 0, 1])
        self.assert_horizontal_seam(horizontal, height=3, width=3)

        rectangle = [[1, 5, 1], [0, 9, 0]]
        rectangular_seam = find_greedy_horizontal_seam(rectangle)
        self.assertEqual(rectangular_seam, [1, 0, 1])
        self.assert_horizontal_seam(rectangular_seam, height=2, width=3)

    def test_horizontal_tie_is_topmost(self) -> None:
        seam = find_greedy_horizontal_seam([[7, 7, 7], [7, 7, 7]])
        self.assertEqual(seam, [0, 0, 0])
        self.assert_horizontal_seam(seam, height=2, width=3)

    def test_negative_finite_values_are_accepted(self) -> None:
        seam = find_greedy_vertical_seam([[-5, -1], [-3, -4]])
        self.assertEqual(seam, [0, 1])
        self.assert_vertical_seam(seam, height=2, width=2)

    def test_invalid_energy_is_rejected(self) -> None:
        invalid_maps = [
            [],
            [[]],
            [1, 2, 3],
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[1, 2], [3]],
            [["a", 1]],
            [[1, float("nan")]],
            [[1, float("inf")]],
            [[float("-inf"), 1]],
            [[1, 0], [2, float("nan")]],
            [[True, 0]],
        ]
        for energy in invalid_maps:
            with self.subTest(energy=energy):
                with self.assertRaises(ValueError):
                    find_greedy_vertical_seam(energy)
                with self.assertRaises(ValueError):
                    find_greedy_horizontal_seam(energy)

    def test_input_is_not_modified(self) -> None:
        energy = copy.deepcopy(COUNTEREXAMPLE)
        snapshot = copy.deepcopy(energy)
        row_ids = [id(row) for row in energy]
        find_greedy_vertical_seam(energy)
        find_greedy_horizontal_seam(energy)
        self.assertEqual(energy, snapshot)
        self.assertEqual([id(row) for row in energy], row_ids)

    def assert_vertical_seam(self, seam: list[int], height: int, width: int) -> None:
        self.assertEqual(len(seam), height)
        self.assert_connected(seam, limit=width)

    def assert_horizontal_seam(self, seam: list[int], height: int, width: int) -> None:
        self.assertEqual(len(seam), width)
        self.assert_connected(seam, limit=height)

    def assert_connected(self, seam: list[int], limit: int) -> None:
        for index, value in enumerate(seam):
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, limit)
            if index:
                self.assertLessEqual(abs(value - seam[index - 1]), 1)


if __name__ == "__main__":
    unittest.main()
