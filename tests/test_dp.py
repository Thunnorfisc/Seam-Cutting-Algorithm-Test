"""Unit tests for the dynamic-programming seam finders."""

from __future__ import annotations

import copy
import itertools
import random
import unittest

import numpy as np

from algorithms.dp import (
    cumulative_energy_vertical,
    find_dp_horizontal_seam,
    find_dp_vertical_seam,
    solve_dp_vertical,
)
from algorithms.dp_numpy import find_dp_horizontal_seam_numpy, find_dp_vertical_seam_numpy
from algorithms.graphcut import solve_vertical
from algorithms.greedy import find_greedy_vertical_seam, vertical_seam_cost

COUNTEREXAMPLE = [
    [1, 2, 100],
    [4, 50, 0],
    [5, 50, 0],
]


def brute_force_min_cost(energy) -> float:
    height, width = len(energy), len(energy[0])
    best = float("inf")
    for seam in itertools.product(range(width), repeat=height):
        if all(abs(a - b) <= 1 for a, b in zip(seam, seam[1:])):
            best = min(best, sum(energy[i][j] for i, j in enumerate(seam)))
    return best


def random_map(rng: random.Random, height: int, width: int, integers: bool = False):
    if integers:
        # Small integers force many ties.
        return [[rng.randint(0, 3) for _ in range(width)] for _ in range(height)]
    return [[rng.uniform(0, 255) for _ in range(width)] for _ in range(height)]


class DpSeamTests(unittest.TestCase):
    def assert_valid(self, seam, length: int, limit: int) -> None:
        self.assertEqual(len(seam), length)
        for index in seam:
            self.assertIs(type(index), int)
            self.assertTrue(0 <= index < limit)
        for a, b in zip(seam, seam[1:]):
            self.assertLessEqual(abs(a - b), 1)

    def test_repo_example_is_the_hand_checked_optimum(self) -> None:
        seam, cost = solve_dp_vertical(COUNTEREXAMPLE)
        self.assertEqual(seam, [1, 2, 2])
        self.assertEqual(cost, 2)
        self.assertEqual(find_dp_vertical_seam(COUNTEREXAMPLE), [1, 2, 2])
        self.assertEqual(vertical_seam_cost(COUNTEREXAMPLE, seam), 2)

    def test_cumulative_table_on_repo_example(self) -> None:
        self.assertEqual(
            cumulative_energy_vertical(COUNTEREXAMPLE),
            [[1, 2, 100], [5, 51, 2], [10, 52, 2]],
        )

    def test_matches_brute_force(self) -> None:
        rng = random.Random(7)
        for case in range(300):
            height, width = rng.randint(1, 5), rng.randint(1, 5)
            energy = random_map(rng, height, width, integers=case % 2 == 0)
            seam, cost = solve_dp_vertical(energy)
            self.assert_valid(seam, height, width)
            self.assertAlmostEqual(cost, brute_force_min_cost(energy))
            self.assertAlmostEqual(vertical_seam_cost(energy, seam), cost)

    def test_cost_equals_graph_cut_and_never_exceeds_greedy(self) -> None:
        rng = random.Random(11)
        for _ in range(40):
            height, width = rng.randint(2, 12), rng.randint(2, 12)
            energy = random_map(rng, height, width)
            seam, cost = solve_dp_vertical(energy)
            _, flow = solve_vertical(energy)
            self.assertAlmostEqual(cost, flow, places=6)
            greedy = find_greedy_vertical_seam(energy)
            self.assertLessEqual(cost, vertical_seam_cost(energy, greedy) + 1e-9)

    def test_horizontal_is_vertical_on_the_transpose(self) -> None:
        rng = random.Random(3)
        for _ in range(50):
            height, width = rng.randint(1, 7), rng.randint(1, 7)
            energy = random_map(rng, height, width, integers=True)
            transposed = [list(column) for column in zip(*energy)]
            seam = find_dp_horizontal_seam(energy)
            self.assert_valid(seam, width, height)
            self.assertEqual(seam, find_dp_vertical_seam(transposed))

    def test_horizontal_repo_example(self) -> None:
        # Rows are [1, 2, 100], [4, 50, 0], [5, 50, 0]. Best left-to-right
        # path: 1 -> 2 -> 0, rows 0, 0, 1.
        self.assertEqual(find_dp_horizontal_seam(COUNTEREXAMPLE), [0, 0, 1])

    def test_ties_keep_the_smaller_index(self) -> None:
        flat = [[5, 5, 5, 5] for _ in range(4)]
        self.assertEqual(find_dp_vertical_seam(flat), [0, 0, 0, 0])
        self.assertEqual(find_dp_horizontal_seam(flat), [0, 0, 0, 0])
        # Two seams cost 2: [0, 0] and [2, 2]. The left one wins.
        self.assertEqual(find_dp_vertical_seam([[1, 9, 1], [1, 9, 1]]), [0, 0])

    def test_single_row_and_single_column(self) -> None:
        self.assertEqual(find_dp_vertical_seam([[3, 1, 2]]), [1])
        self.assertEqual(find_dp_vertical_seam([[3], [1], [2]]), [0, 0, 0])
        self.assertEqual(find_dp_horizontal_seam([[3], [1], [2]]), [1])
        self.assertEqual(find_dp_horizontal_seam([[3, 1, 2]]), [0, 0, 0])

    def test_caller_map_is_not_modified(self) -> None:
        energy = copy.deepcopy(COUNTEREXAMPLE)
        find_dp_vertical_seam(energy)
        find_dp_horizontal_seam(energy)
        cumulative_energy_vertical(energy)
        self.assertEqual(energy, COUNTEREXAMPLE)

    def test_rejects_the_same_maps_as_greedy(self) -> None:
        bad_maps = [
            [],
            [[]],
            [1, 2, 3],
            [[1, 2], [3]],
            [[1, float("nan")]],
            [[1, float("inf")]],
            [[True, False]],
            [["1", "2"]],
            "abc",
            np.zeros((2, 2)),
            [[[1, 2]], [[3, 4]]],
        ]
        for bad in bad_maps:
            for finder in (find_dp_vertical_seam, find_dp_horizontal_seam):
                with self.assertRaises(ValueError):
                    finder(bad)
            with self.assertRaises(ValueError):
                find_greedy_vertical_seam(bad)

    def test_tall_map_does_not_recurse(self) -> None:
        energy = [[1.0, 2.0, 3.0] for _ in range(5000)]
        self.assertEqual(find_dp_vertical_seam(energy), [0] * 5000)


class DpNumpyTests(unittest.TestCase):
    def test_same_seams_as_the_pure_python_finder(self) -> None:
        rng = random.Random(19)
        for case in range(200):
            height, width = rng.randint(1, 12), rng.randint(1, 12)
            energy = random_map(rng, height, width, integers=case % 2 == 0)
            self.assertEqual(find_dp_vertical_seam_numpy(energy), find_dp_vertical_seam(energy))
            self.assertEqual(find_dp_horizontal_seam_numpy(energy), find_dp_horizontal_seam(energy))

    def test_accepts_an_array_and_returns_plain_ints(self) -> None:
        energy = np.array(COUNTEREXAMPLE, dtype=np.float64)
        before = energy.copy()
        seam = find_dp_vertical_seam_numpy(energy)
        self.assertEqual(seam, [1, 2, 2])
        self.assertTrue(all(type(index) is int for index in seam))
        self.assertEqual(find_dp_horizontal_seam_numpy(energy), [0, 0, 1])
        self.assertTrue(np.array_equal(energy, before))

    def test_rejects_bad_maps(self) -> None:
        bad_maps = [
            [],
            [[]],
            [1, 2, 3],
            [[1, 2], [3]],
            [[1, float("nan")]],
            [[True, False]],
            [["1", "2"]],
            np.zeros((2, 2, 3)),
        ]
        for bad in bad_maps:
            with self.assertRaises(ValueError):
                find_dp_vertical_seam_numpy(bad)


if __name__ == "__main__":
    unittest.main()
