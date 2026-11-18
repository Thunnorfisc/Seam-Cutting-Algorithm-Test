"""Dinic must agree with Edmonds-Karp and with brute force."""

import random
import unittest

import numpy as np

from algorithms.graphcut import solve_vertical
from algorithms.graphcut_resize import resize_graphcut
from tests.test_graphcut_flow import brute_force_min_cost


class TestDinic(unittest.TestCase):
    def test_matches_edmonds_karp_and_brute_force(self):
        rng = random.Random(3)
        for _ in range(200):
            h, w = rng.randint(1, 5), rng.randint(1, 5)
            energy = [[rng.choice([0, rng.uniform(0, 10)]) for _ in range(w)] for _ in range(h)]
            d_seam, d_flow = solve_vertical(energy, "dinic")
            e_seam, e_flow = solve_vertical(energy, "edmonds_karp")
            self.assertAlmostEqual(d_flow, brute_force_min_cost(energy))
            self.assertAlmostEqual(d_flow, e_flow)
            self.assertEqual(d_seam, e_seam)     # same residual source side

    def test_larger_maps_agree(self):
        rng = random.Random(4)
        for size in (15, 30):
            energy = [[rng.uniform(0, 255) for _ in range(size)] for _ in range(size)]
            d_seam, d_flow = solve_vertical(energy, "dinic")
            e_seam, e_flow = solve_vertical(energy, "edmonds_karp")
            self.assertEqual(d_seam, e_seam)
            self.assertAlmostEqual(d_flow, e_flow)

    def test_long_path_no_recursion_error(self):
        energy = [[1.0, 2.0] for _ in range(3000)]   # path length > recursion limit
        seam, flow = solve_vertical(energy, "dinic")
        self.assertEqual(seam, [0] * 3000)
        self.assertAlmostEqual(flow, 3000.0)

    def test_resize_same_result_both_algorithms(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(12, 14, 3), dtype=np.uint8)
        a = resize_graphcut(image, 10, 9, max_flow="dinic")
        b = resize_graphcut(image, 10, 9, max_flow="edmonds_karp")
        self.assertTrue(np.array_equal(a, b))

    def test_unknown_algorithm_rejected(self):
        with self.assertRaises(ValueError):
            solve_vertical([[1, 2]], "ford_fulkerson")


if __name__ == "__main__":
    unittest.main()