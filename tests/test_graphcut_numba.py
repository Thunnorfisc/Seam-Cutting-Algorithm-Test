"""The Numba graph cut must return exactly what the pure-Python Dinic returns."""

import random
import unittest

import numpy as np

from algorithms.graphcut import find_graphcut_horizontal_seam, solve_vertical
from tests.test_graphcut_flow import brute_force_min_cost

try:
    from algorithms.graphcut_numba import find_graphcut_horizontal_seam_numba, solve_vertical_numba
    HAVE_NUMBA = True
except (ImportError, OSError):  # Numba not installed, or its library is blocked on this machine
    HAVE_NUMBA = False


@unittest.skipUnless(HAVE_NUMBA, "numba is not available")
class TestNumba(unittest.TestCase):
    def test_matches_python_and_brute_force(self):
        rng = random.Random(5)
        for _ in range(200):
            h, w = rng.randint(1, 5), rng.randint(1, 5)
            energy = [[rng.choice([0, rng.uniform(0, 10)]) for _ in range(w)] for _ in range(h)]
            seam, flow = solve_vertical_numba(energy)
            py_seam, py_flow = solve_vertical(energy)
            self.assertEqual(seam, py_seam)
            self.assertAlmostEqual(flow, py_flow)
            self.assertAlmostEqual(flow, brute_force_min_cost(energy))

    def test_larger_maps_match_python(self):
        rng = random.Random(6)
        for h, w in ((20, 30), (35, 25)):
            energy = [[rng.uniform(0, 255) for _ in range(w)] for _ in range(h)]
            self.assertEqual(solve_vertical_numba(energy)[0], solve_vertical(energy)[0])
            self.assertEqual(find_graphcut_horizontal_seam_numba(energy), find_graphcut_horizontal_seam(energy))

    def test_repo_example(self):
        seam, flow = solve_vertical_numba([[1, 2, 100], [4, 50, 0], [5, 50, 0]])
        self.assertEqual(seam, [1, 2, 2])
        self.assertAlmostEqual(flow, 2)

    def test_single_column_and_long_map(self):
        self.assertEqual(solve_vertical_numba([[5], [7], [1]])[0], [0, 0, 0])
        self.assertEqual(solve_vertical_numba([[1.0, 2.0]] * 3000)[0], [0] * 3000)

    def test_numpy_array_input(self):
        rng = np.random.default_rng(8)
        energy = rng.uniform(0, 255, size=(20, 30))
        before = energy.copy()
        as_list = energy.tolist()
        self.assertEqual(solve_vertical_numba(energy), solve_vertical_numba(as_list))
        self.assertEqual(find_graphcut_horizontal_seam_numba(energy),
                         find_graphcut_horizontal_seam_numba(as_list))
        self.assertTrue(np.array_equal(energy, before))          # caller's array unchanged
        self.assertEqual(solve_vertical_numba(np.array([[1, 2, 100], [4, 50, 0], [5, 50, 0]]))[0],
                         [1, 2, 2])                                # integer arrays work too

    def test_bad_arrays_rejected(self):
        for bad in (np.array([1.0, 2.0]), np.empty((0, 3)), np.array([[1.0, np.nan]]),
                    np.array([[True, False]]), np.zeros((2, 2, 3))):
            with self.assertRaises(ValueError):
                solve_vertical_numba(bad)


if __name__ == "__main__":
    unittest.main()
