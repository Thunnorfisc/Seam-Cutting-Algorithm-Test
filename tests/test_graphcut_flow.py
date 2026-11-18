import itertools
import random
import unittest

from algorithms.graphcut import (
    find_graphcut_vertical_seam,
    find_graphcut_horizontal_seam,
    solve_vertical
)

def brute_force_min_cost(energy) -> float:
    h, w = len(energy), len(energy[0])
    best = float("inf")
    for seam in itertools.product(range(w), repeat = h):
        if all(abs(a - b) <= 1 for a, b in zip(seam, seam[1:])):
            best = min(best, sum(energy[i][j] for i, j in enumerate(seam)))
    return best

def seam_cost(energy, seam) -> float:
    return sum(energy[i][j] for i, j in enumerate(seam))


class TestFlow(unittest.TestCase):
    def assert_valid_vertical(self, energy, seam):
        self.assertEqual(len(seam), len(energy))
        for column in seam:
            self.assertTrue(0 <= column < len(energy[0]))
        for a, b in zip(seam, seam[1:]):
            self.assertLessEqual(abs(a - b), 1)

    def test_repo_example(self):
        energy = [[1, 2, 100], [4, 50, 0], [5, 50, 0]]
        seam, flow = solve_vertical(energy)
        self.assertEqual(seam, [1, 2, 2])
        self.assertAlmostEqual(flow, 2)

    def test_matches_brute_force(self):
        rng = random.Random(1)
        for _ in range(200):
            h, w = rng.randint(1, 5), rng.randint(1, 5)
            energy = [[rng.choice([0, rng.uniform(0, 10)]) for _ in range(w)] for _ in range(h)]
            seam, flow = solve_vertical(energy)
            self.assert_valid_vertical(energy, seam)
            best = brute_force_min_cost(energy)
            self.assertAlmostEqual(flow, best)                  # max-flow = min cut
            self.assertAlmostEqual(seam_cost(energy, seam), best)  # seam read back correctly

    def test_horizontal(self):
        rng = random.Random(2)
        for _ in range(50):
            h, w = rng.randint(1, 5), rng.randint(1, 5)
            energy = [[rng.uniform(0, 10) for _ in range(w)] for _ in range(h)]
            seam = find_graphcut_horizontal_seam(energy)
            self.assertEqual(len(seam), w)
            transposed = [list(c) for c in zip(*energy)]
            self.assertAlmostEqual(
                sum(energy[r][c] for c, r in enumerate(seam)), brute_force_min_cost(transposed)
            )

    def test_input_not_modified(self):
        energy = [[3, 1], [2, 4]]
        find_graphcut_vertical_seam(energy)
        self.assertEqual(energy, [[3, 1], [2, 4]])

    def test_single_column(self):
        self.assertEqual(find_graphcut_vertical_seam([[5], [7], [1]]), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()