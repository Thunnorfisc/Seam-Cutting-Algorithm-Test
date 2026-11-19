"""Integration tests for DP resizing. Images are synthetic arrays."""

from __future__ import annotations

import random
import unittest

import numpy as np

from algorithms.dp import find_dp_horizontal_seam, find_dp_vertical_seam
from algorithms.dp_resize import REMOVAL_ORDER, resize_dp
from algorithms.graphcut import find_graphcut_horizontal_seam, find_graphcut_vertical_seam
from benchmarks.compare_graphcut import build_methods, carve, compare_first_seams
from temporary.energy import compute_energy
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam


def sample_bgr(height: int = 4, width: int = 6) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (np.arange(width, dtype=np.uint8) * 20)[None, :]
    image[:, :, 1] = (np.arange(height, dtype=np.uint8) * 15)[:, None]
    image[0, 0] = (255, 0, 0)
    return image


def noise_bgr(seed: int, height: int, width: int) -> np.ndarray:
    rng = random.Random(seed)
    values = [rng.randrange(256) for _ in range(height * width * 3)]
    return np.array(values, dtype=np.uint8).reshape(height, width, 3)


class DpResizeTests(unittest.TestCase):
    def test_same_removal_order_as_greedy(self) -> None:
        self.assertEqual(REMOVAL_ORDER, "vertical then horizontal")

    def test_one_vertical_removal_matches_finder(self) -> None:
        image = sample_bgr()
        seam = find_dp_vertical_seam(compute_energy(image).tolist())
        expected = remove_vertical_seam(image, seam)
        result = resize_dp(image, image.shape[1] - 1, image.shape[0])
        self.assertTrue(np.array_equal(result, expected))

    def test_one_horizontal_removal_matches_finder(self) -> None:
        image = sample_bgr()
        seam = find_dp_horizontal_seam(compute_energy(image).tolist())
        expected = remove_horizontal_seam(image, seam)
        result = resize_dp(image, image.shape[1], image.shape[0] - 1)
        self.assertTrue(np.array_equal(result, expected))

    def test_both_axes_vertical_first_and_energy_recomputed(self) -> None:
        image = sample_bgr()
        seen: list[tuple[int, int]] = []

        def energy_fn(current):
            seen.append(current.shape[:2])
            return compute_energy(current)

        result = resize_dp(image, 4, 3, energy_fn=energy_fn)
        self.assertEqual(seen, [(4, 6), (4, 5), (4, 4)])
        self.assertEqual(result.shape, (3, 4, 3))
        self.assertEqual(result.dtype, image.dtype)

    def test_equal_target_copies_without_energy(self) -> None:
        image = sample_bgr()

        def energy_fn(current):
            raise AssertionError("energy must not be computed")

        result = resize_dp(image, image.shape[1], image.shape[0], energy_fn=energy_fn)
        self.assertTrue(np.array_equal(result, image))
        self.assertIsNot(result, image)

    def test_gray_and_bgra_images(self) -> None:
        gray = sample_bgr()[:, :, 0].copy()
        self.assertEqual(resize_dp(gray, 3, 2).shape, (2, 3))
        bgra = np.dstack([sample_bgr(), np.full((4, 6), 255, dtype=np.uint8)])
        self.assertEqual(resize_dp(bgra, 3, 2).shape, (2, 3, 4))

    def test_caller_image_is_not_modified(self) -> None:
        image = sample_bgr()
        before = image.copy()
        resize_dp(image, 3, 2)
        self.assertTrue(np.array_equal(image, before))

    def test_bad_targets_are_rejected(self) -> None:
        image = sample_bgr()
        for width, height in ((0, 4), (7, 4), (6, 0), (6, 5), (True, 4), (3.0, 4), ("3", 4)):
            with self.assertRaises(ValueError):
                resize_dp(image, width, height)

    def test_can_shrink_to_one_pixel(self) -> None:
        self.assertEqual(resize_dp(sample_bgr(), 1, 1).shape, (1, 1, 3))

    def test_every_dp_seam_costs_the_same_as_graph_cut(self) -> None:
        """Follow DP's own run and check each seam against the optimum on that energy map."""
        work = noise_bgr(5, 9, 10)
        for _ in range(4):
            energy = compute_energy(work).tolist()
            dp_seam = find_dp_vertical_seam(energy)
            cut_seam = find_graphcut_vertical_seam(energy)
            dp_cost = sum(energy[row][column] for row, column in enumerate(dp_seam))
            cut_cost = sum(energy[row][column] for row, column in enumerate(cut_seam))
            self.assertAlmostEqual(dp_cost, cut_cost, places=6)
            work = remove_vertical_seam(work, dp_seam)
        for _ in range(3):
            energy = compute_energy(work).tolist()
            dp_seam = find_dp_horizontal_seam(energy)
            cut_seam = find_graphcut_horizontal_seam(energy)
            dp_cost = sum(energy[row][column] for column, row in enumerate(dp_seam))
            cut_cost = sum(energy[row][column] for column, row in enumerate(cut_seam))
            self.assertAlmostEqual(dp_cost, cut_cost, places=6)
            work = remove_horizontal_seam(work, dp_seam)


class DpInBenchmarkTests(unittest.TestCase):
    def test_shared_carve_loop_reproduces_resize_dp(self) -> None:
        image = noise_bgr(2, 8, 9)
        carved, timing = carve(image, 6, 6, find_dp_vertical_seam, find_dp_horizontal_seam)
        self.assertTrue(np.array_equal(carved, resize_dp(image, 6, 6)))
        self.assertEqual(timing["seams_removed"], 5)

    def test_first_seam_cost_matches_graph_cut_and_beats_greedy(self) -> None:
        image = noise_bgr(4, 10, 12)
        methods = build_methods(find_dp_vertical_seam, find_dp_horizontal_seam, include_edmonds_karp=False)
        rows = compare_first_seams(image, methods, repeats=1)
        for orientation in ("vertical", "horizontal"):
            cost = {row["method"]: row["cost"] for row in rows if row["orientation"] == orientation}
            self.assertAlmostEqual(cost["dp"], cost["graphcut_dinic"], places=6)
            self.assertLessEqual(cost["dp"], cost["greedy"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
