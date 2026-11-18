"""Integration tests for graph-cut resizing. Images are synthetic arrays."""

from __future__ import annotations

import unittest

import numpy as np

from algorithms.graphcut import find_graphcut_horizontal_seam, find_graphcut_vertical_seam
from algorithms.graphcut_resize import REMOVAL_ORDER, resize_graphcut
from temporary.energy import compute_energy
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam


def sample_bgr(height: int = 4, width: int = 6) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (np.arange(width, dtype=np.uint8) * 20)[None, :]
    image[:, :, 1] = (np.arange(height, dtype=np.uint8) * 15)[:, None]
    image[0, 0] = (255, 0, 0)
    return image


class GraphCutResizeTests(unittest.TestCase):
    def test_same_removal_order_as_greedy(self) -> None:
        self.assertEqual(REMOVAL_ORDER, "vertical then horizontal")

    def test_one_vertical_removal_matches_finder(self) -> None:
        image = sample_bgr()
        seam = find_graphcut_vertical_seam(compute_energy(image).tolist())
        expected = remove_vertical_seam(image, seam)
        result = resize_graphcut(image, image.shape[1] - 1, image.shape[0])
        self.assertTrue(np.array_equal(result, expected))

    def test_one_horizontal_removal_matches_finder(self) -> None:
        image = sample_bgr()
        seam = find_graphcut_horizontal_seam(compute_energy(image).tolist())
        expected = remove_horizontal_seam(image, seam)
        result = resize_graphcut(image, image.shape[1], image.shape[0] - 1)
        self.assertTrue(np.array_equal(result, expected))

    def test_both_axes_vertical_first_and_energy_recomputed(self) -> None:
        image = sample_bgr()
        seen: list[tuple[int, int]] = []

        def energy_fn(current):
            seen.append(current.shape[:2])
            return compute_energy(current)

        result = resize_graphcut(image, 4, 3, energy_fn=energy_fn)
        self.assertEqual(seen, [(4, 6), (4, 5), (4, 4)])
        self.assertEqual(result.shape, (3, 4, 3))
        self.assertEqual(result.dtype, image.dtype)

    def test_equal_target_copies_without_energy(self) -> None:
        image = sample_bgr()

        def energy_fn(current):
            raise AssertionError("energy should not be computed")

        result = resize_graphcut(image, image.shape[1], image.shape[0], energy_fn=energy_fn)
        self.assertTrue(np.array_equal(result, image))
        self.assertIsNot(result, image)

    def test_caller_image_unchanged_and_gray_works(self) -> None:
        image = sample_bgr()[:, :, 1].copy()
        before = image.copy()
        result = resize_graphcut(image, 3, 2)
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(np.array_equal(image, before))

    def test_bad_targets_rejected(self) -> None:
        image = sample_bgr()
        for width, height in ((0, 4), (7, 4), (True, 4), (6, 5)):
            with self.assertRaises(ValueError):
                resize_graphcut(image, width, height)


if __name__ == "__main__":
    unittest.main()