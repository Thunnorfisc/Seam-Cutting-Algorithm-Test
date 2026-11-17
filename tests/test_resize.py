"""Integration tests for greedy resizing. Images are synthetic arrays."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from algorithms.greedy import find_greedy_horizontal_seam, find_greedy_vertical_seam
from algorithms.greedy_resize import REMOVAL_ORDER, resize_greedy
from temporary.energy import compute_energy
from temporary.image_io import load_image, save_image
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam

ROOT = Path(__file__).resolve().parents[1]


def sample_bgr(height: int = 4, width: int = 6) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (np.arange(width, dtype=np.uint8) * 20)[None, :]
    image[:, :, 1] = (np.arange(height, dtype=np.uint8) * 15)[:, None]
    image[0, 0] = (255, 0, 0)
    return image


class ResizeTests(unittest.TestCase):
    def test_removal_order_is_vertical_then_horizontal(self) -> None:
        self.assertEqual(REMOVAL_ORDER, "vertical then horizontal")

    def test_one_vertical_removal_matches_finder(self) -> None:
        image = sample_bgr()
        seam = find_greedy_vertical_seam(compute_energy(image).tolist())
        expected = remove_vertical_seam(image, seam)
        result = resize_greedy(image, image.shape[1] - 1, image.shape[0])
        self.assertEqual(result.shape, (image.shape[0], image.shape[1] - 1, 3))
        self.assertTrue(np.array_equal(result, expected))

    def test_one_horizontal_removal_matches_finder(self) -> None:
        image = sample_bgr()
        seam = find_greedy_horizontal_seam(compute_energy(image).tolist())
        expected = remove_horizontal_seam(image, seam)
        result = resize_greedy(image, image.shape[1], image.shape[0] - 1)
        self.assertEqual(result.shape, (image.shape[0] - 1, image.shape[1], 3))
        self.assertTrue(np.array_equal(result, expected))

    def test_several_vertical_removals_recompute_energy(self) -> None:
        image = sample_bgr()
        seen: list[tuple[int, int]] = []

        def energy_fn(current):
            seen.append(current.shape[:2])
            return compute_energy(current)

        result = resize_greedy(image, image.shape[1] - 2, image.shape[0], energy_fn=energy_fn)
        self.assertEqual(seen, [(4, 6), (4, 5)])
        self.assertEqual(result.shape, (4, 4, 3))

        first = remove_vertical_seam(
            image, find_greedy_vertical_seam(compute_energy(image).tolist())
        )
        expected = remove_vertical_seam(
            first, find_greedy_vertical_seam(compute_energy(first).tolist())
        )
        self.assertTrue(np.array_equal(result, expected))

    def test_both_axes_remove_vertical_seams_first(self) -> None:
        image = sample_bgr()
        seen: list[tuple[int, int]] = []

        def energy_fn(current):
            seen.append(current.shape[:2])
            return compute_energy(current)

        result = resize_greedy(image, 4, 3, energy_fn=energy_fn)
        self.assertEqual(seen, [(4, 6), (4, 5), (4, 4)])
        self.assertEqual(result.shape, (3, 4, 3))
        self.assertEqual(result.dtype, image.dtype)

    def test_equal_target_is_a_copy_and_skips_energy(self) -> None:
        image = sample_bgr()
        calls = 0

        def energy_fn(current):
            nonlocal calls
            calls += 1
            return compute_energy(current)

        result = resize_greedy(image, image.shape[1], image.shape[0], energy_fn=energy_fn)
        self.assertEqual(calls, 0)
        self.assertIsNot(result, image)
        self.assertTrue(np.array_equal(result, image))

    def test_invalid_targets_raise_before_energy(self) -> None:
        image = sample_bgr()
        height, width = image.shape[:2]
        calls = 0

        def energy_fn(current):
            nonlocal calls
            calls += 1
            return compute_energy(current)

        invalid_targets = [
            (0, height),
            (-1, height),
            (width + 1, height),
            (width, 0),
            (width, height + 1),
            (1.5, height),
            (True, height),
            (width, "3"),
        ]
        for target_width, target_height in invalid_targets:
            with self.subTest(target_width=target_width, target_height=target_height):
                with self.assertRaises(ValueError):
                    resize_greedy(image, target_width, target_height, energy_fn=energy_fn)
        self.assertEqual(calls, 0)

    def test_input_dtype_and_channels_are_preserved(self) -> None:
        color = sample_bgr()
        snapshot = color.copy()
        color_result = resize_greedy(color, color.shape[1] - 1, color.shape[0])
        self.assertEqual(color_result.dtype, np.uint8)
        self.assertEqual(color_result.shape[2], 3)
        self.assertTrue(np.array_equal(color, snapshot))

        gray = np.arange(20, dtype=np.float32).reshape(4, 5)
        gray_snapshot = gray.copy()
        gray_result = resize_greedy(gray, 4, 4)
        self.assertEqual(gray_result.shape, (4, 4))
        self.assertEqual(gray_result.dtype, np.float32)
        self.assertTrue(np.array_equal(gray, gray_snapshot))

        bgra = np.zeros((3, 4, 4), dtype=np.uint8)
        bgra[:, :, 0] = 20
        bgra[:, :, 3] = 255
        bgra_result = resize_greedy(bgra, 3, 3)
        self.assertEqual(bgra_result.shape, (3, 3, 4))
        self.assertEqual(bgra_result.dtype, np.uint8)

    def test_constant_image_has_zero_energy_and_is_not_modified(self) -> None:
        image = np.full((4, 5, 3), 30, dtype=np.uint8)
        snapshot = image.copy()
        energy = compute_energy(image)
        self.assertEqual(energy.shape, (4, 5))
        self.assertTrue(np.all(energy == 0))
        self.assertTrue(np.array_equal(image, snapshot))

    def test_bmp_roundtrip_keeps_bgr_pixels(self) -> None:
        image = sample_bgr()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.bmp"
            save_image(path, image)
            loaded = load_image(path)
        self.assertEqual(loaded.shape, image.shape)
        self.assertEqual(loaded.dtype, np.uint8)
        self.assertTrue(np.array_equal(loaded, image))

    def test_seam_removal_rejects_a_broken_path_and_a_width_of_one(self) -> None:
        image = sample_bgr(height=2, width=3)
        with self.assertRaises(ValueError):
            remove_vertical_seam(image, [0, 2])
        narrow = np.zeros((2, 1), dtype=np.uint8)
        with self.assertRaises(ValueError):
            remove_vertical_seam(narrow, [0, 0])

    def test_sources_do_not_call_a_library_resizer(self) -> None:
        text = []
        for folder in ("algorithms", "temporary"):
            for path in (ROOT / folder).glob("*.py"):
                text.append(path.read_text(encoding="utf-8"))
        source = "\n".join(text)
        for banned in ("cv2.resize", "seamCarve", "seam_carve", "content_aware", "contentAware"):
            self.assertNotIn(banned, source)


if __name__ == "__main__":
    unittest.main()
