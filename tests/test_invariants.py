"""Fixed-seed invariant checks for the greedy finder and resizer.

These tests check the local rule and safe resizing. They do not require the
globally minimum seam.
"""

from __future__ import annotations

import copy
import random
import unittest

import numpy as np

from algorithms.greedy import find_greedy_horizontal_seam, find_greedy_vertical_seam
from algorithms.greedy_resize import resize_greedy
from temporary.energy import compute_energy

HEIGHTS = range(1, 9)
WIDTHS = range(1, 10)


class InvariantTests(unittest.TestCase):
    def test_vertical_local_choice_is_stable(self) -> None:
        for seed in (0, 1):
            for height in HEIGHTS:
                for width in WIDTHS:
                    energy = _energy_map(seed, height, width)
                    with self.subTest(seed=seed, height=height, width=width):
                        snapshot = copy.deepcopy(energy)
                        first = find_greedy_vertical_seam(energy)
                        second = find_greedy_vertical_seam(energy)
                        self.assertEqual(first, second)
                        self.assertEqual(energy, snapshot)
                        _assert_local_vertical(self, energy, first)

    def test_horizontal_local_choice_is_stable(self) -> None:
        for seed in (0, 1):
            for height in HEIGHTS:
                for width in WIDTHS:
                    energy = _energy_map(seed, height, width)
                    with self.subTest(seed=seed, height=height, width=width):
                        snapshot = copy.deepcopy(energy)
                        first = find_greedy_horizontal_seam(energy)
                        second = find_greedy_horizontal_seam(energy)
                        self.assertEqual(first, second)
                        self.assertEqual(energy, snapshot)
                        transposed = [list(column) for column in zip(*energy)]
                        _assert_local_vertical(self, transposed, first)

    def test_resize_shape_and_energy_count(self) -> None:
        generator = random.Random(0)
        for height in HEIGHTS:
            for width in WIDTHS:
                pixels = [
                    [[generator.randrange(256) for _ in range(3)] for _ in range(width)]
                    for _ in range(height)
                ]
                image = np.array(pixels, dtype=np.uint8)
                target_width = generator.randrange(1, width + 1)
                target_height = generator.randrange(1, height + 1)
                with self.subTest(
                    height=height,
                    width=width,
                    target_height=target_height,
                    target_width=target_width,
                ):
                    snapshot = image.copy()
                    seen: list[tuple[int, int]] = []

                    def energy_fn(current, seen=seen):
                        seen.append(current.shape[:2])
                        return compute_energy(current)

                    result = resize_greedy(
                        image, target_width, target_height, energy_fn=energy_fn
                    )
                    self.assertEqual(result.shape, (target_height, target_width, 3))
                    self.assertEqual(result.dtype, image.dtype)
                    self.assertTrue(np.array_equal(image, snapshot))
                    self.assertEqual(
                        seen, _expected_energy_shapes(height, width, target_height, target_width)
                    )
                    self.assertEqual(
                        len(seen),
                        (width - target_width) + (height - target_height),
                    )


def _energy_map(seed: int, height: int, width: int) -> list[list[float]]:
    # numpy.random is blocked by Windows application control on this machine.
    generator = random.Random(seed)
    return [
        [generator.uniform(-3.0, 3.0) for _ in range(width)]
        for _ in range(height)
    ]


def _assert_local_vertical(test: unittest.TestCase, energy, seam: list[int]) -> None:
    height = len(energy)
    width = len(energy[0])
    test.assertEqual(len(seam), height)
    test.assertEqual(seam[0], _leftmost_argmin(energy[0]))
    for row in range(height):
        column = seam[row]
        test.assertIsInstance(column, int)
        test.assertNotIsInstance(column, bool)
        test.assertGreaterEqual(column, 0)
        test.assertLess(column, width)
        if row == 0:
            continue
        previous = seam[row - 1]
        test.assertLessEqual(abs(column - previous), 1)
        candidates = [
            candidate
            for candidate in (previous - 1, previous, previous + 1)
            if 0 <= candidate < width
        ]
        chosen = min(candidates, key=lambda candidate: (energy[row][candidate], candidate))
        test.assertEqual(column, chosen)


def _leftmost_argmin(row) -> int:
    best = 0
    for column in range(1, len(row)):
        if row[column] < row[best]:
            best = column
    return best


def _expected_energy_shapes(
    height: int, width: int, target_height: int, target_width: int
) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    while width > target_width:
        shapes.append((height, width))
        width -= 1
    while height > target_height:
        shapes.append((height, width))
        height -= 1
    return shapes


if __name__ == "__main__":
    unittest.main()
