"""Checks for the graph-cut comparison script, on small synthetic images."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from algorithms.graphcut_resize import resize_graphcut
from algorithms.greedy_resize import resize_greedy
from benchmarks.compare_graphcut import build_methods, carve, compare_first_seams, run
from temporary.image_io import save_image


def sample_image(height=10, width=12) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class CompareGraphCutTests(unittest.TestCase):
    def test_carve_matches_the_real_resizers(self):
        image = sample_image()
        methods = build_methods()
        greedy, _ = carve(image, 9, 8, *methods["greedy"])
        dinic, _ = carve(image, 9, 8, *methods["graphcut_dinic"])
        self.assertTrue(np.array_equal(greedy, resize_greedy(image, 9, 8)))
        self.assertTrue(np.array_equal(dinic, resize_graphcut(image, 9, 8)))

    def test_graph_cut_never_costs_more_than_greedy(self):
        rows = compare_first_seams(sample_image(), build_methods(), repeats=1)
        for orientation in ("vertical", "horizontal"):
            cost = {r["method"]: r["cost"] for r in rows if r["orientation"] == orientation}
            self.assertAlmostEqual(cost["graphcut_dinic"], cost["graphcut_edmonds_karp"])
            self.assertLessEqual(cost["graphcut_dinic"], cost["greedy"] + 1e-9)

    def test_dp_slot_needs_both_finders(self):
        with self.assertRaises(ValueError):
            build_methods(dp_find_vertical=lambda e: [0] * len(e))

    def test_run_writes_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.bmp"
            save_image(path, sample_image())
            record = run(path, 10, 9, output_dir=Path(folder) / "out", repeats=1)
            out = Path(folder) / "out"
            for name in ("metrics.json", "summary.md", "seam_greedy.bmp",
                         "resized_greedy.bmp", "resized_graphcut_dinic.bmp"):
                self.assertTrue((out / name).exists(), name)
            self.assertFalse(record["dp_available"])


if __name__ == "__main__":
    unittest.main()