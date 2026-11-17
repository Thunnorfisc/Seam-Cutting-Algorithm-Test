"""The DP comparison slot must stay blocked until a real finder is supplied."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.compare_greedy_dp import run_comparison

COUNTEREXAMPLE = [
    [1, 2, 100],
    [4, 50, 0],
    [5, 50, 0],
]
_MEASUREMENT_KEYS = {
    "cost",
    "costs",
    "dp_cost",
    "dp_seam",
    "elapsed",
    "greedy_cost",
    "greedy_seam",
    "median",
    "seam",
    "seams",
    "timing",
    "timings",
}


class BlockedComparisonTests(unittest.TestCase):
    def test_missing_dp_is_blocked_and_writes_no_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder)
            record = run_comparison(output_dir=output_dir)
            self.assertEqual(record["status"], "blocked")
            self.assertFalse(record["dp_available"])
            self.assertFalse(_measurement_keys(record))
            metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "blocked")
            self.assertFalse(_measurement_keys(metadata))
            self.assertTrue((output_dir / "comparison_notes.md").is_file())
            written = [path.name for path in output_dir.iterdir()]
            self.assertEqual(sorted(written), ["comparison_notes.md", "run_metadata.json"])
            for path in output_dir.iterdir():
                self.assertNotIn(path.suffix.lower(), {".csv", ".png", ".jpg", ".jpeg", ".bmp"})

    def test_one_missing_callable_is_still_blocked(self) -> None:
        record = run_comparison(dp_find_vertical=lambda energy: [0] * len(energy))
        self.assertEqual(record["status"], "blocked")
        self.assertFalse(_measurement_keys(record))

    def test_supplied_finders_return_costs_without_writing_results(self) -> None:
        def hostile(energy):
            energy[0][0] = 999
            return [1, 2, 2]

        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder)
            result = run_comparison(
                COUNTEREXAMPLE,
                dp_find_vertical=hostile,
                dp_find_horizontal=lambda energy: [0, 0, 0],
                output_dir=output_dir,
            )
            self.assertEqual(list(output_dir.iterdir()), [])
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["greedy_seam"], [0, 0, 0])
        self.assertEqual(result["dp_seam"], [1, 2, 2])
        self.assertEqual(result["greedy_cost"], 10)
        self.assertEqual(result["dp_cost"], 2)
        self.assertEqual(COUNTEREXAMPLE[0][0], 1)


def _measurement_keys(value) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _MEASUREMENT_KEYS:
                found.add(key)
            found.update(_measurement_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_measurement_keys(item))
    return found


if __name__ == "__main__":
    unittest.main()
