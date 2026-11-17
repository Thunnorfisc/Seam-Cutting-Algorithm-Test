"""Comparison slot for greedy seams against a future dynamic-programming finder.

The default run records that DP is absent. It does not invent seam costs,
timings, or images, and it does not implement DP.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.greedy import find_greedy_vertical_seam, vertical_seam_cost
from algorithms.greedy_resize import REMOVAL_ORDER
from temporary.energy import ENERGY_FORMULA

RESULTS_DIR = ROOT / "results" / "greedy_dp"
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


def run_comparison(energy=None, dp_find_vertical=None, dp_find_horizontal=None, output_dir=None):
    """Compare finders when both DP callables exist. Otherwise record the blockage.

    File output is only the blocked record. A supplied DP pair returns seam
    coordinates and additive costs in memory and does not write them.
    """
    if dp_find_vertical is None or dp_find_horizontal is None:
        record = _blocked_record()
        if output_dir is not None:
            _write_blocked(Path(output_dir), record)
        return record
    if energy is None:
        raise ValueError("energy map is required when DP finders are supplied")
    return _compare_first_vertical_seam(energy, dp_find_vertical)


def _blocked_record() -> dict:
    return {
        "status": "blocked",
        "dp_available": False,
        "reason": "Item 3 does not implement dynamic programming. No DP seam finder was supplied.",
        "energy_formula": ENERGY_FORMULA,
        "removal_order": REMOVAL_ORDER,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "expected_interface": {
            "dp_find_vertical": "callable(energy_map) -> one column index per row",
            "dp_find_horizontal": "callable(energy_map) -> one row index per column",
        },
    }


def _compare_first_vertical_seam(energy, dp_find_vertical):
    import copy

    pristine = copy.deepcopy(energy)
    greedy_seam = find_greedy_vertical_seam(copy.deepcopy(pristine))
    dp_seam = dp_find_vertical(copy.deepcopy(pristine))
    return {
        "status": "compared",
        "dp_available": True,
        "orientation": "vertical",
        "greedy_seam": list(greedy_seam),
        "dp_seam": list(dp_seam),
        "greedy_cost": vertical_seam_cost(pristine, greedy_seam),
        "dp_cost": vertical_seam_cost(pristine, dp_seam),
    }


def _write_blocked(output_dir: Path, record: dict) -> None:
    _reject_measurement_keys(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison_notes.md").write_text(_blocked_notes(record), encoding="utf-8")


def _blocked_notes(record: dict) -> str:
    return f"""# Greedy versus dynamic programming

Status: blocked.

No dynamic-programming seam finder is present. This comparison does not implement one. Seam paths, measured seam costs, timings, and images are not recorded.

The three comparisons required by Assignment 2 — seams, visual quality, and running time — stay unmeasured until a teammate supplies vertical and horizontal finders. Those finders must use this energy formula and the removal order `{record["removal_order"]}`.

Energy formula: {record["energy_formula"]}

There are no sample photographs in this repository.

## Hand-checked illustration

This is not a DP execution and not a visual result. On the map `[[1, 2, 100], [4, 50, 0], [5, 50, 0]]`, the greedy finder returns `[0, 0, 0]` with cost 10. That greedy result is covered by the unit tests. A hand-checked optimal seam on the same map is `[1, 2, 2]` with cost 2.
"""


def _reject_measurement_keys(record: dict) -> None:
    found = _measurement_keys(record)
    if found:
        raise ValueError(f"blocked record contains measurement fields: {sorted(found)}")


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


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def _git_dirty() -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def main() -> None:
    record = run_comparison(output_dir=RESULTS_DIR)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
