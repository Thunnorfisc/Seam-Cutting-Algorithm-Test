"""Compare graph-cut seam carving with greedy (and DP once a teammate supplies it).

Two measurements on the same image:
1. First seam: every finder gets the same energy map of the original image.
   Records the seam, its cost and the median time to find it.
2. Full resize: every method carves the image to the same target with the
   same schedule (vertical seams, then horizontal; energy recomputed before
   every seam). Records total time, time spent finding seams, and saves the
   output image.

Outputs go to results/graphcut/<image name>/. Lower seam cost is reported as
a number only; judge visual quality by looking at the saved images.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dp import find_dp_horizontal_seam, find_dp_vertical_seam
from algorithms.graphcut import find_graphcut_horizontal_seam, find_graphcut_vertical_seam
from algorithms.greedy import find_greedy_horizontal_seam, find_greedy_vertical_seam
from algorithms.greedy_resize import REMOVAL_ORDER
from temporary.energy import ENERGY_FORMULA, compute_energy
from temporary.image_io import load_image, save_image
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam

RESULTS_DIR = ROOT / "results" / "graphcut"

# BGR colours for the seam overlays.
SEAM_COLOURS = {
    "greedy": (0, 0, 255),
    "graphcut_dinic": (0, 200, 0),
    "graphcut_edmonds_karp": (0, 200, 200),
    "dp": (255, 0, 0),
}


def build_methods(dp_find_vertical=None, dp_find_horizontal=None, include_edmonds_karp=True):
    """Return {name: (vertical finder, horizontal finder)}. DP needs both callables."""
    methods = {
        "greedy": (find_greedy_vertical_seam, find_greedy_horizontal_seam),
        "graphcut_dinic": (
            lambda e: find_graphcut_vertical_seam(e, "dinic"),
            lambda e: find_graphcut_horizontal_seam(e, "dinic"),
        ),
    }
    if include_edmonds_karp:
        methods["graphcut_edmonds_karp"] = (
            lambda e: find_graphcut_vertical_seam(e, "edmonds_karp"),
            lambda e: find_graphcut_horizontal_seam(e, "edmonds_karp"),
        )
    if (dp_find_vertical is None) != (dp_find_horizontal is None):
        raise ValueError("supply both DP finders or neither")
    if dp_find_vertical is not None:
        methods["dp"] = (dp_find_vertical, dp_find_horizontal)
    return methods


def seam_cost(energy, seam, orientation: str) -> float:
    """Return the total energy of ``seam`` on ``energy``, for either orientation."""
    if orientation == "vertical":
        return float(sum(energy[row][col] for row, col in enumerate(seam)))
    return float(sum(energy[row][col] for col, row in enumerate(seam)))


def compare_first_seams(image: np.ndarray, methods: dict, repeats: int = 3) -> list[dict]:
    """Find the first vertical and horizontal seam with every method on one energy map."""
    energy = compute_energy(image).tolist()
    rows = []
    for orientation, index in (("vertical", 0), ("horizontal", 1)):
        for name, finders in methods.items():
            find = finders[index]
            times = []
            for _ in range(repeats):
                # Each run gets a fresh copy so no finder can affect another.
                fresh = [list(r) for r in energy]
                start = time.perf_counter()
                seam = find(fresh)
                times.append(time.perf_counter() - start)
            rows.append({
                "method": name,
                "orientation": orientation,
                "seam": [int(i) for i in seam],
                "cost": seam_cost(energy, seam, orientation),
                "median_seconds": statistics.median(times),
            })
    return rows


def carve(image: np.ndarray, target_width: int, target_height: int, find_vertical, find_horizontal):
    """Shrink with the shared schedule. Returns (image, timing dict)."""
    work = np.array(image, copy=True)
    find_seconds = 0.0
    start = time.perf_counter()
    while work.shape[1] > target_width:
        energy = compute_energy(work).tolist()
        t = time.perf_counter()
        seam = find_vertical(energy)
        find_seconds += time.perf_counter() - t
        work = remove_vertical_seam(work, seam)
    while work.shape[0] > target_height:
        energy = compute_energy(work).tolist()
        t = time.perf_counter()
        seam = find_horizontal(energy)
        find_seconds += time.perf_counter() - t
        work = remove_horizontal_seam(work, seam)
    return work, {
        "total_seconds": time.perf_counter() - start,
        "find_seconds": find_seconds,
        "seams_removed": (image.shape[1] - target_width) + (image.shape[0] - target_height),
    }


def draw_seam(image: np.ndarray, seam, orientation: str, colour) -> np.ndarray:
    """Return a BGR copy of ``image`` with the seam painted in ``colour``."""
    out = np.array(image, copy=True)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=2)
    out = out[:, :, :3].copy()
    for a, b in enumerate(seam):
        if orientation == "vertical":
            out[a, b] = colour
        else:
            out[b, a] = colour
    return out


def run(image_path, target_width, target_height, output_dir=None,
        dp_find_vertical=None, dp_find_horizontal=None,
        include_edmonds_karp=True, edmonds_karp_resize=False, repeats=3) -> dict:
    """Compare every method on one image and write the results to ``output_dir``.

    Measures the first vertical and horizontal seam on the original energy map
    and a full resize to the target, saves seam overlays and resized images as
    BMP, and writes ``metrics.json`` and ``summary.md``. Returns the record.
    Edmonds-Karp runs the full resize only when ``edmonds_karp_resize`` is true.
    """
    image = load_image(image_path)
    height, width = image.shape[:2]
    if not (1 <= target_width <= width and 1 <= target_height <= height):
        raise ValueError(f"target must be within the image size {width}x{height}")
    out = Path(output_dir) if output_dir else RESULTS_DIR / Path(image_path).stem
    out.mkdir(parents=True, exist_ok=True)

    methods = build_methods(dp_find_vertical, dp_find_horizontal, include_edmonds_karp)

    # 1. First seam on identical energy.
    first = compare_first_seams(image, methods, repeats)
    for row in first:
        if row["orientation"] == "vertical":
            save_image(out / f"seam_{row['method']}.bmp",
                       draw_seam(image, row["seam"], "vertical", SEAM_COLOURS[row["method"]]))

    # 2. Full resize. Edmonds-Karp only on request: it gives the same image as
    # Dinic but can take minutes.
    resizes = []
    for name, (find_v, find_h) in methods.items():
        if name == "graphcut_edmonds_karp" and not edmonds_karp_resize:
            continue
        result, timing = carve(image, target_width, target_height, find_v, find_h)
        save_image(out / f"resized_{name}.bmp", result)
        resizes.append({"method": name, **timing})

    record = {
        "image": str(image_path),
        "original_size": [width, height],
        "target_size": [target_width, target_height],
        "energy_formula": ENERGY_FORMULA,
        "removal_order": REMOVAL_ORDER,
        "dp_available": "dp" in methods,
        "first_seams": first,
        "resizes": resizes,
    }
    (out / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(record), encoding="utf-8")
    return record


def summary_markdown(record: dict) -> str:
    """Return the record from ``run`` as Markdown tables."""
    w, h = record["original_size"]
    tw, th = record["target_size"]
    lines = [
        f"# Graph cut comparison: {Path(record['image']).name}",
        "",
        f"{w}x{h} -> {tw}x{th}, removal order: {record['removal_order']}.",
        f"Energy: {record['energy_formula']}",
        "",
        "## First seam (same energy map for every method)",
        "",
        "| Method | Orientation | Seam cost | Median time (s) |",
        "|---|---|---|---|",
    ]
    for r in record["first_seams"]:
        lines.append(f"| {r['method']} | {r['orientation']} | {r['cost']:.2f} | {r['median_seconds']:.4f} |")
    lines += [
        "",
        "## Full resize",
        "",
        "| Method | Seams removed | Total time (s) | Seam-finding time (s) |",
        "|---|---|---|---|",
    ]
    for r in record["resizes"]:
        lines.append(f"| {r['method']} | {r['seams_removed']} | {r['total_seconds']:.3f} | {r['find_seconds']:.3f} |")
    if not record["dp_available"]:
        lines += ["", "DP was not supplied, so it is not in these tables."]
    lines += ["", "Seam overlays: `seam_<method>.bmp`. Outputs: `resized_<method>.bmp`.", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    """Command-line entry point: parse the arguments, run, and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="input image (24-bit BMP if OpenCV is unavailable)")
    parser.add_argument("--width", type=int, required=True, help="target width")
    parser.add_argument("--height", type=int, required=True, help="target height")
    parser.add_argument("--out", help="output folder (default results/graphcut/<image name>)")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats for the first seam")
    parser.add_argument("--no-ek", action="store_true", help="skip Edmonds-Karp entirely")
    parser.add_argument("--ek-resize", action="store_true", help="also run the full resize with Edmonds-Karp")
    parser.add_argument("--no-dp", action="store_true", help="leave dynamic programming out")
    args = parser.parse_args(argv)

    record = run(args.image, args.width, args.height, args.out,
                 dp_find_vertical=None if args.no_dp else find_dp_vertical_seam,
                 dp_find_horizontal=None if args.no_dp else find_dp_horizontal_seam,
                 include_edmonds_karp=not args.no_ek,
                 edmonds_karp_resize=args.ek_resize, repeats=args.repeats)
    print(summary_markdown(record))


if __name__ == "__main__":
    main()
