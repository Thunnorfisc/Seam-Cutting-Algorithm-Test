"""What the web page asks for, as plain functions on NumPy arrays.

Nothing here knows about HTTP. The server in ``frontend/serve.py`` turns
requests into these calls, and ``tests/test_frontend_engine.py`` calls them
directly.

Every method uses the shared ``compute_energy``, the shared seam removal and
the same schedule as the ``resize_*`` functions: vertical seams first, then
horizontal, energy recomputed before every seam.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import OrderedDict

import numpy as np

from algorithms.dp import find_dp_horizontal_seam, find_dp_vertical_seam
from algorithms.dp_numpy import find_dp_horizontal_seam_numpy, find_dp_vertical_seam_numpy
from algorithms.graphcut import find_graphcut_horizontal_seam, find_graphcut_vertical_seam
from algorithms.greedy import find_greedy_horizontal_seam, find_greedy_vertical_seam
from temporary.energy import ENERGY_FORMULA, compute_energy
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam

# The Numba graph cut is optional: without Numba (not installed, or its library
# blocked by Windows application control) the page simply does not list it.
try:
    from algorithms import graphcut_numba
    # Compile now (or load the cached machine code), so no timed run or live
    # preview request pays the compile cost. A few seconds the very first time,
    # well under a second once cached in algorithms/__pycache__.
    graphcut_numba.warm_up()
    HAVE_NUMBA = True
except (ImportError, OSError):  # OSError: Windows blocked Numba's compiled library
    graphcut_numba = None
    HAVE_NUMBA = False

# Seam colours are RGB here (the page draws them). They match the BGR colours
# in benchmarks/compare_graphcut.py: greedy red, DP blue, graph cut green.
#
# ``max_pixels`` is the largest image a method runs on without the caller
# saying "force". One pure-Python Dinic seam takes about 1 s at 120x81 and
# minutes on a photograph; Edmonds-Karp is several times slower again. The
# Numba Dinic finds the same seam about 20-35x faster (about 1 s at 320x217).
METHODS = {
    "greedy": {
        "label": "Greedy",
        "colour": [255, 0, 0],
        "vertical": find_greedy_vertical_seam,
        "horizontal": find_greedy_horizontal_seam,
        "takes_array": False,
        "max_pixels": None,
        "optimal": False,
    },
    "dp": {
        "label": "Dynamic programming",
        "colour": [0, 90, 255],
        "vertical": find_dp_vertical_seam,
        "horizontal": find_dp_horizontal_seam,
        "takes_array": False,
        "max_pixels": None,
        "optimal": True,
    },
    "dp_numpy": {
        "label": "DP, NumPy rows",
        "colour": [150, 80, 255],
        "vertical": find_dp_vertical_seam_numpy,
        "horizontal": find_dp_horizontal_seam_numpy,
        "takes_array": True,
        "max_pixels": None,
        "optimal": True,
    },
    # "graphcut_numba" is inserted here when Numba is available (below).
    "graphcut_dinic": {
        "label": "Graph cut, Dinic",
        "colour": [0, 200, 0],
        "vertical": lambda energy: find_graphcut_vertical_seam(energy, "dinic"),
        "horizontal": lambda energy: find_graphcut_horizontal_seam(energy, "dinic"),
        "takes_array": False,
        "max_pixels": 12_000,
        "optimal": True,
    },
    "graphcut_edmonds_karp": {
        "label": "Graph cut, Edmonds-Karp",
        "colour": [200, 200, 0],
        "vertical": lambda energy: find_graphcut_vertical_seam(energy, "edmonds_karp"),
        "horizontal": lambda energy: find_graphcut_horizontal_seam(energy, "edmonds_karp"),
        "takes_array": False,
        "max_pixels": 4_000,
        "optimal": True,
    },
}


if HAVE_NUMBA:
    # The default graph cut: listed first among the graph-cut methods.
    _numba_spec = {
        "label": "Graph cut, Dinic (Numba)",
        "colour": [0, 170, 140],
        "vertical": graphcut_numba.find_graphcut_vertical_seam_numba,
        "horizontal": graphcut_numba.find_graphcut_horizontal_seam_numba,
        "takes_array": True,
        "max_pixels": 60_000,
        "optimal": True,
    }
    METHODS = {
        **{k: v for k, v in METHODS.items() if not k.startswith("graphcut")},
        "graphcut_numba": _numba_spec,
        **{k: v for k, v in METHODS.items() if k.startswith("graphcut")},
    }


class Cancelled(Exception):
    """Raised inside a carve when its ``progress`` callback asks it to stop."""


class TooLarge(ValueError):
    """The image is over a method's size limit and the caller did not force it."""


def method_table() -> list[dict]:
    """The methods as plain data for the page."""
    return [
        {
            "name": name,
            "label": spec["label"],
            "colour": spec["colour"],
            "max_pixels": spec["max_pixels"],
            "optimal": spec["optimal"],
        }
        for name, spec in METHODS.items()
    ]


def rgba_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """Turn the browser's RGBA bytes into the BGR array the algorithms use. Alpha is dropped."""
    if width < 1 or height < 1 or len(data) != width * height * 4:
        raise ValueError("pixel data does not match the stated size")
    rgba = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(rgba[:, :, 2::-1])


def bgr_to_rgba(image: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = image[:, :, 2::-1]
    rgba[:, :, 3] = 255
    return rgba.tobytes()


def energy_preview(image: np.ndarray) -> tuple[bytes, float]:
    """Return the energy map scaled to one byte per pixel, and its true maximum."""
    energy = compute_energy(image)
    peak = float(energy.max())
    scaled = energy * (255.0 / peak) if peak > 0 else energy
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8).tobytes(), peak


def check_size(name: str, image: np.ndarray, force: bool = False) -> None:
    spec = _spec(name)
    limit = spec["max_pixels"]
    pixels = image.shape[0] * image.shape[1]
    if limit is not None and pixels > limit and not force:
        raise TooLarge(
            f"{spec['label']} is limited to {limit} pixels; this image has {pixels}"
        )


def first_seams(image: np.ndarray, name: str, repeats: int = 3) -> list[dict]:
    """Find the first vertical and horizontal seam on the original image's energy map.

    Every method gets the same map, so the costs can be compared directly.
    The time is seam finding only: the energy map is computed before the
    clock starts.
    """
    spec = _spec(name)
    energy = compute_energy(image)
    rows = []
    for orientation in ("vertical", "horizontal"):
        find = spec[orientation]
        times = []
        seam: list[int] = []
        for _ in range(max(1, repeats)):
            # A fresh copy each run, so no finder can affect another.
            fresh = energy.copy() if spec["takes_array"] else energy.tolist()
            start = time.perf_counter()
            seam = find(fresh)
            times.append(time.perf_counter() - start)
        rows.append({
            "method": name,
            "orientation": orientation,
            "seam": [int(index) for index in seam],
            "cost": seam_cost(energy, seam, orientation),
            "median_seconds": statistics.median(times),
            "repeats": len(times),
        })
    return rows


def seam_cost(energy: np.ndarray, seam, orientation: str) -> float:
    if orientation == "vertical":
        return float(sum(energy[row, column] for row, column in enumerate(seam)))
    return float(sum(energy[row, column] for column, row in enumerate(seam)))


def carve_trace(image: np.ndarray, target_width: int, target_height: int, name: str,
                progress=None) -> dict:
    """Carve ``image`` to the target with one method and record where each seam was.

    Returns the carved image, timings, and ``order``: an array the shape of
    the ORIGINAL image where 0 means the pixel survived and k means it was
    removed by the k-th seam. Seams 1..``vertical_seams`` are vertical, the
    rest are horizontal.

    ``progress(done, total)`` is called after every seam. It may raise
    ``Cancelled`` to stop the run.
    """
    spec = _spec(name)
    height, width = image.shape[:2]
    if not (1 <= target_width <= width and 1 <= target_height <= height):
        raise ValueError(f"target must be within the image size {width}x{height}")

    work = np.array(image, copy=True)
    # ids follows the pixels: ids[y, x] is the original position of the pixel
    # now at (y, x). Removing the same seam from it keeps the two in step.
    ids = np.arange(height * width, dtype=np.int64).reshape(height, width)
    order = np.zeros(height * width, dtype=np.int32)
    vertical_total = width - target_width
    total = vertical_total + (height - target_height)
    done = 0
    find_seconds = 0.0
    energy_seconds = 0.0
    start = time.perf_counter()

    def find_seam(orientation: str) -> list[int]:
        nonlocal find_seconds, energy_seconds
        t = time.perf_counter()
        energy = compute_energy(work)
        prepared = energy if spec["takes_array"] else energy.tolist()
        energy_seconds += time.perf_counter() - t
        t = time.perf_counter()
        seam = spec[orientation](prepared)
        find_seconds += time.perf_counter() - t
        return seam

    while work.shape[1] > target_width:
        seam = find_seam("vertical")
        done += 1
        order[ids[np.arange(work.shape[0]), seam]] = done
        work = remove_vertical_seam(work, seam)
        ids = remove_vertical_seam(ids, seam)
        if progress is not None:
            progress(done, total)

    while work.shape[0] > target_height:
        seam = find_seam("horizontal")
        done += 1
        order[ids[seam, np.arange(work.shape[1])]] = done
        work = remove_horizontal_seam(work, seam)
        ids = remove_horizontal_seam(ids, seam)
        if progress is not None:
            progress(done, total)

    return {
        "method": name,
        "image": work,
        "order": order.reshape(height, width),
        "vertical_seams": vertical_total,
        "horizontal_seams": total - vertical_total,
        "seams_removed": total,
        "total_seconds": time.perf_counter() - start,
        "find_seconds": find_seconds,
        "energy_seconds": energy_seconds,
    }


class SeamRecord:
    """Seams of one image under one method, found on demand and remembered.

    This is what makes dragging a size handle feel live. The schedule is
    vertical seams first, so the first k vertical seams are the same for
    every target: asking for a narrower image only finds the extra ones. The
    horizontal seams depend on the width they start from, so they are
    remembered per width (the last few widths only).

    A result is exactly what ``carve_trace`` and the ``resize_*`` functions
    return for the same target; ``tests/test_frontend_engine.py`` checks it.
    Not timed: use ``carve_trace`` for measurements.
    """

    KEPT_WIDTHS = 6

    def __init__(self, image: np.ndarray, name: str):
        self.spec = _spec(name)
        self.name = name
        self.image = image
        self.height, self.width = image.shape[:2]
        self._pixels = image.reshape((self.height * self.width,) + image.shape[2:])
        self._ids = np.arange(self.height * self.width, dtype=np.int64).reshape(self.height, self.width)
        # v_order[y, x] = k: original pixel (y, x) was removed by vertical seam k. 0 = not removed.
        self._v_order = np.zeros((self.height, self.width), dtype=np.int32)
        self._v_depth = 0
        # target width -> [h_order, depth], numbered from 1 within that width.
        self._horizontal: "OrderedDict[int, list]" = OrderedDict()
        self.lock = threading.Lock()

    def carve(self, target_width: int, target_height: int) -> dict:
        if not (1 <= target_width <= self.width and 1 <= target_height <= self.height):
            raise ValueError(f"target must be within the image size {self.width}x{self.height}")
        vertical = self.width - target_width
        horizontal = self.height - target_height
        with self.lock:
            found = self._extend_vertical(vertical)
            grid = self._grid_after_vertical(vertical)
            if horizontal:
                h_order, more = self._extend_horizontal(target_width, grid, horizontal)
                found += more
                grid = _drop_rows(grid, h_order, horizontal)
                order = np.where(self._v_order <= vertical, self._v_order, 0)
                order += np.where((h_order > 0) & (h_order <= horizontal), h_order + vertical, 0)
            else:
                order = np.where(self._v_order <= vertical, self._v_order, 0)
            return {
                "method": self.name,
                "image": self._pixels[grid],
                "order": order.astype(np.int32),
                "vertical_seams": vertical,
                "horizontal_seams": horizontal,
                "seams_removed": vertical + horizontal,
                "seams_found_now": found,
            }

    def _find(self, work: np.ndarray, orientation: str) -> list[int]:
        energy = compute_energy(work)
        return self.spec[orientation](energy if self.spec["takes_array"] else energy.tolist())

    def _grid_after_vertical(self, depth: int) -> np.ndarray:
        """Original pixel ids, laid out as the image looks after ``depth`` vertical seams."""
        keep = (self._v_order == 0) | (self._v_order > depth)
        return self._ids[keep].reshape(self.height, self.width - depth)

    def _extend_vertical(self, depth: int) -> int:
        if depth <= self._v_depth:
            return 0
        grid = self._grid_after_vertical(self._v_depth)
        work = self._pixels[grid]
        rows = np.arange(self.height)
        start = self._v_depth
        while self._v_depth < depth:
            seam = self._find(work, "vertical")
            self._v_order.flat[grid[rows, seam]] = self._v_depth + 1
            work = remove_vertical_seam(work, seam)
            grid = remove_vertical_seam(grid, seam)
            self._v_depth += 1
        return self._v_depth - start

    def _extend_horizontal(self, target_width: int, grid: np.ndarray, depth: int):
        entry = self._horizontal.get(target_width)
        if entry is None:
            entry = [np.zeros((self.height, self.width), dtype=np.int32), 0]
            self._horizontal[target_width] = entry
            while len(self._horizontal) > self.KEPT_WIDTHS:
                self._horizontal.popitem(last=False)
        self._horizontal.move_to_end(target_width)
        h_order, done = entry
        if depth <= done:
            return h_order, 0
        grid = _drop_rows(grid, h_order, done)
        work = self._pixels[grid]
        columns = np.arange(grid.shape[1])
        start = done
        while done < depth:
            seam = self._find(work, "horizontal")
            h_order.flat[grid[seam, columns]] = done + 1
            work = remove_horizontal_seam(work, seam)
            grid = remove_horizontal_seam(grid, seam)
            done += 1
        entry[1] = done
        return h_order, done - start


def _drop_rows(grid: np.ndarray, h_order: np.ndarray, depth: int) -> np.ndarray:
    """Remove the first ``depth`` horizontal seams from a grid of original pixel ids."""
    if depth == 0:
        return grid
    marks = h_order.flat[grid]
    keep = (marks == 0) | (marks > depth)
    # Each column loses exactly ``depth`` cells, so column-major order reshapes cleanly.
    return grid.T[keep.T].reshape(grid.shape[1], grid.shape[0] - depth).T


def _spec(name: str) -> dict:
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(f"unknown method: {name}") from None


__all__ = [
    "Cancelled",
    "ENERGY_FORMULA",
    "HAVE_NUMBA",
    "METHODS",
    "SeamRecord",
    "TooLarge",
    "bgr_to_rgba",
    "carve_trace",
    "check_size",
    "energy_preview",
    "first_seams",
    "method_table",
    "rgba_to_bgr",
    "seam_cost",
]
