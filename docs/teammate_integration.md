# Teammate integration notes

Read this with `docs/greedy_method.md`. That file is the report narrative. This file is the contract for using or replacing the greedy code. Branch: `LX-Greedy`.

## What you can call

Run commands from the repository root. Dependencies are in `requirements.txt`: `numpy` and `opencv-python`.

```text
py -m pip install -r requirements.txt
py -m unittest tests.test_greedy tests.test_resize tests.test_invariants tests.test_compare
```

| Function | Module | Role |
|---|---|---|
| `find_greedy_vertical_seam(energy)` | `algorithms/greedy.py` | One column index per row |
| `find_greedy_horizontal_seam(energy)` | `algorithms/greedy.py` | One row index per column |
| `vertical_seam_cost(energy, seam)` | `algorithms/greedy.py` | Sum after a vertical seam is chosen |
| `resize_greedy(image, target_width, target_height, energy_fn=None)` | `algorithms/greedy_resize.py` | Shrink only |
| `compute_energy(image)` | `temporary/energy.py` | Temporary shared energy |
| `remove_vertical_seam(image, seam)` | `temporary/seam_remove.py` | Temporary deletion |
| `remove_horizontal_seam(image, seam)` | `temporary/seam_remove.py` | Temporary deletion |
| `load_image(path)` / `save_image(path, image)` | `temporary/image_io.py` | Temporary file I/O |
| `run_comparison(...)` | `benchmarks/compare_greedy_dp.py` | Comparison slot |

`REMOVAL_ORDER` in `algorithms/greedy_resize.py` is the string `vertical then horizontal`. `ENERGY_FORMULA` in `temporary/energy.py` is the string form of the energy rule below. Do not give greedy and DP two different formulas.

## Energy maps passed to the seam finders

`find_greedy_vertical_seam` and `find_greedy_horizontal_seam` take a plain two-dimensional sequence, such as a list of lists. They do not accept a NumPy array. Convert first:

```python
seam = find_greedy_vertical_seam(energy.tolist())
```

`resize_greedy` already does this. Each row must have the same length. Values must be finite real numbers. `True` and `False` are rejected. Empty, ragged, one-dimensional, and three-dimensional maps are rejected. Indexes are zero-based.

A vertical seam has length H and stores a column index. A horizontal seam has length W and stores a row index. Consecutive indexes differ by at most 1. Ties keep the smaller index. On a horizontal seam, that means the topmost row.

## Images

`resize_greedy` accepts a non-empty NumPy array:

- 2D: gray
- shape `(H, W, 3)`: BGR, not RGB
- shape `(H, W, 4)`: BGRA

`target_width` and `target_height` are integers from 1 through the current size. A Boolean is rejected. The function cannot enlarge the image and cannot insert seams. An equal target returns a copy and does not compute energy. The caller's array is not modified.

When both axes shrink, all vertical seams are removed first, then all horizontal seams. Energy is computed on the current image before every seam. A later DP resizer has to use that same order, or the comparison mixes two schedules.

`remove_vertical_seam` refuses a width-1 image. `remove_horizontal_seam` refuses a height-1 image. A seam that skips a column or a row raises `ValueError`.

## Energy formula to match

`compute_energy` copies the image and returns one float64 value per pixel:

`0.114 B + 0.587 G + 0.299 R`, then a 3×3 Sobel in x and y, border `BORDER_REFLECT_101`, then `hypot(gx, gy)`. A gray image skips the color weights.

The `temporary/` package is a stand-in. Replace it with the team's shared functions only if greedy and DP still call one energy function and one pair of removal functions. Do not add a second formula beside it.

On the machine where this was tested, Windows blocks the OpenCV DLL. `compute_energy` then uses the NumPy steps above. `load_image` returns uint8 BGR. Without OpenCV it reads only 24-bit BMP, and `save_image` writes only uint8 BGR BMP.

## How to attach a DP finder

`run_comparison` stays blocked unless both callables are passed. Passing only one is still blocked. No seam costs or timings are written in that case.

```python
from benchmarks.compare_greedy_dp import run_comparison

result = run_comparison(
    energy.tolist(),
    dp_find_vertical=dp_find_vertical,
    dp_find_horizontal=dp_find_horizontal,
)
```

`dp_find_vertical(energy_map)` must return one column index per row. `dp_find_horizontal(energy_map)` must return one row index per column. Give them the same list-of-lists map the greedy finder accepts. Do not modify the caller's map.

With both callables and an energy map, the current driver compares only the first vertical seam. It gives each finder its own copy of that map and returns `greedy_seam`, `dp_seam`, `greedy_cost`, and `dp_cost` in memory. It does not call the horizontal finder, and it does not write those numbers into `results/`. File output exists only for the blocked record: `results/greedy_dp/run_metadata.json` and `results/greedy_dp/comparison_notes.md`.

There is no DP implementation in this repository. Do not fill the comparison with invented costs, timings, or photographs. The map `[[1, 2, 100], [4, 50, 0], [5, 50, 0]]` has a tested greedy seam `[0, 0, 0]` with cost 10. The seam `[1, 2, 2]` with cost 2 was checked by hand. It is not the output of a DP program here.

## What not to do

- Do not call a library content-aware resize or `cv2.resize` to shrink the image.
- Do not compare a greedy seam from the original image with a DP seam from an image that already had a seam removed.
- Do not treat a lower seam energy as a measured judgment of visual quality.
- The report text and the AI-use note are in `docs/greedy_method.md`. This file does not replace them.
