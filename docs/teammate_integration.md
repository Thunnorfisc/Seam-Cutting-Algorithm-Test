# Teammate integration notes

Read this with `docs/greedy_method.md` (greedy, Item 3), `docs/graphcut_method.md` (graph cut, Item 4) and `docs/dp_method.md` (dynamic programming). Those files are the report narratives. This file is the contract for using or replacing the greedy and graph-cut code. The greedy tree is also on `LX-Greedy`; the graph-cut code is on `main`.

## What you can call

Run commands from the repository root. Dependencies are in `requirements.txt`: `numpy` and `opencv-python`.

```text
py -m pip install -r requirements.txt
py -m unittest tests.test_greedy tests.test_resize tests.test_invariants tests.test_compare
py -m unittest tests.test_graphcut_build tests.test_graphcut_flow tests.test_graphcut_resize tests.test_graphcut_dinic tests.test_compare_graphcut
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
| `find_graphcut_vertical_seam(energy, max_flow="dinic")` | `algorithms/graphcut.py` | One column index per row (optimal) |
| `find_graphcut_horizontal_seam(energy, max_flow="dinic")` | `algorithms/graphcut.py` | One row index per column (optimal) |
| `solve_vertical(energy, max_flow="dinic")` | `algorithms/graphcut.py` | `(seam, flow)`; flow equals the seam cost |
| `resize_graphcut(image, target_width, target_height, energy_fn=None, max_flow="dinic")` | `algorithms/graphcut_resize.py` | Shrink only, same contract as `resize_greedy` |
| `run(image_path, target_width, target_height, ...)` | `benchmarks/compare_graphcut.py` | Greedy vs graph cut (vs DP when supplied); writes `results/graphcut/<image>/` |
| `find_dp_vertical_seam(energy)` | `algorithms/dp.py` | One column index per row (optimal) |
| `find_dp_horizontal_seam(energy)` | `algorithms/dp.py` | One row index per column (optimal) |
| `solve_dp_vertical(energy)` | `algorithms/dp.py` | `(seam, cost)` |
| `resize_dp(image, target_width, target_height, energy_fn=None)` | `algorithms/dp_resize.py` | Shrink only, same contract as `resize_greedy` |
| `py frontend\serve.py` | `frontend/` | Comparison web page; see `docs/dp_method.md`, section 8 |

`max_flow` is `"dinic"` or `"edmonds_karp"`. Both return the same seam; Dinic is faster.

`REMOVAL_ORDER` in `algorithms/greedy_resize.py` is the string `vertical then horizontal`. `ENERGY_FORMULA` in `temporary/energy.py` is the string form of the energy rule below. Do not give greedy and DP two different formulas.

## What to replace

`algorithms/greedy.py` is the finished greedy finder. Leave that file unchanged. Leave the removal loops in `algorithms/greedy_resize.py` unchanged. The only edit in that file is the two imports below. `algorithms/graphcut.py` is the finished graph-cut finder; leave it unchanged too.

The graph-cut code reuses private helpers from the greedy files, so renaming or removing them breaks graph cut:

- `algorithms/graphcut.py` imports `_copy_finite_grid` from `algorithms/greedy.py`.
- `algorithms/graphcut_resize.py` imports `REMOVAL_ORDER`, `_energy_rows`, `_require_image`, and `_require_target` from `algorithms/greedy_resize.py`.

The `temporary/` package is a stand-in. Replace these functions with the team's shared versions:

| Keep this call | Current file | The replacement must |
|---|---|---|
| `compute_energy(image)` | `temporary/energy.py` | Return a new float64 map, one finite value per pixel, and leave `image` unchanged. Export `ENERGY_FORMULA` as the same rule in words. |
| `remove_vertical_seam(image, seam)` | `temporary/seam_remove.py` | Return a new array one column narrower, same height, dtype, and channel count. Refuse a width-1 image. Reject a seam whose length is not the height, whose indexes are out of range, or whose consecutive columns differ by more than 1. |
| `remove_horizontal_seam(image, seam)` | `temporary/seam_remove.py` | Return a new array one row shorter, same width, dtype, and channel count. Refuse a height-1 image. Apply the same checks, with seam length equal to the width. |
| `load_image(path)` | `temporary/image_io.py` | Return a uint8 BGR array. Do not resize or carve. |
| `save_image(path, image)` | `temporary/image_io.py` | Write the array and leave it unchanged. |

Keep these names. Then the only code edits are the imports in the next list. Greedy and DP must call one `compute_energy` and one pair of removal functions.

`resize_greedy` uses `compute_energy` only when the caller omits `energy_fn`. It always calls `remove_vertical_seam` and `remove_horizontal_seam`. There is no parameter for the removal functions, so those imports are required.

Change these imports so they point at the shared modules:

`algorithms/greedy_resize.py`

```python
from temporary.energy import compute_energy
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam
```

`benchmarks/compare_greedy_dp.py`

```python
from temporary.energy import ENERGY_FORMULA
```

`tests/test_resize.py`

```python
from temporary.energy import compute_energy
from temporary.image_io import load_image, save_image
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam
```

`tests/test_invariants.py`

```python
from temporary.energy import compute_energy
```

`algorithms/graphcut_resize.py`

```python
from temporary.energy import compute_energy
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam
```

`benchmarks/compare_graphcut.py`

```python
from temporary.energy import ENERGY_FORMULA, compute_energy
from temporary.image_io import load_image, save_image
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam
```

`tests/test_graphcut_resize.py`

```python
from temporary.energy import compute_energy
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam
```

`tests/test_compare_graphcut.py`

```python
from temporary.image_io import save_image
```

`tests/test_greedy.py`, `tests/test_compare.py`, `tests/test_graphcut_build.py`, `tests/test_graphcut_flow.py`, and `tests/test_graphcut_dinic.py` do not import `temporary/`. Leave them unchanged.

After nothing still imports `temporary/`, delete that package. Then run, from the repository root:

```text
py -m unittest tests.test_greedy tests.test_resize tests.test_invariants tests.test_compare
py -m unittest tests.test_graphcut_build tests.test_graphcut_flow tests.test_graphcut_resize tests.test_graphcut_dinic tests.test_compare_graphcut
```

## Energy maps passed to the seam finders

`find_greedy_vertical_seam` and `find_greedy_horizontal_seam` take a plain two-dimensional sequence, such as a list of lists. They do not accept a NumPy array. Convert first:

```python
seam = find_greedy_vertical_seam(energy.tolist())
```

`resize_greedy` and `resize_graphcut` already do this. The graph-cut finders take the same input and validate it with the same function, so they accept and reject exactly the same maps. Each row must have the same length. Values must be finite real numbers. `True` and `False` are rejected. Empty, ragged, one-dimensional, and three-dimensional maps are rejected. Indexes are zero-based.

A vertical seam has length H and stores a column index. A horizontal seam has length W and stores a row index. Consecutive indexes differ by at most 1. For greedy, ties keep the smaller index; on a horizontal seam, that means the topmost row. Graph cut always returns a minimum-cost seam; when several tie, it may pick a different one from DP, but the cost is the same.

## Images

`resize_greedy` accepts a non-empty NumPy array:

- 2D: gray
- shape `(H, W, 3)`: BGR, not RGB
- shape `(H, W, 4)`: BGRA

`target_width` and `target_height` are integers from 1 through the current size. A Boolean is rejected. The function cannot enlarge the image and cannot insert seams. An equal target returns a copy and does not compute energy. The caller's array is not modified.

When both axes shrink, all vertical seams are removed first, then all horizontal seams. Energy is computed on the current image before every seam. `resize_graphcut` follows the same contract and order. A later DP resizer has to use that same order, or the comparison mixes two schedules.

`remove_vertical_seam` refuses a width-1 image. `remove_horizontal_seam` refuses a height-1 image. A seam that skips a column or a row raises `ValueError`.

## Energy formula to match

`compute_energy` copies the image and returns one float64 value per pixel:

`0.114 B + 0.587 G + 0.299 R`, then a 3×3 Sobel in x and y, border `BORDER_REFLECT_101`, then `hypot(gx, gy)`. A gray image skips the color weights.

Use the shared `compute_energy` from "What to replace" for both methods. Do not add a second formula beside it.

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

The DP implementation is `algorithms/dp.py` (`docs/dp_method.md`); `algorithms/dp_resize.py`, `frontend/` and the DP tests also import `temporary/` and need the same import change. Do not fill the comparison with invented costs, timings, or photographs. The map `[[1, 2, 100], [4, 50, 0], [5, 50, 0]]` has a tested greedy seam `[0, 0, 0]` with cost 10. The seam `[1, 2, 2]` with cost 2 was first checked by hand; the graph-cut finder now returns it, and `tests/test_graphcut_flow.py` checks that. `find_dp_vertical_seam` returns it too (`tests/test_dp.py`).

`benchmarks/compare_graphcut.py` is the fuller comparison. Pass the DP pair to `run`:

```python
from benchmarks.compare_graphcut import run

record = run(
    "photo.bmp", target_width, target_height,
    dp_find_vertical=dp_find_vertical,
    dp_find_horizontal=dp_find_horizontal,
)
```

DP is then included in every table and gets a blue seam overlay. Unlike `run_comparison`, this writes measured results to `results/graphcut/<image>/` (`metrics.json`, `summary.md`, `seam_<method>.bmp`, `resized_<method>.bmp`). Graph cut is optimal, so a correct DP must give the same first-seam cost as graph cut on every image; a different cost means one of them has a bug.

## What not to do

- Do not call a library content-aware resize or `cv2.resize` to shrink the image.
- Do not compare a greedy seam from the original image with a DP seam from an image that already had a seam removed.
- Do not treat a lower seam energy as a measured judgment of visual quality.
- Do not run graph cut on large images. One Dinic seam takes about 1 s at 120×81 and minutes at full photo size; Edmonds–Karp is several times slower. Use inputs of about 100×100 or smaller, and only add `--ek-resize` for small images.
- The report text and the AI-use notes are in `docs/greedy_method.md` and `docs/graphcut_method.md`. This file does not replace them.
