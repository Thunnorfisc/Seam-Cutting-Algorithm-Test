# Greedy seam carving (Assignment 2, Item 3)

This note is for the team report and the two-minute presentation. It describes the code on `main` and `LX-Greedy`. It does not submit the assignment. The files to replace are listed in `docs/teammate_integration.md` under "What to replace".

## 1. Scope

Item 3 selects seams with a greedy rule: at each row, take the lowest-energy pixel among the pixels that touch the previous choice. The implementation supports vertical seams and horizontal seams. Resizing only shrinks an image.

Dynamic programming is a teammate's part. This repository does not contain a DP seam finder. The comparison slot in `benchmarks/compare_greedy_dp.py` records that absence. It does not rank the two methods.

## 2. Input and energy

Images are NumPy arrays. A 2D array is gray. A 3-channel array is BGR. A 4-channel array is BGRA. There is no separate RGB mode.

`temporary/energy.py` function `compute_energy` is the temporary shared energy function. A later DP comparison has to call this same function. The formula, also stored as `ENERGY_FORMULA`, is:

`Grayscale 0.114 B + 0.587 G + 0.299 R; 3x3 Sobel dx and dy; BORDER_REFLECT_101; hypot(gx, gy).`

OpenCV is the preferred library for that Sobel and for file loading. On the machine used for the recorded run, Windows application control blocks the OpenCV DLL (WinError 4551), so `compute_energy` runs the same steps in NumPy. `load_image` and `save_image` in `temporary/image_io.py` fall back to 24-bit BMP. `numpy.random` is blocked by the same policy, so the randomized tests use Python's `random.Random`.

Seam deletion is `remove_vertical_seam` and `remove_horizontal_seam` in `temporary/seam_remove.py`. Image loading is `load_image` and `save_image` in `temporary/image_io.py`. All four, plus `compute_energy`, are temporary. `docs/teammate_integration.md` lists each one and the imports to retarget: `algorithms/greedy_resize.py`, `benchmarks/compare_greedy_dp.py`, `tests/test_resize.py`, and `tests/test_invariants.py`.

## 3. Algorithm

`find_greedy_vertical_seam` in `algorithms/greedy.py` returns one column index per row. It does not modify the energy map.

1. In row 0, choose the lowest energy. If several columns tie, choose the smallest column.
2. In each later row, look only at the in-bounds columns `previous - 1`, `previous`, and `previous + 1`.
3. Choose the lowest energy among those columns. A tie keeps the smaller column.
4. Do not add up the path, look at later rows, backtrack, or try every starting column.

`vertical_seam_cost` sums the chosen pixels after the seam exists. That sum is not an input to the next choice.

`find_greedy_horizontal_seam` runs the vertical finder on the transpose. A leftmost tie on the transpose is a topmost tie on the original image. The result has one row index per column.

```text
c = leftmost argmin(E[0])
seam = [c]
for each later row r:
    candidates = in-bounds columns among c-1, c, c+1
    c = argmin of (E[r, c], c)
    append c
return seam

horizontal(E) = vertical(transpose(E))
```

`resize_greedy` in `algorithms/greedy_resize.py` copies the caller's image. `REMOVAL_ORDER` is `vertical then horizontal`. For every seam it computes energy on the current image, asks the greedy finder for a seam, and deletes that seam. An equal target returns a copy and does not compute energy. Targets must be integers from 1 through the original size.

Coordinate trace of the unit-test map:

```text
1   2   100
4  50     0
5  50     0
```

Row 0 chooses column 0 because 1 is smaller than 2 and 100. Row 1 may look at columns 0 and 1 (4 and 50) and chooses column 0. Row 2 again chooses column 0 (5 rather than 50). The seam is `[0, 0, 0]`. Its measured cost is `1 + 4 + 5 = 10`.

## 4. Correctness and limitations

A vertical seam has length H, each index lies in `0 .. W-1`, and consecutive indexes differ by at most 1. A horizontal seam has length W, each index lies in `0 .. H-1`, and the same adjacency rule applies. `resize_greedy` returns an array of the requested height and width, with the caller's dtype and channel count, and it leaves the caller's array unchanged.

Fresh energy is required. After the first removal, the image has changed, so the next seam is chosen on a new map. The resize tests record the shape passed to `compute_energy` and check that the number of calls equals the number of seams removed, and that the width shrinks before the height.

Local optimality does not give the minimum total energy. On the map above, the unit test measures the greedy cost as 10. A hand-checked optimal seam is `[1, 2, 2]`, with cost `2 + 0 + 0 = 2`. That optimal seam was not produced by a DP program in this repository. The tests do not require the greedy finder to return cost 2.

## 5. Complexity

For an image or energy map with H rows and W columns:

- The vertical decision loop scans the first row and then inspects at most three candidates per later row. That loop is `O(W + H)` time. The returned path uses `O(H)` storage.
- The horizontal decision loop is `O(H + W)` time and the path uses `O(W)` storage.
- `find_greedy_vertical_seam` also copies the map while checking that every entry is a finite real number. That copy is `O(HW)`. The horizontal finder builds a transpose, which is another `O(HW)` copy.
- One removal recomputes a full energy map and builds a new image. Both are `O(HW)`. They dominate the `O(W + H)` decision loop.
- Removing k seams, with a full energy recomputation each time, is `O(kHW)` in the worst case. The implementation does not claim that a full resize is `O(H)`.

This repository has no DP implementation, so there is no measured DP runtime. As course background only, a standard full-grid DP seam finder fills an `H` by `W` table and is `O(HW)` time per seam (CSD3130 L02 pp. 2–7; 9. DP1_v4.pdf p. 7). That statement is not a measurement of a teammate's program.

## 6. Experimental methodology

The intended comparison uses the same original image, the same target size, `compute_energy`, and `REMOVAL_ORDER` for both methods. Seam cost would be compared on one frozen energy map, before either image is changed. Full-resize timing would start from fresh in-memory copies and would exclude disk and plotting. Seam-only timing would start from one precomputed map and would exclude energy. The driver would use a warm-up, at least five repeats, and `perf_counter`, then report the median and the raw repeats.

None of that was run. `run_comparison()` was called with no DP finder.

## 7. Results

`results/greedy_dp/run_metadata.json` is the recorded comparison. It was refreshed from commit `f40e2db077ae8026635a4e32956b5684753d9755` on a clean tree. The record says:

- `status`: `blocked`
- `dp_available`: false
- `reason`: Item 3 does not implement dynamic programming. No DP seam finder was supplied.
- Python 3.15.2, platform `Windows-11-10.0.26300-SP0`
- `git_dirty`: false at the time of that record

There is no `seam_comparison.csv`, no timing CSV, and no comparison image. No median was computed. No photograph was resized. The only measured seam cost in the tests is the greedy cost 10 on the numerical map in section 3. The cost-2 seam is a hand-checked illustration, recorded in `results/greedy_dp/comparison_notes.md`, and is not a DP execution.

## 8. Discussion

There is no measured pair of seams on a photograph, no observed distortion, and no measured speed difference. The numerical map shows why a difference is possible: greedy keeps the cheap first pixel and then cannot reach the zero-energy pixels on the right. Whether that difference is visible on real pictures is not known from this repository.

A fair later comparison has to use one energy function and the same vertical-then-horizontal removal order. After the first deletion, the two methods see different images, so later seam costs are not comparable as if they came from one map.

## 9. Testing evidence

Command:

```text
py -m unittest tests.test_greedy tests.test_resize tests.test_invariants tests.test_compare
```

Output from the Phase 6 run:

```text
..............................
----------------------------------------------------------------------
Ran 30 tests in 0.481s

OK
```

30 passed, 0 failed, 0 skipped. Python 3.15.2.

Covered:

- Greedy seam `[0, 0, 0]` with cost 10 on `[[1, 2, 100], [4, 50, 0], [5, 50, 0]]`. The test does not demand cost 2.
- 1×1, one row, one column, ties, diagonal steps, and both borders.
- Rejection of empty, ragged, 1D, 3D, non-numeric, NaN, and infinite maps, including a non-finite value the walk would not visit.
- The caller's energy map and image stay unchanged.
- Seeds 0 and 1, every height from 1 through 8 and width from 1 through 9: 144 vertical seams and 144 horizontal seams. Each seam matches the local rule, stays connected, and repeats.
- Resize sweep on those 72 shapes with seed 0: exact output size, dtype preserved, energy calls equal seams removed, width reduced before height.
- Synthetic BMP roundtrip through `save_image` and `load_image`.
- The comparison record stays `blocked` and does not write a CSV or an image.

Not covered, and not claimed:

- Real photographs.
- A separate RGB channel order.
- Any DP test. None exist in this repository.
- Runtime measurements.

Relevant commits on `LX-Greedy`:

- `f53e8b4` added the local greedy seam finder.
- `9d636e8` added greedy image resizing.
- `e633772` added the fixed-seed invariant tests.
- `f40e2db` recorded the missing DP comparison.

## 10. Sources and AI use

Course materials:

- Assignment 2, pp. 2–4. Page 2 defines connected seams, energy, and repeated removal, and it allows image libraries while forbidding a built-in content-aware resize. Page 3 asks for the greedy comparison. Page 4 requires the presentation limit and the AI declaration.
- Rubrics.pdf, p. 1. Design, implementation, analysis, and documentation are 20 points each. Presentation is 15. Peer evaluation is 5.
- CSD3130 L05, pp. 42–45. A greedy choice does not look at future consequences and is not always optimal.
- CSD3130 L02, pp. 2–7. Operation counts and asymptotic comparison.
- 9. DP1_v4.pdf, p. 7, and 11. Graph 1_v3.1.pdf, pp. 89–92, are course context for dynamic programming and shortest paths. They are not an implementation in this repository.

No external web page was used to set the greedy rule.

AI use: Cursor's coding agent was used for the design, the Python implementation, the tests, the blocked comparison record, and this note. Representative requests were: read each phase plan and complete that phase; implement Item 3 in Python; do not implement dynamic programming; stop after each phase for review. The agent was not used to invent seam costs, timings, or photographs. The Week 13 quiz on 24 November 2026 is closed-book; do not use an AI tool during that quiz.

## Presentation outline (about 2 minutes)

- 15–20 seconds. Greedy picks the cheapest touching pixel at the current row. A DP seam adds that pixel's energy to the cheapest path that can reach it.
- 35–45 seconds. On the 3×3 map, row 0 chooses column 0. The next two rows stay at column 0. The seam is `[0, 0, 0]` and the tested cost is 10. The hand-checked cheaper seam `[1, 2, 2]` costs 2, because the first step would have to accept energy 2.
- 40–50 seconds. Say that the photograph comparison and both timing experiments were not run. The recorded status is `blocked` because this repository has no DP finder. Do not show a DP image or a timing number.
- Remaining time. One vertical decision is `O(W + H)`; one full removal is `O(HW)` because the energy map is rebuilt. The invariant tests checked the local rule on 144 vertical seams and 144 horizontal seams.
