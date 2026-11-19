# Dynamic-programming seam carving (Assignment 2)

This note is for the team report and the presentation. It describes the DP code and the comparison page on `main`. The greedy write-up is `docs/greedy_method.md`, the graph-cut write-up is `docs/graphcut_method.md`, and the shared contract is `docs/teammate_integration.md`.

## 1. Scope

This part finds the seam with the minimum total energy by dynamic programming, for vertical and horizontal seams, and resizes an image by removing such seams one at a time. Resizing only shrinks.

| File | Contents |
|---|---|
| `algorithms/dp.py` | `find_dp_vertical_seam`, `find_dp_horizontal_seam`, `solve_dp_vertical`, `cumulative_energy_vertical` |
| `algorithms/dp_numpy.py` | The same DP with each row filled by NumPy (the speed improvement) |
| `algorithms/dp_resize.py` | `resize_dp` |
| `frontend/` | The comparison web page (section 8) |
| `tests/test_dp.py`, `tests/test_dp_resize.py`, `tests/test_frontend_engine.py` | 37 tests (section 9) |

## 2. The three steps

**a) Energy map.** DP uses the team's shared `compute_energy`, the same function greedy and graph cut call: grayscale `0.114 B + 0.587 G + 0.299 R`, a 3×3 Sobel filter in x and y, border `BORDER_REFLECT_101`, then `hypot(gx, gy)`. A pixel on an edge has a large gradient, so removing it is a large disruption. No second formula was added, so all three methods minimise the same quantity.

**b) Minimum seam by DP.** Section 3.

**c) Remove and repeat.** `resize_dp` removes vertical seams until the width matches, then horizontal seams until the height matches (`REMOVAL_ORDER`, the same schedule as the other two methods). The energy map is recomputed on the current image before every seam, because removing a seam changes the gradients next to it. Removal is the shared `remove_vertical_seam` / `remove_horizontal_seam`.

## 3. The recurrence

For a vertical seam on an H × W energy map `e`, let `M[i][j]` be the smallest total energy of any connected seam that starts in row 0 and ends at pixel `(i, j)`.

```text
M[0][j] = e[0][j]
M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1])     (columns outside the image are skipped)
```

**Why this is correct (optimal substructure).** A seam ending at `(i, j)` must pass through one of the three pixels above it. Whichever one it is, the part of the seam above must itself be the cheapest seam ending at that pixel: if a cheaper one existed, swapping it in would make the whole seam cheaper. So the best seam to `(i, j)` is `e[i][j]` plus the best of the three subproblems.

**Why DP and not plain recursion (overlapping subproblems).** `M[i-1][j]` is needed by `M[i][j-1]`, `M[i][j]` and `M[i][j+1]`. Recursion without a table would recompute it each time, and the number of seams grows like 3^H. The table computes each entry once.

**Recovering the seam.** While filling row `i`, the code stores for every `j` which column of row `i-1` gave the minimum (`came_from`). The smallest entry in the last row is the cost of the best seam. Following the stored columns back up to row 0 gives the seam.

**Worked example.** The repo's test map and its table:

```text
e = [[1,  2, 100],        M = [[ 1,  2, 100],
     [4, 50,   0],             [ 5, 51,   2],
     [5, 50,   0]]             [10, 52,   2]]
```

The last-row minimum is 2 at column 2. It came from column 2 in row 1 (value 2), which came from column 1 in row 0 (value 2). The seam is `[1, 2, 2]` with cost `2 + 0 + 0 = 2`. Greedy returns `[0, 0, 0]` with cost 10 on the same map, because it takes the 1 in row 0 and can never reach the zeros. `docs/greedy_method.md` recorded `[1, 2, 2]` as a hand-checked optimum; it is now the output of a DP program, checked in `tests/test_dp.py`.

**Ties.** When totals are equal the smaller index wins, at every row and at the last-row minimum. This is the rule greedy uses, and it makes the result deterministic. Graph cut may pick a different seam of the same cost.

**Horizontal seams.** The map is transposed and the vertical finder is run on it, as in greedy and graph cut. A leftmost tie on the transpose is a topmost tie on the image.

**Input.** The finders take the same list-of-lists map as greedy and validate it with greedy's `_copy_finite_grid`, so they reject exactly the same inputs. The caller's map is not modified. The fill is a loop, not recursion, so a tall image cannot hit Python's recursion limit.

## 4. Complexity

For an H × W map:

| Step | Time | Space |
|---|---|---|
| Fill the table | `O(HW)`: each entry looks at 3 neighbours | `O(W)` for the running row |
| Stored choices for backtracking | — | `O(HW)` |
| Backtrack | `O(H)` | `O(H)` |
| One seam | `O(HW)` | `O(HW)` |
| Full resize, k seams | `O(kHW)`: energy and DP are both `O(HW)` per seam | `O(HW)` |

A horizontal seam is the same with H and W swapped. DP is optimal like graph cut, and linear in the number of pixels like the energy map itself. No method that reads every pixel can be asymptotically faster.

Greedy's decision loop is `O(W + H)`, but the greedy function as written is also `O(HW)` per call, because it validates and copies the whole map first. The measurements show this: greedy is only about 2× faster than DP, not hundreds of times.

## 5. Measurements

Run on Linux, Python 3.14.9, NumPy 2.5.4, OpenCV not installed (so `compute_energy` used its NumPy path). Median of 5 runs; Dinic 1 run. Rerun on the report machine before quoting.

**One vertical seam, random maps (uniform 0–255 energies):**

| Map | Greedy | DP | DP, NumPy rows | Graph cut (Dinic) |
|---|---|---|---|---|
| 20×20 | 0.09 ms, cost 1127 | 0.12 ms, cost 923 | 0.12 ms | 0.013 s, cost 923 |
| 40×40 | 0.22 ms, cost 2486 | 0.40 ms, cost 1986 | 0.23 ms | 0.104 s, cost 1986 |
| 60×60 | 0.50 ms, cost 3581 | 0.91 ms, cost 2546 | 0.54 ms | 0.182 s, cost 2546 |
| 100×100 | 1.59 ms, cost 6506 | 2.65 ms, cost 4354 | 0.73 ms | not run |
| 200×200 | 5.36 ms, cost 11679 | 10.25 ms, cost 9011 | 1.67 ms | not run |
| 400×400 | 21.4 ms, cost 24241 | 46.7 ms, cost 17722 | 4.48 ms | not run |
| 800×800 | 86.5 ms, cost 51353 | 210.8 ms, cost 36037 | 14.3 ms | not run |

- DP and graph cut have the same cost on every map where both ran. The NumPy version returned the identical seam to the pure-Python DP on every map.
- DP time grows 4× when the pixel count grows 4×, which is the `O(HW)` prediction.
- On 300 random 8×8 maps, greedy found the optimal seam 67 times, and its seam cost 1.27× the optimum on average.

**Drawn test picture** (the page's built-in scene, scaled to 100×67, 25 vertical seams removed):

| | Greedy | DP | DP, NumPy rows | Graph cut (Dinic) |
|---|---|---|---|---|
| First vertical seam cost | 1686.53 | 1386.90 | 1386.90 | 1386.90 |
| First vertical seam time | 0.98 ms | 1.84 ms | 0.49 ms | 0.417 s |
| Full resize, total | 0.033 s | 0.058 s | 0.023 s | 10.2 s |
| Full resize, seam finding only | 0.021 s | 0.044 s | 0.013 s | 10.1 s |

This is a drawn scene, not a photograph, and one image. No photograph was measured for this note.

## 6. Discussion

**Seams.** Greedy commits to the cheapest pixel in row 0 and cannot recover if that leads into an edge lower down. DP considers every seam and picks the cheapest. On the test picture greedy's first seam cost 1.22× the optimum; on random maps about 1.27× on average.

**Visual quality.** Lower seam energy is not, by itself, a measure of visual quality. Judge it by looking at the resized images side by side on the page. DP removes the least total gradient, which usually means flat regions such as sky go first.

**Running time.** DP gives graph cut's answer at roughly greedy's price: about 2× greedy, and about 200× faster than Dinic at 60×60, with the gap growing. For plain seam carving, DP is the method to use.

**Limitations.**
- Removing a seam creates new neighbours, and the energy this adds is not counted (backward energy only). Forward energy would count it; it was left out so that all three methods minimise the same thing.
- Removing many seams one at a time recomputes the whole energy map each time.

**Possible improvements (report item 5d).**
- NumPy row fill. Done (`algorithms/dp_numpy.py`): same seams, about 15× faster at 800×800.
- Recompute energy only near the removed seam instead of over the whole image.
- Forward energy.
- A compiled implementation.

## 7. Using the code

```python
from algorithms.dp import find_dp_vertical_seam, find_dp_horizontal_seam, solve_dp_vertical
from algorithms.dp_resize import resize_dp

seam = find_dp_vertical_seam(energy.tolist())        # one column per row
seam, cost = solve_dp_vertical(energy.tolist())      # cost == minimum seam energy
out = resize_dp(image, target_width, target_height)  # shrink only
```

The benchmark now includes DP by default:

```text
py benchmarks\compare_graphcut.py photo.bmp --width 90 --height 81
```

`benchmarks/compare_greedy_dp.py` is unchanged. Its default run still writes the "blocked" record in `results/greedy_dp/`, which is now out of date; passing the two DP finders to `run_comparison` gives the comparison in memory.

## 8. The comparison page

```text
py frontend\serve.py
```

or double-click `open-page.bat` in the repository root (`./open-page.sh` on Linux and macOS). The page opens in the browser at `http://127.0.0.1:8765/`; closing the terminal window stops it. It needs only Python and NumPy. The browser decodes the picture and sends raw pixels, so PNG and JPEG work without OpenCV. The server listens on this computer only and writes nothing to disk.

1. Drop an image (or use the built-in test picture). "Working size" scales it down in the browser first, so the slow methods finish. That step is plain scaling, not seam carving.
2. Pick the target width and height.
3. Tick the methods and press Run.

The page shows:
- the energy map;
- every method's first seam on the original image, with its energy and the time to find it. All methods get the same energy map here, so these numbers compare like for like;
- each method's resized image, with total time, seam-finding time and time per seam;
- "All seams removed by …": every pixel that method removed, drawn at its place in the original, with a slider and Play to step through the seams in order.

Graph cut is skipped above 12,000 pixels (Dinic) or 4,000 pixels (Edmonds–Karp) unless the "run graph cut above its size limit" box is ticked. `http://127.0.0.1:8765/?demo=100` loads the test picture and runs straight away.

The page calls the same seam finders, the same `compute_energy` and the same removal functions as the `resize_*` functions; `tests/test_frontend_engine.py` checks that its output equals theirs.

## 9. Testing evidence

```text
py -m unittest tests.test_dp tests.test_dp_resize tests.test_frontend_engine
```

37 tests pass, and the 54 earlier tests still pass alongside them (91 in total), on the Linux machine in section 5. Rerun with `py` on Windows before quoting.

Covered:
- **Finder** (`test_dp`). The 3×3 map returns `[1, 2, 2]` with cost 2, and its table is checked entry by entry. On 300 random maps up to 5×5, half of them small integers to force ties, the DP cost equals the brute-force minimum over every connected seam. On 40 maps up to 12×12 the DP cost equals the graph-cut max-flow and never exceeds greedy. Also: horizontal seams, tie-breaking, single-row and single-column maps, the caller's map unchanged, the same rejected inputs as greedy, and a 5000-row map.
- **NumPy version** (`test_dp`). Identical seams to the pure-Python finder on 200 maps, including ties.
- **Resize** (`test_dp_resize`). Same order and energy recomputation as greedy, an equal target copies without computing energy, gray and BGRA images, bad targets rejected, shrinking to 1×1. Every seam along a DP resize costs the same as graph cut's seam on that energy map. The benchmark's shared `carve` loop reproduces `resize_dp`.
- **Page back end** (`test_frontend_engine`). Carving through the page reproduces `resize_greedy`, `resize_dp` and `resize_graphcut` exactly; the removed-pixel map accounts for every pixel; cancelling works; the HTTP API round trip, its error replies and the graph-cut size limit.

Not covered:
- The page's JavaScript has no automated tests. It was checked by running the demo in a browser.
- Full-size photographs, and timing stability across machines.

## 10. Sources and AI use

- Assignment 2 brief, the dynamic-programming item: the three steps in section 2.
- The DP recurrence for seam carving: Avidan and Shamir, "Seam Carving for Content-Aware Image Resizing" (SIGGRAPH 2007), and the course DP material cited in `docs/greedy_method.md`, section 5.

AI use: Claude wrote the DP finders, the NumPy variant, `resize_dp`, the comparison page and its server, their tests, and this note, and ran the tests and the measurements in section 5. The measurements come from actual runs, as stated there; none were invented.

## Presentation outline (about 2 minutes)

- **20 s.** Energy map: Sobel gradient per pixel. Bright = edge = costly to remove.
- **40 s.** `M[i][j]` = cheapest seam ending at `(i, j)` = own energy + the smallest of the three entries above. Fill top to bottom, take the smallest in the last row, walk back up. Show the 3×3 map: greedy 10, DP 2.
- **20 s.** Remove the seam, recompute energy, repeat. `O(HW)` per seam.
- **40 s.** The page: same first-seam energy as graph cut, about 2× greedy's time, hundreds of times faster than graph cut. Show the three resized images side by side.
