# Graph-cut seam carving (Assignment 2, Item 4)

This note is for the team report and the presentation. It describes the graph-cut code on `main`. For the files and imports that other parts depend on, read `docs/teammate_integration.md`. The greedy write-up is `docs/greedy_method.md`.

## 1. Scope

Item 4 finds seams as a minimum s–t cut and solves it with max-flow. There are three implementations of the same graph:

- Edmonds–Karp in pure Python, which is the method the brief names.
- Dinic in pure Python, added as an improvement. It is the default in `algorithms/graphcut.py`.
- The same Dinic compiled with Numba (`algorithms/graphcut_numba.py`). It is the default graph cut on the comparison page when Numba is installed.

All three return the same seam. The code handles vertical and horizontal seams. Resizing only shrinks.

DP is in `algorithms/dp.py` (`docs/dp_method.md`). On every map tested, DP and graph cut find the same minimum seam energy (section 6).

| File | Contents |
|---|---|
| `algorithms/graphcut.py` | `FlowGraph`, `SeamGraph`, `edmonds_karp`, `dinic`, `solve_vertical`, `find_graphcut_vertical_seam`, `find_graphcut_horizontal_seam` |
| `algorithms/graphcut_resize.py` | `resize_graphcut` |
| `algorithms/graphcut_numba.py` | `solve_vertical_numba`, `find_graphcut_vertical_seam_numba`, `find_graphcut_horizontal_seam_numba`, `warm_up` (optional; needs Numba) |
| `benchmarks/compare_graphcut.py` | Greedy vs graph cut (and DP when supplied): seams, costs, timings, output images |
| `tests/test_graphcut_*.py`, `tests/test_compare_graphcut.py` | 30 tests (section 9) |
| `frontend/engine.py` | Registers the Numba graph cut on the comparison page as `graphcut_numba` |

## 2. Input and energy

The graph-cut code uses the same energy function as greedy: the shared `compute_energy` and `ENERGY_FORMULA`. It also uses the same removal functions and the same `REMOVAL_ORDER` (`vertical then horizontal`). Energy is recomputed on the current image before every seam.

The finders take the same list-of-lists energy map as the greedy finders. They validate it with greedy's `_copy_finite_grid`, so both reject exactly the same inputs. The Numba finders also accept a 2D NumPy array directly; an array is checked with the same rules (non-empty, 2D, finite real numbers, not Boolean) and copied, so the caller's array is unchanged.

The capacity of the edge that removes pixel `(i, j)` is its backward energy `e[i][j]`. The brief also mentions forward energy, `|I[i, j+1] − I[i, j−1]|`. It is not used, so that greedy, graph cut and DP all minimise the same quantity.

## 3. Graph construction (vertical seam)

Pixel `(i, j)` becomes node `i·(W+1) + j`. Column `W` is a dummy column (see below). The source is node `H·(W+1)` and the sink is the next node. "∞" is the total energy plus 1, which is larger than any finite cut, so the arithmetic stays finite.

| Edge | Capacity | Purpose |
|---|---|---|
| `s → (i, 0)` | ∞ | Column 0 is always on the source side |
| `(i, W) → t` | ∞ | The dummy column is always on the sink side |
| `(i, j) → (i, j+1)` for `j = 0..W-1` | `e[i][j]` | Cutting this edge means removing pixel `(i, j)` |
| `(i, j+1) → (i, j)` | ∞ | The source side of each row is a prefix, so each row is cut exactly once |
| `(i, j) → (i−1, j−1)`, `(i, j) → (i+1, j−1)` for `j = 1..W-1` | ∞ | Neighbouring rows' cut positions differ by at most 1, so the seam is connected |

**Why the dummy column is needed.** The brief ties the last real column to the sink. That means the last column can never be removed. The repo's test map `[[1, 2, 100], [4, 50, 0], [5, 50, 0]]` shows the problem. Its optimal seam `[1, 2, 2]` (cost 2) uses the last column, so the brief's graph returns cost 10 instead. With the dummy column, every real pixel has its own capacity edge, and the set of seams equals the DP's search space.

**Why a finite cut is a seam.** Let the source side of row `i` be the prefix of columns `0..a_i`.
- The backward ∞ edges rule out any source side that isn't a prefix.
- The diagonal edge `(i+1, j) → (i, j−1)` would be cut at infinite cost if `j ≤ a_{i+1}` but `j − 1 > a_i`. So `a_{i+1} ≤ a_i + 1`, and the other diagonal gives `a_i ≤ a_{i+1} + 1`.
- Each row then crosses exactly one finite edge, `(i, a_i) → (i, a_i+1)`. So the cut capacity is `Σ e[i][a_i]`, which is the seam's energy.

Every connected seam corresponds to exactly one finite cut, so the minimum cut is the minimum-energy seam. `tests/test_graphcut_build.py` checks this by listing every seam on small maps, before any max-flow code runs.

**Storage.** `FlowGraph` keeps flat lists `edge_to` and `capacity`, plus a list of edge ids for each node. Each `add_edge` call stores the forward edge at an even index and its reverse (capacity 0) at the next index. So the reverse of edge `e` is `e ^ 1`, and the tail of `e` is `edge_to[e ^ 1]`.

The Numba version builds exactly the same edges with the same capacities, but stores them as flat NumPy arrays so Numba can compile the loops: `edge_to` and `capacity` arrays with the same `e ^ 1` pairing, and the edges leaving node `u` listed in `adjacent[start[u] : start[u + 1]]` (compressed sparse row form). `build_seam_graph` creates all edges of one kind at once with array operations instead of a Python loop.

A horizontal seam is found by transposing the energy map and running the vertical finder, as greedy does.

## 4. Max-flow and seam recovery

**Edmonds–Karp.**
1. Run a BFS from `s` over edges with capacity left above `EPSILON = 1e-9`.
2. Walk back from `t` to find the smallest remaining capacity on the path (the bottleneck).
3. Subtract the bottleneck from each edge on the path and add it to each reverse edge.
4. Repeat until no path reaches `t`.

**Dinic.** Each phase has two parts:
1. One BFS labels every node with its distance from `s` (its level).
2. Paths that only go from level `k` to level `k+1` are pushed until this layered graph is saturated.

The path search is iterative, not recursive. A seam path is at least H nodes long, so a recursive DFS would hit Python's recursion limit on tall images. `next_edge` (the current-arc pointer) and setting `level[u] = -1` at dead ends mean no edge or node is retried within a phase.

**Seam recovery.**
1. After max-flow, one more BFS from `s` in the residual graph finds the source side S.
2. In each row, the rightmost pixel in S is the seam pixel. Column 0 is always in S, so this always exists.

By the max-flow min-cut theorem, the flow value equals the cut capacity, which equals the seam's cost.

**Why both algorithms give the same seam.** The set of nodes reachable from `s` after any maximum flow is always the same set. So Edmonds–Karp and Dinic return identical seams, not just equal costs, and `tests/test_graphcut_dinic.py` checks this.

When several seams tie for the minimum, graph cut may pick a different one from a DP. The cost is still the same.

**The Numba version.** `dinic_arrays` and `reachable_from` in `algorithms/graphcut_numba.py` are the same Dinic and the same seam recovery, step for step, written with preallocated arrays (level, current-arc pointer, BFS queue, path) so that Numba compiles them to machine code. Because the graph and the steps are the same, it returns the same seam as the pure-Python Dinic, not just one of equal cost. Reading back the seam is one array operation: S is a prefix of each row, so the number of the row's pixels in S, minus 1, is the seam column.

Numba compiles a function the first time it is called. `cache=True` saves the machine code in `algorithms/__pycache__/`, so later processes load it (about 0.25–0.5 s) instead of compiling again (a few seconds). `warm_up()` makes that first call on a 2×2 map. `frontend/engine.py` calls it when the server starts, so no timed run and no live-preview request includes compile time. If Numba cannot be imported (not installed, or its library blocked by Windows application control), the page leaves the method out and everything else works as before.

**Why `EPSILON` exists.** Float subtraction can leave residual capacities like `1e-14`. Counting those as usable would create fake augmenting paths and give a wrong S.

## 5. Complexity

For an H × W energy map, the graph has V = H(W+1) + 2 nodes. It has about 4HW edges, and twice that once reverse edges are counted. Both are `O(HW)`.

| Step | Time | Space |
|---|---|---|
| Build the graph | `O(HW)` | `O(HW)`, with a large constant (about 8HW edge entries, each in two Python lists) |
| Edmonds–Karp | `O(V·E²)`, i.e. `O((HW)³)`, worst case | `O(V)` for BFS |
| Dinic | `O(V²·E)`, i.e. `O((HW)³)`, worst case; far fewer phases in practice | `O(V)` |
| Seam recovery | `O(HW)` | `O(V)` |
| Full resize, k seams | k × (energy `O(HW)` + build + max-flow) | `O(HW)` |

The Numba version has the same complexity; compiling changes only the constant factor.

For comparison, a full-grid DP seam is `O(HW)` per seam (course background, not a measurement; see `docs/greedy_method.md`, section 5). Greedy's decision loop is `O(W + H)`. Graph cut has the same optimal answer as DP but a much higher cost per seam.

## 6. Measurements

These were run in a Linux workspace with OpenCV available, not on the Windows machine used for the greedy record. Treat them as indicative, and rerun on the report machine.

**One vertical seam, random maps (uniform 0–255 energies):**

| Map | Greedy | Dinic | Edmonds–Karp |
|---|---|---|---|
| 20×20 | 0.15 ms, cost 1360 | 0.009 s, cost 948 | 0.022 s, cost 948 |
| 40×40 | 0.25 ms, cost 2448 | 0.11 s, cost 1648 | 0.36 s, cost 1648 |
| 60×60 | 0.56 ms, cost 4210 | 0.61 s, cost 2800 | 3.2 s, cost 2800 |

On 300 random 8×8 maps, greedy found the optimal seam 68 times. On average its seam cost 1.26× the optimum.

**Photograph.** The photograph is Wikimedia Commons `Broadway_tower_edit.jpg`, 960×651. Check its licence page before putting it in the report.

It was reduced to 120×81 by averaging 8×8 pixel blocks before carving. This only prepares a smaller input; no seam carving was done at this step. At full size, one Dinic seam would take minutes. Thirty vertical seams were then removed, giving 90×81:

| | Greedy | Dinic | Edmonds–Karp |
|---|---|---|---|
| First vertical seam cost | 5323.55 | 1620.08 | 1620.08 |
| First vertical seam time | 0.0015 s | 1.07 s | 13.6 s |
| First horizontal seam cost | 435.41 | 392.91 | 392.91 |
| First horizontal seam time | 0.0031 s | 0.83 s | 2.44 s |
| Full resize, 30 seams | 0.049 s | 38.9 s | not run |

The output images (seam overlays and resized images) were not committed to `results/`. To regenerate them:

```text
py benchmarks\compare_graphcut.py broadway_tower_small.bmp --width 90 --height 81
```

**Simple test picture.** A drawn scene (sky, two clouds, a tower, a person, grass), 400×300, was reduced to 100×75 the same way and carved to 75×75 (25 vertical seams). The first vertical seam cost 318.4 with greedy and 302.2 with both graph-cut algorithms (Dinic 0.60 s, Edmonds–Karp 1.38 s). The full resize took 0.03 s with greedy and 15.2 s with Dinic.

**Pure Python vs Numba, one vertical seam** (Broadway Tower photo scaled to each size; Linux workspace):

| Size | Pure-Python Dinic | Numba Dinic | Speed-up | Same seam |
|---|---|---|---|---|
| 120×81 | 1.27 s | 0.060 s | 21× | yes (checked) |
| 320×217 | 34.1 s | 1.31 s | 26× | yes (cost) |

**Report machine (Windows), simple test picture at full size, 400×300, comparison page.** Pure-Python Dinic took 122.5 s for the first vertical seam. The other methods, from one Run of the page:

| | Greedy | DP | DP, NumPy rows | Graph cut, Numba Dinic |
|---|---|---|---|---|
| First vertical seam energy | 584.20 (1.39× the minimum) | 421.40 | 421.40 | 421.40 |
| First vertical seam time | 46.99 ms | 94.73 ms | 18.69 ms | 7.704 s |
| First horizontal seam energy | 0.00 | 0.00 | 0.00 | 0.00 |
| First horizontal seam time | 0.127 s | 0.123 s | 19.16 ms | 71.91 ms |
| Full resize 400 → 200 wide (200 seams), total | 11.2 s | 18.2 s | 4.575 s | 2080.2 s |
| Of which seam finding | 8.843 s | 16.0 s | 3.307 s | 2074.8 s |
| Per seam | 44.21 ms | 80.04 ms | 16.53 ms | 10.4 s |

On the same machine, Numba made the first seam about 16× faster than pure Python (122.5 s → 7.7 s). The report machine ran greedy about 3× slower than the Linux workspace (47 ms against 15 ms for the same seam), which is why its Numba seam took 7.7 s against 3.5 s there. A teammate's C++ Dinic was reported at 188 s for the same 400 → 200 resize; it was not rerun on this machine, so treat that as approximate.

**Profile of one Numba Dinic seam** (400×300, first vertical seam): 187 phases (each one full BFS), 1,920 augmenting paths, 142 million edge checks while searching for paths, and 22.5 million nodes processed by the BFS. Building the graph took about 50 ms and the input checks about 25 ms, against about 3.4 s for max-flow.

## 7. Discussion

**Seams.** On the photograph, greedy's first seam starts at the top-left. Because it can't look ahead, it zig-zags down through the cloud edges, which have high energy. Its cost is 3.3× the optimum. The graph-cut seam runs down the plain sky to the right of the tower.

**Visual quality.** Both resized images keep the tower intact, since its stone texture has the highest energy in the picture.
- Greedy took most of the width from the left, which squashed the clouds and moved the person closer to the edge.
- Graph cut took the width from the empty sky on the right. The clouds and the person keep their spacing, but the right tower ends up almost touching the image edge.

Lower seam energy is not, by itself, a measure of visual quality. This is one image at reduced size.

**Why the photo's seams hug the right edge.** The first 17 graph-cut seams all ran down the plain sky right of the tower, which looks like a crop. This is the true minimum, not a bug: a separate DP run on the same energy map gave the same cost (1620.08), and the cheapest seam left of the tower costs 2573.82. The plain right sky has an average energy of about 15, the cloudy left sky about 62, because the soft cloud edges are gradients to the Sobel filter. Removing plain sky leaves plain sky, so later seams stay there until that strip is used up, then move left.

**On the simple test picture it is the other way round.** Graph cut spread its seams around (right of the tower, between the cloud and the person, along the left edge), and the tower, clouds and person all stay intact. Greedy cropped the left side: the top row's energies are almost equal, greedy breaks ties toward the leftmost column, so it slides down the left border and pushes the person against the edge.

**Zero-energy horizontal seams.** On the 400×300 picture every method finds a horizontal seam of energy 0.00. With the Sobel border handling (`BORDER_REFLECT_101`), a border row of smooth sky has no gradient at all, so its energy is exactly 0. It is a limitation of the energy function, not of the seam finders. It also explains why the Numba horizontal seam took 72 ms: a zero-cost cut is found almost at once.

**Running time.** Graph cut was hundreds of times slower than greedy on every map. On the report machine, the 200-seam resize took 2080 s with the Numba graph cut against 4.6 s with the NumPy DP, about 450× slower for the same seams. Dinic was about 2.5–13× faster than Edmonds–Karp, and the gap grew with size. Max-flow costs far more than linear time, while the DP fills one table entry per pixel. So for plain seam carving, graph cut gives the DP's answer at a much higher price.

**Why graph cut is still worth having.** The graph formulation extends to problems the row-by-row DP does not handle directly:
- Forward energy, which costs the new edges created by removing a pixel.
- Video, where the seam becomes a surface through a 3D volume.

**Improvements (report item 5d).**
- Dinic over Edmonds–Karp. Done: about 2.5–13× faster.
- A compiled implementation. Done with Numba: about 16–26× faster on the same machine, with identical seams. Compiling changes the constant factor, not the complexity.
- Three small Dinic changes were tried and measured, then not adopted. Each gave identical seams. Average over 10 seams at 400×300 (Linux workspace): 5.01 s per seam for the current version, 3.98 s with all three. The changes were (1) backing up only to the first saturated edge after a push instead of restarting from the source, (2) measuring levels as distance to the sink, and (3) stopping the BFS as soon as it reaches the source. The profile above explains the small gain: re-walking paths (change 1) is under 1% of the work; the time goes into 187 full BFS phases, which only change 3 reduces. Timings varied by about 15% between identical runs.
- Not done: storing the graph as a pixel grid (neighbours computed from `(i, j)` instead of looked up), Boykov–Kolmogorov max-flow (designed for image grids; keeps its search trees between augmentations instead of a full BFS per phase), and reusing the previous flow between seams.

Multithreading helps little for augmenting-path methods, because each path depends on the previous one. Push–relabel algorithms parallelise better.

## 8. Using the code

```python
from algorithms.graphcut import find_graphcut_vertical_seam, find_graphcut_horizontal_seam, solve_vertical
from algorithms.graphcut_resize import resize_graphcut

seam = find_graphcut_vertical_seam(energy.tolist())             # one column per row
seam, flow = solve_vertical(energy.tolist(), "edmonds_karp")    # flow == seam cost
out = resize_graphcut(image, target_width, target_height)       # max_flow="dinic" by default
```

`max_flow` accepts `"dinic"` or `"edmonds_karp"`. Anything else raises `ValueError`.

With Numba installed (it is in `requirements.txt`; `install.bat` / `install.sh` install it and compile the code once):

```python
from algorithms.graphcut_numba import find_graphcut_vertical_seam_numba, solve_vertical_numba, warm_up

warm_up()                                                      # compile before timing anything
seam = find_graphcut_vertical_seam_numba(energy)               # NumPy array or list of lists
seam, flow = solve_vertical_numba(energy)                      # same seam as the pure-Python Dinic
```

## 9. Testing evidence

Command:

```text
py -m unittest tests.test_graphcut_build tests.test_graphcut_flow tests.test_graphcut_resize tests.test_graphcut_dinic tests.test_compare_graphcut tests.test_graphcut_numba
```

On branch `Will-JITPy`, the whole repository suite (104 tests) passes with Python 3.10 in a Linux VM on the report machine. That VM has no Numba, so the 7 Numba-dependent tests were skipped, which also checks that everything works without it. With Numba 0.68 (Linux workspace), the graph-cut, DP, resize, front-end and benchmark tests (86) all pass, including the Numba ones. Rerun with `py` on Windows, with Numba installed, before quoting these in the report.

Covered:
- **Construction** (`test_graphcut_build`). Every seam on the 3×3 repo map and on 30 random maps up to 4×4 is checked: a cut is finite exactly when the seam is connected, and a finite cut equals the seam's energy. The 3×3 map's minimum is 2.
- **Max-flow** (`test_graphcut_flow`). On 200 random maps up to 5×5, about a quarter of them zero energies, max-flow = brute-force minimum = cost of the recovered seam. The 3×3 map returns `[1, 2, 2]` with cost 2. Also covers horizontal seams, single-column maps, and that the caller's map is unchanged.
- **Dinic** (`test_graphcut_dinic`). Matches Edmonds–Karp (same seam and same flow) and brute force on 200 maps and on 15×15 and 30×30 maps. A 3000-row map runs without a recursion error. Full resizes with the two algorithms are identical. Unknown algorithm names are rejected.
- **Resize** (`test_graphcut_resize`). Same order and energy recomputation as greedy, an equal target copies without computing energy, gray images work, the caller's image is unchanged, and bad targets are rejected.
- **Benchmark** (`test_compare_graphcut`). The shared `carve` loop reproduces `resize_greedy` and `resize_graphcut` exactly. Graph cut never costs more than greedy. The DP slot requires both finders. Output files are written.
- **Numba** (`test_graphcut_numba`, 6 tests). Same seam and flow as the pure-Python Dinic, and the brute-force minimum, on 200 random maps; identical vertical and horizontal seams on 20×30 and 35×25 maps; the 3×3 map gives `[1, 2, 2]`; a 3000-row map; NumPy arrays give the same result as lists and are not modified; 1D, empty, NaN, Boolean and 3D arrays are rejected. The class is skipped when Numba cannot be imported.
- **Comparison page** (`test_frontend_engine`, Numba parts). Carving with `graphcut_numba` reproduces `resize_graphcut`; its first seams equal the pure-Python Dinic's; it is listed before the pure-Python Dinic, has a higher size limit (60,000 pixels against 12,000), and the live preview gives the same picture with both.

Not covered:
- Full-size photographs.
- Timing stability across machines.

## 10. Sources and AI use

- Assignment 2 brief, Item 4: the graph construction, Ford–Fulkerson / Edmonds–Karp, and BFS on the residual graph for the cut.
- Max-flow min-cut theorem, Edmonds–Karp and Dinic: standard algorithms course material.
- The dummy-column correction and the use of Dinic were added in this implementation and are explained in sections 3 and 4.

AI use: Claude was used to break Item 4 into steps, to draft the graph, max-flow, resize and benchmark code and their tests, to draft the Numba version and measure it, and to write this note and the integration-doc updates. I reviewed each draft, placed the code into the repository, and ran the tests. The measurements in section 6 come from actual runs, as stated there; none were invented.

## Presentation outline (about 2 minutes)

- **20 s.** Each pixel becomes a node. Cutting the edge to a pixel's right neighbour means removing that pixel. Infinite backward and diagonal edges force every finite cut to be one connected seam.
- **30 s.** Min cut = max flow, so running max-flow gives the cheapest seam. Mention the dummy column, using the 3×3 map: without it the answer is 10, with it the correct answer is 2.
- **30 s.** Edmonds–Karp vs Dinic: same seam, Dinic up to about 13× faster. Compiling Dinic with Numba: same seam again, about 16× faster on the report machine (122.5 s → 7.7 s for one 400×300 seam). Still far slower than DP: 2080 s against 4.6 s for 200 seams.
- **40 s.** Broadway Tower: show the greedy and graph-cut seam overlays and the two resized images. Graph-cut seam cost is 1620 vs greedy's 5324, but 38.9 s vs 0.05 s for 30 seams. Graph cut = DP's answer at a much higher cost; its value is generality (forward energy, video).
