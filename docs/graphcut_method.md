# Graph-cut seam carving (Assignment 2, Item 4)

This note is for the team report and the presentation. It describes the graph-cut code on `main`. For the files and imports that other parts depend on, read `docs/teammate_integration.md`. The greedy write-up is `docs/greedy_method.md`.

## 1. Scope

Item 4 finds seams as a minimum s–t cut and solves it with max-flow. Two max-flow algorithms are implemented:

- Edmonds–Karp, which is the method the brief names.
- Dinic, added as an improvement. It is the default.

Both return the same seam. The code handles vertical and horizontal seams. Resizing only shrinks.

The repository still has no DP seam finder. `benchmarks/compare_graphcut.py` has a slot for one (section 7).

| File | Contents |
|---|---|
| `algorithms/graphcut.py` | `FlowGraph`, `SeamGraph`, `edmonds_karp`, `dinic`, `solve_vertical`, `find_graphcut_vertical_seam`, `find_graphcut_horizontal_seam` |
| `algorithms/graphcut_resize.py` | `resize_graphcut` |
| `benchmarks/compare_graphcut.py` | Greedy vs graph cut (and DP when supplied): seams, costs, timings, output images |
| `tests/test_graphcut_*.py`, `tests/test_compare_graphcut.py` | 24 tests (section 9) |

## 2. Input and energy

The graph-cut code uses the same energy function as greedy: the shared `compute_energy` and `ENERGY_FORMULA`. It also uses the same removal functions and the same `REMOVAL_ORDER` (`vertical then horizontal`). Energy is recomputed on the current image before every seam.

The finders take the same list-of-lists energy map as the greedy finders. They validate it with greedy's `_copy_finite_grid`, so both reject exactly the same inputs.

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

## 7. Discussion

**Seams.** On the photograph, greedy's first seam starts at the top-left. Because it can't look ahead, it zig-zags down through the cloud edges, which have high energy. Its cost is 3.3× the optimum. The graph-cut seam runs down the plain sky to the right of the tower.

**Visual quality.** Both resized images keep the tower intact, since its stone texture has the highest energy in the picture.
- Greedy took most of the width from the left, which squashed the clouds and moved the person closer to the edge.
- Graph cut took the width from the empty sky on the right. The clouds and the person keep their spacing, but the right tower ends up almost touching the image edge.

Lower seam energy is not, by itself, a measure of visual quality. This is one image at reduced size.

**Running time.** Graph cut was hundreds of times slower than greedy on every map. Dinic was about 2.5–13× faster than Edmonds–Karp, and the gap grew with size. Max-flow costs far more than linear time, while the DP fills one table entry per pixel. So for plain seam carving, graph cut gives the DP's answer at a much higher price.

**Why graph cut is still worth having.** The graph formulation extends to problems the row-by-row DP does not handle directly:
- Forward energy, which costs the new edges created by removing a pixel.
- Video, where the seam becomes a surface through a 3D volume.

**Possible improvements (report item 5d).**
- Dinic over Edmonds–Karp. Done.
- Boykov–Kolmogorov max-flow, which is designed for image grids.
- Reusing the previous residual graph between seams, instead of rebuilding.
- A compiled implementation instead of pure Python.

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

## 9. Testing evidence

Command:

```text
py -m unittest tests.test_graphcut_build tests.test_graphcut_flow tests.test_graphcut_resize tests.test_graphcut_dinic tests.test_compare_graphcut
```

24 tests pass, at commit `f30ad81` plus the uncommitted benchmark files. The greedy suite (30 tests) still passes alongside them. These results come from Python 3.10 in a Linux VM on the report machine. Rerun with `py` on Windows before quoting them in the report.

Covered:
- **Construction** (`test_graphcut_build`). Every seam on the 3×3 repo map and on 30 random maps up to 4×4 is checked: a cut is finite exactly when the seam is connected, and a finite cut equals the seam's energy. The 3×3 map's minimum is 2.
- **Max-flow** (`test_graphcut_flow`). On 200 random maps up to 5×5, about a quarter of them zero energies, max-flow = brute-force minimum = cost of the recovered seam. The 3×3 map returns `[1, 2, 2]` with cost 2. Also covers horizontal seams, single-column maps, and that the caller's map is unchanged.
- **Dinic** (`test_graphcut_dinic`). Matches Edmonds–Karp (same seam and same flow) and brute force on 200 maps and on 15×15 and 30×30 maps. A 3000-row map runs without a recursion error. Full resizes with the two algorithms are identical. Unknown algorithm names are rejected.
- **Resize** (`test_graphcut_resize`). Same order and energy recomputation as greedy, an equal target copies without computing energy, gray images work, the caller's image is unchanged, and bad targets are rejected.
- **Benchmark** (`test_compare_graphcut`). The shared `carve` loop reproduces `resize_greedy` and `resize_graphcut` exactly. Graph cut never costs more than greedy. The DP slot requires both finders. Output files are written.

Not covered:
- DP. None exists in the repository yet.
- Full-size photographs.
- Timing stability across machines.

## 10. Sources and AI use

- Assignment 2 brief, Item 4: the graph construction, Ford–Fulkerson / Edmonds–Karp, and BFS on the residual graph for the cut.
- Max-flow min-cut theorem, Edmonds–Karp and Dinic: standard algorithms course material.
- The dummy-column correction and the use of Dinic were added in this implementation and are explained in sections 3 and 4.

AI use: Claude was used to break Item 4 into steps, to draft the graph, max-flow, resize and benchmark code and their tests, and to write this note and the integration-doc updates. I reviewed each draft, placed the code into the repository, and ran the tests. The measurements in section 6 come from actual runs, as stated there; none were invented.

## Presentation outline (about 2 minutes)

- **20 s.** Each pixel becomes a node. Cutting the edge to a pixel's right neighbour means removing that pixel. Infinite backward and diagonal edges force every finite cut to be one connected seam.
- **30 s.** Min cut = max flow, so running max-flow gives the cheapest seam. Mention the dummy column, using the 3×3 map: without it the answer is 10, with it the correct answer is 2.
- **30 s.** Edmonds–Karp vs Dinic: same seam, Dinic up to about 13× faster. Both are still seconds per seam.
- **40 s.** Broadway Tower: show the greedy and graph-cut seam overlays and the two resized images. Graph-cut seam cost is 1620 vs greedy's 5324, but 38.9 s vs 0.05 s for 30 seams. Graph cut = DP's answer at a much higher cost; its value is generality (forward energy, video).
