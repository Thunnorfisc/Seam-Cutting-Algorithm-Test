"""The graph-cut seam finder of ``algorithms/graphcut.py``, compiled with Numba.

Same graph, same Dinic and same seam recovery, so the seams are identical to
``find_graphcut_vertical_seam(energy, "dinic")``; only the speed differs (about
16-26x faster than pure Python on the same machine). The complexity is
unchanged: compiling removes a constant factor.

The graph is stored as flat NumPy arrays so Numba can compile the loops.
``edge_to`` and ``capacity`` keep the pairing where the reverse of edge ``e`` is
``e ^ 1``, and the edges leaving node ``u`` are ``adjacent[start[u]:start[u + 1]]``
(compressed sparse row form).

Numba compiles a function the first time it is called. ``cache=True`` stores
the machine code in ``__pycache__`` so later processes load it instead. Call
``warm_up()`` before timing anything. This module needs Numba;
``frontend/engine.py`` and the tests leave it out when Numba cannot be imported.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from algorithms.greedy import _copy_finite_grid

EPSILON = 1e-9  # residual capacity at or below this counts as used up (as in graphcut.py)

def build_seam_graph(energy: np.ndarray):
    """Build the vertical-seam graph of ``graphcut.SeamGraph`` as flat arrays.

    ``energy`` is a 2D float64 array. Returns ``(start, adjacent, edge_to,
    capacity, source, sink)``. Edge ``2k`` is a forward edge and ``2k + 1`` its
    reverse. Each kind of edge is added for every pixel at once with array
    operations instead of a Python loop.
    """
    h, w = energy.shape
    gw = w + 1
    source, sink = h * gw, h * gw + 1
    inf = float(energy.sum()) + 1.0

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    node = rows * gw + cols

    tails, heads, caps = [], [], []

    def add(u, v, c):
        """Append the edges ``u -> v`` with capacity ``c``; the arguments broadcast together."""
        u, v = np.broadcast_arrays(u, v)
        tails.append(u.ravel()); heads.append(v.ravel())
        caps.append(np.broadcast_to(c, u.shape).astype(np.float64).ravel())

    add(np.full(h, source), np.arange(h) * gw, inf)    # s -> column 0
    add(np.arange(h) * gw + w, np.full(h, sink), inf)  # dummy column -> t
    add(node, node + 1, energy)                        # remove (i, j)
    add(node + 1, node, inf)                           # backward, infinite
    if w > 1:
        inner = node[:, 1:]                            # columns 1..W-1
        add(inner[1:], inner[1:] - gw - 1, inf)        # (i,j) -> (i-1,j-1)
        add(inner[:-1], inner[:-1] + gw - 1, inf)      # (i,j) -> (i+1,j-1)

    tail = np.concatenate(tails); head = np.concatenate(heads); cap = np.concatenate(caps)
    m = tail.size
    edge_to = np.empty(2 * m, np.int64); edge_tail = np.empty(2 * m, np.int64)
    capacity = np.zeros(2 * m, np.float64)
    edge_to[0::2], edge_to[1::2] = head, tail
    edge_tail[0::2], edge_tail[1::2] = tail, head
    capacity[0::2] = cap

    node_count = h * gw + 2
    adjacent = np.argsort(edge_tail, kind="stable").astype(np.int64)
    start = np.zeros(node_count + 1, np.int64)
    np.cumsum(np.bincount(edge_tail, minlength=node_count), out=start[1:])
    return start, adjacent, edge_to, capacity, source, sink

@njit(cache=True)
def dinic_arrays(start, adjacent, edge_to, capacity, source, sink, eps):
    """Dinic's max flow, step for step as ``graphcut.dinic``, on the flat arrays.

    Preallocated arrays hold the levels, current-arc pointers, BFS queue and
    path, so Numba can compile it. Changes ``capacity`` in place into the
    residual graph and returns the total flow.
    """
    n = start.size - 1
    level = np.empty(n, np.int64)
    next_edge = np.empty(n, np.int64)
    queue = np.empty(n, np.int64)
    path = np.empty(n, np.int64)
    total = 0.0
    while True:
        level[:] = -1
        level[source] = 0
        head, tail = 0, 1
        queue[0] = source
        while head < tail:
            u = queue[head]; head += 1
            for k in range(start[u], start[u + 1]):
                e = adjacent[k]; v = edge_to[e]
                if level[v] == -1 and capacity[e] > eps:
                    level[v] = level[u] + 1
                    queue[tail] = v; tail += 1
        if level[sink] == -1:
            return total

        next_edge[:] = start[:-1]                 # current-arc pointers
        while True:
            depth = 0
            u = source
            while u != sink:
                found = False
                while next_edge[u] < start[u + 1]:
                    e = adjacent[next_edge[u]]; v = edge_to[e]
                    if capacity[e] > eps and level[v] == level[u] + 1:
                        found = True
                        break
                    next_edge[u] += 1
                if found:
                    path[depth] = e; depth += 1
                    u = v
                else:
                    if u == source:
                        break
                    level[u] = -1
                    depth -= 1
                    u = edge_to[path[depth] ^ 1]
                    next_edge[u] += 1
            if u != sink:
                break
            bottleneck = capacity[path[0]]
            for d in range(1, depth):
                if capacity[path[d]] < bottleneck:
                    bottleneck = capacity[path[d]]
            for d in range(depth):
                capacity[path[d]] -= bottleneck
                capacity[path[d] ^ 1] += bottleneck
            total += bottleneck

@njit(cache=True)
def reachable_from(start, adjacent, edge_to, capacity, source, eps):
    """BFS over edges with capacity left; True for the source side S."""
    n = start.size - 1
    seen = np.zeros(n, np.bool_)
    queue = np.empty(n, np.int64)
    seen[source] = True
    queue[0] = source
    head, tail = 0, 1
    while head < tail:
        u = queue[head]; head += 1
        for k in range(start[u], start[u + 1]):
            e = adjacent[k]; v = edge_to[e]
            if not seen[v] and capacity[e] > eps:
                seen[v] = True
                queue[tail] = v; tail += 1
    return seen


# Error message for an invalid energy map, shared by every check below.
_BAD_MAP = "energy map must be a non-empty 2D array of finite real numbers"


def _as_energy_array(energy) -> np.ndarray:
    """Return a float64 copy of the map. NumPy arrays skip the list round trip."""
    if isinstance(energy, np.ndarray):
        if (energy.ndim != 2 or energy.size == 0 or energy.dtype == np.bool_
                or not np.issubdtype(energy.dtype, np.number) or np.iscomplexobj(energy)):
            raise ValueError(_BAD_MAP)
        grid = np.array(energy, dtype=np.float64)    # a copy: caller's array unchanged
        if not np.isfinite(grid).all():
            raise ValueError(_BAD_MAP)
        return grid
    # Lists of lists keep greedy's exact validation rules.
    return np.asarray(_copy_finite_grid(energy), dtype=np.float64)


def solve_vertical_numba(energy) -> tuple[list[int], float]:
    """Return (seam, flow value), like graphcut.solve_vertical with Dinic.

    ``energy`` may be a list of lists or a 2D NumPy array.
    """
    grid = _as_energy_array(energy)
    h, w = grid.shape
    start, adjacent, edge_to, capacity, source, sink = build_seam_graph(grid)
    flow = dinic_arrays(start, adjacent, edge_to, capacity, source, sink, EPSILON)
    seen = reachable_from(start, adjacent, edge_to, capacity, source, EPSILON)
    # S is a prefix of each row: its length minus one is the seam column.
    in_s = seen[: h * (w + 1)].reshape(h, w + 1)[:, :w]
    seam = (in_s.sum(axis=1) - 1).tolist()
    return seam, flow


def find_graphcut_vertical_seam_numba(energy) -> list[int]:
    """Return one column index per row; the same seam as the pure-Python Dinic."""
    return solve_vertical_numba(energy)[0]


def find_graphcut_horizontal_seam_numba(energy) -> list[int]:
    """Return one row index per column, by solving the transposed map."""
    return find_graphcut_vertical_seam_numba(_as_energy_array(energy).T)


def warm_up() -> None:
    """Compile (or load from the cache) every Numba function now.

    Call this once at start-up so no timed run includes compile time.
    """
    solve_vertical_numba(np.array([[1.0, 2.0], [3.0, 4.0]]))
