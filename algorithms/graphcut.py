"""Minimum-energy seam selection by graph cut (minimum s-t cut / maximum flow).

For a vertical seam, pixel ``(i, j)`` becomes node ``i * (W + 1) + j``, where
column ``W`` is a dummy column. The edges are:

    s -> (i, 0)             infinite   column 0 is always on the source side
    (i, W) -> t             infinite   the dummy column is always on the sink side
    (i, j) -> (i, j + 1)    e[i][j]    cutting it removes pixel (i, j)
    (i, j + 1) -> (i, j)    infinite   each row is cut exactly once
    (i, j) -> (i +- 1, j - 1)  infinite   neighbouring rows' cuts differ by <= 1

so every finite cut is one connected seam, and its capacity is the seam's
energy. By the max-flow min-cut theorem a maximum flow gives the minimum cut.
One more BFS in the residual graph then finds the source side S, and the
rightmost pixel of S in each row is the seam pixel. The dummy column lets the
last real column be removed; the brief's graph, which ties that column to the
sink, cannot remove it.

Two max-flow algorithms share the graph: Edmonds-Karp, O(V E^2), and Dinic,
O(V^2 E) in the worst case (the default). Both return the same seam. With
V = H(W + 1) + 2 and E about 4HW, both are cubic in the number of pixels.

The input contract is the greedy finder's: a list-of-lists energy map,
validated by the same function, never modified. When several seams tie for
the minimum, graph cut may pick a different one from DP; the cost is the same.
"""

from __future__ import annotations
from collections import deque
from collections.abc import Sequence
from algorithms.greedy import _copy_finite_grid

EPSILON = 1e-9 # Residual capacity at/below this counts as used up

class FlowGraph:
    """Residual graph stored as flat edge lists.

    ``add_edge`` stores the forward edge at an even index and its reverse
    (capacity 0) at the next odd index, so the reverse of edge ``e`` is ``e ^ 1``
    and the tail of ``e`` is ``edge_to[e ^ 1]``.
    """
    def __init__(self, node_count: int) -> None:
        """Create ``node_count`` nodes with no edges."""
        self.node_count = node_count
        self.adjacent: list[list[int]] = [[] for _ in range(node_count)] #edge ids leaving each node
        self.edge_to:  list[int] = [] #head node of each edge
        self.capacity: list[float] = [] # remaining (residual) capacity

    def add_edge(self, u: int, v: int, cap: float) -> None:
        """Add the edge ``u -> v`` with capacity ``cap``, and its reverse ``v -> u`` with capacity 0."""
        self.adjacent[u].append(len(self.edge_to))
        self.edge_to.append(v)
        self.capacity.append(cap)

        self.adjacent[v].append(len(self.edge_to))
        self.edge_to.append(u)
        self.capacity.append(0.0)

class SeamGraph:
    """The vertical-seam graph of one energy map, plus the ids needed to read the seam back.

    The grid is ``height`` rows by ``width + 1`` columns; the last column is the
    dummy column. "Infinite" capacity is the total energy plus 1, larger than any
    finite cut, so the arithmetic stays finite.
    """
    def __init__(self, energy: Sequence) -> None:
        """Validate ``energy`` (greedy's rules) and build the graph. ``energy`` is not modified."""
        rows = _copy_finite_grid(energy)
        self.height = len(rows)
        self.width = len(rows[0])
        self.grid_width = self.width + 1
        self.source = self.height * self.grid_width
        self.sink = self.source + 1
        # Larger than any finite cut, so it is never chosen
        self.infinity = sum(sum(row) for row in rows) + 1.0
        self.graph  = FlowGraph(self.source + 2)
        self._build(rows)

    def node(self, row: int, column: int) -> int:
        """Return the node id of grid cell ``(row, column)``."""
        return row * self.grid_width + column

    def _build(self, rows: list[list[float]]) -> None:
        """Add the source, sink, capacity, backward and diagonal edges listed in the module docstring."""
        g, inf = self.graph, self.infinity
        h, w = self.height, self.width

        for i in range(h):
            # Column 0 always on source side
            g.add_edge(self.source, self.node(i, 0), inf)
            # dummy column always on sink side
            g.add_edge(self.node(i, w), self.sink, inf)

            for j in range(w):
                # Cutting (i,j)->(i,j+1)
                g.add_edge(self.node(i, j), self.node(i, j + 1), rows[i][j])
                # Backward infinite edge so each row is cut once
                g.add_edge(self.node(i, j + 1), self.node(i, j), inf)

            for j in range(1, w):
                # Diagonal infinite edges, neighbouring rows cut differ by <= 1
                if i > 0:
                    g.add_edge(self.node(i, j), self.node(i - 1, j - 1), inf)
                if i < h - 1:
                    g.add_edge(self.node(i, j), self.node(i + 1, j - 1), inf)

def _bfs(graph: FlowGraph, source: int, sink: int | None = None) -> list[int]:
    """Breadth-first search over edges with residual capacity above ``EPSILON``.

    Returns ``parent_edge``: for each node, the id of the edge used to reach it,
    -1 if not reached, and -2 for the source. With ``sink`` given it stops as soon
    as the sink is reached (an augmenting path); without it, it marks everything
    reachable (the source side S of the cut).
    """
    parent_edge = [-1] * graph.node_count
    parent_edge[source] = -2
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for e in graph.adjacent[u]:
            v = graph.edge_to[e]
            if parent_edge[v] == -1 and graph.capacity[e] > EPSILON:
                parent_edge[v] = e
                if v == sink:
                    return parent_edge
                queue.append(v)
    return parent_edge

def edmonds_karp(graph: FlowGraph, source: int, sink: int) -> float:
    """Maximum flow by shortest augmenting paths, one BFS per path.

    Changes ``graph.capacity`` into the final residual graph and returns the total
    flow, which equals the minimum cut capacity. O(V E^2).
    """
    total_flow = 0.0
    while True:
        parent_edge = _bfs(graph, source, sink)
        if parent_edge[sink] == -1:
            return total_flow # no augmenting path left

        # Walk back from sink to source to find the bottleneck
        bottleneck = float("inf")
        v = sink
        while v != source:
            e = parent_edge[v]
            bottleneck = min(bottleneck, graph.capacity[e])
            v = graph.edge_to[e ^ 1] # tail of e = head of its reverse

        # Push the bottleneck along the same path
        v = sink
        while v != source:
            e = parent_edge[v]
            graph.capacity[e] -= bottleneck
            graph.capacity[e ^ 1] += bottleneck
            v = graph.edge_to[e ^ 1]

        total_flow += bottleneck

def dinic(graph: FlowGraph, source: int, sink: int) -> float:
    """Maximum flow by Dinic's algorithm.

    Each phase runs one BFS that gives every node a level (its distance from the
    source), then pushes paths that only step from level k to level k + 1 until
    that layered graph is saturated. ``next_edge`` (the current-arc pointer) and
    setting a dead end's level to -1 mean no edge or node is retried within a
    phase. The path search is iterative, so tall images cannot hit Python's
    recursion limit. Changes ``graph.capacity`` into the final residual graph and
    returns the total flow. O(V^2 E) worst case, far fewer phases in practice.
    """
    adjacent, edge_to, capacity = graph.adjacent, graph.edge_to, graph.capacity
    total_flow = 0.0

    while True:
        # Phase 1: BFS levels (distance from source using edges with capacity)
        level = [-1] * graph.node_count
        level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for e in adjacent[u]:
                v = edge_to[e]
                if level[v] == -1 and capacity[e] > EPSILON:
                    level[v] = level[u] + 1
                    queue.append(v)
        if level[sink] == -1:
            return total_flow # sink unreachable: flow is maximum

        # Phase 2: Blocking flow
        next_edge = [0] * graph.node_count
        while True:
            path: list[int] = []
            u = source
            while u != sink:
                edges = adjacent[u]
                while next_edge[u] < len(edges):
                    e = edges[next_edge[u]]
                    v = edge_to[e]
                    if capacity[e] > EPSILON and level[v] == level[u] + 1:
                        break
                    next_edge[u] += 1
                else:
                    # Dead end
                    if u == source:
                        break
                    level[u] = -1
                    e = path.pop()
                    u = edge_to[e ^ 1]
                    next_edge[u] += 1
                    continue
                path.append(e)
                u = v

            if u != sink:
                break

            bottleneck = min(capacity[e] for e in path)
            for e in path:
                capacity[e] -= bottleneck
                capacity[e ^ 1] += bottleneck
            total_flow += bottleneck

# Names accepted by the ``max_flow`` argument.
MAX_FLOW_ALGORITHMS = {"edmonds_karp": edmonds_karp, "dinic": dinic}

def solve_vertical(energy: Sequence, max_flow: str = "dinic") -> tuple[list[int], float]:
    """Return ``(seam, flow)`` for a minimum-energy vertical seam.

    ``seam`` has one column index per row; ``flow`` is the maximum flow, which
    equals the seam's energy. ``max_flow`` is ``"dinic"`` or ``"edmonds_karp"``,
    and both give the same seam. Raises ``ValueError`` for another name or an
    invalid energy map. ``energy`` is not modified.
    """
    if max_flow not in MAX_FLOW_ALGORITHMS:
        raise ValueError(f"max_flow must be one of {sorted(MAX_FLOW_ALGORITHMS)}")
    sg = SeamGraph(energy)
    flow = MAX_FLOW_ALGORITHMS[max_flow](sg.graph, sg.source, sg.sink)

    # Source side S (everything still reachable in the residual graph)
    reachable = _bfs(sg.graph, sg.source)
    seam = []
    for i in range(sg.height):
        # S is a prefix of each row, so scan from the right for its last pixel
        column = sg.width - 1
        while reachable[sg.node(i, column)] == -1:
            column -= 1
        seam.append(column)
    return seam, flow

def find_graphcut_vertical_seam(energy: Sequence, max_flow: str = "dinic") -> list[int]:
    """Return one column index per row for a minimum-energy top-to-bottom seam."""
    return solve_vertical(energy, max_flow)[0]


def find_graphcut_horizontal_seam(energy: Sequence, max_flow: str = "dinic") -> list[int]:
    """Return one row index per column by running the vertical finder on the transpose."""
    rows = _copy_finite_grid(energy)
    transposed = [list(column) for column in zip(*rows)]
    return find_graphcut_vertical_seam(transposed, max_flow)
