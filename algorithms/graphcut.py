from __future__ import annotations
from collections import deque
from collections.abc import Sequence
from algorithms.greedy import _copy_finite_grid

EPSILON = 1e-9 # Residual capacity at/below this counts as used up

class FlowGraph:
    def __init__(self, node_count: int) -> None:
        self.node_count = node_count
        self.adjacent: list[list[int]] = [[] for _ in range(node_count)] #edge ids leaving each node
        self.edge_to:  list[int] = [] #head node of each edge
        self.capacity: list[float] = [] # remaining (residual) capacity

    def add_edge(self, u: int, v: int, cap: float) -> None:
        self.adjacent[u].append(len(self.edge_to))
        self.edge_to.append(v)
        self.capacity.append(cap)

        self.adjacent[v].append(len(self.edge_to))
        self.edge_to.append(u)
        self.capacity.append(0.0)

class SeamGraph:
    def __init__(self, energy: Sequence) -> None:
        rows = _copy_finite_grid(energy)
        self.height = len(rows)
        self.width = len(rows[0])
        self.grid_width = self.width + 1
        self.source = self.height * self.grid_width
        self.sink = self.source + 1
        # Infiinite edge
        self.infinity = sum(sum(row) for row in rows) + 1.0
        self.graph  = FlowGraph(self.source + 2)
        self._build(rows)

    def node(self, row: int, column: int) -> int:
        return row * self.grid_width + column

    def _build(self, rows: list[list[float]]) -> None:
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

MAX_FLOW_ALGORITHMS = {"edmonds_karp": edmonds_karp, "dinic": dinic}

def solve_vertical(energy: Sequence, max_flow: str = "dinic") -> tuple[list[int], float]:
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
    return solve_vertical(energy, max_flow)[0]


def find_graphcut_horizontal_seam(energy: Sequence, max_flow: str = "dinic") -> list[int]:
    rows = _copy_finite_grid(energy)
    transposed = [list(column) for column in zip(*rows)]
    return find_graphcut_vertical_seam(transposed, max_flow)