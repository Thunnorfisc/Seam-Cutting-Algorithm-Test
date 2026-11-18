from __future__ import annotations
from collections.abc import Sequence
from algorithms.greedy import _copy_finite_grid

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
            #dummy column always on sink side
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