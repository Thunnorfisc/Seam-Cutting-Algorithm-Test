import itertools
import random
import unittest

from algorithms.graphcut import SeamGraph

def cut_capacity(sg: SeamGraph, seam) -> float:
    source_side = {sg.source}
    for i, column in enumerate(seam):
        source_side.update(sg.node(i, j) for j in range(column + 1))
    g = sg.graph
    total = 0.0
    for u in source_side:
        for e in g.adjacent[u]:
            if e % 2 == 0 and g.edge_to[e] not in source_side: # Forward edges
                total += g.capacity[e]
    return total

def is_connected(seam) -> bool:
    return all(abs(a - b) <= 1 for a, b in zip(seam, seam[1:]))

class TestBuild(unittest.TestCase):
    def check_all_seams(self, energy):
        sg = SeamGraph(energy)
        w = len(energy[0])
        best = None
        for seam in itertools.product(range(w), repeat=len(energy)):
            cut = cut_capacity(sg, seam)
            if is_connected(seam):
                cost = sum(energy[i][j] for i, j in enumerate(seam))
                self.assertAlmostEqual(cut, cost)
                best = cost if best is None else min(best, cost)
            else:
                self.assertGreaterEqual(cut, sg.infinity)
        return best

    def test_repo_example(self):
        energy = [[1, 2, 500], [4, 50, 0], [5, 50, 0]]
        self.assertEqual(self.check_all_seams(energy), 2) # seam [1, 2, 3]

    def test_random_small_maps(self):
        rng = random.Random(0)
        for _ in range(30):
            h, w = rng.randint(1, 4), rng.randint(1, 4)
            self.check_all_seams([[rng.uniform(0, 10) for _ in range(w)] for _ in range(h)])

    def test_node_countd(self):
        sg = SeamGraph([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(sg.graph.node_count, 2 * 4 + 2)

if (__name__ == "__main__"):
    unittest.main()