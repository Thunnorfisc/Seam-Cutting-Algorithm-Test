# Greedy versus dynamic programming

Status: blocked.

No dynamic-programming seam finder is present. This comparison does not implement one. Seam paths, measured seam costs, timings, and images are not recorded.

The three comparisons required by Assignment 2 — seams, visual quality, and running time — stay unmeasured until a teammate supplies vertical and horizontal finders. Those finders must use this energy formula and the removal order `vertical then horizontal`.

Energy formula: Grayscale 0.114 B + 0.587 G + 0.299 R; 3x3 Sobel dx and dy; BORDER_REFLECT_101; hypot(gx, gy).

There are no sample photographs in this repository.

## Hand-checked illustration

This is not a DP execution and not a visual result. On the map `[[1, 2, 100], [4, 50, 0], [5, 50, 0]]`, the greedy finder returns `[0, 0, 0]` with cost 10. That greedy result is covered by the unit tests. A hand-checked optimal seam on the same map is `[1, 2, 2]` with cost 2.
