"""Resize an image by repeatedly removing minimum-energy seams found by graph cut.

Same contract and schedule as ``resize_greedy`` and ``resize_dp``: shrink only,
vertical seams until the width matches, then horizontal seams until the height
matches, energy recomputed on the current image before every seam.
"""

from __future__ import annotations

import numpy as np

from algorithms.graphcut import find_graphcut_horizontal_seam, find_graphcut_vertical_seam
from algorithms.greedy_resize import (
    REMOVAL_ORDER,
    _energy_rows,
    _require_image,
    _require_target,
)
from temporary.energy import compute_energy
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam

__all__ = ["REMOVAL_ORDER", "resize_graphcut"]

def resize_graphcut(image, target_width: int, target_height: int, energy_fn=None, max_flow: str = "dinic"):
    """Return a new image reduced to the target size. ``image`` is not modified.

    ``max_flow`` is ``"dinic"`` (default) or ``"edmonds_karp"``; both give the same
    image. Slow: one pure-Python Dinic seam takes about a second at 120x81, so
    keep images small (or use ``algorithms/graphcut_numba.py``).
    """
    _require_image(image)
    height, width = image.shape[:2]
    target_width = _require_target("target_width", target_width, width)
    target_height = _require_target("target_height", target_height, height)
    if energy_fn is None:
        energy_fn = compute_energy

    work = np.array(image, copy=True)
    while work.shape[1] > target_width:
        seam = find_graphcut_vertical_seam(_energy_rows(work, energy_fn), max_flow)
        work = remove_vertical_seam(work, seam)

    while work.shape[0] > target_height:
        seam = find_graphcut_horizontal_seam(_energy_rows(work, energy_fn), max_flow)
        work = remove_horizontal_seam(work, seam)
    return work
