"""Resize an image by repeatedly removing greedy seams.

There is no dynamic-programming removal order in this repository. When both
axes shrink, vertical seams are removed until the width matches, then
horizontal seams are removed until the height matches.
"""

from __future__ import annotations

import numpy as np

from algorithms.greedy import find_greedy_horizontal_seam, find_greedy_vertical_seam
from temporary.energy import compute_energy
from temporary.seam_remove import remove_horizontal_seam, remove_vertical_seam

REMOVAL_ORDER = "vertical then horizontal"


def resize_greedy(image, target_width: int, target_height: int, energy_fn=None):
    """Return a new image reduced to the target size. ``image`` is not modified.

    Targets are integers from 1 through the current size. An equal target
    copies the pixels and does not compute energy. Energy is computed on the
    current image before every seam.
    """
    _require_image(image)
    height, width = image.shape[:2]
    target_width = _require_target("target_width", target_width, width)
    target_height = _require_target("target_height", target_height, height)
    if energy_fn is None:
        energy_fn = compute_energy

    work = np.array(image, copy=True)
    while work.shape[1] > target_width:
        if work.shape[1] <= 1:
            raise ValueError("cannot remove a vertical seam from a width-1 image")
        seam = find_greedy_vertical_seam(_energy_rows(work, energy_fn))
        updated = remove_vertical_seam(work, seam)
        if updated.shape[0] != work.shape[0] or updated.shape[1] != work.shape[1] - 1:
            raise ValueError("vertical removal did not shrink width by one")
        work = updated

    while work.shape[0] > target_height:
        if work.shape[0] <= 1:
            raise ValueError("cannot remove a horizontal seam from a height-1 image")
        seam = find_greedy_horizontal_seam(_energy_rows(work, energy_fn))
        updated = remove_horizontal_seam(work, seam)
        if updated.shape[1] != work.shape[1] or updated.shape[0] != work.shape[0] - 1:
            raise ValueError("horizontal removal did not shrink height by one")
        work = updated
    return work


def _energy_rows(image: np.ndarray, energy_fn) -> list:
    energy = np.asarray(energy_fn(image), dtype=np.float64)
    if energy.shape != image.shape[:2]:
        raise ValueError("energy shape must match the current image")
    if not np.isfinite(energy).all():
        raise ValueError("energy map must contain finite real numbers")
    return energy.tolist()


def _require_image(image) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return
    raise ValueError("image must be gray, BGR, or BGRA")


def _require_target(name: str, value: object, limit: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, np.integer):
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1 or value > limit:
        raise ValueError(f"{name} must be between 1 and {limit}")
    return value
