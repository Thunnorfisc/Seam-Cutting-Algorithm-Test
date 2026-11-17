"""TEMPORARY stand-in for the team's shared seam removal.

Replace this module when the shared implementation is available. Each function
returns a new array and keeps the dtype and channel count of the input.
"""

from __future__ import annotations

import numpy as np


def remove_vertical_seam(image: np.ndarray, seam) -> np.ndarray:
    """Delete one pixel from every row and shift the rest left."""
    _require_image(image)
    height, width = image.shape[:2]
    if width <= 1:
        raise ValueError("cannot remove a vertical seam from a width-1 image")
    columns = _validated_seam(seam, expected_length=height, limit=width)
    removed = np.empty((height, width - 1) + image.shape[2:], dtype=image.dtype)
    for row, column in enumerate(columns):
        removed[row, :column] = image[row, :column]
        removed[row, column:] = image[row, column + 1 :]
    return removed


def remove_horizontal_seam(image: np.ndarray, seam) -> np.ndarray:
    """Delete one pixel from every column and shift the rest up."""
    _require_image(image)
    height, width = image.shape[:2]
    if height <= 1:
        raise ValueError("cannot remove a horizontal seam from a height-1 image")
    rows = _validated_seam(seam, expected_length=width, limit=height)
    removed = np.empty((height - 1, width) + image.shape[2:], dtype=image.dtype)
    for column, row in enumerate(rows):
        removed[:row, column] = image[:row, column]
        removed[row:, column] = image[row + 1 :, column]
    return removed


def _require_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty gray, BGR, or BGRA array")
    if min(image.shape[:2]) < 1:
        raise ValueError("image must be a non-empty gray, BGR, or BGRA array")


def _validated_seam(seam, expected_length: int, limit: int) -> list[int]:
    try:
        length = len(seam)
    except TypeError as error:
        raise ValueError("seam must be a sequence of integer indexes") from error
    if length != expected_length:
        raise ValueError("seam length does not match the image")
    indexes: list[int] = []
    previous: int | None = None
    for raw in seam:
        index = _as_index(raw, limit)
        if previous is not None and abs(index - previous) > 1:
            raise ValueError("seam breaks connectivity")
        indexes.append(index)
        previous = index
    return indexes


def _as_index(value: object, limit: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("seam index must be an integer")
    if isinstance(value, np.integer):
        value = int(value)
    if not isinstance(value, int):
        raise ValueError("seam index must be an integer")
    if value < 0 or value >= limit:
        raise ValueError("seam index out of range")
    return value
