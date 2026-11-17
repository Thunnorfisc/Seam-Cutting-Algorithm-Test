"""TEMPORARY stand-in for the team's image loading and saving.

OpenCV is used when its DLL loads. Windows application control blocks that
DLL on this machine, so 24-bit BMP files are read and written directly.
``load_image`` returns a 3-channel BGR array. This module does not scale
images and does not carve seams.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np


def load_image(path) -> np.ndarray:
    """Read an image file and return uint8 BGR pixels."""
    opencv = _load_opencv()
    if opencv is not None:
        loaded = opencv.imread(str(path), opencv.IMREAD_COLOR)
        if loaded is None:
            raise ValueError(f"could not read image: {path}")
        return loaded
    return _read_bmp(path)


def save_image(path, image: np.ndarray) -> None:
    """Write ``image`` to ``path`` without changing the array."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    opencv = _load_opencv()
    if opencv is not None:
        if not opencv.imwrite(str(path), image):
            raise ValueError(f"could not write image: {path}")
        return
    _write_bmp(path, image)


_OPENCV = None
_OPENCV_TRIED = False


def _load_opencv():
    global _OPENCV, _OPENCV_TRIED
    if _OPENCV_TRIED:
        return _OPENCV
    _OPENCV_TRIED = True
    try:
        import cv2
    except (ImportError, OSError):
        sys.modules.pop("cv2", None)
        _OPENCV = None
        return None
    _OPENCV = cv2
    return cv2


def _write_bmp(path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("without OpenCV, only uint8 BGR images can be saved as BMP")
    height, width = image.shape[:2]
    row_stride = (width * 3 + 3) & ~3
    pixel_size = row_stride * height
    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<I", header, 2, 54 + pixel_size)
    struct.pack_into("<I", header, 10, 54)
    struct.pack_into("<I", header, 14, 40)
    struct.pack_into("<i", header, 18, width)
    struct.pack_into("<i", header, 22, height)
    struct.pack_into("<HH", header, 26, 1, 24)
    struct.pack_into("<I", header, 34, pixel_size)
    pixels = bytearray(pixel_size)
    ordered = np.ascontiguousarray(image)
    for row in range(height):
        start = row * row_stride
        source = ordered[height - 1 - row].tobytes()
        pixels[start : start + width * 3] = source
    Path(path).write_bytes(header + pixels)


def _read_bmp(path) -> np.ndarray:
    data = Path(path).read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError(f"could not read image: {path}")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    header_size = struct.unpack_from("<I", data, 14)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    planes, bit_count = struct.unpack_from("<HH", data, 26)
    if header_size < 40 or planes != 1 or bit_count != 24 or width <= 0 or height == 0:
        raise ValueError(f"could not read image: {path}")
    top_down = height < 0
    height = abs(height)
    row_stride = (width * 3 + 3) & ~3
    image = np.empty((height, width, 3), dtype=np.uint8)
    for file_row in range(height):
        start = pixel_offset + file_row * row_stride
        row_bytes = data[start : start + width * 3]
        if len(row_bytes) != width * 3:
            raise ValueError(f"could not read image: {path}")
        image_row = file_row if top_down else height - 1 - file_row
        image[image_row] = np.frombuffer(row_bytes, dtype=np.uint8).reshape(width, 3)
    return image
