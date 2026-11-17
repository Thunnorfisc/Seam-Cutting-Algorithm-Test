"""TEMPORARY stand-in for the team's shared energy function.

Replace this module when the shared implementation is available. Greedy and any
future dynamic-programming comparison must call this same function.

Energy is the gradient magnitude of a grayscale image: 3x3 Sobel in x and y,
OpenCV's default border (BORDER_REFLECT_101), then hypot(gx, gy). A BGR or
BGRA image is converted with the standard weights
0.114 B + 0.587 G + 0.299 R. A 2D gray image skips that conversion.

OpenCV is used when its DLL loads. On this machine Windows application
control blocks that DLL (WinError 4551), so the same steps run in NumPy.
"""

from __future__ import annotations

import sys

import numpy as np

ENERGY_FORMULA = (
    "Grayscale 0.114 B + 0.587 G + 0.299 R; 3x3 Sobel dx and dy; "
    "BORDER_REFLECT_101; hypot(gx, gy)."
)

_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def compute_energy(image: np.ndarray) -> np.ndarray:
    """Return a float64 energy map with one value per pixel. ``image`` is not modified."""
    source = _copied_image(image)
    opencv = _load_opencv()
    if opencv is not None:
        return _energy_opencv(opencv, source)
    return _energy_numpy(source)


def _copied_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim == 2:
        pass
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        pass
    else:
        raise ValueError("image must be gray, BGR, or BGRA")
    return np.array(image, copy=True, order="C")


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


def _energy_opencv(cv2, image: np.ndarray) -> np.ndarray:
    gray = _gray_opencv(cv2, image)
    gradient_x = cv2.Sobel(
        gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT
    )
    gradient_y = cv2.Sobel(
        gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT
    )
    return np.hypot(gradient_x, gradient_y)


def _gray_opencv(cv2, image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _energy_numpy(image: np.ndarray) -> np.ndarray:
    gray = _gray_numpy(image)
    padded = np.pad(gray, 1, mode="reflect")
    gradient_x = np.zeros(gray.shape, dtype=np.float64)
    gradient_y = np.zeros(gray.shape, dtype=np.float64)
    for row_offset in range(3):
        for column_offset in range(3):
            window = padded[
                row_offset : row_offset + gray.shape[0],
                column_offset : column_offset + gray.shape[1],
            ]
            gradient_x += _SOBEL_X[row_offset, column_offset] * window
            gradient_y += _SOBEL_Y[row_offset, column_offset] * window
    return np.hypot(gradient_x, gradient_y)


def _gray_numpy(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64, copy=False)
    pixels = image.astype(np.float64, copy=False)
    blue = pixels[:, :, 0]
    green = pixels[:, :, 1]
    red = pixels[:, :, 2]
    return 0.114 * blue + 0.587 * green + 0.299 * red
