"""How the image is turned on screen: a rotation (degrees, clockwise) and an optional left-right mirror.

Display only. The scan, the contours and every measurement stay in the scanner's own pixel grid; the main
view, the ED | ES view, saved figures and exported images all show the same turned picture. Everything here
uses one matrix, so the four never disagree about which way is clockwise.

Screen coordinates have x to the right and y downward (rows), as in the viewer.
"""

from __future__ import annotations

import numpy as np


def normalise(angle: float) -> float:
    """An angle in (-180, 180]."""
    a = float(angle) % 360.0
    return a - 360.0 if a > 180.0 else a


def is_identity(angle: float, flip: bool) -> bool:
    return not flip and abs(normalise(angle)) < 1e-9


def matrix_xy(angle: float, flip: bool) -> np.ndarray:
    """2x2 matrix acting on (x, y) offsets from the turning centre: mirror left-right first (if asked),
    then rotate clockwise on screen by `angle` degrees."""
    a = np.deg2rad(normalise(angle))
    c, s = np.cos(a), np.sin(a)
    f = -1.0 if flip else 1.0
    return np.array([[c * f, -s], [s * f, c]])


def matrix_yx(angle: float, flip: bool) -> np.ndarray:
    """The same matrix acting on (y, x) offsets, napari's axis order."""
    m = matrix_xy(angle, flip)
    return np.array([[m[1, 1], m[1, 0]], [m[0, 1], m[0, 0]]])


def napari_affine(angle: float, flip: bool, centre_yx, ndim: int = 4) -> np.ndarray:
    """(ndim+1) x (ndim+1) affine for a napari layer, turning the last two (y, x) world axes about
    `centre_yx` (world units, i.e. after the layer's scale)."""
    a = np.eye(ndim + 1)
    m = matrix_yx(angle, flip)
    c = np.asarray(centre_yx, dtype=float)
    a[ndim - 2:ndim, ndim - 2:ndim] = m
    a[ndim - 2:ndim, ndim] = c - m @ c
    return a


def transform_points_xy(pts, angle: float, flip: bool, centre_xy, spacing_xy=(1.0, 1.0)) -> np.ndarray:
    """Turn (n, 2) pixel points (x, y) about `centre_xy` (pixels). Done in millimetres, so non-square
    pixels still turn as a rigid picture."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    s = np.asarray(spacing_xy, dtype=float)
    c = np.asarray(centre_xy, dtype=float)
    return ((p - c) * s) @ matrix_xy(angle, flip).T / s + c


def pixel_matrix_xy(angle: float, flip: bool, spacing_xy=(1.0, 1.0)) -> np.ndarray:
    """matrix_xy expressed in pixel units: S^-1 M S, with S the pixel size. Equal to matrix_xy for square
    pixels."""
    s = np.diag(np.asarray(spacing_xy, dtype=float))
    return np.linalg.inv(s) @ matrix_xy(angle, flip) @ s


def orient_image(img_yx: np.ndarray, angle: float, flip: bool, order: int = 1, cval: float = 0.0,
                 spacing_xy=(1.0, 1.0)) -> np.ndarray:
    """A 2D image (rows = y, columns = x) turned about its centre, same size. order=0 for label images."""
    if is_identity(angle, flip):
        return img_yx
    from scipy import ndimage

    h, w = img_yx.shape
    c_yx = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    m_xy = pixel_matrix_xy(angle, flip, spacing_xy)
    m = np.array([[m_xy[1, 1], m_xy[1, 0]], [m_xy[0, 1], m_xy[0, 0]]])  # (y, x) order
    inv = np.linalg.inv(m)  # affine_transform maps output coordinates back to input coordinates
    offset = c_yx - inv @ c_yx
    return ndimage.affine_transform(img_yx, inv, offset=offset, order=order, cval=cval, mode="constant")


def grow_box(box, shape, angle: float, flip: bool) -> tuple[int, int, int, int]:
    """For a turned picture: the source region (xa, xb, ya, yb) that fills the window `box` once turned,
    clipped to the image. The window itself is unchanged."""
    if abs(normalise(angle)) < 1e-9:
        return tuple(int(v) for v in box)
    xa, xb, ya, yb = box
    cx, cy = (xa + xb) / 2.0, (ya + yb) / 2.0
    half = 0.5 * float(np.hypot(xb - xa, yb - ya)) + 1
    return (int(max(0, np.floor(cx - half))), int(min(shape[0], np.ceil(cx + half))),
            int(max(0, np.floor(cy - half))), int(min(shape[1], np.ceil(cy + half))))
