"""Curve contours: convert between pixel masks and smooth closed curves.

Segment-style editing works on curves with a few draggable control points; the
AI models and the UCL labels are pixel masks. This module converts masks to
curves (for display and editing), curves back to masks (for calculation), and
measures curve areas exactly (shoelace formula) instead of counting pixels.

Coordinates are (x, y) in pixel units, matching the (x, y, slice) arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate
from skimage import draw, measure


@dataclass
class Curve:
    """A closed curve defined by control points; `dense` is the smooth outline drawn through them."""

    control: np.ndarray  # (n, 2) control points (x, y)
    dense: np.ndarray  # (m, 2) points along the smooth curve

    def area_px(self) -> float:
        return polygon_area(self.dense)


def polygon_area(pts: np.ndarray) -> float:
    """Area enclosed by a closed polygon (shoelace formula), in pixel units."""
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _outer_boundary(mask: np.ndarray) -> np.ndarray | None:
    """Longest boundary of a binary mask at the 0.5 level, as (x, y) points."""
    if not mask.any():
        return None
    padded = np.pad(mask.astype(float), 1)
    cs = measure.find_contours(padded, 0.5)
    if not cs:
        return None
    c = max(cs, key=len) - 1.0  # undo padding; (row=x, col=y) because arrays are (x, y)
    return c[:, [0, 1]]


def resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """n points evenly spaced by arc length along a closed polyline."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def spline_through(control: np.ndarray, n_dense: int = 400) -> np.ndarray:
    """Smooth closed curve through the control points (centripetal Catmull-Rom).

    Centripetal parametrisation does not overshoot or loop when control points
    are unevenly spaced (e.g. clustered at a tip), unlike a uniform cubic spline.
    """
    n = len(control)
    if n < 4:
        return control.copy()
    p = np.vstack([control[-1:], control, control[:2]])  # wrap for a closed curve
    out = []
    per_seg = max(2, n_dense // n)
    for i in range(1, n + 1):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        tt = np.linspace(t1, t2, per_seg, endpoint=False)[:, None]
        a1 = (t1 - tt) / (t1 - t0) * p0 + (tt - t0) / (t1 - t0) * p1
        a2 = (t2 - tt) / (t2 - t1) * p1 + (tt - t1) / (t2 - t1) * p2
        a3 = (t3 - tt) / (t3 - t2) * p2 + (tt - t2) / (t3 - t2) * p3
        b1 = (t2 - tt) / (t2 - t0) * a1 + (tt - t0) / (t2 - t0) * a2
        b2 = (t3 - tt) / (t3 - t1) * a2 + (tt - t1) / (t3 - t1) * a3
        out.append((t2 - tt) / (t2 - t1) * b1 + (tt - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def _adaptive_control(b: np.ndarray, n_start: int, tol_px: float, max_points: int) -> np.ndarray:
    """Control points chosen from the outline so the spline stays within tol_px of it.

    Starts from n_start evenly spaced points, then repeatedly adds a point at the
    outline position the spline misses most. Points therefore gather at tips and
    sharp bends (where even spacing cuts corners) and stay sparse on smooth wall.
    """
    from scipy.spatial import cKDTree

    closed = np.vstack([b, b[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])[:-1]
    targets = np.linspace(0, s[-1] + np.hypot(*(b[0] - b[-1])), n_start, endpoint=False)
    idx = sorted(set(int(np.argmin(np.abs(s - x))) for x in targets))
    while True:
        ctrl = b[idx]
        dense = spline_through(ctrl, max(400, 8 * len(b)))
        dist, _ = cKDTree(dense).query(b)
        worst = int(np.argmax(dist))
        if dist[worst] <= tol_px or len(idx) >= max_points or worst in idx:
            return ctrl
        idx = sorted(idx + [worst])


def _smooth_outline(b: np.ndarray, sigma_px: float = 1.5) -> np.ndarray:
    """Remove the pixel staircase from a closed outline (Gaussian along the curve, wrap-around)."""
    from scipy.ndimage import gaussian_filter1d

    seg = np.hypot(*np.diff(np.vstack([b, b[:1]]), axis=0).T)
    step = max(float(np.median(seg)), 1e-6)
    sig = sigma_px / step
    sm = np.column_stack([gaussian_filter1d(b[:, 0], sig, mode="wrap"), gaussian_filter1d(b[:, 1], sig, mode="wrap")])
    # Smoothing a closed curve shrinks it; rescale about the centroid to keep the original area
    # (without this, small end-systolic cavities read up to ~5% small).
    a0, a1 = polygon_area(b), polygon_area(sm)
    if a1 > 0:
        c = sm.mean(axis=0)
        sm = c + (sm - c) * np.sqrt(a0 / a1)
    return sm


def _even_with_tips(b: np.ndarray, n_points: int, tip_tol_px: float, min_gap_px: float, max_extra: int) -> np.ndarray:
    """Evenly spaced control points, plus a few at genuinely sharp tips.

    b is a smoothed outline. A tip point is added where the curve through the even
    points misses the outline by more than tip_tol_px, and only if it is at least
    min_gap_px from every existing point, so points never bunch up.
    """
    from scipy.spatial import cKDTree

    closed = np.vstack([b, b[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    perim = s[-1]
    n = int(max(8, min(n_points, perim / max(min_gap_px, 1e-6))))
    pos = list(np.linspace(0, perim, n, endpoint=False))
    for _ in range(max_extra):
        ctrl = np.column_stack([np.interp(pos, s, closed[:, 0]), np.interp(pos, s, closed[:, 1])])
        dense = spline_through(ctrl, max(400, 4 * len(b)))
        dist, _ = cKDTree(dense).query(b)
        order = np.argsort(dist)[::-1]
        added = False
        for i in order[: max(1, len(order) // 20)]:
            if dist[i] <= tip_tol_px:
                break
            cand = s[i]
            gaps = np.abs(np.array(pos) - cand)
            gaps = np.minimum(gaps, perim - gaps)
            if gaps.min() >= min_gap_px:
                pos = sorted(pos + [cand])
                added = True
                break
        if not added:
            break
    return np.column_stack([np.interp(pos, s, closed[:, 0]), np.interp(pos, s, closed[:, 1])])


def mask_to_curve(
    mask: np.ndarray,
    n_points: int | None = 18,
    tol_px: float | None = 1.5,
    max_points: int = 26,
    min_gap_px: float = 4.0,
    smooth_px: float = 1.5,
) -> Curve | None:
    """Fit a smooth closed curve to a mask's outer boundary.

    Default: the outline is smoothed (removing the pixel staircase), then n_points
    control points are spaced evenly along it, plus extra points only at genuinely
    sharp tips (where the curve would miss the outline by more than tol_px), never
    closer than min_gap_px to another point, up to max_points in total. This keeps
    points easy to grab while still following tips (author feedback, 2026-10-07:
    points clustered at pixel steps were hard to edit).
    tol_px=None: exactly n_points evenly spaced. n_points=None: the raw sub-pixel outline.
    smooth_px: how much to smooth the outline first. The epicardium is anatomically
    smooth (no papillary muscles), so the editor smooths it more (3 px) to iron out
    small model wobbles; the endocardium keeps 1.5 px so real detail survives.
    Holes inside the mask are ignored here; `mask_area_by_curve` accounts for them.
    """
    b = _outer_boundary(mask)
    if b is None or len(b) < 4:
        return None
    if n_points is None:
        return Curve(control=b, dense=b)
    if len(b) <= n_points:
        ctrl = resample_closed(b, min(n_points, len(b)))
        return Curve(control=ctrl, dense=spline_through(ctrl))
    sm = _smooth_outline(b, smooth_px)
    if tol_px is None:
        ctrl = resample_closed(sm, n_points)
    else:
        ctrl = _even_with_tips(sm, n_points, tol_px, min_gap_px, max(0, max_points - n_points))
    return Curve(control=ctrl, dense=spline_through(ctrl))


def curve_to_mask(curve: Curve, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a curve: pixels whose centres fall inside it."""
    rr, cc = draw.polygon(curve.dense[:, 0], curve.dense[:, 1], shape)
    m = np.zeros(shape, dtype=bool)
    m[rr, cc] = True
    return m


def _filled(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.binary_fill_holes(mask)


def mask_area_by_curve(mask: np.ndarray, n_points: int | None = 18, tol_px: float | None = 1.5) -> float:
    """Area (pixel units) of a mask measured from a curve fitted to its outline.

    Interior holes (e.g. a papillary muscle drawn as an island) are subtracted by
    pixel count, so the labeling convention is preserved.
    """
    from scipy import ndimage

    comps, n = ndimage.label(mask)
    total = 0.0
    for k in range(1, n + 1):
        part = comps == k
        c = mask_to_curve(part, n_points, tol_px)
        if c is None:  # a speck too small to outline: count its pixels
            total += float(part.sum())
            continue
        holes = int(_filled(part).sum() - part.sum())
        total += c.area_px() - holes
    return total
