"""Tests for curve contours."""

import numpy as np
import pytest

from tracecmr.contours import curve_to_mask, mask_area_by_curve, mask_to_curve, polygon_area


def disc(r, shape=(128, 128), centre=(63.3, 64.7)):
    x, y = np.mgrid[: shape[0], : shape[1]]
    return (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= r**2


def test_polygon_area_square():
    assert polygon_area(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)) == pytest.approx(4.0)


@pytest.mark.parametrize("r", [6, 12, 25])
def test_curve_area_close_to_true_circle(r):
    true = np.pi * r**2
    m = disc(r)
    for n in (16, 24, 32):
        assert mask_area_by_curve(m, n) == pytest.approx(true, rel=0.06 if r == 6 else 0.03)


def test_round_trip_mask_curve_mask():
    m = disc(15)
    c = mask_to_curve(m, 24)
    back = curve_to_mask(c, m.shape)
    overlap = (m & back).sum() / (m | back).sum()
    assert overlap > 0.95


def test_holes_and_components_preserved():
    m = disc(20) & ~disc(4)  # ring-like: island hole in the middle
    two = m | disc(5, centre=(10, 10))
    assert mask_area_by_curve(m, 32) == pytest.approx(m.sum(), rel=0.04)
    assert mask_area_by_curve(two, 32) > mask_area_by_curve(m, 32)


def test_empty():
    assert mask_to_curve(np.zeros((10, 10), bool)) is None
    assert mask_area_by_curve(np.zeros((10, 10), bool)) == 0.0


def _spline_reference(control, n_dense):
    """Centripetal Catmull-Rom one segment at a time (the original, slower formulation)."""
    n = len(control)
    p = np.vstack([control[-1:], control, control[:2]])
    out, per_seg = [], max(2, n_dense // n)
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


def test_spline_through_all_segments_at_once_matches_one_at_a_time():
    """spline_through runs on every redraw of a dragged dot, so it computes every segment in one go; the curve
    must be the same as segment by segment, including with a repeated point."""
    from tracecmr.contours import spline_through

    rng = np.random.default_rng(1)
    for k in range(40):
        n = int(rng.integers(4, 28))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        c = np.column_stack([rng.uniform(5, 20, n) * np.cos(ang) + 50, rng.uniform(5, 20, n) * np.sin(ang) + 60])
        if k % 5 == 0:
            c[2] = c[1]
        for n_dense in (200, 400, 33):
            assert np.allclose(spline_through(c, n_dense), _spline_reference(c, n_dense), atol=1e-9)
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.array_equal(spline_through(tri), tri)  # fewer than 4 points: returned as they are
