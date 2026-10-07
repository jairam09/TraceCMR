"""Tests for curve editing logic (no screen needed)."""

import numpy as np

from ventrimap import labels as L
from ventrimap.app.curve_editor import curves_from_labels, insert_point, labels_from_curves
from ventrimap.synthetic import ring_phantom


def test_insert_point_goes_on_nearest_edge():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    out = insert_point(sq, np.array([5.0, -1.0]))
    assert np.allclose(out[1], [5, -1]) and len(out) == 5


def test_round_trip_labels_curves_labels():
    ph = ring_phantom(shape=(64, 64, 3), spacing=(0.1, 0.1, 1.0), blood_radius_mm=1.5, wall_mm=1.0)
    lab4 = ph.label[..., None]
    curves = curves_from_labels(lab4, [0])
    assert {(0, 1, "endo"), (0, 1, "epi")} <= set(curves)
    back = labels_from_curves(curves[(0, 1, "endo")], curves[(0, 1, "epi")], lab4.shape[:2])
    orig = lab4[:, :, 1, 0]
    for v in (L.LV_BLOODPOOL, L.LV_MYOCARDIUM):
        a, b = orig == v, back == v
        assert (a & b).sum() / (a | b).sum() > 0.9


def test_smooth_points_reduces_roughness_keeps_area_and_respects_selection():
    from ventrimap.app.curve_editor import smooth_points
    from ventrimap.contours import polygon_area

    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    r = 20 + np.where(np.arange(24) % 2, 2.0, -2.0)  # zig-zag circle
    pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
    rough = lambda p: np.std(np.hypot(*p.T))
    out = smooth_points(pts)
    assert rough(out) < 0.5 * rough(pts)
    assert abs(polygon_area(out) - polygon_area(pts)) < 1e-6 * polygon_area(pts)
    which = np.zeros(24, bool)
    which[:4] = True
    part = smooth_points(pts, which)
    assert np.allclose(part[4:], pts[4:]) and not np.allclose(part[:4], pts[:4])


def test_snap_to_edge_recovers_bright_tip_and_never_shrinks():
    from ventrimap.refine import snap_slice

    ph = ring_phantom(shape=(96, 96, 1), spacing=(0.1, 0.1, 1.0), blood_radius_mm=2.0, wall_mm=1.0)
    lab = ph.label[:, :, 0].copy()
    img = np.where(lab == L.LV_BLOODPOOL, 1000.0, np.where(lab == L.LV_MYOCARDIUM, 200.0, 50.0))
    c = (np.array(lab.shape) - 1) / 2
    x, y = np.mgrid[: lab.shape[0], : lab.shape[1]]
    tip = (abs(y - c[1]) <= 1) & (x > c[0] + 19) & (x <= c[0] + 22)  # bright tip the "model" missed
    img[tip & (lab == L.LV_MYOCARDIUM)] = 1000.0
    out = snap_slice(lab, img, (0.1, 0.1))
    assert (out[tip & (lab == L.LV_MYOCARDIUM)] == L.LV_BLOODPOOL).all()
    assert ((lab == L.LV_BLOODPOOL) <= (out == L.LV_BLOODPOOL)).all()  # never shrinks
    assert ((lab > 0) == (out > 0)).all()  # epicardium unchanged
