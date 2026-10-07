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
