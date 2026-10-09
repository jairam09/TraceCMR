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
