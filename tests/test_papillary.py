"""Tests for papillary-convention conversion."""

import numpy as np

from ventrimap import labels as L
from ventrimap.papillary import smooth_papillary, smooth_slice
from ventrimap.synthetic import ring_phantom


def ring_with_papillary():
    ph = ring_phantom(shape=(96, 96, 1), spacing=(0.1, 0.1, 1.0), blood_radius_mm=2.0, wall_mm=1.0)
    lab = ph.label[:, :, 0].copy()
    c = (np.array(lab.shape) - 1) / 2
    x, y = np.mgrid[: lab.shape[0], : lab.shape[1]]
    # a papillary muscle: a 0.6 mm blob of muscle attached to the inside of the wall at 3 o'clock
    bump = (x - (c[0] + 16)) ** 2 + (y - c[1]) ** 2 <= 6**2
    lab[bump & (lab == L.LV_BLOODPOOL)] = L.LV_MYOCARDIUM
    return ph, lab


def test_papillary_bump_moves_into_blood_pool_and_lv_is_unchanged():
    ph, lab = ring_with_papillary()
    out = smooth_slice(lab, (0.1, 0.1))
    clean = ph.label[:, :, 0]
    lv_before, lv_after = lab > 0, out > 0
    assert (lv_before == lv_after).all()  # epicardium untouched
    # the blood pool recovers almost all of the area the papillary muscle took
    lost = (clean == L.LV_BLOODPOOL).sum() - (lab == L.LV_BLOODPOOL).sum()
    regained = (out == L.LV_BLOODPOOL).sum() - (lab == L.LV_BLOODPOOL).sum()
    assert regained > 0.8 * lost
    # and does not eat into the real wall
    assert (out == L.LV_BLOODPOOL).sum() <= (clean == L.LV_BLOODPOOL).sum() * 1.03


def test_smooth_ring_is_unchanged_and_4d_works():
    ph = ring_phantom(shape=(64, 64, 3), spacing=(0.1, 0.1, 1.0), blood_radius_mm=1.5, wall_mm=1.0)
    out = smooth_papillary(ph.label, ph.spacing)
    assert ((out == L.LV_BLOODPOOL) != (ph.label == L.LV_BLOODPOOL)).sum() <= 0.02 * (ph.label == L.LV_BLOODPOOL).sum()
    four = np.stack([ph.label, np.zeros_like(ph.label)], -1)
    assert smooth_papillary(four, ph.spacing).shape == four.shape
