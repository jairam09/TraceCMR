import numpy as np
import pytest

from ventrimap import labels as L
from ventrimap.synthetic import ring_phantom, tapered_phantom
from ventrimap.wall import (
    SEGMENTS_PER_LEVEL,
    compute_wall,
    heart_wall,
    segment_means,
    slice_thickness,
)

SP = (0.1, 0.1)


def ring2d(blood=2.0, wall=1.0, shape=(128, 128), spacing=(0.1, 0.1, 1.0), center_mm=None):
    return ring_phantom(
        shape=(*shape, 1), spacing=spacing, blood_radius_mm=blood, wall_mm=wall, center_mm=center_mm
    ).label[:, :, 0]


def test_ring_thickness_equals_wall_at_every_angle():
    st = slice_thickness(ring2d(wall=1.0), SP)
    assert np.all(np.isfinite(st.thickness_mm))
    # Sub-pixel borders: within a third of a pixel (0.033 mm) on a 0.1 mm grid.
    assert np.max(np.abs(st.thickness_mm - 1.0)) < 0.04
    assert np.mean(st.thickness_mm) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("wall", [0.6, 1.0, 1.5])
def test_different_wall_widths(wall):
    st = slice_thickness(ring2d(wall=wall), SP)
    assert np.nanmean(st.thickness_mm) == pytest.approx(wall, abs=0.02)


def test_offcentre_ring_unchanged():
    st = slice_thickness(ring2d(center_mm=(4.5, 8.0)), SP)
    assert np.nanmean(st.thickness_mm) == pytest.approx(1.0, abs=0.02)


def test_spacing_scales_thickness():
    lab = ring2d(spacing=(0.1, 0.1, 1.0))
    a = np.nanmean(slice_thickness(lab, (0.1, 0.1)).thickness_mm)
    b = np.nanmean(slice_thickness(lab, (0.2, 0.2)).thickness_mm)
    assert b == pytest.approx(2 * a, rel=0.01)


def test_no_blood_pool_returns_none():
    lab = ring2d()
    lab[lab == L.LV_BLOODPOOL] = L.LV_MYOCARDIUM
    assert slice_thickness(lab, SP) is None


def test_papillary_bump_increases_local_thickness():
    """A blob of myocardium attached inside the wall (papillary muscle labeled as myocardium)."""
    lab = ring2d()
    cx = cy = 64
    xx, yy = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    bump = np.hypot(xx - (cx + 17), yy - cy) <= 4  # sits on the inner wall at angle 0
    lab[bump & (lab == L.LV_BLOODPOOL)] = L.LV_MYOCARDIUM
    st = slice_thickness(lab, SP)
    at0 = st.thickness_mm[np.argmin(np.abs(st.angles_deg - 0))]
    at180 = st.thickness_mm[np.argmin(np.abs(st.angles_deg - 180))]
    assert at0 > at180 + 0.3  # inflated where the papillary muscle attaches


def test_segment_means_shape_and_values():
    st = slice_thickness(ring2d(), SP)
    for n in (4, 6):
        seg = segment_means(st, n)
        assert seg.shape == (n,)
        assert np.allclose(seg, 1.0, atol=0.03)


def test_heart_levels_follow_aha_layout():
    ci = tapered_phantom(n_slices=9, apical_cap=True, base_first=False)
    hw = heart_wall(ci, base_end="last")
    assert set(hw.levels) == {"basal", "mid", "apical"}
    for level, res in hw.levels.items():
        assert res.segments_mm.shape == (SEGMENTS_PER_LEVEL[level],)
        assert np.allclose(res.segments_mm, 1.0, atol=0.04)


def test_thickening_on_phantom():
    # ED: wall 1.0 mm. ES: same outer ring area per slice, smaller cavity -> thicker wall.
    ed = tapered_phantom(n_slices=9, base_blood_mm=2.5, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=9, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.5)
    ed.papillary = es.papillary = "myocardium"
    r = compute_wall(ed, es)
    for level in ("basal", "mid", "apical"):
        assert np.allclose(r.thickening_pct[level], 50.0, atol=5.0)
    rows = r.rows()
    assert len(rows) == 6 + 6 + 4
    assert any("papillary" in f for f in r.flags)


def test_mixed_conventions_refused():
    ed = tapered_phantom()
    es = tapered_phantom()
    ed.papillary, es.papillary = "myocardium", "blood_pool"
    with pytest.raises(ValueError, match="papillary"):
        compute_wall(ed, es)
