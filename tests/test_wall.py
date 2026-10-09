import numpy as np
import pytest

from tracecmr import labels as L
from tracecmr.synthetic import ring_phantom, tapered_phantom
from tracecmr.wall import (
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
    assert at0 > at180 + 0.15  # still inflated where the papillary muscle attaches (centerline: less than rays)
    rays = slice_thickness(lab, SP, method="rays")
    assert rays.thickness_mm[np.argmin(np.abs(rays.angles_deg - 0))] > rays.thickness_mm[np.argmin(np.abs(rays.angles_deg - 180))] + 0.3


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


def _oval(a, b, wall=1.0, sp=0.05, n=200):
    """Elliptical cavity with a wall of exactly `wall` mm everywhere (all points within `wall` of the cavity)."""
    from scipy import ndimage as ndi

    x = (np.arange(n) - n / 2) * sp
    X, Y = np.meshgrid(x, x, indexing="ij")
    endo = (X / a) ** 2 + (Y / b) ** 2 <= 1
    epi = ndi.distance_transform_edt(~endo) * sp <= wall
    lab = np.zeros((n, n), np.uint8)
    lab[epi] = L.LV_MYOCARDIUM
    lab[endo] = L.LV_BLOODPOOL
    return lab


@pytest.mark.parametrize("a,b", [(2.6, 1.3), (3.0, 1.0)])
def test_centerline_is_right_on_oval_hearts_where_rays_overestimate(a, b):
    """Uniform 1 mm wall around an oval cavity: the centerline reads 1 mm all round; rays read thick where the
    wall is oblique to them (the reason the centerline method is the default)."""
    lab = _oval(a, b)
    c = slice_thickness(lab, (0.05, 0.05))
    assert np.all(np.isfinite(c.thickness_mm))
    assert np.nanmax(np.abs(c.thickness_mm - 1.0)) < 0.05
    r = slice_thickness(lab, (0.05, 0.05), method="rays")
    assert np.nanmax(r.thickness_mm) > 1.15


def test_centerline_ignores_enclosed_papillary_and_unknown_method():
    lab = ring2d()
    lab[62:66, 62:66] = L.LV_MYOCARDIUM  # muscle floating in the cavity, not touching the wall
    assert np.nanmean(slice_thickness(lab, SP).thickness_mm) == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError):
        slice_thickness(lab, SP, method="nope")


def _ring_with_rv(rv_side="left", thick_at_deg=None, n=128, sp=0.1):
    """Ring LV (blood 2 mm, wall 1 mm) with an RV crescent on one side; optional 1.6 mm wall patch at an angle
    (raw angle: 0 = +x, 90 = +y, which is down on the image)."""
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    c = n / 2
    r = np.hypot(xx - c, yy - c) * sp
    ang = np.rad2deg(np.arctan2(yy - c, xx - c)) % 360
    lab = np.zeros((n, n), np.uint8)
    epi_r = np.full_like(r, 3.0)
    if thick_at_deg is not None:
        d = np.abs((ang - thick_at_deg + 180) % 360 - 180)
        epi_r = np.where(d < 25, 3.6, 3.0)
    lab[r <= epi_r] = L.LV_MYOCARDIUM
    lab[r <= 2.0] = L.LV_BLOODPOOL
    side = {"left": 180, "right": 0, "up": 270, "down": 90}[rv_side]
    d_rv = np.abs((ang - side + 180) % 360 - 180)
    lab[(r > epi_r + 0.15) & (r < epi_r + 1.2) & (d_rv < 50)] = L.RV_BLOODPOOL
    return lab


def test_septal_angle_finds_the_rv_side():
    from tracecmr.wall import septal_angle

    for side, want in (("left", 180), ("down", 90), ("right", 0)):
        sept, (a, b) = septal_angle(_ring_with_rv(side), SP)
        assert abs((sept - want + 180) % 360 - 180) < 5
        assert 60 < abs((a - b + 180) % 360 - 180) < 130  # insertion points either side of the septum
    lab = ring2d()
    assert septal_angle(lab, SP) is None  # no RV


def _heart(lab2d, n_slices=6):
    from tracecmr.io import CardiacImage

    lab = np.repeat(lab2d[:, :, None], n_slices, axis=2)
    return CardiacImage(image=np.zeros(lab.shape, np.float32), label=lab, spacing=(0.1, 0.1, 1.0),
                        subject_id="t", phase="ED", papillary="blood_pool")


def test_segments_follow_the_septum_in_aha_order():
    """RV on the left, thick wall at the top (anterior in the standard view from the apex): segment 1 is thickest.
    Rotating the whole heart moves nothing; mirroring swaps anterior with inferior."""
    for rv_side, top in (("left", 270), ("down", 180)):  # second case: the same heart turned 90 degrees
        ci = _heart(_ring_with_rv(rv_side, thick_at_deg=top))
        from tracecmr.wall import septum_by_level

        sept = septum_by_level(ci, "first")
        assert set(sept) == {"basal", "mid", "apical"}
        hw = heart_wall(ci, "first", septum=sept)
        for level in ("basal", "mid"):
            seg = hw.levels[level].segments_mm
            assert int(np.nanargmax(seg)) == 0, (rv_side, level, seg)  # anterior = segment 1
        assert int(np.nanargmax(hw.levels["apical"].segments_mm)) == 0
        assert "septum" in hw.reference
        hm = heart_wall(ci, "first", septum=sept, mirror=True)
        assert int(np.nanargmax(hm.levels["mid"].segments_mm)) == 3  # seen from the base: the patch reads inferior


def test_compute_wall_flags_septum_or_image_axis():
    ci = _heart(_ring_with_rv("left"))
    es = _heart(_ring_with_rv("left"))
    es.phase = "ES"
    r = compute_wall(ci, es, base_end="first")
    assert any("RV insertion" in f for f in r.flags)
    plain = _heart(ring2d())
    plain_es = _heart(ring2d())
    r2 = compute_wall(plain, plain_es, base_end="first")
    assert any("no RV traced" in f for f in r2.flags)


def test_septum_set_by_hand_wins_over_the_rv():
    from tracecmr.wall import septum_by_level

    ci = _heart(_ring_with_rv("left"))  # RV says 180 degrees
    auto = septum_by_level(ci, "first")
    assert all(abs((a - 180 + 180) % 360 - 180) < 5 for a in auto.values())
    manual = septum_by_level(ci, "first", manual={i: 90.0 for i in range(6)})
    assert all(abs(a - 90) < 1e-6 for a in manual.values())
    # only the mid slices set by hand: mid follows the hand, the others keep the RV
    lv = septum_by_level(ci, "first", manual={2: 90.0, 3: 90.0})
    assert abs(lv["mid"] - 90) < 1e-6 and abs((lv["basal"] - 180 + 180) % 360 - 180) < 5
