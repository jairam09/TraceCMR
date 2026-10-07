import numpy as np
import pytest

from mcmr import labels as L
from mcmr.synthetic import ring_phantom, ring_phantom_truth
from mcmr.volumes import (
    MYOCARDIAL_DENSITY_G_PER_ML,
    check_comparable,
    compute_function,
    slice_areas_mm2,
    volume_ul,
)

SHAPE = (240, 240, 8)
SPACING = (0.05, 0.05, 1.0)  # fine pixels -> small pixelation error


# ES wall chosen so myocardial area is conserved, like a real heart:
# outer_ES^2 - 1.5^2 = 3.5^2 - 2.5^2  ->  outer_ES = sqrt(8.25) ~ 2.872 mm
ES_WALL = 8.25**0.5 - 1.5


def phantoms(papillary="myocardium"):
    ed = ring_phantom(shape=SHAPE, spacing=SPACING, blood_radius_mm=2.5, wall_mm=1.0, phase="ED")
    es = ring_phantom(shape=SHAPE, spacing=SPACING, blood_radius_mm=1.5, wall_mm=ES_WALL, phase="ES")
    ed.papillary = es.papillary = papillary
    return ed, es


def test_volumes_match_exact_geometry():
    ed, es = phantoms()
    r = compute_function(ed, es)
    edv = ring_phantom_truth(SHAPE, SPACING, 2.5, 1.0)["blood_ul"]
    esv = ring_phantom_truth(SHAPE, SPACING, 1.5, ES_WALL)["blood_ul"]
    assert r.edv_ul == pytest.approx(edv, rel=0.01)
    assert r.esv_ul == pytest.approx(esv, rel=0.02)
    assert r.ef_pct == pytest.approx(100 * (edv - esv) / edv, abs=1.0)
    assert r.sv_ul == pytest.approx(r.edv_ul - r.esv_ul)
    assert r.flags == []


def test_mass_uses_density():
    ed, _ = phantoms()
    myo = ring_phantom_truth(SHAPE, SPACING, 2.5, 1.0)["myocardium_ul"]
    assert volume_ul(ed, (L.LV_MYOCARDIUM,)) * MYOCARDIAL_DENSITY_G_PER_ML == pytest.approx(
        myo * 1.05, rel=0.01
    )


def test_halving_pixel_spacing_quarters_area():
    ed, _ = phantoms()
    a = slice_areas_mm2(ed, (L.LV_BLOODPOOL,))
    ed.spacing = (SPACING[0] / 2, SPACING[1] / 2, SPACING[2])
    assert np.allclose(slice_areas_mm2(ed, (L.LV_BLOODPOOL,)), a / 4)


def test_scar_counts_toward_mass_not_blood():
    ed, es = phantoms()
    before = compute_function(ed, es)
    ed.label[ed.label == L.LV_MYOCARDIUM] = L.SCAR  # relabel myocardium as scar
    after = compute_function(ed, es)
    assert after.mass_ed_mg == pytest.approx(before.mass_ed_mg)
    assert after.edv_ul == pytest.approx(before.edv_ul)


def test_swapped_phases_are_flagged():
    ed, es = phantoms()
    r = compute_function(es, ed)
    assert any("swapped" in f for f in r.flags)


def test_mass_mismatch_is_flagged():
    ed, es = phantoms()
    es.label[es.label == L.LV_MYOCARDIUM] = L.BACKGROUND
    es.label[:, :, :4][ring_phantom(shape=SHAPE, spacing=SPACING, blood_radius_mm=1.5, wall_mm=ES_WALL).label[:, :, :4] == L.LV_MYOCARDIUM] = L.LV_MYOCARDIUM
    r = compute_function(ed, es)
    assert any("mass differs" in f for f in r.flags)


def test_mixed_papillary_conventions_refused():
    ed, es = phantoms()
    es.papillary = "blood_pool"
    with pytest.raises(ValueError, match="papillary"):
        compute_function(ed, es)


def test_check_comparable():
    a = compute_function(*phantoms("myocardium"))
    b = compute_function(*phantoms("blood_pool"))
    check_comparable(a, a)
    with pytest.raises(ValueError, match="not directly comparable"):
        check_comparable(a, b)
    check_comparable(a, b, allow_mismatch=True)


def test_unknown_convention_is_flagged():
    r = compute_function(*phantoms(None))
    assert "papillary convention unknown" in r.flags


def test_bad_convention_value_rejected():
    with pytest.raises(ValueError, match="papillary must be"):
        ring_phantom().__class__(image=np.zeros((2, 2, 2)), spacing=(1, 1, 1), papillary="trabeculae")
