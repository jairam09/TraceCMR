"""Tests for the app's Session (no screen needed)."""

import numpy as np
import pytest

from ventrimap.app.session import Session
from ventrimap.synthetic import tapered_phantom
from ventrimap.volumes import compute_function


def make_session(base_first=True):
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0, base_first=base_first)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3, base_first=base_first)
    image = np.stack([ed.image, es.image, np.zeros_like(ed.image)], axis=-1)
    label = np.stack([ed.label, es.label, np.zeros_like(ed.label)], axis=-1)
    return Session(image=image, label=label, spacing=ed.spacing, subject_id="phantom",
                   papillary="blood_pool", ed_frame=0, es_frame=1), ed, es


def test_auto_ed_es_and_order():
    s, _, _ = make_session(base_first=False)
    s.ed_frame = s.es_frame = 2
    assert "frame 0" in s.auto_ed_es() and (s.ed_frame, s.es_frame) == (0, 1)
    s.auto_order()
    nz = s.image.shape[2]
    assert s.order == list(range(nz - 1, -1, -1))  # stored apex-first, so reversed
    assert s.order_warnings() == []


def test_reversed_order_warns_but_still_computes():
    s, _, _ = make_session(base_first=True)
    s.flip()
    assert any("reversed" in w for w in s.order_warnings())
    fn, wall, notes = s.compute()
    assert any("reversed" in n for n in notes)
    assert fn.ef_pct > 0


def test_volumes_match_direct_calculation_and_exclusion_changes_them():
    s, ed, es = make_session()
    fn, wall, _ = s.compute()
    direct = compute_function(ed, es)
    assert fn.edv_ul == pytest.approx(direct.edv_ul)
    assert wall is not None and set(wall.thickening_pct) == {"basal", "mid", "apical"}
    s.set_included(s.order[0], False)  # both phases
    fn2, _, _ = s.compute()
    assert fn2.edv_ul < fn.edv_ul


def test_move_reorders():
    s, _, _ = make_session()
    nz = s.image.shape[2]
    s.move(nz - 1, 0)
    assert s.order[0] == nz - 1 and len(s.order) == nz


def test_positions_contradicted_warns():
    s, _, _ = make_session()
    s.slice_positions_mm = [float(z) for z in range(s.image.shape[2])]
    s.move(5, 1)
    assert any("slice positions" in w for w in s.order_warnings())


def test_no_contours_message():
    s, _, _ = make_session()
    s.label[:] = 0
    assert "No contours" in s.auto_ed_es()
    with pytest.raises(ValueError):
        s.compute()


def test_per_phase_inclusion_and_suggestions():
    s, _, _ = make_session()
    sug = s.suggest_slices()
    assert set(sug) == {"ED", "ES"}
    assert all(inc for inc, why in sug["ED"].values() if why.startswith("wall covers"))
    fn_all, _, _ = s.compute()
    base = s.order[0]
    s.set_included(base, False, "ES")
    assert base in s.included["ED"] and base not in s.included["ES"]
    fn_es, _, _ = s.compute()
    assert fn_es.edv_ul == fn_all.edv_ul and fn_es.esv_ul < fn_all.esv_ul
    s.label[:, :, base, s.es_frame] = 0
    s.apply_suggestions()
    assert base not in s.included["ES"] and s.reasons["ES"][base] == "no contour"
