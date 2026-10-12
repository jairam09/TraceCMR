"""Tests for the app's Session (no screen needed)."""

import numpy as np
import pytest

from tracecmr.app.session import Session
from tracecmr.synthetic import tapered_phantom
from tracecmr.volumes import compute_function


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


def test_per_slice_ed_es_frames():
    """Separately acquired slices: each slice's own ED/ES frame is used for the volumes."""
    from tracecmr.synthetic import tapered_phantom

    ed = tapered_phantom(n_slices=6, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=6, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    # frames 0..3; slice 2's cine is shifted by one frame (its ES is frame 3, the others' frame 2)
    img = np.stack([ed.image, ed.image, es.image, ed.image], -1)
    lab = np.stack([ed.label, ed.label, es.label, ed.label], -1)
    shifted_img, shifted_lab = img.copy(), lab.copy()
    shifted_img[:, :, 2, 2], shifted_lab[:, :, 2, 2] = ed.image[:, :, 2], ed.label[:, :, 2]
    shifted_img[:, :, 2, 3], shifted_lab[:, :, 2, 3] = es.image[:, :, 2], es.label[:, :, 2]
    ref = Session(image=img, label=lab, spacing=ed.spacing, ed_frame=0, es_frame=2)
    s = Session(image=shifted_img, label=shifted_lab, spacing=ed.spacing, ed_frame=0, es_frame=2,
                separate_slices=True)
    s.set_phase_frame("ES", 3, z=2)
    assert s.t_for(2, "ES") == 3 and s.t_for(1, "ES") == 2 and s.frames_in_use() == [0, 2, 3]
    assert s.compute()[0].ef_pct == pytest.approx(ref.compute()[0].ef_pct)
    s.set_phase_frame("ES", 2)  # whole stack: clears the slice override
    assert s.t_for(2, "ES") == 2 and not s.es_by_slice
    # per-slice detection finds the shifted slice
    msg = s.auto_ed_es()
    assert s.t_for(2, "ES") == 3 and s.t_for(1, "ES") == 2 and "per slice" in msg


def test_model_contours_kept_apart_from_corrections(tmp_path, monkeypatch):
    """The model's own output is saved next to the user's corrected contours (training data)."""
    from tracecmr import autocontour as ac

    ed = tapered_phantom(n_slices=6, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    img = np.stack([ed.image, ed.image], -1)
    monkeypatch.setattr(ac, "available", lambda: {"ucl": "ready"})
    monkeypatch.setattr(ac, "run_pairs", lambda image, spacing, pairs, model="ucl", unsure_out=None:
                        np.stack([ed.label[:, :, z] for z, t in pairs], axis=2))
    s = Session(image=img, label=np.zeros(img.shape, np.uint8), spacing=ed.spacing, subject_id="m")
    nz = img.shape[2]
    s.autocontour_pairs([(z, 0) for z in range(nz)])
    assert s.model_ran[:, 0].all() and not s.model_ran[:, 1].any()
    s.label[:, :, 2, 0] = 0  # the user corrects (here: clears) one slice
    p = s.save_contours(tmp_path / "m.tracecmr.npz")
    s2 = Session(image=img, label=np.zeros(img.shape, np.uint8), spacing=ed.spacing, subject_id="m")
    s2.load_contours(p)
    assert np.array_equal(s2.model_label[:, :, 2, 0], ed.label[:, :, 2]) and not s2.label[:, :, 2, 0].any()
    assert s2.model_ran[:, 0].all()


def test_cutoffs_grow_blood_only_inside_and_trim_the_outer_edge():
    from tracecmr import autocontour as ac

    prob = np.zeros((20, 20, 1, 3), np.float32)
    prob[..., 0] = 1.0
    yy, xx = np.mgrid[:20, :20]
    r = np.hypot(yy - 10, xx - 10)[..., None]
    prob[r < 7] = (0.0, 1.0, 0.0)  # myocardium ring
    prob[r < 4] = (0.0, 0.0, 1.0)  # blood
    prob[(r >= 4) & (r < 5)] = (0.0, 0.6, 0.4)  # corner pixels the model is unsure about
    prob[(r >= 6.5) & (r < 7)] = (0.3, 0.7, 0.0)  # soft outer edge
    prob[1, 1, 0] = (0.0, 0.5, 0.5)  # an unsure speck outside the LV
    base = ac.labels_from_probs(prob)
    grown = ac.labels_from_probs(prob, cutoff=0.35)
    assert (grown == 2).sum() > (base == 2).sum() and grown[1, 1, 0] == 0
    trimmed = ac.labels_from_probs(prob, epi_cutoff=0.9)
    assert (trimmed > 0).sum() < (base > 0).sum() and (trimmed == 2).sum() == (base == 2).sum()


def test_es_not_set_on_some_slices_is_flagged():
    ed = tapered_phantom(n_slices=6, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=6, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="p", ed_frame=0, es_frame=0, separate_slices=True)
    s.set_phase_frame("ES", 1, z=2)  # ES set on one slice only
    missing = s.es_not_set()
    assert 2 not in missing and len(missing) >= 3
    _, _, notes = s.compute()
    assert "ES isn't set" in notes[0]
    rows = {r["slice"]: r for r in s.per_slice()}
    assert not rows[2]["es_not_set"] and rows[missing[0]]["es_not_set"]
