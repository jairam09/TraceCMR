"""Tests for base/apex detection and level assignment.

The last test compares against visual (anatomical) labels for 16 UCL mice in
tests/data/ucl_orientation_truth.csv; it runs only with real data:
    TRACECMR_UCL_DATA=data/ucl/Data pytest tests/test_orientation.py
"""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from tracecmr import labels as L
from tracecmr.orientation import (
    OrientationUncertain,
    assign_levels,
    detect_base_apex,
    detect_for_heart,
    to_base_first,
)
from tracecmr.synthetic import ring_phantom, tapered_phantom


@pytest.mark.parametrize("cap", [True, False])
def test_detects_base_first_and_apex_first(cap):
    assert detect_base_apex(tapered_phantom(apical_cap=cap, base_first=True)).base_end == "first"
    assert detect_base_apex(tapered_phantom(apical_cap=cap, base_first=False)).base_end == "last"


def test_cylinder_is_refused_not_guessed():
    with pytest.raises(OrientationUncertain, match="too similar"):
        detect_base_apex(ring_phantom())


def test_contradicting_cap_is_refused():
    ci = tapered_phantom(apical_cap=True, base_first=True)
    # Put a myocardium-only cap at the BASE end as well as removing it at the apex: contradiction.
    ci.label[:, :, -1][ci.label[:, :, -1] == L.LV_MYOCARDIUM] = L.BACKGROUND
    ci.label[:, :, 0][ci.label[:, :, 0] == L.LV_BLOODPOOL] = L.LV_MYOCARDIUM
    with pytest.raises(OrientationUncertain):
        detect_base_apex(ci)


def test_to_base_first_reverses_and_reports_order():
    ci = tapered_phantom(base_first=False)
    out, order = to_base_first(ci)
    assert order == list(range(ci.image.shape[2] - 1, -1, -1))
    assert detect_base_apex(out).base_end == "first"
    assert np.array_equal(out.label[:, :, 0], ci.label[:, :, -1])


def test_manual_override():
    out, order = to_base_first(ring_phantom(), base_end="last")
    assert order[0] == 9


def test_levels_split_into_thirds_plus_cap():
    ci = tapered_phantom(n_slices=8, apical_cap=True, base_first=True)
    levels = assign_levels(ci)
    assert [levels[z] for z in range(9)] == (
        ["basal"] * 3 + ["mid"] * 3 + ["apical"] * 2 + ["apical cap"]
    )


UCL_DIR = os.environ.get("TRACECMR_UCL_DATA")
TRUTH = Path(__file__).parent / "data" / "ucl_orientation_truth.csv"


def test_heart_level_detection_falls_back_to_es():
    flat_ed = ring_phantom()  # undecidable
    es = tapered_phantom(base_first=False)
    r = detect_for_heart(flat_ed, es)
    assert r.base_end == "last" and r.evidence["decided_from"] == "ES"


def test_heart_level_refuses_when_ed_and_es_disagree():
    with pytest.raises(OrientationUncertain, match="opposite"):
        detect_for_heart(tapered_phantom(base_first=True), tapered_phantom(base_first=False))


def _truth_rows():
    return [r for r in csv.DictReader(open(TRUTH)) if (Path(UCL_DIR) / f"{r['subject_id']}.h5").exists()]


@pytest.mark.skipif(not UCL_DIR, reason="set TRACECMR_UCL_DATA to a folder of UCL .h5 files")
def test_matches_visual_labels_on_ucl():
    from tracecmr.datasets import load_ucl

    rows = _truth_rows()
    assert rows, "none of the truth-file mice are in TRACECMR_UCL_DATA"
    for row in rows:
        p = Path(UCL_DIR) / f"{row['subject_id']}.h5"
        r = detect_for_heart(load_ucl(p, "ED"), load_ucl(p, "ES"))
        assert r.base_end == row["base_end"], row["subject_id"]


@pytest.mark.skipif(not UCL_DIR, reason="set TRACECMR_UCL_DATA to a folder of UCL .h5 files")
def test_single_phase_may_refuse_but_never_errs():
    from tracecmr.datasets import load_ucl

    for row in _truth_rows():
        for phase in ("ED", "ES"):
            try:
                got = detect_base_apex(load_ucl(Path(UCL_DIR) / f"{row['subject_id']}.h5", phase))
            except OrientationUncertain:
                continue
            assert got.base_end == row["base_end"], f"{row['subject_id']} {phase}"
