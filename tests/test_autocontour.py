"""Tests for auto-contour helpers (the models themselves are optional installs)."""

import os

import numpy as np
import pytest

from tracecmr import autocontour as ac
from tracecmr import labels as L
from tracecmr.synthetic import ring_phantom


def test_fit_and_undo_round_trip():
    for shape in [(256, 256, 3), (180, 200, 3), (300, 150, 2)]:
        a = np.random.rand(*shape)
        f = ac._fit(a, 192)
        assert f.arr.shape[:2] == (192, 192)
        back = f.undo(f.arr)
        sx, sy = min(192, shape[0]), min(192, shape[1])
        x0, y0 = max((shape[0] - 192) // 2, 0), max((shape[1] - 192) // 2, 0)
        assert np.allclose(back[x0:x0 + sx, y0:y0 + sy], a[x0:x0 + sx, y0:y0 + sy])


def test_disagreement_and_sanity_flags():
    ph = ring_phantom(shape=(64, 64, 5), spacing=(0.1, 0.1, 1.0), blood_radius_mm=1.5, wall_mm=1.0)
    a = ph.label[..., None].copy()
    b = a.copy()
    b[:, :, 2, 0] = 0  # second model misses slice 2
    flags = ac.disagreement(a, b, [0])
    assert set(flags) == {(0, 2)} and flags[(0, 2)] == 0.0
    c = a.copy()
    c[:, :, 3, 0][c[:, :, 3, 0] == L.LV_MYOCARDIUM] = 0  # blood without wall
    assert ac.sanity_flags(c, [0])[(0, 3)] == "blood pool but no wall"


def test_keep_main_component_drops_specks():
    lab = np.zeros((40, 40, 3), np.uint8)
    lab[10:20, 10:20, :] = L.LV_BLOODPOOL
    lab[35, 35, 1] = L.LV_MYOCARDIUM
    out = ac._keep_main_component(lab)
    assert out[35, 35, 1] == 0 and out[15, 15, 1] == L.LV_BLOODPOOL


def test_unknown_or_missing_model_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACECMR_UCL_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        ac.run(np.zeros((8, 8, 2, 1)), (0.1, 0.1, 1.0), "nope")
    with pytest.raises(RuntimeError):
        ac.run(np.zeros((8, 8, 2, 1)), (0.1, 0.1, 1.0), "ucl")


UCL_DIR = os.environ.get("TRACECMR_UCL_DIR")
DATA = os.environ.get("TRACECMR_UCL_DATA")


@pytest.mark.skipif(not (UCL_DIR and DATA and ac.available()["ucl"] == "ready"), reason="needs the UCL model and data")
def test_ucl_model_matches_expert_on_a_test_mouse():
    import h5py

    from tracecmr.datasets.ucl import find_ed_es

    p = os.path.join(DATA, "Mouse6.h5")  # one of UCL's 25 held-out test mice
    with h5py.File(p) as f:
        img, gt = f["Images"][:].astype(np.float32), f["Masks"][:]
    t = find_ed_es(p)["ED"]
    lab = ac.run(img, (0.1, 0.1, 1.0), "ucl", frames=[t])[..., t]
    a, b = lab == L.LV_BLOODPOOL, gt[..., t] == 2
    assert 2 * (a & b).sum() / (a.sum() + b.sum()) > 0.85


def test_epi_cutoff_never_erases_a_faint_slice():
    import numpy as np
    from tracecmr import autocontour as ac

    prob = np.zeros((20, 20, 2, 3), np.float32)
    prob[..., 0] = 1.0
    yy, xx = np.mgrid[:20, :20]
    r = np.hypot(yy - 10, xx - 10)
    prob[r < 6, 0] = (0.0, 1.0, 0.0)  # slice 0: confident LV
    prob[r < 3, 0] = (0.0, 0.0, 1.0)
    prob[r < 4, 1] = (0.3, 0.4, 0.3)  # slice 1: faint apex, LV but only 70% sure
    lab = ac.labels_from_probs(prob, 0.5, 0.9)
    assert (lab[:, :, 1] > 0).sum() == (r < 4).sum()  # kept, not erased
    assert (lab[:, :, 0] > 0).sum() == (r < 6).sum()
