"""Smoke test of the napari panel: open a scan, compute, check the numbers.

Skipped unless napari and a display (real or Xvfb) are available.
"""

import os

import numpy as np
import pytest

napari = pytest.importorskip("napari")
if not os.environ.get("DISPLAY") and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    pytest.skip("needs a display (run under xvfb-run)", allow_module_level=True)

from ventrimap.app._widget import VentriMapWidget  # noqa: E402
from ventrimap.app.session import Session  # noqa: E402
from ventrimap.synthetic import tapered_phantom  # noqa: E402


def test_panel_computes_and_reorders():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = VentriMapWidget(viewer)
        panel.load_session(s)
        assert panel.list.topLevelItemCount() == ed.image.shape[2]
        panel.compute()
        assert float(panel.res_labels["ef_pct"].text()) > 0
        panel._flip()
        assert "reversed" in panel.lbl_order.text()
        # an edit in the napari layer reaches the results
        ef_before = panel.res_labels["edv_ul"].text()
        data = viewer.layers["contours"].data
        data[0][data[0] == 1] = 2  # turn ED blood pool into myocardium on every slice
        viewer.layers["contours"].data = data
        panel.compute()
        assert "need a drawn blood pool" in panel.lbl_notes.text() or panel.res_labels["edv_ul"].text() != ef_before
    finally:
        viewer.close()


def test_dragging_a_control_point_changes_volume():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = VentriMapWidget(viewer)
        panel.load_session(s)
        panel.compute()
        edv0 = float(panel.res_labels["edv_ul"].text())
        pts = viewer.layers["control points"]
        data = np.asarray(pts.data).copy()
        idx = [i for i, k in enumerate(panel.editor._keys) if k[0] == 0 and k[2] == "endo"]
        z = panel.editor._keys[idx[0]][1]
        same = [i for i in idx if panel.editor._keys[i][1] == z]
        c = data[same, 2:].mean(0)
        for i in same:  # enlarge one slice's endocardium by 30%
            data[i, 2:] = c + 1.3 * (data[i, 2:] - c)
        pts.data = data
        assert (0, z, "endo") in panel.editor.edited
        panel.compute()
        assert float(panel.res_labels["edv_ul"].text()) > edv0
        panel.rb_pixels.setChecked(True)
        assert viewer.layers["contours"].visible and not viewer.layers["control points"].visible
    finally:
        viewer.close()


def test_set_ed_es_from_current_frame():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    img = np.stack([ed.image, ed.image, es.image, es.image], -1)
    lab = np.stack([ed.label, np.zeros_like(ed.label), es.label, np.zeros_like(es.label)], -1)
    s = Session(image=img, label=lab, spacing=ed.spacing, subject_id="phantom", ed_frame=0, es_frame=0)
    viewer = napari.Viewer(show=False)
    try:
        panel = VentriMapWidget(viewer)
        panel.load_session(s)
        panel._show_frame(2)
        panel.btn_set_es.click()
        assert panel.session.es_frame == 2 and "ES" in panel.lbl_frame.text()
        panel._show_frame(0)
        panel.btn_set_ed.click()
        assert panel.session.ed_frame == 0
        panel.compute()
        assert float(panel.res_labels["ef_pct"].text()) > 0
    finally:
        viewer.close()


def test_autocontour_button_with_a_stand_in_model(monkeypatch):
    """The real models are optional installs; a stand-in returns phantom contours on every frame."""
    from ventrimap import autocontour as ac

    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    img = np.stack([ed.image, ed.image, es.image, ed.image], -1)
    truth = np.stack([ed.label, ed.label, es.label, ed.label], -1)

    def fake_run(image, spacing, model="ucl", frames=None, progress=None):
        out = np.zeros(image.shape, np.uint8)
        for t in frames:
            out[..., t] = truth[..., t]
        out[:, :, 3, 0] = 0  # a missed slice on frame 0, so sanity checks have something to flag
        return out

    monkeypatch.setattr(ac, "available", lambda: {"ucl": "ready", "calico": "not installed"})
    monkeypatch.setattr(ac, "run", fake_run)
    s = Session(image=img, label=np.zeros(img.shape, np.uint8), spacing=ed.spacing, subject_id="raw")
    viewer = napari.Viewer(show=False)
    try:
        panel = VentriMapWidget(viewer)
        panel.load_session(s)
        panel.autocontour(sync=True)
        assert "Contoured 4 frame(s)" in panel.lbl_auto.text()
        assert panel.session.ed_frame in (1, 3) and panel.session.es_frame == 2  # from the volume curve (frame 0 lost a slice)
        assert len(panel.vfig.axes) == 1  # volume curve drawn
        panel.compute()
        assert float(panel.res_labels["ef_pct"].text()) > 0
    finally:
        viewer.close()


def test_smooth_button_and_undo():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = VentriMapWidget(viewer)
        panel.load_session(s)
        ed_ = panel.editor
        z = int(viewer.dims.current_step[1])
        before = {k: v.copy() for k, v in ed_.curves.items()}
        n_points = len(ed_.points.data)
        panel._smooth_points()
        here = {k for k in before if (k[0], k[1]) == (0, z)}
        assert ed_.edited == here and (0, z) in ed_.hidden
        assert len(ed_.points.data) == n_points - sum(len(before[k]) for k in here)  # only this slice's dots gone
        assert all(np.allclose(before[k], ed_.curves[k]) for k in before if k not in here)
        # dragging a point on another slice keeps the hidden slice's contours
        data = np.asarray(ed_.points.data).copy()
        data[0, 2] += 1.0
        ed_.points.data = data
        assert all(k in ed_.curves for k in here)
        panel._edit_points()
        assert (0, z) not in ed_.hidden
        assert ed_.undo()
    finally:
        viewer.close()
