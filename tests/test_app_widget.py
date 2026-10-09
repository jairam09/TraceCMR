"""Smoke test of the napari panel: open a scan, compute, check the numbers.

Skipped unless napari and a display (real or Xvfb) are available.
"""

import os

import numpy as np
import pytest

napari = pytest.importorskip("napari")
if not os.environ.get("DISPLAY") and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    pytest.skip("needs a display (run under xvfb-run)", allow_module_level=True)

from tracecmr.app._widget import TraceCMRWidget  # noqa: E402
from tracecmr.app.session import Session  # noqa: E402
from tracecmr.synthetic import tapered_phantom  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402


def test_panel_computes_and_reorders():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
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
        panel = TraceCMRWidget(viewer)
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
        panel = TraceCMRWidget(viewer)
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
    from tracecmr import autocontour as ac

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
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        panel.chk_all_frames.setChecked(True)
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
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        ed_ = panel.editor
        z = int(viewer.dims.current_step[1])
        before = {k: v.copy() for k, v in ed_.curves.items()}
        n_points = len(ed_.points.data)
        panel._smooth_points()  # Endo is selected: only the endocardium is smoothed
        here = {(0, z, "endo")}
        assert ed_.edited == here and (0, z, "endo") in ed_.hidden
        assert len(ed_.points.data) == n_points - len(before[(0, z, "endo")])  # only its dots are gone
        assert all(np.allclose(before[k], ed_.curves[k]) for k in before if k not in here)
        # editing another slice keeps the hidden slice's contours
        other = next(k for k in ed_.curves if k[:2] != (0, z))
        ed_.move_point(other, 0, ed_.curves[other][0] + 1.0)
        assert all(k in ed_.curves for k in here)
        panel._edit_points()
        assert not any(k[:2] == (0, z) for k in ed_.hidden)
        assert ed_.undo()
    finally:
        viewer.close()


def test_smoothed_contour_stays_smoothed_after_phase_change_elsewhere():
    """Smooth on one slice, set ES on another slice, come back: the dots stay hidden."""
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image, es.image], -1),
                label=np.stack([ed.label, es.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        e = panel.editor
        z = int(viewer.dims.current_step[1])
        for target in (0, 1):  # smooth the endo, then the epi
            panel.cmb_add.setCurrentIndex(target)
            panel._smooth_points()
        smoothed = {k: e.curves[k].copy() for k in ((0, z, "endo"), (0, z, "epi"))}
        assert set(smoothed) <= e.hidden
        panel.chk_per_slice.setChecked(True)
        panel.step_slice(1)
        panel.set_phase_frame("ES", 2)  # re-shows the curves
        panel.step_slice(-1)
        assert set(smoothed) <= e.hidden
        assert all(np.allclose(e.curves[k], v) for k, v in smoothed.items())
        assert len(e.points.data) == 0  # epi is selected and smoothed: no dots
        # contours replaced on this slice (e.g. auto-contour) come back with dots
        lab = panel.session.label.copy()
        lab[:, :, z, 0] = 0
        lab[:, :, z, 0][ed.label[:, :, z] > 0] = ed.label[:, :, z][ed.label[:, :, z] > 0]
        lab[:, :, z, 0] = np.roll(lab[:, :, z, 0], 1, axis=0)
        panel.session.label = lab
        panel.viewer.layers["contours"].data = lab.transpose(3, 2, 1, 0).copy()
        panel._mode_changed()
        assert not any(k[:2] == (0, z) for k in e.hidden)
    finally:
        viewer.close()


def test_zoom_buttons_and_fit():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    s = Session(image=np.stack([ed.image, ed.image], -1), label=np.stack([ed.label, ed.label], -1),
                spacing=ed.spacing, subject_id="phantom", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        z0 = viewer.scene.camera.zoom
        panel.btn_zoom_in.click()
        assert viewer.scene.camera.zoom == pytest.approx(z0 * 1.25)
        panel.btn_zoom_out.click()
        panel.btn_zoom_out.click()
        assert viewer.scene.camera.zoom < z0
        step = viewer.dims.current_step
        panel.btn_zoom_fit.click()  # back to the heart view, same slice
        assert viewer.scene.camera.zoom == pytest.approx(z0, rel=0.01)
        assert viewer.dims.current_step == step
    finally:
        viewer.close()


def test_model_folder_choice(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACECMR_UCL_DIR", raising=False)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        status = panel.set_model_dir("ucl", str(tmp_path), remember=False)  # empty folder: not a model
        assert status != "ready" and "not found" in panel.lbl_auto.text()
        assert os.environ["TRACECMR_UCL_DIR"] == str(tmp_path)
        assert "(not installed)" in panel.cmb_model.currentText()
    finally:
        viewer.close()



class _Ev:
    """Minimal stand-in for a napari mouse event."""

    def __init__(self, world, button=1, screen=(0.0, 0.0)):
        self.position, self.button, self.pos, self.type = tuple(world), button, screen, "mouse_press"


def _drive(editor, viewer, ev, moves=()):
    gen = editor._on_mouse(viewer, ev)
    try:
        next(gen)
        for world, screen in moves:
            ev.type, ev.position, ev.pos = "mouse_move", tuple(world), screen
            next(gen)
        ev.type = "mouse_release"
        next(gen)
    except StopIteration:
        pass


def _world(editor, t, z, p):
    sc = editor.scale
    return (t, z * sc[1], p[1] * sc[2], p[0] * sc[3])


def test_segment_style_mouse_editing_and_undo():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        e = panel.editor
        t, z = (int(v) for v in viewer.dims.current_step[:2])
        key = (t, z, "endo")
        n0 = len(e.curves[key])
        p0 = e.curves[key][0].copy()
        # drag the first dot outward
        c = e.curves[key].mean(0)
        target = c + 1.3 * (p0 - c)
        _drive(e, viewer, _Ev(_world(e, t, z, p0)), [(_world(e, t, z, target), (5, 5))])
        assert np.allclose(e.curves[key][0], target) and key in e.edited
        # click on the line between dots 2 and 3: adds a dot
        mid = spline_mid = None
        from tracecmr.contours import spline_through
        dense = spline_through(e.curves[key], 400)
        per = max(2, 400 // len(e.curves[key]))
        spline_mid = dense[2 * per + per // 2]
        _drive(e, viewer, _Ev(_world(e, t, z, spline_mid)))
        assert len(e.curves[key]) == n0 + 1
        # right-click a dot deletes it
        _drive(e, viewer, _Ev(_world(e, t, z, e.curves[key][5]), button=2))
        assert len(e.curves[key]) == n0
        # three undos bring the original contour back
        for _ in range(3):
            panel.undo()
        assert len(e.curves[key]) == n0 and np.allclose(e.curves[key][0], p0)
        # trash can clears the selected contour (Endo), then undo
        panel.clear_slice()
        assert key not in e.curves and (t, z, "epi") in e.curves
        panel.undo()
        assert key in e.curves
        # draw a new endocardium on a cleared slice: 4 clicks, right-click closes
        panel.clear_slice()
        ring = [c + 4 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 5)[:-1]]
        for q in ring:
            _drive(e, viewer, _Ev(_world(e, t, z, q)))
        assert e.pending is not None and len(e.pending[3]) == 4
        _drive(e, viewer, _Ev(_world(e, t, z, c + 30), button=2))
        assert key in e.curves and len(e.curves[key]) == 4 and e.pending is None
        panel.compute()  # the drawn contour feeds the numbers
        assert float(panel.res_labels["edv_ul"].text()) > 0
    finally:
        viewer.close()


def test_right_drag_pans_without_editing():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    s = Session(image=np.stack([ed.image, ed.image], -1), label=np.stack([ed.label, ed.label], -1),
                spacing=ed.spacing, subject_id="phantom", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        e = panel.editor
        t, z = (int(v) for v in viewer.dims.current_step[:2])
        before = {k: v.copy() for k, v in e.curves.items()}
        centre = viewer.scene.camera.center
        p = e.curves[(t, z, "endo")][0]
        _drive(e, viewer, _Ev(_world(e, t, z, p), button=2, screen=(100, 100)),
               [(_world(e, t, z, p), (140, 100)), (_world(e, t, z, p), (180, 120))])
        assert viewer.scene.camera.center != centre
        assert all(np.array_equal(before[k], e.curves[k]) for k in before)
        assert viewer.scene.camera.mouse_pan is False  # left-drag edits, not pans
    finally:
        viewer.close()


def test_left_drag_never_pans_unless_pan_is_on():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    s = Session(image=np.stack([ed.image, ed.image], -1), label=np.stack([ed.label, ed.label], -1),
                spacing=ed.spacing, subject_id="phantom", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        e = panel.editor
        t, z = (int(v) for v in viewer.dims.current_step[:2])
        centre = viewer.scene.camera.center
        corner = np.array([1.0, 1.0])  # far from the heart
        _drive(e, viewer, _Ev(_world(e, t, z, corner), screen=(10, 10)), [(_world(e, t, z, corner), (60, 40))])
        assert viewer.scene.camera.center == centre and e.pending is None  # editing never pans
        panel.btn_pan.setChecked(True)  # Pan on: left-drag moves the image
        _drive(e, viewer, _Ev(_world(e, t, z, corner), screen=(10, 10)), [(_world(e, t, z, corner), (60, 40))])
        assert viewer.scene.camera.center != centre
        panel.btn_pan.setChecked(False)
        centre = viewer.scene.camera.center
        # a drag that starts a little off a dot (beyond the click radius, inside the grab radius) moves that dot
        key = (t, z, "endo")
        p0 = e.curves[key][0].copy()
        c = e.curves[key].mean(0)
        off = p0 + (p0 - c) / np.linalg.norm(p0 - c) * e._tol_px() * 1.6
        _drive(e, viewer, _Ev(_world(e, t, z, off), screen=(10, 10)), [(_world(e, t, z, off + 2), (30, 30))])
        assert not np.allclose(e.curves[key][0], p0) and viewer.scene.camera.center == centre  # no pan

        # drawing: a misplaced point can be dragged before the contour is closed
        panel.clear_slice()  # endo gone on this slice
        ring = [c + 4 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 5)[:-1]]
        for q in ring:
            _drive(e, viewer, _Ev(_world(e, t, z, q)))
        first = np.array(e.pending[3][0])
        _drive(e, viewer, _Ev(_world(e, t, z, first)), [(_world(e, t, z, first + 1.5), (20, 20))])
        assert np.allclose(e.pending[3][0], first + 1.5) and len(e.pending[3]) == 4
    finally:
        viewer.close()


def test_series_panel_lists_study_and_switches(tmp_path):
    from test_study import _write

    _write(tmp_path / "a", "cine_sax", 3, zs=range(4))
    _write(tmp_path / "b", "cine_4ch", 4, zs=[0], orient=[1, 0, 0, 0, 0, 1])
    for k in range(3):  # one cine per slice, as in Segment workflows
        _write(tmp_path / f"s{k}", f"SAX_S-{k}", 10 + k, zs=[[2, 0, 1][k]])
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.open_study(tmp_path)
        # stack row, the multi-slice SAX and the 4CH (the stack's slices are in the slice list)
        assert panel.series_panel.list.count() == 3
        assert panel.list.topLevelItemCount() == 3
        s = panel.session
        assert s.image.shape[2] == 3 and s.separate_slices and panel.chk_per_slice.isChecked()
        assert s.slice_names == ["11 SAX_S-1", "12 SAX_S-2", "10 SAX_S-0"]  # sorted by slice position
        # clicking a slice row shows that slice of the stack
        part = next(x for x in panel.study if x.stack is not None and x.slice_index == 2)
        panel.open_series(panel.study.index(part))
        assert panel.current_slice() == 2
        # per-slice ED/ES: set ES on this slice only
        panel._show_frame(3)
        panel.btn_set_es.click()
        assert s.t_for(2, "ES") == 3 and s.t_for(0, "ES") == 0
        panel.timebar.grab()  # paints without error
        # an edit survives switching away and back
        viewer.layers["contours"].data[0, 1, 2, 2] = 1
        stack_i = panel.current_series
        lax = next(i for i, x in enumerate(panel.study) if x.kind == "Long axis")
        panel.open_series(lax)
        assert panel.session.image.shape[2] == 1
        panel.open_series(stack_i)
        assert panel.session.label[2, 2, 1, 0] == 1 and panel.session.t_for(2, "ES") == 3
    finally:
        viewer.close()



def test_autocontour_only_the_frame_being_viewed(monkeypatch):
    """Default: ED = ES = frame 0 until set; Auto-contour does the viewed frame only (one model pass)."""
    from tracecmr import autocontour as ac

    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    img = np.stack([ed.image, ed.image, es.image, ed.image], -1)
    truth = np.stack([ed.label, ed.label, es.label, ed.label], -1)
    calls = []

    def fake_pairs(image, spacing, pairs, model="ucl"):
        calls.append(list(pairs))
        return np.stack([truth[:, :, z, t] for z, t in pairs], axis=2)

    monkeypatch.setattr(ac, "available", lambda: {"ucl": "ready", "calico": "not installed"})
    monkeypatch.setattr(ac, "run_pairs", fake_pairs)
    s = Session(image=img, label=np.zeros(img.shape, np.uint8), spacing=ed.spacing, subject_id="raw")
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel._testing = True
        panel.load_session(s)
        assert s.ed_frame == 0 and s.es_frame == 0
        panel._show_frame(2)
        panel.autocontour(sync=True)
        assert calls == [[(z, 2) for z in range(img.shape[2])]]  # every slice, only frame 2
        assert s.label[..., 2].any() and not s.label[..., [0, 1, 3]].any()
        assert panel.current_frame() == 2  # stays on the frame just contoured
        # per-slice mode at a slice's ES frame: each slice's own ES
        panel.chk_per_slice.setChecked(True)
        s.set_phase_frame("ES", 2)
        s.set_phase_frame("ES", 3, z=1)
        assert panel.autocontour_pairs()[1] == (1, 3) and panel.autocontour_pairs()[0] == (0, 2)
    finally:
        viewer.close()


def test_hand_drawn_rv_and_rv_volumes():
    from tracecmr import labels as L

    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    lab = np.stack([ed.label, es.label], -1)
    lab[2:10, 2:12, 1:7, 0] = L.RV_BLOODPOOL  # an RV beside the LV, bigger at ED
    lab[2:8, 2:12, 1:7, 1] = L.RV_BLOODPOOL
    s = Session(image=np.stack([ed.image, es.image], -1), label=lab, spacing=ed.spacing, subject_id="rv",
                papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        z = int(viewer.dims.current_step[1])
        assert (0, z, "rv") in panel.editor.curves  # yellow RV curve is editable
        panel.btn_rv.click()
        assert set(k[2] for k in panel.editor._keys) == {"rv"}
        panel.compute()
        assert panel.box_rv.isVisibleTo(panel) and float(panel.rv_labels["rvef_pct"].text()) > 0
        assert not hasattr(panel, "btn_auto_rv")  # no RV model in the app
    finally:
        viewer.close()

def test_contour_focus_and_per_slice_results():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        e = panel.editor
        t, z = (int(v) for v in viewer.dims.current_step[:2])
        assert set(k[2] for k in e._keys) == {"endo"}  # only the selected contour's dots are shown
        panel.btn_epi.click()
        assert set(k[2] for k in e._keys) == {"epi"} and e.add_to == "epi"
        # with Epi selected, a click on an endocardium dot grabs nothing
        p = e.curves[(t, z, "endo")][0]
        assert e.hit(t, z, p, 0.5) is None or e.hit(t, z, p, 0.5)[1][2] == "epi"
        panel._smooth_points()
        assert e.edited == {(t, z, "epi")}
        # per-slice table: one row per slice with ED and ES, plus the total
        panel.compute()
        rows = s.per_slice()
        assert panel.box_slices.isVisibleTo(panel) and panel.tbl_slices.rowCount() == len(rows) + 1
        assert sum(r["edv_ul"] for r in rows if r["included"]) == pytest.approx(
            float(panel.res_labels["edv_ul"].text()), rel=0.02)
        # leaving a slice out greys it and changes the total
        item = panel.list.topLevelItem(1)
        zz = item.data(0, Qt.ItemDataRole.UserRole)
        item.setCheckState(1, Qt.CheckState.Unchecked)
        panel._refresh_slice_table()
        assert not next(r for r in s.per_slice() if r["slice"] == zz)["included"]
        total = float(panel.tbl_slices.item(panel.tbl_slices.rowCount() - 1, 1).text())
        assert total == pytest.approx(sum(r["edv_ul"] for r in s.per_slice() if r["slice"] in s.included["ED"]), abs=0.1)
    finally:
        viewer.close()


def test_save_and_reload_contours_and_data_folder(tmp_path, monkeypatch):
    from qtpy.QtCore import QSettings
    from test_study import _write

    monkeypatch.setattr(QSettings, "value", lambda self, key, *a, **k: {
        "folders/contours": str(tmp_path / "saved"), "folders/data": str(tmp_path / "data")}.get(key))
    monkeypatch.setattr(QSettings, "setValue", lambda self, *a: None)
    for k in range(3):
        _write(tmp_path / "data" / "MouseA" / f"s{k}", f"SAX_S-{k}", 10 + k, zs=[k])
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        assert panel.series_panel.mice.count() == 1  # MouseA listed from the data folder
        panel.open_path(tmp_path / "data" / "MouseA")
        s = panel.session
        assert s.subject_id.startswith("MouseA") and s.image.shape[2] == 3
        s.label[2:6, 2:6, 1, 0] = 1  # some contour
        panel.viewer.layers["contours"].data = s.label.transpose(3, 2, 1, 0).copy()
        panel._show_frame(3)
        panel.chk_per_slice.setChecked(True)
        panel.set_phase_frame("ES", 3)
        path = panel.save_contours()
        assert path is not None and path.exists()
        assert "✓" in panel.series_panel.mice.item(0).text()
        # reopen: contours and the per-slice ES come back
        panel.open_path(tmp_path / "data" / "MouseA")
        s2 = panel.session
        assert s2 is not s and s2.label[3, 3, 1, 0] == 1 and s2.t_for(panel.current_slice(), "ES") == 3
        assert "Loaded your saved contours" in panel.lbl_auto.text()
        # no slice scrollbar under the image
        assert not viewer.window._qt_viewer.dims.isVisible()
    finally:
        viewer.close()


def test_results_kept_apart_from_raw_scans(tmp_path, monkeypatch):
    """Contours, tables and images go to <results>/<mouse>/, never into the raw scans folder."""
    from qtpy.QtCore import QSettings
    from test_study import _write

    store = {"folders/contours": str(tmp_path / "results"), "folders/data": str(tmp_path / "data")}
    monkeypatch.setattr(QSettings, "value", lambda self, key, *a, **k: store.get(key))
    monkeypatch.setattr(QSettings, "setValue", lambda self, key, v: store.__setitem__(key, v))
    for k in range(3):
        _write(tmp_path / "data" / "MouseA" / f"s{k}", f"SAX_S-{k}", 10 + k, zs=[k])
    raw_before = {p: p.stat().st_mtime for p in (tmp_path / "data").rglob("*")}
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.open_path(tmp_path / "data" / "MouseA")
        s = panel.session
        s.label[4:12, 4:12, :, 0] = 2
        s.label[6:10, 6:10, :, 0] = 1
        panel.viewer.layers["contours"].data = s.label.transpose(3, 2, 1, 0).copy()
        path = panel.save_contours()
        folder = panel.results_dir()
        assert path.parent == folder and folder.parent == tmp_path / "results"
        imgs = sorted(p.name for p in (folder / "images").glob("*.png"))
        assert "montage_ED.png" in imgs and any(n.startswith("slice") for n in imgs)
        # the raw scans are untouched and nothing new appeared among them
        assert {p: p.stat().st_mtime for p in (tmp_path / "data").rglob("*")} == raw_before
        # results inside the raw folder (or vice versa) are refused
        assert not panel.set_results_dir(tmp_path / "data" / "out")
        assert store["folders/contours"] == str(tmp_path / "results")
        # raw view hides the contours without deleting them; H/Raw again brings them back
        panel.btn_raw.setChecked(True)
        assert not viewer.layers["curves"].visible and not viewer.layers["control points"].visible
        assert panel.session.label.any()
        panel.btn_raw.setChecked(False)
        assert viewer.layers["curves"].visible
        # files saved under the old name (VentriMap) still load
        old_name = path.with_name(path.name.replace(".tracecmr.npz", ".ventrimap.npz"))
        path.rename(old_name)
        panel.open_path(tmp_path / "data" / "MouseA")
        assert panel.session.label.any() and "Loaded your saved contours" in panel.lbl_auto.text()
        path = panel._find_saved(panel.session)
        # contours saved by older versions (flat file in the results folder) still load
        legacy = panel._legacy_contours_path(s)
        path.rename(legacy)
        panel.open_path(tmp_path / "data" / "MouseA")
        assert panel.session.label.any() and "Loaded your saved contours" in panel.lbl_auto.text()
    finally:
        viewer.close()


def test_auto_lv_never_overwrites_finished_slices(monkeypatch):
    """Contour slice 2 (with an RV) by hand, go to another slice, Auto LV: slice 2 stays as it was."""
    from tracecmr import autocontour as ac
    from tracecmr import labels as LB

    ed = tapered_phantom(n_slices=6, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    img = np.stack([ed.image, ed.image], -1)
    calls = []

    def fake_pairs(image, spacing, pairs, model="ucl"):
        calls.append(list(pairs))
        return np.stack([ed.label[:, :, z] for z, t in pairs], axis=2)

    monkeypatch.setattr(ac, "available", lambda: {"ucl": "ready"})
    monkeypatch.setattr(ac, "run_pairs", fake_pairs)
    lab = np.zeros(img.shape, np.uint8)
    mine = ed.label[:, :, 2].copy()
    mine[mine == LB.LV_BLOODPOOL] = LB.LV_MYOCARDIUM  # 'my' slice 2: different from what the model gives
    mine[2:6, 2:6] = LB.RV_BLOODPOOL
    lab[:, :, 2, 1] = mine
    s = Session(image=img, label=lab, spacing=ed.spacing, subject_id="m")
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel._testing = True
        panel.load_session(s)
        panel._show_frame(1)
        step = list(viewer.dims.current_step); step[1] = 3; viewer.dims.current_step = tuple(step)
        panel.autocontour(sync=True)
        s = panel.session
        assert 2 not in [z for z, _ in calls[-1]]  # slice 2 was not sent to the model
        assert np.array_equal(s.label[:, :, 2, 1], mine)  # and is unchanged, RV included
        assert s.label[:, :, 3, 1].any()
        # every slice done: nothing happens unless the user asks to redo the viewed slice
        n = len(calls)
        panel.autocontour(sync=True, redo_current=False)
        assert len(calls) == n
        step[1] = 2; viewer.dims.current_step = tuple(step)
        panel.autocontour(sync=True, redo_current=True)
        assert calls[-1] == [(2, 1)]
        assert (s.label[:, :, 2, 1] == LB.RV_BLOODPOOL).any()  # hand-drawn RV kept
        # Ctrl+Z reverts the redo
        panel.undo()
        assert np.array_equal(panel.session.label[:, :, 2, 1], mine)
    finally:
        viewer.close()


def test_results_are_live_and_contrast_controls():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        panel._refresh_slice_table()  # what the timer does
        assert panel.res_labels["ef_pct"].text() not in ("–", "")  # no Compute press needed
        layer = viewer.layers["image"]
        base = tuple(layer.contrast_limits)
        panel.sld_contrast.setValue(50)
        c = tuple(layer.contrast_limits)
        assert (c[1] - c[0]) < (base[1] - base[0])  # more contrast = narrower window
        panel.sld_brightness.setValue(50)
        assert tuple(layer.contrast_limits)[0] < c[0]  # brighter = window moves down
        panel.adjust_contrast(10, -10)
        assert panel.sld_contrast.value() == 60 and panel.sld_brightness.value() == 40
        panel._reset_contrast()
        assert np.allclose(layer.contrast_limits, base)
    finally:
        viewer.close()


def test_save_as_asks_once_then_remembers(tmp_path, monkeypatch):
    """First save opens a save window (rename / choose folder); later saves go to the same file; reopening
    the mouse loads from wherever it was saved."""
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QFileDialog
    from test_study import _write

    store = {"folders/contours": str(tmp_path / "results"), "folders/data": str(tmp_path / "data")}
    monkeypatch.setattr(QSettings, "value", lambda self, key, *a, **k: store.get(key))
    monkeypatch.setattr(QSettings, "setValue", lambda self, key, v: store.__setitem__(key, v))
    asked = []
    custom = tmp_path / "elsewhere" / "My mouse A.tracecmr.npz"

    def fake_dialog(*a, **k):
        asked.append(a[2])  # the suggested path
        return str(custom), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
    for k in range(3):
        _write(tmp_path / "data" / "MouseA" / f"s{k}", f"SAX_S-{k}", 10 + k, zs=[k])
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel._testing = True
        panel.open_path(tmp_path / "data" / "MouseA")
        s = panel.session
        s.label[4:12, 4:12, :, 0] = 2
        s.label[6:10, 6:10, :, 0] = 1
        panel.viewer.layers["contours"].data = s.label.transpose(3, 2, 1, 0).copy()
        assert panel._save_clicked() == custom and custom.exists() and len(asked) == 1
        assert str(tmp_path / "results") in asked[0]  # suggested: the mouse's results folder
        assert (custom.parent / "My mouse A_images").is_dir()
        panel._save_clicked()  # Ctrl+S again: no window, same file
        assert len(asked) == 1
        panel._save_clicked(save_as=True)  # Save as: asks again
        assert len(asked) == 2
        panel.open_path(tmp_path / "data" / "MouseA")
        assert panel.session.label.any() and custom.name in panel.lbl_auto.text()
    finally:
        viewer.close()


def test_auto_lv_marks_base_and_apex_until_edited(monkeypatch):
    from tracecmr import autocontour as ac

    ed = tapered_phantom(n_slices=6, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    img = np.stack([ed.image, ed.image], -1)
    monkeypatch.setattr(ac, "available", lambda: {"ucl": "ready"})
    monkeypatch.setattr(ac, "run_pairs", lambda image, spacing, pairs, model="ucl":
                        np.stack([ed.label[:, :, z] for z, t in pairs], axis=2))
    s = Session(image=img, label=np.zeros(img.shape, np.uint8), spacing=ed.spacing, subject_id="m")
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel._testing = True
        panel.load_session(s)
        panel.autocontour(sync=True)
        s = panel.session
        zs = [z for z in s.order if s.label[:, :, z, 0].any()]
        base, apex = zs[0], zs[-1]
        assert s.flags[(0, base)].startswith("base slice") and s.flags[(0, apex)].startswith("apex slice")
        assert any(it.text(0).startswith("⚠") for it in panel._items())
        # edit the base slice: its note clears
        e = panel.editor
        key = next(k for k in e.curves if k[:2] == (0, base))
        e.move_point(key, 0, e.curves[key][0] + 0.5)
        panel._sync_labels()
        assert (0, base) not in s.flags and (0, apex) in s.flags
    finally:
        viewer.close()


def test_grow_and_shrink_contour():
    """Grow pushes the selected outline out (never past the epi), Shrink pulls it in, Undo reverts, dots stay hidden."""
    from tracecmr.contours import polygon_area

    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        e = panel.editor
        z = int(viewer.dims.current_step[1])
        endo, epi = (0, z, "endo"), (0, z, "epi")
        a0, epi0 = polygon_area(e.curves[endo]), e.curves[epi].copy()
        panel._smooth_points()  # dots hidden: growing keeps them hidden
        panel.btn_grow.click()
        a1 = polygon_area(e.curves[endo])
        assert a1 > a0 and endo in e.hidden and endo in e.edited
        assert np.allclose(e.curves[epi], epi0)  # only the selected contour moves
        panel.btn_shrink.click()
        panel.btn_shrink.click()
        assert polygon_area(e.curves[endo]) < a1
        assert e.undo() and e.undo()
        assert np.isclose(polygon_area(e.curves[endo]), a1)
        for _ in range(40):  # never past the epicardium
            panel.grow_contour(0.5)
        from tracecmr.app.curve_editor import _inside
        assert _inside(e.curves[epi], e.curves[endo]).all()
        assert "further" in panel.lbl_auto.text() or "Grew" in panel.lbl_auto.text()
        panel._sync_labels()
        lab = s.label[:, :, z, 0]
        assert (lab == 2).any()  # some myocardium is left
        panel.cmb_add.setCurrentIndex(1)  # Epi: grows outward
        a_epi = polygon_area(e.curves[epi])
        panel.btn_grow.click()
        assert polygon_area(e.curves[epi]) > a_epi
    finally:
        viewer.close()


def test_offset_points_either_winding():
    from tracecmr.app.curve_editor import offset_points
    from tracecmr.contours import polygon_area

    th = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    c = np.column_stack([10 + 5 * np.cos(th), 10 + 5 * np.sin(th)])
    for pts in (c, c[::-1]):
        assert polygon_area(offset_points(pts, 1.0)) > polygon_area(pts)
        assert polygon_area(offset_points(pts, -1.0)) < polygon_area(pts)
        assert np.allclose(np.hypot(*(offset_points(pts, 1.0) - 10).T), 6.0, atol=0.05)


def test_fit_button_pulls_an_oversized_endo_back_to_the_blood():
    from tracecmr.contours import polygon_area

    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        e = panel.editor
        z = int(viewer.dims.current_step[1])
        key = (0, z, "endo")
        a0 = polygon_area(e.curves[key])
        for _ in range(6):
            panel.grow_contour(0.5)
        assert polygon_area(e.curves[key]) > a0 * 1.15
        panel.btn_fit.click()
        assert abs(polygon_area(e.curves[key]) - a0) / a0 < 0.12, panel.lbl_auto.text()
        assert "gave" in panel.lbl_auto.text()
        assert e.undo()
        assert polygon_area(e.curves[key]) > a0 * 1.15
    finally:
        viewer.close()


def test_wall_button_draws_thickness_lines_on_the_slice():
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        active = viewer.layers.selection.active
        panel.btn_wall.setChecked(True)
        layer = viewer.layers["wall thickness"]
        assert layer.visible and 30 <= len(layer.data) <= 45
        assert viewer.layers.selection.active is active  # editing keeps working
        t, z = (int(v) for v in viewer.dims.current_step[:2])
        lengths = []
        dz, dy, dx = layer.scale[1:]
        for d in layer.data:
            assert int(d[0, 0]) == t and int(d[0, 1]) == z
            lengths.append(np.hypot((d[1, 2] - d[0, 2]) * dy, (d[1, 3] - d[0, 3]) * dx))
        assert np.median(lengths) == pytest.approx(1.0, abs=0.08)  # the phantom wall is 1 mm
        assert "mean" in panel.lbl_auto.text()
        panel.step_slice(1)
        assert all(int(d[0, 1]) != z for d in layer.data)  # follows the slice
        panel.set_wall_mode("thickening")  # phantom: ED wall 1.0 mm, ES wall 1.3 mm -> about +30% all round
        assert 30 <= len(layer.data) <= 45
        import re
        m = re.search(r"mean (-?\d+)%", panel.lbl_auto.text())
        assert m and 22 <= int(m.group(1)) <= 38, panel.lbl_auto.text()
        assert "thinning" not in panel.lbl_auto.text()
        panel.set_wall_mode("thickness")
        assert "mm" in panel.lbl_auto.text() and "Thickening" not in panel.lbl_auto.text()
        panel.btn_wall.setChecked(False)
        assert not layer.visible
    finally:
        viewer.close()


def test_undo_after_live_results_puts_the_labels_back():
    """Fit (or any edit), let the results update (labels synced), then Undo: the labels, results and wall lines
    must follow the undone outline, not keep the edit."""
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        z = int(viewer.dims.current_step[1])
        before = int((s.label[:, :, z, 0] == 1).sum())
        for _ in range(6):
            panel.grow_contour(0.5)
        panel.compute()  # live results: the edit is written into the labels
        grown = int((panel.session.label[:, :, z, 0] == 1).sum())
        assert grown > before * 1.1
        edv_grown = panel.last[0].as_row()["edv_ul"]
        for _ in range(6):
            panel.undo()
        panel.compute()
        after = int((panel.session.label[:, :, z, 0] == 1).sum())
        assert abs(after - before) <= max(4, 0.03 * before), (before, grown, after)
        assert panel.last[0].as_row()["edv_ul"] < edv_grown
    finally:
        viewer.close()


def test_set_septum_by_click_without_an_rv(tmp_path):
    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        panel.compute()
        assert not panel.last[1].ed.reference.startswith("septum")  # no RV: image axis
        t, z = (int(v) for v in viewer.dims.current_step[:2])
        panel.start_set_septum()
        assert panel.btn_wall.isChecked() and panel.editor.click_hook is not None
        from scipy import ndimage as ndi

        cx, cy = ndi.center_of_mass(s.label[:, :, z, t] == 1)
        before = {k: v.copy() for k, v in panel.editor.curves.items()}
        _drive(panel.editor, viewer, _Ev(_world(panel.editor, t, z, (cx - 15, cy))))  # a real click, left of the cavity
        assert panel.editor.click_hook is None
        assert all(np.array_equal(before[k], panel.editor.curves[k]) for k in before)  # the click edited nothing
        assert abs(((s.septum_manual[z] - 180) + 180) % 360 - 180) < 2
        wall = panel.last[1]
        assert wall.ed.reference.startswith("septum")
        assert any("by hand" in f for f in wall.flags)
        assert "set by hand" in panel.lbl_auto.text()
        path = s.save_contours(tmp_path / "m.tracecmr.npz")
        s2 = Session(image=s.image, label=np.zeros_like(s.label), spacing=s.spacing)
        s2.load_contours(path)
        assert s2.septum_manual == s.septum_manual
        panel.start_set_septum()
        panel.cancel_click()
        assert panel.editor.click_hook is None
        panel.reset_septum(all_slices=True)
        assert not s.septum_manual and not panel.last[1].ed.reference.startswith("septum")
    finally:
        viewer.close()


def test_reopening_keeps_the_exact_points_no_creeping_tips(tmp_path):
    """Open, edit, save, reopen, many times: a sharp-tipped RV must stay exactly as drawn (rebuilding the points
    from pixels on every open used to round the tips a little each time). Smoothed slices stay smoothed."""
    from tracecmr.app.session import Session as S

    ed = tapered_phantom(n_slices=8, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=8, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    s = Session(image=np.stack([ed.image, es.image], -1), label=np.stack([ed.label, es.label], -1),
                spacing=ed.spacing, subject_id="phantom", papillary="blood_pool", ed_frame=0, es_frame=1)
    viewer = napari.Viewer(show=False)
    try:
        panel = TraceCMRWidget(viewer)
        panel.load_session(s)
        e = panel.editor
        t, z = (int(v) for v in viewer.dims.current_step[:2])
        epi = e.curves[(t, z, "epi")]
        c = epi.mean(axis=0)
        r = np.hypot(*(epi - c).T).mean()
        ang = np.linspace(200, 340, 9)  # a thin crescent with sharp tips below the LV
        outer = np.column_stack([c[0] + (r + 7) * np.cos(np.deg2rad(ang)), c[1] - (r + 7) * np.sin(np.deg2rad(ang))])
        inner = np.column_stack([c[0] + (r + 2) * np.cos(np.deg2rad(ang[::-1][1:-1])),
                                 c[1] - (r + 2) * np.sin(np.deg2rad(ang[::-1][1:-1]))])
        rv = np.vstack([outer, inner])
        e.push_undo()
        e.curves[(t, z, "rv")] = rv.copy()
        e.edited.add((t, z, "rv"))
        panel.cmb_add.setCurrentIndex(1)
        panel._smooth_points()  # epi smoothed: its points hidden
        path = tmp_path / "m.tracecmr.npz"
        for cycle in range(5):
            panel.save_contours(quiet=True, path=path)
            s2 = S(image=s.image, label=np.zeros_like(s.label), spacing=s.spacing, subject_id="phantom")
            s2.load_contours(path)
            panel.load_session(s2)
            e = panel.editor
            viewer.dims.current_step = (t, z, *viewer.dims.current_step[2:])
            assert np.allclose(e.curves[(t, z, "rv")], rv, atol=1e-4), f"RV moved after reopen {cycle + 1}"
            assert (t, z, "epi") in e.hidden  # still smoothed
            ek = (t, z, "endo")
            e.move_point(ek, 0, e.curves[ek][0] + 0.01)  # an edit on the slice: it is re-rendered from the points
        # a slice changed outside the editor after saving (e.g. Auto LV) falls back to its pixels
        s3 = S(image=s.image, label=np.zeros_like(s.label), spacing=s.spacing)
        s3.load_contours(path)
        s3.label[:, :, z, t] = 0
        assert not any(k[:2] == (t, z) for k in s3.curves_still_valid())
    finally:
        viewer.close()
