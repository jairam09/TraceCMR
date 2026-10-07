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
