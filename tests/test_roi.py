"""Regions of interest: masks, numbers, contrast-to-noise, and that they are saved with the mouse."""

import numpy as np

from tracecmr.app.roi import cnr, outline, roi_mask, roi_stats, snr


def test_circle_mask_area_and_stats():
    img = np.zeros((40, 40))
    img[:, :] = 100.0
    roi = {"kind": "circle", "t": 0, "z": 0, "cx": 20.0, "cy": 20.0, "r_mm": 1.0}
    m = roi_mask(roi, img.shape, (0.1, 0.1))  # radius 10 px
    assert abs(m.sum() - np.pi * 100) / (np.pi * 100) < 0.03
    st = roi_stats(img, m, (0.1, 0.1))
    assert st["mean"] == 100 and st["sd"] == 0 and abs(st["area_mm2"] - np.pi * 0.01 * 100) < 0.1


def test_circle_is_round_in_millimetres_with_rectangular_pixels():
    m = roi_mask({"kind": "circle", "cx": 20, "cy": 20, "r_mm": 1.0}, (60, 60), (0.1, 0.2))
    xs, ys = np.nonzero(m)
    assert (xs.max() - xs.min()) > 1.8 * (ys.max() - ys.min())  # twice as many pixels across x as y


def test_freehand_square():
    roi = {"kind": "freehand", "pts": [[5, 5], [15, 5], [15, 15], [5, 15]]}
    m = roi_mask(roi, (30, 30), (0.1, 0.1))
    assert 80 <= m.sum() <= 121
    assert m[10, 10] and not m[2, 2]
    assert outline(roi, (0.1, 0.1)).shape == (4, 2)


def test_contrast_and_signal_to_noise():
    rng = np.random.default_rng(0)
    img = np.zeros((60, 60))
    img[:20] = 800 + rng.normal(0, 5, (20, 60))  # "scar"
    img[20:40] = 300 + rng.normal(0, 5, (20, 60))  # "remote"
    img[40:] = rng.normal(0, 50, (20, 60))  # "air": noise SD 50
    sp = (0.1, 0.1)
    box = lambda x0: {"kind": "freehand", "pts": [[x0 + 2, 5], [x0 + 16, 5], [x0 + 16, 55], [x0 + 2, 55]]}
    a, b, n = (roi_stats(img, roi_mask(box(x), img.shape, sp), sp) for x in (0, 20, 40))
    assert abs(cnr(a["mean"], b["mean"], n["sd"]) - 10) < 0.7
    assert abs(snr(a["mean"], n["sd"]) - 16) < 1.2
    assert np.isnan(cnr(1, 2, 0))


def test_rois_are_saved_with_the_contours(tmp_path):
    from tracecmr.app.session import Session
    from tracecmr.synthetic import tapered_phantom

    ed = tapered_phantom(n_slices=4, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    s = Session(image=np.stack([ed.image, ed.image], -1), label=np.stack([ed.label, ed.label], -1),
                spacing=ed.spacing, subject_id="p", papillary="blood_pool", ed_frame=0, es_frame=1)
    s.rois = [{"name": "scar", "kind": "circle", "t": 0, "z": 1, "cx": 10.0, "cy": 12.0, "r_mm": 0.3,
               "colour": "#FF9F40"}]
    path = s.save_contours(tmp_path / "p.tracecmr.npz")
    s2 = Session(image=s.image, label=np.zeros_like(s.label), spacing=s.spacing, subject_id="p",
                 papillary="blood_pool", ed_frame=0, es_frame=1)
    s2.load_contours(path)
    assert s2.rois == s.rois
