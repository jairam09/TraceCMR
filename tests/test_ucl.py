"""Tests for the UCL loader.

Most tests build a tiny fake file in the UCL format, so they run anywhere.
The last test runs on a real UCL file only if you point to the data folder:
    VENTRIMAP_UCL_DATA=data/ucl/Data pytest tests/test_ucl.py
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from ventrimap import labels as L
from ventrimap.datasets import UCL_SPACING, find_ed_es, labeled_frames, load_ucl
from ventrimap.synthetic import ring_phantom


def make_fake_ucl(path: Path, ed_frame=0, es_frame=3, n_frames=6) -> Path:
    """Write a UCL-format file: big LV at ED, small LV at ES, other frames unlabeled."""
    ed = ring_phantom(shape=(64, 64, 5), spacing=UCL_SPACING, blood_radius_mm=2.0)
    es = ring_phantom(shape=(64, 64, 5), spacing=UCL_SPACING, blood_radius_mm=1.0, wall_mm=1.5)
    images = np.zeros((64, 64, 5, n_frames), dtype=np.uint16)
    masks = np.zeros_like(images, dtype=np.uint8)
    to_ucl = {L.BACKGROUND: 0, L.LV_MYOCARDIUM: 1, L.LV_BLOODPOOL: 2}
    for t, ph in ((ed_frame, ed), (es_frame, es)):
        images[..., t] = (ph.image * 1000).astype(np.uint16)
        for std, ucl in to_ucl.items():
            masks[..., t][ph.label == std] = ucl
    with h5py.File(path, "w") as f:
        f["Images"] = images
        f["Masks"] = masks
        f["Frames_with_ROI"] = np.array(sorted([ed_frame, es_frame]), dtype=np.int32)
    return path


def test_labeled_frames_and_ed_es(tmp_path):
    p = make_fake_ucl(tmp_path / "Mouse1.h5", ed_frame=4, es_frame=1)
    assert labeled_frames(p) == [1, 4]
    assert find_ed_es(p) == {"ED": 4, "ES": 1}


def test_partial_artifact_frame_is_not_chosen_as_es(tmp_path):
    """Some UCL mice have extra frames contoured on only a few slices (Mouse91).

    A frame with 2 of 5 slices contoured has the smallest blood pool but is not ES.
    """
    p = make_fake_ucl(tmp_path / "Mouse91.h5", ed_frame=0, es_frame=3)
    with h5py.File(p, "r+") as f:
        masks = f["Masks"][:]
        masks[..., 4] = masks[..., 3]
        masks[:, :, 2:, 4] = 0  # artifact frame: only 2 of 5 slices contoured
        f["Masks"][...] = masks
        del f["Frames_with_ROI"]
        f["Frames_with_ROI"] = np.array([0, 3, 4], dtype=np.int32)
    assert find_ed_es(p) == {"ED": 0, "ES": 3}


def test_load_ed_remaps_labels_and_sets_spacing(tmp_path):
    p = make_fake_ucl(tmp_path / "Mouse1.h5")
    ci = load_ucl(p, "ED")
    assert ci.subject_id == "Mouse1" and ci.phase == "ED"
    assert ci.spacing == UCL_SPACING
    assert ci.image.shape == (64, 64, 5)
    present = ci.labels_present()
    assert set(present) == {L.BACKGROUND, L.LV_MYOCARDIUM, L.LV_BLOODPOOL}
    # ED blood pool is bigger than ES blood pool
    es = load_ucl(p, "ES")
    assert present[L.LV_BLOODPOOL] > es.labels_present()[L.LV_BLOODPOOL]


def test_unlabeled_frame_has_no_label(tmp_path):
    p = make_fake_ucl(tmp_path / "Mouse1.h5")
    ci = load_ucl(p, 2)
    assert ci.label is None and ci.phase == "frame2"


def test_bad_frame_raises(tmp_path):
    p = make_fake_ucl(tmp_path / "Mouse1.h5")
    with pytest.raises(IndexError):
        load_ucl(p, 99)
    with pytest.raises(ValueError):
        load_ucl(p, "mid")


UCL_DIR = os.environ.get("VENTRIMAP_UCL_DATA")


@pytest.mark.skipif(not UCL_DIR, reason="set VENTRIMAP_UCL_DATA to a folder of UCL .h5 files")
def test_real_ucl_files_are_plausible():
    files = sorted(Path(UCL_DIR).glob("Mouse*.h5"))
    assert files, f"no Mouse*.h5 files in {UCL_DIR}"
    for p in files:
        ed, es = load_ucl(p, "ED"), load_ucl(p, "ES")
        edv = ed.labels_present()[L.LV_BLOODPOOL] * ed.voxel_volume_mm3
        esv = es.labels_present()[L.LV_BLOODPOOL] * es.voxel_volume_mm3
        ef = (edv - esv) / edv
        # Broad physiological ranges for mice, including disease models.
        assert 15 < edv < 300, f"{p.name}: EDV {edv:.1f} uL"
        assert 0.05 < ef < 0.9, f"{p.name}: EF {ef:.2f}"
