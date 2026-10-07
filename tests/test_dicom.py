"""DICOM reader tests on synthetic series with known geometry.

Each frame's pixels encode its own slice and time index, so the tests can check
that sorting put every frame in the right place.
"""

import random

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from mcmr.dicom import load_dicom_cine

ROWS, COLS = 20, 24  # non-square so a transposition bug would show
ORIENT = [1, 0, 0, 0, 1, 0]  # rows along x, columns along y -> normal along z


def _base(sop_class="1.2.840.10008.5.1.4.1.1.4"):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.Rows, ds.Columns = ROWS, COLS
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    return ds


def _pixels(z, t):
    arr = np.full((ROWS, COLS), 100 * z + t, dtype=np.uint16)
    arr[0, 1] = 999  # marker at row 0, column 1 -> x=1, y=0 after loading
    return arr


def write_classic_series(folder, n_slices=4, n_times=5, gap=0.2, thickness=0.8, spacing=(0.12, 0.1)):
    folder.mkdir(parents=True, exist_ok=True)
    order = [(z, t) for z in range(n_slices) for t in range(n_times)]
    random.Random(0).shuffle(order)  # files on disk in random order
    for i, (z, t) in enumerate(order):
        ds = _base()
        ds.ImageOrientationPatient = ORIENT
        ds.ImagePositionPatient = [0.0, 0.0, z * (thickness + gap)]
        ds.PixelSpacing = list(spacing)  # (row spacing, column spacing)
        ds.SliceThickness = thickness
        ds.TriggerTime = 10.0 * t
        ds.InstanceNumber = i + 1
        ds.PixelData = _pixels(z, t).tobytes()
        ds.save_as(folder / f"IM_{i:04d}.dcm", enforce_file_format=True)
    return folder


def test_classic_series_sorted_into_slices_and_times(tmp_path):
    cine = load_dicom_cine(write_classic_series(tmp_path / "s"))
    assert cine.data.shape == (COLS, ROWS, 4, 5)  # (x, y, slice, time)
    for z in range(4):
        for t in range(5):
            assert cine.data[5, 5, z, t] == 100 * z + t
    assert cine.data[1, 0, 0, 0] == 999  # orientation preserved, not transposed
    assert np.allclose(cine.times_ms, [0, 10, 20, 30, 40])


def test_spacing_from_headers_includes_gap(tmp_path):
    cine = load_dicom_cine(write_classic_series(tmp_path / "s", spacing=(0.12, 0.1)))
    dx, dy, dz = cine.spacing
    assert dx == pytest.approx(0.1)  # column spacing -> x
    assert dy == pytest.approx(0.12)  # row spacing -> y
    assert dz == pytest.approx(1.0)  # 0.8 thickness + 0.2 gap


def test_frame_returns_cardiac_image(tmp_path):
    cine = load_dicom_cine(write_classic_series(tmp_path / "s"))
    ci = cine.frame(2, papillary="blood_pool")
    assert ci.image.shape == (COLS, ROWS, 4)
    assert ci.papillary == "blood_pool"
    assert ci.image[5, 5, 3] == 302
    with pytest.raises(IndexError):
        cine.frame(9)


def test_incomplete_series_refused(tmp_path):
    folder = write_classic_series(tmp_path / "s")
    sorted(folder.glob("*.dcm"))[0].unlink()  # one frame missing
    with pytest.raises(ValueError):
        load_dicom_cine(folder)


def test_non_dicom_files_ignored(tmp_path):
    folder = write_classic_series(tmp_path / "s")
    (folder / "notes.txt").write_text("not a dicom")
    assert load_dicom_cine(folder).data.shape == (COLS, ROWS, 4, 5)


def test_enhanced_multiframe(tmp_path):
    n_slices, n_times = 3, 4
    ds = _base(sop_class="1.2.840.10008.5.1.4.1.1.4.1")
    frames, pf_items = [], []
    for z in range(n_slices):
        for t in range(n_times):
            frames.append(_pixels(z, t))
            item = Dataset()
            pp = Dataset(); pp.ImagePositionPatient = [0.0, 0.0, z * 1.0]
            cs = Dataset(); cs.NominalCardiacTriggerDelayTime = 12.5 * t
            item.PlanePositionSequence = Sequence([pp])
            item.CardiacSynchronizationSequence = Sequence([cs])
            pf_items.append(item)
    order = list(range(len(frames)))
    random.Random(1).shuffle(order)
    shared = Dataset()
    pm = Dataset(); pm.PixelSpacing = [0.1, 0.1]
    po = Dataset(); po.ImageOrientationPatient = ORIENT
    shared.PixelMeasuresSequence = Sequence([pm])
    shared.PlaneOrientationSequence = Sequence([po])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    ds.PerFrameFunctionalGroupsSequence = Sequence([pf_items[i] for i in order])
    ds.NumberOfFrames = len(frames)
    ds.PixelData = np.stack([frames[i] for i in order]).tobytes()
    folder = tmp_path / "enh"
    folder.mkdir()
    ds.save_as(folder / "cine.dcm", enforce_file_format=True)

    cine = load_dicom_cine(folder)
    assert cine.data.shape == (COLS, ROWS, n_slices, n_times)
    for z in range(n_slices):
        for t in range(n_times):
            assert cine.data[5, 5, z, t] == 100 * z + t
    assert np.allclose(cine.times_ms, [0, 12.5, 25, 37.5])
    assert cine.spacing[2] == pytest.approx(1.0)
