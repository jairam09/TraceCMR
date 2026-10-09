"""Study scanning: DICOM files grouped into series, kinds guessed, single-slice cines combined."""

import numpy as np
from pydicom.uid import generate_uid

from test_dicom import ORIENT, _base, _pixels

from tracecmr.app.study import combine, guess_kind, scan_dicom_folder


def _write(folder, desc, number, zs, n_times=5, orient=ORIENT):
    folder.mkdir(parents=True, exist_ok=True)
    uid = generate_uid()
    i = 0
    for z in zs:
        for t in range(n_times):
            ds = _base()
            ds.SeriesInstanceUID = uid
            ds.SeriesDescription = desc
            ds.SeriesNumber = number
            ds.ImageOrientationPatient = orient
            ds.ImagePositionPatient = [0.0, 0.0, z * 1.0]
            ds.PixelSpacing = [0.1, 0.1]
            ds.TriggerTime = 10.0 * t
            ds.InstanceNumber = i + 1
            ds.PixelData = _pixels(z, t).tobytes()
            ds.save_as(folder / f"{number}_{i:03d}.dcm", enforce_file_format=True)
            i += 1


def test_guess_kind():
    assert guess_kind("cine_sax") == "Short axis"
    assert guess_kind("CINE 4CH") == "Long axis"
    assert guess_kind("LGE_PSIR") == "LGE"
    assert guess_kind("localizer") == "Other"


def test_scan_groups_series_and_combines_slices(tmp_path):
    _write(tmp_path / "a", "cine_sax", 3, zs=range(4))
    _write(tmp_path / "b", "cine_4ch", 4, zs=[0], orient=[1, 0, 0, 0, 0, 1])
    for k in range(3):  # one cine per short-axis slice, saved out of order
        _write(tmp_path / f"s{k}", f"cine_sa_slice{k}", 10 + k, zs=[[2, 0, 1][k]])
    series = scan_dicom_folder(tmp_path)
    assert len(series) == 5
    sax = [s for s in series if s.kind == "Short axis"]
    assert {s.n_slices for s in sax} == {4, 1}
    assert any(s.kind == "Long axis" for s in series)
    assert all(s.thumb is not None for s in series)
    full = next(s for s in sax if s.n_slices == 4)
    sess = full.load()
    assert sess.image.shape[2:] == (4, 5)
    singles = [s for s in sax if s.n_slices == 1]
    stack = combine(singles)
    assert stack.n_slices == 3 and stack.kind == "Short axis"
    img = stack.load().image
    # slices sorted by position: pixel values encode the slice index (100*z + t)
    assert [int(img[5, 5, z, 0]) for z in range(3)] == [0, 100, 200]


def test_any_slice_count_repeats_and_uneven_frames(tmp_path):
    from tracecmr.app.study import auto_stacks

    zs = [0, 1, 2, 3, 4, 5]  # six slices, not eight
    for k, z in enumerate(zs):
        _write(tmp_path / f"s{k}", f"SAX_S-{z}", 10 + k, zs=[z], n_times=5 if k != 2 else 6)  # slice 2: 6 frames
    _write(tmp_path / "rep", "SAX_S-3 repeat", 30, zs=[3])  # slice position 3 scanned again later
    series = scan_dicom_folder(tmp_path)
    stacks = auto_stacks(series)
    assert len(stacks) == 1
    st = stacks[0]
    assert st.n_slices == 6 and st.n_frames == 5
    assert any(p.name.startswith("30 ") for p in st.parts)  # the later scan of position 3 is used
    assert any(s.name.endswith("(earlier scan)") for s in series)  # the older one stays listed
    sess = st.load()
    assert sess.image.shape[2:] == (6, 5) and sess.separate_slices
    assert [int(sess.image[5, 5, z, 0]) for z in range(6)] == [0, 100, 200, 300, 400, 500]
    assert "frames 5–6" in sess.subject_id
