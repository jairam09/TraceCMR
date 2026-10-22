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
    for name in ("SAX_S-0.0", "cine_sa_slice1", "cine SA3", "short axis stack"):
        assert guess_kind(name) == "Short axis", name
    assert guess_kind("LAX_S0.0 4chamber") == guess_kind("2ch cine") == "Long axis"
    # whole words only: a saturation band, a sagittal scan or a template is not a short axis or LGE
    for name in ("DTI_EPI_seg_30dir_sat", "T2_sagittal", "FLASH template", "Cine_brightblood_IG_FLASH_flc"):
        assert guess_kind(name) == "Other", name


def test_first_series_opened_is_a_cine_stack_whatever_its_name():
    from tracecmr.app.study import Series, best_first_series

    loc = Series("1 Localizer", "Other", "", n_slices=3, n_frames=1)
    sat = Series("2 DTI sat", "Other", "", n_slices=5, n_frames=1)
    cine = Series("3 Cine FLASH", "Other", "", n_slices=8, n_frames=16)
    lax = Series("4 LAX cine", "Long axis", "", n_slices=1, n_frames=16)
    sax = Series("5 SAX cine", "Short axis", "", n_slices=8, n_frames=16)
    assert best_first_series([loc, sat, cine, lax, sax]) == 4  # a short-axis cine stack first
    assert best_first_series([loc, sat, lax, cine]) == 3  # else any multi-slice cine, not the long axis
    assert best_first_series([loc, sat]) == 0


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


def _write_bruker_enhanced(folder, desc, number, z, n_times=6, slope=2.0, orient=ORIENT):
    """Like ParaVision's DICOM export: one enhanced file per short-axis slice, orientation, position, pixel
    spacing and rescale in the shared functional groups, only the cardiac timing per frame."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    from test_dicom import _base

    folder.mkdir(parents=True, exist_ok=True)
    ds = _base(sop_class="1.2.840.10008.5.1.4.1.1.4.1")
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = desc
    ds.SeriesNumber = number
    shared = Dataset()
    pm = Dataset(); pm.PixelSpacing = [0.2, 0.2]
    po = Dataset(); po.ImageOrientationPatient = orient
    pp = Dataset(); pp.ImagePositionPatient = [0.0, 0.0, float(z)]
    pv = Dataset(); pv.RescaleSlope = slope; pv.RescaleIntercept = 0; pv.RescaleType = "US"
    shared.PixelMeasuresSequence = Sequence([pm])
    shared.PlaneOrientationSequence = Sequence([po])
    shared.PlanePositionSequence = Sequence([pp])
    shared.PixelValueTransformationSequence = Sequence([pv])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    items = []
    for t in range(n_times):
        it = Dataset()
        cs = Dataset(); cs.NominalCardiacTriggerDelayTime = 8.0 * t
        it.CardiacSynchronizationSequence = Sequence([cs])
        items.append(it)
    ds.PerFrameFunctionalGroupsSequence = Sequence(items)
    ds.NumberOfFrames = n_times
    ds.PixelData = np.stack([_pixels(z, t) for t in range(n_times)]).tobytes()
    ds.save_as(folder / f"E{number}_P1_EnIm1.dcm", enforce_file_format=True)


def test_bruker_dicom_export_one_enhanced_file_per_slice_stacks(tmp_path):
    from tracecmr.app.study import auto_stacks

    # scan numbers and hand-typed labels disagree with the geometry; slices must follow the positions
    for k, (number, z) in enumerate([(8, 2), (9, 0), (10, 1)]):
        _write_bruker_enhanced(tmp_path / f"{number}", f"SAX_S-{k}.0 GAD", number, z)
    _write_bruker_enhanced(tmp_path / "16", "LAX_S0.0 4chamber", 16, 0, orient=[1, 0, 0, 0, 0, 1])
    series = scan_dicom_folder(tmp_path)
    sax = [s for s in series if s.kind == "Short axis"]
    assert len(sax) == 3 and all(s.n_slices == 1 and s.n_frames == 6 and s.geometry for s in sax)
    stacks = auto_stacks(series)
    assert len(stacks) == 1 and stacks[0].n_slices == 3
    sess = stacks[0].load()
    assert sess.image.shape[2:] == (3, 6)
    # rescale applied (slope 2) and slices sorted by position: value = 2 * (100 * z + t)
    assert [float(sess.image[5, 5, z, 1]) for z in range(3)] == [2.0, 202.0, 402.0]
    assert sess.frame_times_ms == [0, 8, 16, 24, 32, 40]
    assert sess.spacing[0] == sess.spacing[1] == 0.2
