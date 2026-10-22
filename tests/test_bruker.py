"""Bruker ParaVision study folders read straight from 2dseq + visu_pars. The test studies are written here in the
same layout and parameter syntax as ParaVision 360 (checked against a real cine study): @n*(v) runs, (a, <b>)
tuples, <string> arrays, values continued on the next lines."""

import numpy as np
import pytest

from tracecmr import bruker
from tracecmr.app.study import auto_stacks, scan_bruker_study

ORIENT = [-0.82786543589881, -0.21216093516643, -0.5192557728455, 0.346023277876701, -0.921749128945765,
          -0.175061230588571, -0.441482401927342, -0.324601726546589, 0.836496866647707]


def _arr(vals, per_line=6):
    vals = [str(v) for v in vals]
    return "\n".join(" ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line))


def write_scan(study, number, name, data, slope=2.5, groups=(("FG_CARDIAC_MOVIE",),), positions=None,
               orient=ORIENT, extent=(25.6, 25.6), word="_16BIT_SGN_INT", times=True):
    """data: (x, y, slice, time) integers. groups: frame groups fastest first, by name."""
    nx, ny, nz, nt = data.shape
    d = study / str(number) / "pdata" / "1"
    d.mkdir(parents=True)
    sizes = {"FG_CARDIAC_MOVIE": nt, "FG_SLICE": nz}
    order = [g[0] for g in groups]
    frames = []
    shape = [sizes[g] for g in order]
    for f in range(int(np.prod(shape))):
        idx = np.unravel_index(f, shape[::-1])[::-1]  # first group fastest
        z = idx[order.index("FG_SLICE")] if "FG_SLICE" in order else 0
        t = idx[order.index("FG_CARDIAC_MOVIE")] if "FG_CARDIAC_MOVIE" in order else 0
        frames.append(data[:, :, z, t].T.ravel())  # stored x fastest
    np.asarray(frames, dtype=np.int16 if word == "_16BIT_SGN_INT" else np.int32).tofile(d / "2dseq")
    n = len(frames)
    if positions is None:
        positions = [[10.5, 17.6, 4.0]]
    fg = " ".join(f"({sizes[g]}, <{g}>, <>, 0, 2)" for g in order)
    txt = [
        "##TITLE=Parameter List, ParaVision 360 V3.5",
        "$$ @vis= ignored comment line",
        "##$VisuCoreFrameCount=" + str(n),
        "##$VisuCoreDim=2",
        f"##$VisuCoreSize=( 2 )\n{nx} {ny}",
        f"##$VisuCoreExtent=( 2 )\n{extent[0]} {extent[1]}",
        "##$VisuCoreFrameThickness=( 1 )\n1",
        f"##$VisuCoreOrientation=( {len(positions)}, 9 )\n" + _arr(list(orient) * len(positions)),
        f"##$VisuCorePosition=( {len(positions)}, 3 )\n" + _arr([v for p in positions for v in p]),
        f"##$VisuCoreDataSlope=( {n} )\n@{n}*({slope})",
        f"##$VisuCoreDataOffs=( {n} )\n@{n}*(0)",
        f"##$VisuCoreWordType={word}",
        "##$VisuCoreByteOrder=littleEndian",
        f"##$VisuFGOrderDescDim={len(order)}",
        f"##$VisuFGOrderDesc=( {len(order)} )\n{fg}",
        f"##$VisuSeriesExperimentComment=( 65 )\n<{name}>",
        "##$VisuAcquisitionProtocol=( 65 )\n<Cine_brightblood_IG_FLASH_flc>",
        "##$VisuCardiacHeartRate=450",
    ]
    if times:
        txt.append(f"##$VisuCardiacMovieFrameTime=( {nt} )\n" + " ".join(f"({8 * i}, {8 * i})" for i in range(nt)))
    txt.append("##END=")
    (d / "visu_pars").write_text("\n".join(txt) + "\n")
    (study / str(number) / "acqp").write_text(f"##$ACQ_scan_name=( 64 )\n<{name} (E{number})>\n##END=\n")


def test_parameter_syntax(tmp_path):
    p = tmp_path / "visu_pars"
    p.write_text("##$A=( 3 )\n@3*(4.5)\n##$B=( 1 ) (16, <FG_CARDIAC_MOVIE>, <>, 0, 2)\n"
                 "##$C=( 65 )\n<SAX_S-4.5 GAD19min>\n##$D=littleEndian\n##$E=( 2, 3 )\n1 2 3\n4 5 6\n##$F=450\n##END=\n")
    v = bruker.read_params(p)
    assert v["A"] == [4.5, 4.5, 4.5]
    assert v["B"] == [(16, "FG_CARDIAC_MOVIE", "", 0, 2)]
    assert v["C"] == "SAX_S-4.5 GAD19min" and v["D"] == "littleEndian" and v["F"] == 450
    assert v["E"] == [1, 2, 3, 4, 5, 6]


def test_single_slice_cine_reads_pixels_slope_and_times(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 3000, size=(32, 24, 1, 16))
    write_scan(tmp_path, 8, "SAX_S-4.5 GAD19min", data, slope=2.5)
    sc = bruker.read_scan(tmp_path / "8")
    assert (sc.n_slices, sc.n_frames, sc.size) == (1, 16, (32, 24))
    assert sc.pixel_mm == pytest.approx((0.8, 25.6 / 24))
    assert sc.name == "SAX_S-4.5 GAD19min" and sc.frame_times_ms[:3] == [0, 8, 16]
    img = bruker.load_scan(sc)
    assert img.shape == (32, 24, 1, 16)
    assert np.allclose(img, data * 2.5)  # x stays x, y stays y, every frame scaled


def test_multislice_frame_groups_in_either_order(tmp_path):
    data = np.arange(8 * 6 * 3 * 4).reshape(8, 6, 3, 4) % 1000
    normal = np.array(ORIENT[6:])
    pos = [list(np.array([1.0, 2.0, 3.0]) + k * normal) for k in range(3)]
    write_scan(tmp_path, 1, "movie fastest", data, slope=1, groups=(("FG_CARDIAC_MOVIE",), ("FG_SLICE",)), positions=pos)
    write_scan(tmp_path, 2, "slice fastest", data, slope=1, groups=(("FG_SLICE",), ("FG_CARDIAC_MOVIE",)), positions=pos)
    for n in (1, 2):
        sc = bruker.read_scan(tmp_path / str(n))
        assert (sc.n_slices, sc.n_frames) == (3, 4)
        assert np.array_equal(bruker.load_scan(sc), data)
        assert np.allclose(np.diff(sc.slice_distances()), 1.0)


def test_study_of_single_slice_scans_becomes_one_stack_sorted_by_position(tmp_path):
    """Like a real study: localizers, then one cine per short-axis slice scanned in a shuffled order with
    hand-typed names that disagree with the geometry, then a long-axis cine."""
    normal = np.array(ORIENT[6:])
    base = np.array([7.447, 15.373, 9.896])  # the most basal slice, at 0 mm along the normal
    rng = np.random.default_rng(1)
    slices = {}
    for k, number in enumerate([12, 8, 10, 9, 11]):  # scan numbers out of position order
        dist = -float(k)  # 0, -1, -2, ...
        data = np.full((16, 16, 1, 6), 100 * (k + 1)) + rng.integers(0, 5, (16, 16, 1, 6))
        slices[dist] = data
        write_scan(tmp_path, number, f"SAX_S-{4.5 if number == 8 else number} GAD", data, slope=1.0,
                   positions=[list(base + dist * normal)])
    write_scan(tmp_path, 16, "LAX_S0.0 4chamber", np.ones((16, 16, 1, 6), int),
               orient=[1, 0, 0, 0, 1, 0, 0, 0, 1])
    loc = tmp_path / "1" / "pdata" / "1"
    write_scan(tmp_path, 1, "1_Localizer", np.ones((16, 16, 3, 1), int), groups=(("FG_SLICE",),),
               positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], times=False)
    assert loc.joinpath("2dseq").exists()
    assert bruker.is_study(tmp_path) and not bruker.is_study(tmp_path / "8")

    series = scan_bruker_study(tmp_path)
    kinds = {s.name: s.kind for s in series}
    assert kinds["E16 LAX_S0.0 4chamber"] == "Long axis" and kinds["E1 1_Localizer"] == "Other"
    stacks = auto_stacks(series)
    assert len(stacks) == 1 and stacks[0].n_slices == 5
    s = stacks[0].load()
    assert s.image.shape == (16, 16, 5, 6)
    means = [float(s.image[:, :, z, 0].mean()) for z in range(5)]
    expect = [float(slices[d][:, :, 0, 0].mean()) for d in sorted(slices)]  # most negative distance first
    assert means == pytest.approx(expect)
    assert s.spacing == pytest.approx((1.6, 1.6, 1.0))  # 25.6 mm over 16 pixels; slices 1 mm apart
    assert s.frame_times_ms == [0, 8, 16, 24, 32, 40]


def test_a_scan_without_images_is_skipped(tmp_path):
    (tmp_path / "3").mkdir()
    (tmp_path / "3" / "acqp").write_text("##END=\n")
    write_scan(tmp_path, 4, "SAX one", np.ones((8, 8, 1, 4), int))
    assert [s.number for s in bruker.scan_study(tmp_path)] == [4]


def test_slices_that_do_not_step_along_their_normal_still_get_a_thickness(tmp_path):
    """Positions that move across the slice plane instead of along its normal (seen in a public ParaVision 5.1
    test set) gave a slice gap of 0, so every volume would be 0: use the distance between slice centres."""
    data = np.ones((8, 6, 3, 2), int)
    in_plane = np.array(ORIENT[:3])  # the read direction: within the slice plane
    pos = [list(np.array([1.0, 2.0, 3.0]) + 1.5 * k * in_plane) for k in range(3)]
    write_scan(tmp_path, 1, "cine", data, slope=1, groups=(("FG_CARDIAC_MOVIE",), ("FG_SLICE",)), positions=pos)
    cine = bruker.load_cine(bruker.read_scan(tmp_path / "1"))
    assert abs(cine.spacing[2] - 1.5) < 1e-6


def test_spectroscopy_is_skipped_with_a_clear_reason(tmp_path):
    write_scan(tmp_path, 1, "cine", np.ones((8, 6, 1, 4), int), slope=1)
    write_scan(tmp_path, 2, "press", np.ones((8, 6, 1, 1), int), slope=1)
    vp = tmp_path / "2" / "pdata" / "1" / "visu_pars"
    text = vp.read_text().split("\n##$")
    vp.write_text("\n##$".join(t for t in text if not t.startswith(("VisuCoreOrientation", "VisuCorePosition"))))
    skipped = []
    scans = bruker.scan_study(tmp_path, skipped=skipped)
    assert [s.number for s in scans] == [1]
    assert skipped == [("2", "parameters not understood (ValueError: not an image series (no slice geometry; "
                              "spectroscopy, for example))")]
