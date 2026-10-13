"""ED/ES timing across slices (tracecmr.drift)."""

import numpy as np

from tracecmr import labels as L
from tracecmr.app.session import Session
from tracecmr.drift import phase_timing, read_jcamp, timing_params, write_csv


def _cine(nz=8, nt=20, drift_frames=0.0, noise=0.0, seed=0):
    """A stack of discs whose radius beats through one cycle; slice z reaches ES `drift_frames * z / (nz-1)`
    frames later than slice 0 (what a drifting heart rate does to a serially acquired stack)."""
    rng = np.random.default_rng(seed)
    n = 48
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - n / 2, yy - n / 2)
    label = np.zeros((n, n, nz, nt), np.uint8)
    for z in range(nz):
        lag = drift_frames * z / max(nz - 1, 1)
        base = 12 - 0.6 * z  # tapering towards the apex
        for t in range(nt):
            phase = 2 * np.pi * (t - 8 - lag) / nt  # ES (smallest) near frame 8 + lag
            r = base * (0.8 + 0.2 * -np.cos(phase)) + (rng.normal(0, noise) if noise else 0)
            label[..., z, t][rr <= r + 3] = L.LV_MYOCARDIUM
            label[..., z, t][rr <= r] = L.LV_BLOODPOOL
    image = label.astype(np.float32)
    return Session(image=image, label=label, spacing=(0.1, 0.1, 1.0), subject_id="cine", ed_frame=0, es_frame=8)


def test_no_drift_means_no_difference():
    r = phase_timing(_cine(drift_frames=0))
    assert len(r.slices) == 8 and r.es_spread_frames == 0
    assert abs(r.delta_ef) < 1e-6
    assert r.ef_global > 0


def test_drift_moves_es_with_slice_order_and_raises_ef():
    r = phase_timing(_cine(drift_frames=4))
    assert r.es_spread_frames >= 2  # conservative: frames tied with the extreme go to the global frame
    assert r.es_trend is not None and r.es_trend > 0.9  # ES comes later slice by slice
    assert r.delta_ef > 0.5  # one global ES leaves some slices off their smallest
    assert r.edv_per_slice >= r.edv_global and r.esv_per_slice <= r.esv_global
    assert "trending with slice order" in r.summary()


def test_noise_alone_scatters_without_a_trend():
    r = phase_timing(_cine(drift_frames=0, noise=0.6, seed=3), smooth=3)
    assert r.es_trend is None or abs(r.es_trend) < 0.6


def test_too_few_frames_is_reported_not_guessed():
    s = _cine()
    s.label[..., 2:8] = 0  # only frames 0, 1 and 8.. drawn on... still a full set elsewhere
    s.label[..., :, 9:] = 0
    s.label[..., :, 1] = 0
    r = phase_timing(s, min_frames=4)
    assert r.slices == [] and "Not enough" in r.summary()


def test_csv_and_frame_times(tmp_path):
    s = _cine(drift_frames=4)
    s.frame_times_ms = [i * 6.0 for i in range(20)]
    r = phase_timing(s)
    assert r.frame_ms == 6.0 and r.es_spread_ms == r.es_spread_frames * 6.0
    text = write_csv(r, tmp_path / "t.csv", "cine").read_text()
    assert "es_shift_frames" in text and "delta_ef_pts" in text


def test_read_paravision_parameters(tmp_path):
    scan = tmp_path / "5"
    (scan / "pdata" / "1").mkdir(parents=True)
    (scan / "method").write_text(
        "##TITLE=Parameter List\n##$Method=<Bruker:IgFLASH>\n##$PVM_RepetitionTime=6.2\n"
        "##$PVM_TriggerModule=On\n##$PVM_NMovieFrames=20\n##$PVM_SliceThick=1\n"
        "##$IgHeartRate=( 3 )\n512 498 476\n$$ comment\n##END=\n")
    (scan / "pdata" / "1" / "visu_pars").write_text("##$VisuAcqRepetitionTime=( 1 )\n6.2\n##END=\n")
    p = read_jcamp(scan / "method")
    assert p["Method"] == "Bruker:IgFLASH" and p["PVM_RepetitionTime"] == 6.2
    assert p["IgHeartRate"] == ["512", "498", "476"]
    t = timing_params(scan)
    assert "method:PVM_TriggerModule" in t and "method:IgHeartRate" in t and "method:PVM_NMovieFrames" in t
    assert "1/visu_pars:VisuAcqRepetitionTime" in t
    assert not any(k.endswith("PVM_SliceThick") for k in t)
