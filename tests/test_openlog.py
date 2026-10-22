"""What a problem report says about the scan folders opened: what was found, what was skipped and why, without
images or animal names."""

import numpy as np

from test_bruker import write_scan
from test_study import _write

from tracecmr.app import openlog
from tracecmr.app.study import scan_bruker_study, scan_dicom_folder


def test_paravision_study_notes_what_was_listed_and_skipped(tmp_path):
    openlog.clear()
    study = tmp_path / "20260730_Mouse_Jones_44"
    write_scan(study, 8, "SAX_S-0.0 mouse Jones", np.ones((16, 16, 1, 6), int))
    (study / "9" / "pdata" / "1").mkdir(parents=True)  # a scan with no reconstructed images
    write_scan(study, 10, "LAX", np.ones((16, 16, 1, 6), int))
    (study / "10" / "pdata" / "1" / "visu_pars").write_text("##$VisuCoreDim=2\n##END=\n")  # parameters cut short
    found = scan_bruker_study(study)
    assert [s.pv.number for s in found] == [8]
    text = "\n".join(openlog.notes())
    assert "E8 [Cine_brightblood_IG_FLASH_flc]: 2D 16x16, 1 slice(s) x 6 frame(s)" in text
    assert "ParaVision 360 V3.5" in text or "ParaVision ?" in text
    assert "E9: skipped, no reconstructed images" in text
    assert "E10: skipped, parameters not understood (KeyError" in text
    assert "Jones" not in text and "Mouse" not in text  # no animal or study names


def test_dicom_folder_notes_unread_files_without_their_names(tmp_path):
    openlog.clear()
    _write(tmp_path / "Mouse_Jones", "SAX cine Jones", 7, [0, 1], n_times=3)
    (tmp_path / "Mouse_Jones" / "notes_Jones.txt").write_text("not an image")
    series = scan_dicom_folder(tmp_path)
    assert len(series) == 1
    text = "\n".join(openlog.notes())
    assert "7 files, 6 images in 1 series, 1 not read as DICOM" in text
    assert "1 .txt" in text
    assert "series 7 (Short axis): 6 file(s) -> 2 slice(s) x 3 frame(s)" in text
    assert "Jones" not in text


def test_paths_in_error_messages_are_hidden():
    msg = "[Errno 2] No such file or directory: 'C:\\\\Users\\\\me\\\\Mouse 3\\\\8\\\\pdata'"
    assert openlog.scrub(msg) == "[Errno 2] No such file or directory: '<path>'"
    assert openlog.scrub("cannot read /Users/me/Mouse3/a.dcm now") == "cannot read <path> now"
    assert openlog.scrub("software PV 6.0.1/7, ratio 1/2") == "software PV 6.0.1/7, ratio 1/2"


def test_problem_report_has_the_folder_notes_and_a_support_address(tmp_path, monkeypatch):
    from tracecmr.app import support
    from tracecmr.app.branding import SUPPORT_EMAIL

    monkeypatch.setenv("TRACECMR_LOG", str(tmp_path / "log.txt"))
    openlog.clear()
    openlog.note("ParaVision study: 3 scan folder(s) with parameters or images")
    text = support.diagnostics()
    assert "--- scan folders opened this session ---" in text and "ParaVision study: 3" in text
    assert SUPPORT_EMAIL == "tracecmr.support@gmail.com" and support.support_email() == SUPPORT_EMAIL
