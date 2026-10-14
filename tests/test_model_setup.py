"""One-click auto-contour setup: downloads are checked and never left half-written, pip progress is read,
and failures come back as plain messages. No network: files come from file:// URLs and pip is faked."""

import hashlib
import io

import pytest

from tracecmr import model_setup as ms


def _files(tmp_path, payloads):
    src = tmp_path / "src"
    src.mkdir()
    out = []
    for rel, data, good_hash in payloads:
        p = src / rel.replace("/", "_")
        p.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest() if good_hash else "0" * 64
        out.append((rel, p.as_uri(), len(data), sha))
    return out


def test_downloads_model_files_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "UCL_FILES", _files(tmp_path, [("unet3plus.py", b"print(1)\n", True),
                                                           ("ModelWeights/SEG459.h5", b"x" * 300_000, True)]))
    monkeypatch.setattr(ms, "tensorflow_present", lambda: True)
    monkeypatch.setattr(ms, "_check_import", lambda: None)
    monkeypatch.setattr(ms, "log_path", lambda: tmp_path / "log.txt")
    dest = tmp_path / "ucl"
    steps = list(ms.run_setup(dest))
    assert steps[-1].stage == "done"
    assert (dest / "unet3plus.py").read_bytes() == b"print(1)\n"
    assert (dest / "ModelWeights" / "SEG459.h5").stat().st_size == 300_000
    assert not list(dest.rglob("*.part"))
    assert ms.model_files_ok(dest)
    # a second run skips what's already there
    again = list(ms.run_setup(dest))
    assert not [s for s in again if s.text.startswith("Downloading")]


def test_bad_hash_leaves_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "UCL_FILES", _files(tmp_path, [("ModelWeights/SEG459.h5", b"y" * 1000, False)]))
    monkeypatch.setattr(ms, "log_path", lambda: tmp_path / "log.txt")
    dest = tmp_path / "ucl"
    with pytest.raises(ms.SetupError, match="didn't match"):
        list(ms.run_setup(dest))
    assert not (dest / "ModelWeights" / "SEG459.h5").exists()
    assert not list(dest.rglob("*.part"))


def test_cancel_stops_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "UCL_FILES", _files(tmp_path, [("ModelWeights/SEG459.h5", b"z" * 2_000_000, True)]))
    monkeypatch.setattr(ms, "log_path", lambda: tmp_path / "log.txt")
    with pytest.raises(ms.SetupError, match="Cancelled"):
        list(ms.run_setup(tmp_path / "ucl", cancelled=lambda: True))
    assert not list((tmp_path / "ucl").rglob("*.h5*"))


def test_no_internet_message(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "UCL_FILES", [("unet3plus.py", "file:///nonexistent/nowhere.py", None, None)])
    monkeypatch.setattr(ms, "log_path", lambda: tmp_path / "log.txt")
    with pytest.raises(ms.SetupError, match="Check the internet connection"):
        list(ms.run_setup(tmp_path / "ucl"))


class _FakeProc:
    def __init__(self, lines, rc):
        self.stdout = io.StringIO("".join(line + "\n" for line in lines))
        self._rc = rc

    def wait(self):
        return self._rc

    def poll(self):
        return self._rc

    def terminate(self):
        pass

    def kill(self):
        pass


def test_pip_progress_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "log_path", lambda: tmp_path / "log.txt")
    monkeypatch.setattr(ms, "_pip_has_raw_progress", lambda: True)
    lines = ["Collecting tensorflow==2.21.0",
             "Downloading tensorflow-2.21.0-cp312-cp312-win_amd64.whl (351.0 MB)",
             "Progress 0 of 351000000", "Progress 175500000 of 351000000", "Progress 351000000 of 351000000",
             "Installing collected packages: tensorflow", "Successfully installed tensorflow-2.21.0"]
    monkeypatch.setattr(ms.subprocess, "Popen", lambda *a, **k: _FakeProc(lines, 0))
    steps = list(ms._pip_install(lambda: False))
    half = [s for s in steps if s.total == 351000000 and s.done == 175500000]
    assert half and any("unpacking" in s.text for s in steps)
    monkeypatch.setattr(ms.subprocess, "Popen", lambda *a, **k: _FakeProc(["ERROR: [Errno 28] No space left on device"], 1))
    with pytest.raises(ms.SetupError, match="disk space"):
        list(ms._pip_install(lambda: False))
    monkeypatch.setattr(ms.subprocess, "Popen", lambda *a, **k: _FakeProc(["ERROR: could not connect"], 1))
    with pytest.raises(ms.SetupError, match="Installing TensorFlow failed"):
        list(ms._pip_install(lambda: False))


def test_pinned_versions_and_published_hash():
    assert ms.TENSORFLOW == "tensorflow==2.21.0"
    rel, url, size, sha = ms.UCL_FILES[-1]
    assert rel == "ModelWeights/SEG459.h5" and size == 156_245_288
    assert sha == "3aac032c3725e843736eb01595131bddd64d34bf12d5a02dba16b177048e0d92"
    assert url.startswith("https://media.githubusercontent.com/media/mrphys/")
