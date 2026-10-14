"""One-click auto-contour setup (for installs without the bundled model): downloads are checked and never left
half-written, pip progress is read, and failures come back as plain messages. No network: files come from
file:// URLs and pip is faked."""

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
    monkeypatch.setattr(ms, "UCL_FILES", _files(tmp_path, [("UCL-LICENSE.txt", b"GPL\n", True),
                                                           ("ucl_seg459.onnx", b"x" * 300_000, True)]))
    monkeypatch.setattr(ms, "runtime_present", lambda: True)
    checked = []
    monkeypatch.setattr(ms, "_check_runs", lambda p: checked.append(p))
    monkeypatch.setattr(ms, "log_path", lambda: tmp_path / "log.txt")
    dest = tmp_path / "ucl"
    steps = list(ms.run_setup(dest))
    assert steps[-1].stage == "done" and checked == [dest / "ucl_seg459.onnx"]
    assert (dest / "UCL-LICENSE.txt").read_bytes() == b"GPL\n"
    assert (dest / "ucl_seg459.onnx").stat().st_size == 300_000
    assert not list(dest.rglob("*.part"))
    assert ms.model_files_ok(dest)
    # a second run skips what's already there
    again = list(ms.run_setup(dest))
    assert not [s for s in again if s.text.startswith("Downloading")]


def test_bad_hash_leaves_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "UCL_FILES", _files(tmp_path, [("ucl_seg459.onnx", b"y" * 1000, False)]))
    monkeypatch.setattr(ms, "log_path", lambda: tmp_path / "log.txt")
    dest = tmp_path / "ucl"
    with pytest.raises(ms.SetupError, match="didn't match"):
        list(ms.run_setup(dest))
    assert not (dest / "ucl_seg459.onnx").exists()
    assert not list(dest.rglob("*.part"))


def test_cancel_stops_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "UCL_FILES", _files(tmp_path, [("ucl_seg459.onnx", b"z" * 2_000_000, True)]))
    monkeypatch.setattr(ms, "log_path", lambda: tmp_path / "log.txt")
    with pytest.raises(ms.SetupError, match="Cancelled"):
        list(ms.run_setup(tmp_path / "ucl", cancelled=lambda: True))
    assert not list((tmp_path / "ucl").rglob("*.onnx*"))


def test_no_internet_message(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "UCL_FILES", [("UCL-LICENSE.txt", "file:///nonexistent/nowhere.txt", None, None)])
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
    lines = ["Collecting onnxruntime<2,>=1.20",
             "Downloading onnxruntime-1.29.0-cp312-cp312-win_amd64.whl (12.7 MB)",
             "Progress 0 of 12700000", "Progress 6350000 of 12700000", "Progress 12700000 of 12700000",
             "Installing collected packages: onnxruntime", "Successfully installed onnxruntime-1.29.0"]
    monkeypatch.setattr(ms.subprocess, "Popen", lambda *a, **k: _FakeProc(lines, 0))
    steps = list(ms._pip_install(lambda: False))
    half = [s for s in steps if s.total == 12700000 and s.done == 6350000]
    assert half and any("unpacking" in s.text for s in steps) and all(s.stage == "runtime" for s in steps)
    monkeypatch.setattr(ms.subprocess, "Popen", lambda *a, **k: _FakeProc(["ERROR: [Errno 28] No space left on device"], 1))
    with pytest.raises(ms.SetupError, match="disk space"):
        list(ms._pip_install(lambda: False))
    monkeypatch.setattr(ms.subprocess, "Popen", lambda *a, **k: _FakeProc(["ERROR: could not connect"], 1))
    with pytest.raises(ms.SetupError, match="Installing ONNX Runtime failed"):
        list(ms._pip_install(lambda: False))


def test_pinned_model_release():
    """The model comes from TraceCMR's ucl-model-1 pre-release, pinned by size and SHA-256."""
    rel, url, size, sha = ms.UCL_FILES[0]
    assert rel == "ucl_seg459.onnx" and size == 18_931_904
    assert sha == "7afe39b3870c5ae66cd94103b7b189cdc8153e86c88b1481cb074414bb4f908a"
    assert url == "https://github.com/jairam09/TraceCMR/releases/download/ucl-model-1/ucl_seg459.onnx"
    assert ms.RUNTIME.startswith("onnxruntime") and 20 <= ms.DOWNLOAD_MB <= 50


def test_backend_choice(tmp_path, monkeypatch):
    """ONNX when the model and ONNX Runtime are there (installer, or one-click setup); TensorFlow only as the
    fallback for older setups; otherwise not set up."""
    from tracecmr import autocontour as ac

    monkeypatch.setenv("TRACECMR_UCL_DIR", str(tmp_path / "ucl"))
    monkeypatch.delenv("TRACECMR_UCL_ONNX", raising=False)
    monkeypatch.delenv("TRACECMR_UCL_BACKEND", raising=False)
    monkeypatch.setattr(ac.sys, "prefix", str(tmp_path / "prefix"))
    have = {"onnxruntime": True, "tensorflow": True}
    monkeypatch.setattr(ac, "_present", lambda m: have[m])
    assert ac.ucl_backend() is None and ac.available()["ucl"] != "ready"
    bundled = tmp_path / "prefix" / "models" / "ucl_seg459.onnx"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"0" * 2_000_000)
    assert ac.ucl_onnx_path() == bundled and ac.ucl_backend() == "onnx" and ac.available()["ucl"] == "ready"
    have["onnxruntime"] = False
    assert ac.ucl_backend() is None  # the model alone isn't enough
    (tmp_path / "ucl" / "ModelWeights").mkdir(parents=True)
    (tmp_path / "ucl" / "unet3plus.py").write_text("# authors' code")
    (tmp_path / "ucl" / "ModelWeights" / "SEG459.h5").write_bytes(b"0" * 2_000_000)
    assert ac.ucl_backend() == "tensorflow"


def test_eight_orientation_average_is_orientation_proof(monkeypatch):
    """With TTA, turning or mirroring the input turns the result the same way, nothing else, even for a model
    that is not symmetric at all."""
    import numpy as np

    from tracecmr import autocontour as ac

    rng = np.random.default_rng(0)
    w = rng.normal(size=(3,))

    def lopsided(x):  # depends on position (left-right ramp) and on the pixel: a model with a favourite direction
        ramp = np.linspace(0, 1, x.shape[1])[None, :, None, None]
        z = x * w[None, None, None, :] + ramp * np.array([1.0, -1.0, 0.5])
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    monkeypatch.setattr(ac, "_load_ucl", lambda: lopsided)
    vol = rng.random((40, 40, 3)).astype(np.float32)
    monkeypatch.setenv("TRACECMR_TTA", "1")
    p = ac._ucl_probs(vol, (0.1, 0.1, 1.0))
    q = ac._ucl_probs(np.rot90(vol, 1, axes=(0, 1)).copy(), (0.1, 0.1, 1.0))
    assert np.allclose(np.rot90(p, 1, axes=(0, 1)), q, atol=1e-5)
    monkeypatch.setenv("TRACECMR_TTA", "0")
    p0 = ac._ucl_probs(vol, (0.1, 0.1, 1.0))
    q0 = ac._ucl_probs(np.rot90(vol, 1, axes=(0, 1)).copy(), (0.1, 0.1, 1.0))
    assert not np.allclose(np.rot90(p0, 1, axes=(0, 1)), q0, atol=1e-3)  # one pass: orientation matters
