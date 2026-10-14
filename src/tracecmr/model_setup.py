"""One-click setup for auto-contouring on copies that don't have the model yet (installers from 0.3.5 on
include it, so this is for older installs and source installs).

It fetches two things, about 40 MB in all:
  - UCL's published model (Shah et al., JCMR 2026, GPL-3.0) converted unchanged to ONNX, from TraceCMR's
    model release, checked against the SHA-256 pinned in ucl_model.json (see scripts/convert_ucl_onnx.py)
  - ONNX Runtime, the small library that runs it, from PyPI into TraceCMR's own Python

Scans are never uploaded. `run_setup` is a generator, so the app can show progress and cancel between steps:

    for step in run_setup(dest):
        print(step.stage, step.done, step.total, step.text)

It skips whatever is already in place and finishes by loading the model in a fresh process, so "ready" means it
really runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

PIN = json.loads((Path(__file__).with_name("ucl_model.json")).read_text(encoding="utf-8"))
ONNX_NAME = PIN["file"]

# (relative path, url, expected size in bytes or None, expected sha256 or None)
UCL_FILES = [
    (ONNX_NAME, PIN["url"], int(PIN["size"]), PIN["sha256"]),
    ("UCL-LICENSE.txt", PIN["license_url"], None, None),
]

RUNTIME = "onnxruntime>=1.20,<2"  # also in pyproject's "auto" extra, which the installers include
DOWNLOAD_MB = round(int(PIN["size"]) / 1e6) + 22  # the model + ONNX Runtime (15 MB on Windows, 21 on macOS)


@dataclass
class Step:
    stage: str  # "model", "runtime", "check", "done"
    done: int  # progress within the stage (bytes, or 0 when unknown)
    total: int  # 0 = unknown (show a busy bar)
    text: str  # what the user sees


class SetupError(RuntimeError):
    """A step failed; the message says what happened and what to do."""


def default_dir() -> Path:
    """Where the model goes: the folder autocontour looks in when no other folder was chosen."""
    return Path.home() / ".tracecmr" / "models" / "ucl"


def log_path() -> Path:
    return Path.home() / "Documents" / "TraceCMR results" / "logs" / "auto_contour_setup.txt"


def _log(msg: str) -> None:
    try:
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except OSError:
        pass


def python_exe() -> str:
    """The app's own Python, with a console-less launcher swapped for python.exe so pip's output can be read."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def model_files_ok(dest: Path) -> bool:
    for rel, _url, size, sha in UCL_FILES:
        p = dest / rel
        if not p.exists() or p.stat().st_size == 0:
            return False
        if size is not None and p.stat().st_size != size:
            return False
    return True


def _present(module: str) -> bool:
    importlib.invalidate_caches()
    return importlib.util.find_spec(module) is not None


def runtime_present() -> bool:
    """ONNX Runtime is installed in this Python (found on disk; not imported, so the app stays light)."""
    return _present("onnxruntime")


def tensorflow_present() -> bool:  # older setups ran the model with TensorFlow; still used if it is there
    return _present("tensorflow")


def is_ready(dest: Path | None = None) -> bool:
    from . import autocontour as ac

    if dest is None:
        return ac.ucl_backend() is not None
    return runtime_present() and model_files_ok(Path(dest))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, target: Path, expect_size: int | None, expect_sha: str | None,
              cancelled: Callable[[], bool], label: str) -> Iterator[Step]:
    """Stream url to target (through a .part file, so a broken download never looks finished)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "TraceCMR"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or (expect_size or 0)
            done = 0
            last = 0.0
            while True:
                if cancelled():
                    raise SetupError("Cancelled.")
                block = r.read(1 << 18)
                if not block:
                    break
                f.write(block)
                done += len(block)
                now = time.time()
                if now - last > 0.15:
                    last = now
                    yield Step("model", done, total, f"Downloading the model: {label}")
    except SetupError:
        part.unlink(missing_ok=True)
        raise
    except OSError as e:  # URLError and HTTPError are OSErrors
        part.unlink(missing_ok=True)
        _log(f"download failed {url}: {e!r}")
        raise SetupError(f"Couldn't download {label} ({e}). Check the internet connection and try again.") from e
    size = part.stat().st_size
    if expect_size is not None and size != expect_size:
        part.unlink(missing_ok=True)
        raise SetupError(f"The download of {label} was incomplete ({size:,} of {expect_size:,} bytes). Try again.")
    if expect_sha is not None:
        yield Step("model", size, size, f"Checking {label}")
        if _sha256(part) != expect_sha:
            part.unlink(missing_ok=True)
            raise SetupError(f"{label} didn't match the published file. Try again; if it keeps happening, tell "
                             f"the TraceCMR team.")
    os.replace(part, target)
    _log(f"downloaded {url} -> {target} ({size:,} bytes)")


def _pip_has_raw_progress() -> bool:
    """pip 24.1+ can print machine-readable download progress ("--progress-bar raw")."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run([python_exe(), "-m", "pip", "install", "--help"], capture_output=True, text=True,
                           timeout=60, creationflags=flags)
        return "raw" in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def _pip_install(cancelled: Callable[[], bool], spec: str = RUNTIME) -> Iterator[Step]:
    """pip install ONNX Runtime into this Python, reporting pip's own progress lines."""
    cmd = [python_exe(), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
           "--progress-bar", "raw" if _pip_has_raw_progress() else "off", spec]
    _log("running: " + " ".join(cmd))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                creationflags=flags, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except OSError as e:
        raise SetupError(f"Couldn't start the installer ({e}).") from e
    tail: list[str] = []
    text = "Installing ONNX Runtime (about 20 MB)"
    try:
        for line in proc.stdout:
            if cancelled():
                proc.terminate()
                raise SetupError("Cancelled.")
            line = line.strip()
            if not line:
                continue
            tail = (tail + [line])[-25:]
            if line.startswith("Progress "):  # --progress-bar raw: "Progress <done> of <total>"
                try:
                    _, done, _, total = line.split()
                    yield Step("runtime", int(done), int(total), text)
                except ValueError:
                    pass
                continue
            _log("pip: " + line)
            if line.startswith(("Collecting", "Downloading")) and ".metadata" not in line:
                pkg = line.split()[1].split("/")[-1][:60]
                text = f"Installing ONNX Runtime: getting {pkg}"
                yield Step("runtime", 0, 0, text)
            elif line.startswith("Installing collected packages"):
                text = "Installing ONNX Runtime: unpacking"
                yield Step("runtime", 0, 0, text)
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        detail = " ".join(tail[-4:])
        _log(f"pip failed ({rc})")
        if "No space left" in detail or "Errno 28" in detail:
            raise SetupError("Not enough disk space (it needs about 100 MB). Free some space and try again.")
        raise SetupError("Installing ONNX Runtime failed. Check the internet connection and try again. "
                         f"Details are in {log_path()}.")


def _check_runs(model: Path) -> None:
    """Load the model with ONNX Runtime in a fresh process (the same way the model worker will), with a limit."""
    code = ("import sys, numpy as np, onnxruntime as ort; s = ort.InferenceSession(sys.argv[1], "
            "providers=['CPUExecutionProvider']); x = np.zeros((1, 256, 256, 1), np.float32); "
            "p = s.run(None, {s.get_inputs()[0].name: x})[0]; print(ort.__version__, p.shape)")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run([python_exe(), "-c", code, str(model)], capture_output=True, text=True, timeout=300,
                           creationflags=flags)
    except subprocess.TimeoutExpired as e:
        raise SetupError("The model took too long to start. Restart TraceCMR and try Auto LV.") from e
    _log(f"check rc={r.returncode} out={r.stdout.strip()!r} err={r.stderr.strip()[-500:]!r}")
    if r.returncode != 0:
        raise SetupError(f"The model couldn't start on this computer. Details are in {log_path()}.")


def run_setup(dest: Path | None = None, cancelled: Callable[[], bool] = lambda: False) -> Iterator[Step]:
    """Download the model (if missing), install ONNX Runtime (if missing), and check the model runs."""
    dest = Path(dest) if dest else default_dir()
    _log(f"setup started; model folder {dest}; python {python_exe()}")
    if not model_files_ok(dest):
        for rel, url, size, sha in UCL_FILES:
            target = dest / rel
            if target.exists() and (size is None or target.stat().st_size == size):
                continue
            label = f"UCL model ({round(size / 1e6)} MB)" if rel.endswith(".onnx") else Path(rel).name
            yield from _download(url, target, size, sha, cancelled, label)
    yield Step("model", 1, 1, "Model ready")
    if not runtime_present():
        yield Step("runtime", 0, 0, "Installing ONNX Runtime (about 20 MB)")
        yield from _pip_install(cancelled)
    yield Step("check", 0, 0, "Checking that the model runs")
    _check_runs(dest / ONNX_NAME)
    _log("setup finished")
    yield Step("done", 1, 1, "Auto-contouring is ready")
