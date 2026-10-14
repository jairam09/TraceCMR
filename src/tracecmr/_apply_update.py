"""Install a TraceCMR patch after the app has closed (started by tracecmr.updater; standard library only).

It runs as a copy outside the package, because it replaces the package:

    python -I apply_update.py apply_update.json

1. wait for TraceCMR (the process that started it) to end
2. copy every file of the installed version aside (the list in its RECORD)
3. pip install the new wheel, offline, TraceCMR only (--no-deps --no-index)
4. import the new version in a fresh Python and check its version number
5. if 3 or 4 failed: remove what was installed and put the copied files back, then check again
6. write the result for the next start, and open TraceCMR again

Everything it does goes to Documents/TraceCMR results/logs/update_log.txt.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
A: dict = {}


def log(msg: str) -> None:
    try:
        p = Path(A.get("log") or "update_log.txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  [helper] {msg}\n")
    except OSError:
        pass


# ---------------------------------------------------------------- 1. wait for the app

def _alive_posix(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a finished process its parent hasn't collected yet still answers: count it as gone
        r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip()) and not r.stdout.strip().startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


def wait_for_exit(pid: int, timeout: float) -> bool:
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
        if not handle:
            return True  # already gone
        try:
            return k32.WaitForSingleObject(handle, int(timeout * 1000)) == 0  # WAIT_OBJECT_0
        finally:
            k32.CloseHandle(handle)
    end = time.time() + timeout
    while time.time() < end:
        if not _alive_posix(pid):
            return True
        time.sleep(0.25)
    return False


# ---------------------------------------------------------------- 2. and 5. copy aside, put back

def _dist_info(site: Path, dist: str) -> Path | None:
    for d in site.glob("*.dist-info"):
        if d.name.split("-")[0].lower().replace("_", "-") == dist:
            return d
    return None


def _record_files(site: Path, info: Path) -> list[Path]:
    out = []
    rec = info / "RECORD"
    with open(rec, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0]:
                p = Path(os.path.normpath(site / row[0]))
                out.append(p)
    return out


def backup(site: Path, dist: str, into: Path) -> dict:
    """Copy every installed file of the current version to `into`; returns {original path: copy}."""
    info = _dist_info(site, dist)
    if info is None:
        raise RuntimeError("the installed version's file list (RECORD) wasn't found")
    if into.exists():
        shutil.rmtree(into)
    into.mkdir(parents=True)
    saved = {}
    for i, p in enumerate(_record_files(site, info)):
        if p.is_file():
            dst = into / f"{i:05d}_{p.name}"
            shutil.copy2(p, dst)
            saved[str(p)] = str(dst)
    (into / "files.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")
    log(f"copied {len(saved)} files of the installed version aside")
    return saved


def restore(site: Path, dist: str, saved: dict) -> None:
    """Remove whatever version is installed now and put the copied files back."""
    info = _dist_info(site, dist)
    if info is not None:
        try:
            for p in _record_files(site, info):
                if str(p) not in saved and p.is_file():
                    p.unlink()
        except OSError as e:
            log(f"restore: removing the new files: {e!r}")
        if str(info) not in {str(Path(k).parent) for k in saved}:
            shutil.rmtree(info, ignore_errors=True)
    for orig, copy in saved.items():
        Path(orig).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(copy, orig)
    log(f"put back {len(saved)} files of the previous version")


# ---------------------------------------------------------------- 3. and 4. install, check

def run(cmd: list[str], timeout: float, env: dict | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW,
                           env=env, stdin=subprocess.DEVNULL)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired as e:
        return -9, f"timed out after {timeout:.0f} s: {e}"
    except OSError as e:
        return -1, repr(e)


def pip_install(python: str, wheel: str) -> tuple[int, str]:
    return run([python, "-m", "pip", "install", "--no-deps", "--no-index", "--force-reinstall", "--no-compile",
                "--disable-pip-version-check", "--no-input", wheel], 600)


def check_import(python: str, code: str) -> tuple[bool, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "QT_"))}
    env["QT_QPA_PLATFORM"] = "offscreen"
    rc, out = run([python, "-I", "-c", code], 600, env)
    found = ""
    for line in out.splitlines():
        if line.startswith("VERSION "):
            found = line
    return rc == 0, found or out.strip()[-800:]


# ---------------------------------------------------------------- 6. reopen

def relaunch(cmd: list[str] | None, cwd: str) -> None:
    if not cmd:
        return
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    try:
        kw: dict = {"cwd": cwd if os.path.isdir(cwd) else None, "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "close_fds": True, "env": env}
        if os.name == "nt":
            subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                             **kw)
        else:
            subprocess.Popen(cmd, start_new_session=True, **kw)
        log(f"reopened TraceCMR: {cmd}")
    except OSError as e:
        log(f"couldn't reopen TraceCMR ({cmd}): {e!r}")


def main(argfile: str) -> int:
    A.update(json.loads(Path(argfile).read_text(encoding="utf-8")))
    site, dist, python = Path(A["site"]), A["dist"], A["python"]
    result = {"from": A["from"], "to": A["expect"], "ok": False, "rolled_back": False, "message": "",
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    log(f"update {A['from']} -> {A['expect']} with {A['wheel']}")
    saved = None
    reopen = True
    try:
        if not wait_for_exit(int(A["pid"]), 120):
            result["message"] = "TraceCMR didn't close, so the update wasn't installed. Try again."
            log(result["message"])
            reopen = False
            return 1
        for extra in A.get("also_wait", []):  # Windows: the tracecmr-app.exe launcher that started the app
            if not wait_for_exit(int(extra), 30):
                log(f"process {extra} still running after 30 s; installing anyway")
        time.sleep(0.5)  # let the system release the app's files
        saved = backup(site, dist, Path(A["result"]).parent / "previous_version")
        rc, out = pip_install(python, A["wheel"])
        log(f"pip exit {rc}: {out.strip()[-1500:]}")
        if rc != 0:
            raise RuntimeError(f"pip install failed ({rc}): {out.strip()[-300:]}")
        ok, found = check_import(python, A["check"])
        log(f"check: ok={ok} {found}")
        if not ok or A["expect"] not in found.split():
            raise RuntimeError(f"the new version didn't start: {found[-300:]}")
        result.update(ok=True, message=f"Updated to TraceCMR {A['expect']}.", notes=A.get("notes", []))
        return 0
    except Exception as e:
        log(f"FAILED: {e!r}")
        result["detail"] = str(e)[-600:]
        if saved is not None:
            try:
                restore(site, dist, saved)
                ok, found = check_import(python, A["check"])
                log(f"after putting the previous version back: ok={ok} {found}")
                result["rolled_back"] = True
                result["restored_ok"] = ok
            except Exception as e2:
                log(f"restore FAILED: {e2!r}")
                result["restored_ok"] = False
        result["message"] = (f"The update to {A['expect']} couldn't be installed, so TraceCMR is still "
                             f"{A['from']}. Nothing else changed.")
        return 1
    finally:
        try:
            Path(A["result"]).write_text(json.dumps(result, indent=1), encoding="utf-8")
        except OSError as e:
            log(f"couldn't write the result: {e!r}")
        if result["ok"]:
            shutil.rmtree(Path(A["result"]).parent / "previous_version", ignore_errors=True)
        if reopen:
            relaunch(A.get("relaunch"), A.get("cwd", ""))
        log("done")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
