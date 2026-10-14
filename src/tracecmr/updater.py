"""Patch updates: TraceCMR checks GitHub for a newer version and installs it in place, without a new installer.

Each release on github.com/jairam09/TraceCMR carries `latest.json` (the manifest), the TraceCMR wheel (well under
1 MB: only TraceCMR's own code, not Python, napari or Qt) and the full installers. The app reads the manifest,
downloads the wheel, checks its SHA-256 and, once TraceCMR has closed, a small helper (`_apply_update.py`, copied
out of the package first because it replaces the package) installs it with pip, offline, without touching any
other package. It keeps a copy of the old version and puts it back if the new one doesn't import, then opens
TraceCMR again. The next start reads the helper's result and says what happened.

A patch can only change TraceCMR itself. If a release needs other packages (a new library or newer napari), the
manifest says so and the app offers the new installer instead.

    up = check()                      # None when up to date
    wheel = download(up)              # ~/.tracecmr/updates/tracecmr-X-py3-none-any.whl
    schedule(up, wheel, relaunch=True)
    ...                               # TraceCMR closes; launch.main calls apply_scheduled()

Nothing about the user or their scans is sent: it is one plain download from GitHub, like opening the page.
TRACECMR_UPDATE_URL points the app at another manifest (testing); TRACECMR_NO_UPDATE_CHECK=1 turns the check off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = "jairam09/TraceCMR"
MANIFEST_URL = f"https://github.com/{REPO}/releases/latest/download/latest.json"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
DIST = "tracecmr"


class UpdateError(RuntimeError):
    """Something went wrong; the message is written for the user."""


@dataclass
class Update:
    version: str
    notes: list[str] = field(default_factory=list)
    wheel_url: str = ""
    wheel_name: str = ""
    sha256: str = ""
    size: int = 0
    requires: list[str] = field(default_factory=list)
    requires_python: str = ""
    installer_url: str = ""  # this computer's full installer
    page: str = RELEASES_PAGE


# ---------------------------------------------------------------- where things are

def current_version() -> str:
    from . import __version__

    return __version__


def updates_dir() -> Path:
    return Path.home() / ".tracecmr" / "updates"


def result_path() -> Path:
    """Written by the helper; read (and then renamed) by the next start."""
    return updates_dir() / "last_update.json"


def log_path() -> Path:
    return Path.home() / "Documents" / "TraceCMR results" / "logs" / "update_log.txt"


def _log(msg: str) -> None:
    try:
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except OSError:
        pass


def python_exe() -> str:
    """This Python, with a console-less pythonw.exe swapped for python.exe (pip's output must be readable)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def site_dir() -> Path | None:
    """The folder TraceCMR is installed in (site-packages), or None when it isn't installed from a wheel."""
    try:
        from importlib.metadata import PackageNotFoundError, distribution

        try:
            dist = distribution(DIST)
        except PackageNotFoundError:
            return None
        return Path(dist.locate_file("")).resolve()
    except Exception:
        return None


def install_kind() -> str:
    """"app": a normal install (the installers, or pip install of a wheel) that can patch itself.
    "dev": running from a source folder (git clone, pip install -e): update with git pull instead.
    "readonly": installed where this user can't write (e.g. by an administrator): use the installer."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution(DIST)
    except PackageNotFoundError:
        return "dev"
    try:
        info = json.loads(dist.read_text("direct_url.json") or "{}")
        if info.get("dir_info", {}).get("editable"):
            return "dev"
    except (ValueError, OSError):
        pass
    import tracecmr

    pkg = Path(tracecmr.__file__).resolve().parent
    site = site_dir()
    if site is None or pkg.parent != site:
        return "dev"  # the code that's running isn't the installed copy
    meta = getattr(dist, "_path", None)  # .dist-info for a wheel install; .egg-info in a source folder
    if meta is not None and not str(meta).endswith(".dist-info"):
        return "dev"
    if any((d / "pyproject.toml").exists() for d in (pkg.parent, pkg.parent.parent)):
        return "dev"  # a source checkout (src/ layout or flat)
    if not os.access(site, os.W_OK) or not os.access(pkg, os.W_OK):
        return "readonly"
    return "app"


def platform_key() -> str:
    return "windows" if sys.platform.startswith("win") else ("macos" if sys.platform == "darwin" else "linux")


# ---------------------------------------------------------------- versions and requirements

def _packaging():
    try:
        import packaging.requirements
        import packaging.specifiers
        import packaging.version

        return packaging
    except ImportError:
        from pip._vendor import packaging  # pip carries its own copy

        import pip._vendor.packaging.requirements  # noqa: F401
        import pip._vendor.packaging.specifiers  # noqa: F401
        import pip._vendor.packaging.version  # noqa: F401

        return packaging


def _vkey(v: str):
    try:
        return _packaging().version.Version(v)
    except Exception:
        return tuple(int(x) for x in re.findall(r"\d+", v)[:4])


def newer(candidate: str, than: str) -> bool:
    try:
        return _vkey(candidate) > _vkey(than)
    except TypeError:
        return False


def missing_requirements(requires: list[str], extras: tuple[str, ...] = ("", "app")) -> list[str]:
    """The requirements of the new version (its Requires-Dist lines) this Python doesn't meet. Only TraceCMR's
    core and its "app" extra count: that's what the installers carry."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        pk = _packaging()
    except ImportError:
        return []  # can't judge; pip --no-deps installs only TraceCMR, and the import check catches the rest
    out = []
    for line in requires:
        try:
            req = pk.requirements.Requirement(line)
        except Exception:
            continue
        if req.marker is not None and not any(req.marker.evaluate({"extra": e}) for e in extras):
            continue
        try:
            have = version(req.name)
        except PackageNotFoundError:
            out.append(f"{line} (not installed)")
            continue
        if req.specifier and not req.specifier.contains(have, prereleases=True):
            out.append(f"{line} (have {have})")
    return out


def python_ok(requires_python: str) -> bool:
    if not requires_python:
        return True
    try:
        pk = _packaging()
        return pk.specifiers.SpecifierSet(requires_python).contains(
            ".".join(map(str, sys.version_info[:3])), prereleases=True)
    except Exception:
        return True


def can_patch(up: Update) -> list[str]:
    """Why this update can't be installed as a patch here (empty list = it can)."""
    why = []
    if not python_ok(up.requires_python):
        why.append(f"needs Python {up.requires_python}")
    why += missing_requirements(up.requires)
    return why


# ---------------------------------------------------------------- the manifest

def manifest_url() -> str:
    return os.environ.get("TRACECMR_UPDATE_URL") or MANIFEST_URL


def _open(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": f"TraceCMR/{current_version()}",
                                               "Cache-Control": "no-cache"})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_manifest(url: str | None = None, timeout: float = 15) -> tuple[dict, str]:
    url = url or manifest_url()
    try:
        with _open(url, timeout) as r:
            data = json.loads(r.read(1_000_000).decode("utf-8"))
    except (OSError, ValueError) as e:  # URLError/HTTPError are OSErrors
        raise UpdateError(f"Couldn't check for updates ({e}). Check the internet connection.") from e
    if not isinstance(data, dict) or "version" not in data:
        raise UpdateError("The update information from GitHub didn't make sense. Try again later.")
    return data, url


def parse_manifest(data: dict, url: str) -> Update:
    wheel = data.get("wheel") or {}
    inst = data.get("installers") or {}
    join = lambda u: urllib.parse.urljoin(url, u) if u else ""  # noqa: E731  (relative links: test manifests)
    notes = data.get("notes") or []
    return Update(version=str(data["version"]), notes=[str(n) for n in notes][:8],
                  wheel_url=join(wheel.get("url", "")), wheel_name=str(wheel.get("name", "")),
                  sha256=str(wheel.get("sha256", "")).lower(), size=int(wheel.get("size") or 0),
                  requires=list(data.get("requires") or []), requires_python=str(data.get("requires_python") or ""),
                  installer_url=join(inst.get(platform_key(), "")), page=join(data.get("page", "")) or RELEASES_PAGE)


def check(url: str | None = None) -> Update | None:
    """The newest release if it's newer than this copy, else None."""
    data, url = fetch_manifest(url)
    up = parse_manifest(data, url)
    _log(f"checked {url}: latest {up.version}, this copy {current_version()}")
    return up if newer(up.version, current_version()) else None


# ---------------------------------------------------------------- download

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(up: Update, progress: Callable[[int, int], None] | None = None,
             cancelled: Callable[[], bool] = lambda: False) -> Path:
    """Fetch the wheel into the updates folder (via a .part file) and check its size and SHA-256."""
    if not up.wheel_url or not up.wheel_name.endswith(".whl") or len(up.sha256) != 64:
        raise UpdateError("This release has no patch file. Download the new installer instead.")
    name = Path(up.wheel_name).name
    target = updates_dir() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _sha256(target) == up.sha256:
        return target
    part = target.with_name(name + ".part")
    try:
        with _open(up.wheel_url, 60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or up.size
            done = 0
            while True:
                if cancelled():
                    raise UpdateError("Cancelled.")
                block = r.read(1 << 16)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if progress:
                    progress(done, total)
    except UpdateError:
        part.unlink(missing_ok=True)
        raise
    except OSError as e:
        part.unlink(missing_ok=True)
        _log(f"download failed {up.wheel_url}: {e!r}")
        raise UpdateError(f"Couldn't download the update ({e}). Check the internet connection and try again.") from e
    if up.size and part.stat().st_size != up.size:
        n = part.stat().st_size
        part.unlink(missing_ok=True)
        raise UpdateError(f"The download was incomplete ({n:,} of {up.size:,} bytes). Try again.")
    if _sha256(part) != up.sha256:
        part.unlink(missing_ok=True)
        raise UpdateError("The downloaded update didn't match the published file, so it wasn't used. Try again.")
    os.replace(part, target)
    _log(f"downloaded {up.wheel_url} -> {target}")
    return target


# ---------------------------------------------------------------- install (after TraceCMR closes)

_scheduled: dict | None = None


def relaunch_command() -> tuple[list[str], str]:
    """How to open TraceCMR again, the way its shortcut does, and the folder to start in."""
    home = Path.home()
    cwd = str(home / "Documents") if (home / "Documents").is_dir() else str(home)
    if sys.platform == "darwin":
        app = home / "Applications" / "TraceCMR.app"
        if app.is_dir():
            return ["open", str(app)], cwd
    scripts = Path(sysconfig.get_path("scripts"))
    exe = scripts / ("tracecmr-app.exe" if os.name == "nt" else "tracecmr-app")
    if exe.exists():
        return [str(exe)], cwd
    return [python_exe(), "-m", "tracecmr.app"], cwd


def schedule(up: Update, wheel: Path, relaunch: bool) -> None:
    """Install `wheel` as soon as TraceCMR closes (and open it again if `relaunch`)."""
    global _scheduled
    _scheduled = {"version": up.version, "wheel": str(wheel), "relaunch": bool(relaunch), "notes": list(up.notes)}
    _log(f"scheduled {up.version} (relaunch={relaunch})")


def scheduled() -> dict | None:
    return _scheduled


def cancel_scheduled() -> None:
    global _scheduled
    _scheduled = None


def apply_scheduled() -> bool:
    """Called by the app as its very last step: start the helper if an update is waiting."""
    s = _scheduled
    if not s:
        return False
    try:
        start_helper(Path(s["wheel"]), s["version"], relaunch=s["relaunch"], notes=s.get("notes"))
        return True
    except Exception as e:  # never stand in the way of closing
        _log(f"couldn't start the update helper: {e!r}")
        return False


IMPORT_CHECK = ("import tracecmr, tracecmr.app._widget, tracecmr.app.compare; "
                "from importlib.metadata import version; print('VERSION', tracecmr.__version__, version('tracecmr'))")


def _launcher_pids() -> list[int]:
    """Windows: the desktop shortcut runs Scripts/tracecmr-app.exe, a small launcher that starts Python and waits
    for it. The launcher file can't be replaced while it runs, so the helper waits for it as well."""
    if os.name != "nt":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        ppid = os.getppid()
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, ppid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return []
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if not k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return []
            name = Path(buf.value).name.lower()
        finally:
            k32.CloseHandle(h)
        return [ppid] if name.startswith("tracecmr") else []
    except Exception:
        return []


def start_helper(wheel: Path, expect: str, relaunch: bool = True, pid: int | None = None,
                 relaunch_cmd: list[str] | None = None, notes: list[str] | None = None) -> subprocess.Popen:
    """Copy the helper out of the package and start it, detached: it waits for process `pid` (this one) to end,
    then installs `wheel`. Returns the helper's process (tests wait on it)."""
    site = site_dir()
    if site is None:
        raise UpdateError("TraceCMR isn't installed from a package here, so it can't patch itself.")
    d = updates_dir()
    d.mkdir(parents=True, exist_ok=True)
    helper = d / "apply_update.py"
    shutil.copyfile(Path(__file__).with_name("_apply_update.py"), helper)
    cmd, cwd = relaunch_command()
    if relaunch_cmd is not None:
        cmd = relaunch_cmd
    args = {"pid": os.getpid() if pid is None else pid, "also_wait": _launcher_pids() if pid is None else [],
            "python": python_exe(), "wheel": str(wheel),
            "from": current_version(), "expect": expect, "site": str(site), "dist": DIST,
            "relaunch": cmd if relaunch else None, "cwd": cwd, "result": str(result_path()),
            "log": str(log_path()), "check": IMPORT_CHECK, "notes": list(notes or [])}
    argfile = d / "apply_update.json"
    argfile.write_text(json.dumps(args, indent=1), encoding="utf-8")
    result_path().unlink(missing_ok=True)
    popen = [python_exe(), "-I", str(helper), str(argfile)]
    kw: dict = {"cwd": str(d), "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL, "close_fds": True}
    _log(f"starting helper: {popen} (from {current_version()} to {expect}, relaunch={relaunch})")
    if os.name == "nt":
        base = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:  # outlive the app even if it runs inside a job that closes with it
            return subprocess.Popen(popen, creationflags=base | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
        except OSError:
            return subprocess.Popen(popen, creationflags=base, **kw)
    return subprocess.Popen(popen, start_new_session=True, **kw)


# ---------------------------------------------------------------- after a restart

def read_result() -> dict | None:
    """What the helper did last time, once: the file is renamed after reading so the message shows once."""
    p = result_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    try:
        os.replace(p, p.with_name("last_update.shown.json"))
    except OSError:
        pass
    return data if isinstance(data, dict) else None


def clean_downloads(keep: str | None = None) -> None:
    """Remove downloaded wheels other than `keep` (they are small, but don't pile up)."""
    d = updates_dir()
    if not d.is_dir():
        return
    for p in d.glob("*.whl"):
        if keep is None or p.name != keep:
            try:
                p.unlink()
            except OSError:
                pass
