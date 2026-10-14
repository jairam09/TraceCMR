"""Patch updates without a network or a real install: version order, the manifest, requirement checks, checked
downloads, and the helper's copy-aside / put-back on a stand-in site-packages folder."""

import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tracecmr import updater as up

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "updates_dir", lambda: tmp_path / "updates")
    monkeypatch.setattr(up, "log_path", lambda: tmp_path / "update_log.txt")
    up.cancel_scheduled()
    yield tmp_path
    up.cancel_scheduled()


def _helper():
    spec = importlib.util.spec_from_file_location("apply_update", ROOT / "src" / "tracecmr" / "_apply_update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scripts(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_version_order():
    assert up.newer("0.3.10", "0.3.9")
    assert up.newer("0.4.0", "0.3.99")
    assert not up.newer("0.3.4", "0.3.4")
    assert not up.newer("0.3.3", "0.3.4")


def test_manifest_links_and_installer_for_this_computer(monkeypatch):
    data = {"version": "9.9.9", "notes": ["One.", "Two."],
            "wheel": {"name": "tracecmr-9.9.9-py3-none-any.whl", "url": "tracecmr-9.9.9-py3-none-any.whl",
                      "sha256": "A" * 64, "size": 10},
            "installers": {"windows": "https://x/TraceCMR-Windows.exe", "macos": "https://x/TraceCMR-macOS.pkg"}}
    monkeypatch.setattr(up, "platform_key", lambda: "macos")
    u = up.parse_manifest(data, "file:///tmp/patches/latest.json")
    assert u.wheel_url == "file:///tmp/patches/tracecmr-9.9.9-py3-none-any.whl"
    assert u.sha256 == "a" * 64 and u.installer_url.endswith("macOS.pkg") and u.notes == ["One.", "Two."]


def test_check_reads_a_manifest_file(tmp_path):
    m = tmp_path / "latest.json"
    m.write_text(json.dumps({"version": "99.0.0", "wheel": {}}))
    assert up.check(m.as_uri()).version == "99.0.0"
    m.write_text(json.dumps({"version": "0.0.1", "wheel": {}}))
    assert up.check(m.as_uri()) is None
    with pytest.raises(up.UpdateError, match="internet"):
        up.check((tmp_path / "missing.json").as_uri())
    m.write_text("<html>not json</html>")
    with pytest.raises(up.UpdateError):
        up.check(m.as_uri())


def test_requirements_this_python_does_not_meet():
    assert up.missing_requirements(["numpy>=1.0"]) == []
    assert up.missing_requirements(["numpy>=99"])[0].startswith("numpy>=99 (have ")
    assert "not installed" in up.missing_requirements(["surely-not-a-package-xyz>=1"])[0]
    assert up.missing_requirements(['surely-not-a-package-xyz; extra == "dev"']) == []  # dev extra: not ours
    assert up.missing_requirements(['numpy>=99; extra == "app"'])  # the app extra counts
    assert up.can_patch(up.Update(version="1", requires_python=">=99"))  # needs a newer Python


def _wheel_update(tmp_path, data: bytes, sha=None, size=None):
    src = tmp_path / "src" / "tracecmr-9.9.9-py3-none-any.whl"
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(data)
    return up.Update(version="9.9.9", wheel_url=src.as_uri(), wheel_name=src.name,
                     sha256=sha or hashlib.sha256(data).hexdigest(), size=len(data) if size is None else size)


def test_download_is_checked(tmp_path):
    seen = []
    w = up.download(_wheel_update(tmp_path, b"w" * 200_000), progress=lambda d, t: seen.append((d, t)))
    assert w.read_bytes() == b"w" * 200_000 and seen[-1] == (200_000, 200_000)
    with pytest.raises(up.UpdateError, match="didn't match"):
        up.download(_wheel_update(tmp_path, b"x" * 1000, sha="0" * 64))
    with pytest.raises(up.UpdateError, match="incomplete"):
        up.download(_wheel_update(tmp_path, b"y" * 1000, size=5000))
    assert not list((tmp_path / "updates").glob("*.part"))
    with pytest.raises(up.UpdateError, match="installer"):
        up.download(up.Update(version="1"))


def test_this_checkout_counts_as_a_source_install():
    assert up.install_kind() == "dev"  # tests run from the source folder: never patch it


def test_scheduled_update_waits_for_close(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(up, "start_helper", lambda *a, **k: started.append((a, k)))
    assert not up.apply_scheduled()
    up.schedule(up.Update(version="9.9.9", notes=["n"]), tmp_path / "w.whl", relaunch=True)
    assert up.scheduled()["relaunch"] and up.apply_scheduled()
    assert started[0][0][1] == "9.9.9" and started[0][1]["notes"] == ["n"]


def test_result_is_read_once(home):
    up.updates_dir().mkdir(parents=True)
    up.result_path().write_text(json.dumps({"ok": True, "to": "9.9.9"}))
    assert up.read_result()["to"] == "9.9.9"
    assert up.read_result() is None


# ---------------------------------------------------------------- the helper, on a stand-in install

def _fake_install(site: Path, version: str, files: dict[str, str]):
    info = site / f"tracecmr-{version}.dist-info"
    info.mkdir(parents=True)
    rows = []
    for rel, text in files.items():
        p = site / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        rows.append(f"{rel},sha256=x,{len(text)}")
    (info / "METADATA").write_text(f"Name: tracecmr\nVersion: {version}\n")
    rows += [f"tracecmr-{version}.dist-info/METADATA,,", f"tracecmr-{version}.dist-info/RECORD,,"]
    (info / "RECORD").write_text("\n".join(rows) + "\n")


def _args(tmp_path, site, pid):
    a = {"pid": pid, "python": sys.executable, "wheel": "w.whl", "from": "1.0", "expect": "2.0", "site": str(site),
         "dist": "tracecmr", "relaunch": None, "cwd": str(tmp_path), "result": str(tmp_path / "u" / "last.json"),
         "log": str(tmp_path / "log.txt"), "check": "", "notes": ["new"]}
    (tmp_path / "u").mkdir(exist_ok=True)
    f = tmp_path / "args.json"
    f.write_text(json.dumps(a))
    return f


def _run_helper(tmp_path, monkeypatch, check_ok: bool):
    h = _helper()
    site = tmp_path / "site"
    _fake_install(site, "1.0", {"tracecmr/__init__.py": "v1", "tracecmr/old_only.py": "old"})
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()

    def fake_pip(python, wheel):  # what pip does: remove 1.0's files, write 2.0's
        import shutil

        (site / "tracecmr" / "old_only.py").unlink()
        shutil.rmtree(site / "tracecmr-1.0.dist-info")
        _fake_install(site, "2.0", {"tracecmr/__init__.py": "v2", "tracecmr/new_only.py": "new"})
        return 0, "Successfully installed tracecmr-2.0"

    monkeypatch.setattr(h, "pip_install", fake_pip)
    calls = []

    def fake_check(python, code):
        calls.append(1)
        v = (site / "tracecmr" / "__init__.py").read_text()
        good = v == "v1" or check_ok
        return good, f"VERSION {'2.0' if v == 'v2' else '1.0'}" if good else "ImportError: broken"

    monkeypatch.setattr(h, "check_import", fake_check)
    rc = h.main(str(_args(tmp_path, site, done.pid)))
    return rc, site, json.loads((tmp_path / "u" / "last.json").read_text())


def test_helper_installs(tmp_path, monkeypatch):
    rc, site, res = _run_helper(tmp_path, monkeypatch, check_ok=True)
    assert rc == 0 and res["ok"] and res["notes"] == ["new"]
    assert (site / "tracecmr" / "__init__.py").read_text() == "v2"
    assert not (tmp_path / "u" / "previous_version").exists()


def test_helper_puts_the_old_version_back(tmp_path, monkeypatch):
    rc, site, res = _run_helper(tmp_path, monkeypatch, check_ok=False)
    assert rc == 1 and not res["ok"] and res["rolled_back"] and res["restored_ok"]
    assert (site / "tracecmr" / "__init__.py").read_text() == "v1"
    assert (site / "tracecmr" / "old_only.py").read_text() == "old"
    assert not (site / "tracecmr" / "new_only.py").exists()
    assert [d.name for d in site.glob("*.dist-info")] == ["tracecmr-1.0.dist-info"]
    assert "still 1.0" in res["message"]


def test_helper_waits_for_the_app():
    h = _helper()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert not h.wait_for_exit(p.pid, 0.6)
    finally:
        p.kill()
        p.wait()
    assert h.wait_for_exit(p.pid, 5)


# ---------------------------------------------------------------- release files

def _real_ish_wheel(tmp_path) -> Path:
    w = tmp_path / "tracecmr-0.3.4-py3-none-any.whl"
    with zipfile.ZipFile(w, "w") as z:
        z.writestr("tracecmr/__init__.py", '__version__ = "0.3.4"\n')
        z.writestr("tracecmr/app/__init__.py", "")
        z.writestr("tracecmr-0.3.4.dist-info/METADATA",
                   "Metadata-Version: 2.1\nName: tracecmr\nVersion: 0.3.4\nRequires-Python: >=3.10\n"
                   "Requires-Dist: numpy>=1.24\nRequires-Dist: napari[pyqt6]>=0.5; extra == \"app\"\n\nReadme\n")
        z.writestr("tracecmr-0.3.4.dist-info/RECORD", "")
    return w


def test_release_manifest_and_notes(tmp_path):
    rm = _scripts("release_manifest")
    w = _real_ish_wheel(tmp_path)
    meta = rm.wheel_metadata(w)
    assert meta["version"] == "0.3.4" and len(meta["requires"]) == 2
    m = rm.manifest(w, "https://example/v0.3.4/", ["Note."])
    assert m["wheel"]["url"].endswith("/v0.3.4/tracecmr-0.3.4-py3-none-any.whl")
    assert m["wheel"]["sha256"] == hashlib.sha256(w.read_bytes()).hexdigest()
    notes = rm.changelog_notes(ROOT / "CHANGELOG.md", "0.3.4")
    assert notes, "CHANGELOG.md needs a '## 0.3.4' section: the release and the update card show it"


def test_test_patches_are_valid_wheels(tmp_path):
    mp = _scripts("make_test_patches")
    w = _real_ish_wheel(tmp_path)
    out = mp.rewrite(w, tmp_path, "0.3.98", broken=True)
    with zipfile.ZipFile(out) as z:
        assert b'__version__ = "0.3.98"' in z.read("tracecmr/__init__.py")
        assert b"deliberately broken" in z.read("tracecmr/app/__init__.py")
        assert b"Version: 0.3.98" in z.read("tracecmr-0.3.98.dist-info/METADATA")
        for line in z.read("tracecmr-0.3.98.dist-info/RECORD").decode().splitlines():
            name, digest, size = line.split(",")
            if digest:
                data = z.read(name)
                want = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                assert digest == "sha256=" + want and int(size) == len(data)
