"""Patch updates on a real install, start to finish, without the window (run with the installed app's Python).

    python scripts/check_update.py patches/ update_report.json

`patches/` comes from scripts/make_test_patches.py (versions X:needs, Y:broken, Z good, Y < Z). For each pretend
release it does what the app does: read the manifest, decide, download and check the wheel, then start the helper
the way TraceCMR does when it closes (here it waits for a stand-in process instead of the app):

    needs   -> must be refused as a patch (the app offers the installer)
    broken  -> the helper installs it, the import check fails, the previous version is put back and still works
    good    -> installed, imports, reports its new version; the helper opens "TraceCMR" again (a stand-in)

With --github it also asks the real release on GitHub (the app's default address) whether this copy is current.
Exit code 0 only if everything behaved.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PATCHES = Path(sys.argv[1]).resolve()
REPORT = Path(sys.argv[2]).resolve()
GITHUB = "--github" in sys.argv

from tracecmr import updater  # noqa: E402

steps: list[dict] = []


def step(name, fn):
    rec = {"name": name, "ok": True}
    t0 = time.time()
    try:
        rec["note"] = fn()
    except Exception as e:
        import traceback

        rec["ok"] = False
        rec["error"] = f"{e!r}\n{traceback.format_exc()[-1500:]}"
    rec["seconds"] = round(time.time() - t0, 1)
    steps.append(rec)
    print(("OK   " if rec["ok"] else "FAIL ") + name, json.dumps(rec.get("note", ""))[:400], rec.get("error", ""),
          flush=True)
    return rec["ok"]


def installed_version() -> str:
    """Ask a fresh Python, the way the app would start next time."""
    env_code = "import tracecmr, tracecmr.app._widget; print(tracecmr.__version__)"
    import os

    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    r = subprocess.run([updater.python_exe(), "-I", "-c", env_code], capture_output=True, text=True, env=env,
                       timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"TraceCMR doesn't import: {r.stderr[-800:]}")
    return r.stdout.strip().splitlines()[-1]


def manifests() -> dict[str, str]:
    out = {}
    for p in sorted(PATCHES.glob("latest-*.json")):
        m = json.loads(p.read_text())
        kind = "needs" if "(needs)" in m["notes"][0] else ("broken" if "(broken)" in m["notes"][0] else "good")
        out[kind] = p.as_uri()
    return out


def run_patch(url: str, label: str) -> dict:
    up = updater.check(url)
    if up is None:
        raise RuntimeError(f"{label}: the check said this copy ({updater.current_version()}) is current")
    why = updater.can_patch(up)
    if why:
        raise RuntimeError(f"{label}: refused as a patch: {why}")
    wheel = updater.download(up)
    marker = updater.updates_dir() / f"reopened-{up.version}.txt"
    marker.unlink(missing_ok=True)
    stand_in = subprocess.Popen([updater.python_exe(), "-c", "import time; time.sleep(4)"])
    reopen = [updater.python_exe(), "-c", f"open(r'{marker}', 'w').write('reopened')"]
    t0 = time.time()
    helper = updater.start_helper(wheel, up.version, relaunch=True, pid=stand_in.pid, relaunch_cmd=reopen)
    helper.wait(timeout=1500)
    took = time.time() - t0
    for _ in range(40):  # the stand-in "TraceCMR" is started by the helper as its last step
        if marker.exists():
            break
        time.sleep(0.25)
    result = json.loads(updater.result_path().read_text())
    return {"version": up.version, "result": result, "helper_exit": helper.returncode,
            "reopened": marker.exists(), "seconds": round(took, 1)}


def main() -> None:
    start = updater.current_version()
    kinds = manifests()
    print("patches:", kinds, flush=True)

    def where():
        kind = updater.install_kind()
        if kind != "app":
            raise RuntimeError(f"install kind is {kind!r}, expected 'app'")
        return {"version": start, "site": str(updater.site_dir()), "python": updater.python_exe(),
                "reopen_with": updater.relaunch_command()}

    step("installed_copy_can_patch_itself", where)

    def needs():
        up = updater.check(kinds["needs"])
        why = updater.can_patch(up)
        if not why:
            raise RuntimeError("a release that needs numpy>=99 was accepted as a patch")
        if not up.wheel_url:
            raise RuntimeError("no wheel url")
        return {"offered": up.version, "refused_because": why}

    step("needs_new_packages_is_refused", needs)

    def broken():
        out = run_patch(kinds["broken"], "broken")
        r = out["result"]
        if r.get("ok") or not r.get("rolled_back") or not r.get("restored_ok"):
            raise RuntimeError(f"expected a rollback: {out}")
        v = installed_version()
        if v != start:
            raise RuntimeError(f"after the rollback TraceCMR is {v}, expected {start}")
        if not out["reopened"]:
            raise RuntimeError("TraceCMR wasn't reopened after the failed update")
        return {**out, "after": v}

    step("broken_patch_is_rolled_back", broken)

    def good():
        out = run_patch(kinds["good"], "good")
        r = out["result"]
        if not r.get("ok"):
            raise RuntimeError(f"the update failed: {out}")
        v = installed_version()
        if v != out["version"]:
            raise RuntimeError(f"after the update TraceCMR is {v}, expected {out['version']}")
        if not out["reopened"]:
            raise RuntimeError("TraceCMR wasn't reopened after the update")
        return {**out, "after": v}

    step("good_patch_installs", good)

    def notice():
        r = updater.read_result()
        if not r or not r.get("ok"):
            raise RuntimeError(f"the next start wouldn't say it updated: {r}")
        if updater.read_result() is not None:
            raise RuntimeError("the update message would show twice")
        return r["message"]

    step("next_start_reads_the_result_once", notice)

    if GITHUB:
        def github():
            data, url = updater.fetch_manifest(updater.MANIFEST_URL)
            up = updater.parse_manifest(data, url)
            return {"latest_on_github": up.version, "wheel": up.wheel_url, "installer": up.installer_url}

        step("github_release_answers", github)

    ok = all(s["ok"] for s in steps)
    REPORT.write_text(json.dumps({"ok": ok, "start": start, "steps": steps}, indent=2))
    print(json.dumps({"ok": ok, "failed": [s["name"] for s in steps if not s["ok"]]}), flush=True)
    sys.exit(0 if ok else 1)


main()
