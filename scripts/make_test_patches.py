"""Make pretend TraceCMR releases from a real wheel, to test patch updates without publishing anything.

    python scripts/make_test_patches.py tracecmr-0.3.4-py3-none-any.whl patches/ 0.3.5 0.3.98:broken 0.4.0:needs

For each version it writes patches/tracecmr-<version>-py3-none-any.whl (the same code with that version number)
and patches/latest-<version>.json, a manifest that links the wheel by a relative path, so the app can be pointed
at it with TRACECMR_UPDATE_URL=file:///.../latest-<version>.json.

    :broken  the wheel installs but `import tracecmr.app` fails (the updater must put the old version back)
    :needs   the manifest asks for numpy>=99 (the app must offer the full installer instead of patching)

Standard library only, so the installed app's own Python can run it.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_manifest import manifest, record_hash  # noqa: E402


def rewrite(base: Path, out_dir: Path, version: str, broken: bool) -> Path:
    with zipfile.ZipFile(base) as z:
        names = z.namelist()
        old_info = next(n.split("/")[0] for n in names if n.split("/")[0].endswith(".dist-info"))
        dist = old_info.split("-")[0]
        new_info = f"{dist}-{version}.dist-info"
        files: dict[str, bytes] = {}
        for n in names:
            data = z.read(n)
            new = new_info + n[len(old_info):] if n.startswith(old_info + "/") else n
            if new == f"{new_info}/METADATA":
                data = re.sub(rb"(?m)^Version: [^\r\n]*", f"Version: {version}".encode(), data, count=1)
            elif new == f"{dist}/__init__.py":
                data, count = re.subn(rb'(?m)^__version__ = "[^"]*"', f'__version__ = "{version}"'.encode(), data,
                                      count=1)
                if count != 1:
                    raise SystemExit(f"no __version__ line in {new}")
            elif new == f"{dist}/app/__init__.py" and broken:
                data += b'\nraise ImportError("test patch: deliberately broken")\n'
            files[new] = data
    rec_name = f"{new_info}/RECORD"
    lines = [f"{n},{record_hash(d)},{len(d)}" for n, d in files.items() if n != rec_name]
    files[rec_name] = ("\n".join(lines + [f"{rec_name},,"]) + "\n").encode()
    tag = base.name.split("-", 2)[2]  # py3-none-any.whl
    out = out_dir / f"{dist}-{version}-{tag}"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in files.items():
            z.writestr(n, d)
    return out


def main() -> int:
    base, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[3:]:
        version, _, kind = spec.partition(":")
        whl = rewrite(base, out_dir, version, broken=(kind == "broken"))
        m = manifest(whl, "", [f"Test patch {version}" + (f" ({kind})" if kind else ""),
                               "Made by scripts/make_test_patches.py"])
        if kind == "needs":
            m["requires"] = m["requires"] + ["numpy>=99"]
        (out_dir / f"latest-{version}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"{whl.name}  {m['wheel']['sha256'][:12]}  {kind or 'good'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
