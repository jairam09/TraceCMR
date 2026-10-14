"""Write latest.json (what the app's updater reads) and the release notes for a GitHub release.

    python scripts/release_manifest.py dist/tracecmr-0.3.4-py3-none-any.whl --tag v0.3.4 \
        --out dist/latest.json --notes-md dist/notes.md

The notes are the bullets under "## 0.3.4" in CHANGELOG.md. Fails if the wheel's version isn't the tag's, or if
the changelog has no section for it.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

REPO = "jairam09/TraceCMR"
INSTALLERS = {"windows": "TraceCMR-Windows.exe", "macos": "TraceCMR-macOS.pkg"}


def wheel_metadata(wheel: Path) -> dict:
    """Name, Version, Requires-Python and every Requires-Dist line of a wheel."""
    with zipfile.ZipFile(wheel) as z:
        meta_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        text = z.read(meta_name).decode("utf-8")
    head = text.split("\n\n", 1)[0]
    out = {"requires": []}
    for line in head.splitlines():
        key, _, val = line.partition(": ")
        if key == "Name":
            out["name"] = val.strip()
        elif key == "Version":
            out["version"] = val.strip()
        elif key == "Requires-Python":
            out["requires_python"] = val.strip()
        elif key == "Requires-Dist":
            out["requires"].append(val.strip())
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def changelog_notes(changelog: Path, version: str) -> list[str]:
    """The bullets of the "## <version>" section (a heading like "## 0.3.4 (2026-10-14)" also matches)."""
    lines = changelog.read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(rf"^##\s+\[?{re.escape(version)}\]?(\s|$)", l)), None)
    if start is None:
        return []
    notes = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("- "):
            notes.append(line[2:].strip())
    return notes


def manifest(wheel: Path, url_base: str, notes: list[str], installers: dict | None = None,
             page: str = "") -> dict:
    meta = wheel_metadata(wheel)
    return {"name": "TraceCMR", "version": meta["version"], "published": date.today().isoformat(),
            "notes": notes,
            "wheel": {"name": wheel.name, "url": url_base + wheel.name, "sha256": sha256(wheel),
                      "size": wheel.stat().st_size},
            "requires": meta["requires"], "requires_python": meta.get("requires_python", ""),
            "installers": installers or {}, "page": page}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--notes-md", type=Path)
    ap.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    a = ap.parse_args()
    version = a.tag.lstrip("v")
    meta = wheel_metadata(a.wheel)
    if meta["version"] != version:
        print(f"wheel is {meta['version']} but the tag is {a.tag}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(a.wheel) as z:
        init = z.read("tracecmr/__init__.py").decode("utf-8")
    if f'__version__ = "{version}"' not in init:
        print(f"tracecmr/__init__.py in the wheel doesn't say {version}", file=sys.stderr)
        return 1
    notes = changelog_notes(a.changelog, version)
    if not notes:
        print(f"CHANGELOG.md has no '## {version}' section with bullets", file=sys.stderr)
        return 1
    base = f"https://github.com/{REPO}/releases/download/{a.tag}/"
    m = manifest(a.wheel, base, notes, {k: base + v for k, v in INSTALLERS.items()},
                 f"https://github.com/{REPO}/releases/tag/{a.tag}")
    a.out.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(m, indent=2))
    if a.notes_md:
        body = "\n".join(f"- {n}" for n in notes)
        body += ("\n\n**New install:** download TraceCMR-Windows.exe (Windows) or TraceCMR-macOS.pkg (Mac, Apple "
                 "silicon) below.\n**Already have TraceCMR 0.3.4 or later?** It offers this update when it starts "
                 "(or Help → Check for updates).\n")
        a.notes_md.write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
