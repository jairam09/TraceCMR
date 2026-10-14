"""Download the pinned UCL model (ONNX) and its licence for the installers, and check them.

    python scripts/fetch_ucl_model.py OUT_DIR

Reads src/tracecmr/ucl_model.json (the file name, URL, size and SHA-256 the app also checks) and writes
OUT_DIR/ucl_seg459.onnx and OUT_DIR/UCL-LICENSE.txt. Fails if the size or SHA-256 differ from the pin.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

PIN = json.loads((Path(__file__).resolve().parents[1] / "src" / "tracecmr" / "ucl_model.json").read_text())


def fetch(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "TraceCMR-build"})
    with urllib.request.urlopen(req, timeout=120) as r, open(target, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    model = out / PIN["file"]
    fetch(PIN["url"], model)
    fetch(PIN["license_url"], out / "UCL-LICENSE.txt")
    size = model.stat().st_size
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    if size != int(PIN["size"]) or sha != PIN["sha256"]:
        print(f"{model.name}: got {size} bytes, sha256 {sha}; the pin says {PIN['size']} / {PIN['sha256']}",
              file=sys.stderr)
        return 1
    if "GNU GENERAL PUBLIC LICENSE" not in (out / "UCL-LICENSE.txt").read_text(errors="replace")[:2000]:
        print("UCL-LICENSE.txt is not the GPL text", file=sys.stderr)
        return 1
    print(f"{model.name}: {size:,} bytes, sha256 {sha} (matches the pin)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
