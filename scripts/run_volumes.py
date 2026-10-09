"""Compute EDV, ESV, SV, EF and LV mass for every UCL mouse in a folder.

Example:
    python scripts/run_volumes.py --ucl-dir data/ucl/Data --out results/ucl_volumes.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from tracecmr.datasets import load_ucl
from tracecmr.volumes import compute_function


def mouse_number(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucl-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/ucl_volumes.csv"))
    args = ap.parse_args()

    files = sorted(args.ucl_dir.glob("Mouse*.h5"), key=mouse_number)
    if not files:
        raise SystemExit(f"no Mouse*.h5 files in {args.ucl_dir}")

    rows = []
    for p in files:
        r = compute_function(load_ucl(p, "ED"), load_ucl(p, "ES"))
        rows.append(r.as_row())
        flag = f"  <- {r.as_row()['flags']}" if r.flags else ""
        print(f"{r.subject_id:9s} EDV {r.edv_ul:6.1f}  ESV {r.esv_ul:6.1f}  EF {r.ef_pct:5.1f}%  "
              f"mass {r.mass_ed_mg:6.1f} mg{flag}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for key, unit in (("edv_ul", "uL"), ("esv_ul", "uL"), ("ef_pct", "%"), ("mass_ed_mg", "mg")):
        v = np.array([row[key] for row in rows])
        print(f"{key:11s} median {np.median(v):6.1f} {unit}  range {v.min():6.1f} to {v.max():6.1f}")
    print(f"papillary convention: {rows[0]['papillary']}   ({len(rows)} mice)  -> {args.out}")


if __name__ == "__main__":
    main()
