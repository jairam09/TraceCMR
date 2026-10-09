"""Regional wall thickness and systolic thickening for every UCL mouse in a folder.

Writes one CSV row per mouse, level and segment, plus bullseye plots
(end-diastolic thickness and systolic thickening) for each mouse.

Example:
    python scripts/run_wall.py --ucl-dir data/ucl/Data --out results/wall
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from tracecmr.datasets import load_ucl
from tracecmr.orientation import OrientationUncertain
from tracecmr.viz import save_bullseye
from tracecmr.wall import compute_wall


def mouse_number(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucl-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/wall"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    files = sorted(args.ucl_dir.glob("Mouse*.h5"), key=mouse_number)
    if not files:
        raise SystemExit(f"no Mouse*.h5 files in {args.ucl_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in files:
        try:
            r = compute_wall(load_ucl(p, "ED"), load_ucl(p, "ES"))
        except OrientationUncertain as e:
            print(f"{p.stem:9s} skipped: {e}")
            continue
        rows.extend(r.rows())
        edwt = np.nanmean([row["edwt_mm"] for row in r.rows()])
        th = np.nanmean([row["thickening_pct"] for row in r.rows()])
        warn = [f for f in r.flags if f.startswith("negative")]
        print(f"{r.subject_id:9s} EDWT {edwt:4.2f} mm  thickening {th:5.1f}%" + (f"  <- {'; '.join(warn)}" if warn else ""))
        if not args.no_plots:
            ed_vals = {lv: res.segments_mm for lv, res in r.ed.levels.items()}
            save_bullseye(ed_vals, args.out / f"{r.subject_id}_edwt.png", f"{r.subject_id} ED wall thickness",
                          "mm", vmin=0.4, vmax=1.6)
            save_bullseye(r.thickening_pct, args.out / f"{r.subject_id}_thickening.png",
                          f"{r.subject_id} systolic thickening", "%", vmin=-20, vmax=100, cmap="RdYlGn", fmt="{:.0f}")

    with open(args.out / "wall_thickness.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out / 'wall_thickness.csv'}")
    print("note: segments are numbered from the image x-axis; papillary convention:", rows[0]["papillary"])


if __name__ == "__main__":
    main()
