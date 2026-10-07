"""Whole-cohort summary: volumes, EF, mass and regional wall metrics, one row per mouse.

Built for validation runs on the UCL dataset (130 mice). Mice whose orientation
cannot be decided are listed with the reason instead of being guessed.

Example:
    python scripts/run_cohort.py --ucl-dir data/ucl/Data --out results/cohort.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from ventrimap.datasets import load_ucl
from ventrimap.orientation import OrientationUncertain
from ventrimap.volumes import compute_function
from ventrimap.wall import compute_wall

LEVELS = ("basal", "mid", "apical")


def mouse_number(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def summarise(path: Path) -> dict:
    ed, es = load_ucl(path, "ED"), load_ucl(path, "ES")
    row = {"mouse": mouse_number(path), "subject_id": ed.subject_id, "status": "ok"}
    f = compute_function(ed, es)
    row.update({k: f.as_row()[k] for k in ("edv_ul", "esv_ul", "sv_ul", "ef_pct", "mass_ed_mg", "mass_es_mg")})
    row["volume_flags"] = "; ".join(f.flags)
    try:
        w = compute_wall(ed, es)
    except OrientationUncertain as e:
        row["status"] = f"wall skipped: {e}"
        return row
    for lv in LEVELS:
        if lv in w.ed.levels:
            row[f"edwt_{lv}_mm"] = float(np.nanmean(w.ed.levels[lv].segments_mm))
            row[f"thick_{lv}_pct"] = float(np.nanmean(w.thickening_pct[lv]))
    edwt = np.concatenate([w.ed.levels[lv].segments_mm for lv in LEVELS if lv in w.ed.levels])
    row["edwt_mean_mm"] = float(np.nanmean(edwt))
    # Regional metrics on mid + apical rings only (basal ring is the least reliable, see methods.md).
    th = np.concatenate([w.thickening_pct[lv] for lv in ("mid", "apical") if lv in w.thickening_pct])
    wt = np.concatenate([w.ed.levels[lv].segments_mm for lv in ("mid", "apical") if lv in w.ed.levels])
    row["thick_midapical_pct"] = float(np.nanmean(th))
    row["thick_min_seg_pct"] = float(np.nanmin(th))
    row["thick_spread_pct"] = float(np.nanmedian(th) - np.nanmin(th))
    row["edwt_min_ratio"] = float(np.nanmin(wt) / np.nanmedian(wt))
    row["wall_flags"] = "; ".join(w.flags)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucl-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/cohort.csv"))
    args = ap.parse_args()

    files = sorted(args.ucl_dir.glob("Mouse*.h5"), key=mouse_number)
    if not files:
        raise SystemExit(f"no Mouse*.h5 files in {args.ucl_dir}")
    rows = []
    for p in files:
        try:
            rows.append(summarise(p))
        except Exception as e:  # report and continue: one bad file must not stop a cohort run
            rows.append({"mouse": mouse_number(p), "subject_id": p.stem, "status": f"error: {e}"})
        r = rows[-1]
        print(f"{r['subject_id']:9s} {r['status'][:60]:12s} EF {r.get('ef_pct', float('nan')):5.1f}")

    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} mice -> {args.out}")


if __name__ == "__main__":
    main()
