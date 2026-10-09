"""How much do results change when pixel contours are converted to curves?

For every UCL mouse: volumes, EF and mass from pixel counts (current method)
versus from curves fitted to the same contours, at several numbers of control
points. Writes one row per mouse and setting.

Example:
    python scripts/curve_conversion_test.py --ucl-dir data/ucl/Data --out results/curve_test.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from tracecmr import labels as L
from tracecmr.contours import mask_area_by_curve
from tracecmr.datasets import load_ucl

SETTINGS = {"pixels": "pixels", "outline": (None, None), "default": (18, 1.5), "32pt": (32, None),
            "24pt": (24, None), "16pt": (16, None), "12pt": (12, None)}
DENSITY = 1.05  # g/mL


def areas(ci, setting):
    blood, lv = [], []
    for z in range(ci.label.shape[2]):
        b = ci.label[:, :, z] == L.LV_BLOODPOOL
        o = b | (ci.label[:, :, z] == L.LV_MYOCARDIUM)
        if setting == "pixels":
            blood.append(float(b.sum()))
            lv.append(float(o.sum()))
        else:
            blood.append(mask_area_by_curve(b, *SETTINGS[setting]))
            lv.append(mask_area_by_curve(o, *SETTINGS[setting]))
    return np.array(blood), np.array(lv)


def measures(ed, es, setting):
    dx, dy, dz = ed.spacing
    vox = dx * dy * dz
    b_ed, o_ed = areas(ed, setting)
    b_es, o_es = areas(es, setting)
    edv, esv = b_ed.sum() * vox, b_es.sum() * vox
    return {
        "edv_ul": edv,
        "esv_ul": esv,
        "ef_pct": 100 * (edv - esv) / edv,
        "mass_ed_mg": DENSITY * (o_ed - b_ed).sum() * vox,
        "mass_es_mg": DENSITY * (o_es - b_es).sum() * vox,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucl-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/curve_test.csv"))
    args = ap.parse_args()
    files = sorted(args.ucl_dir.glob("Mouse*.h5"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    rows = []
    for p in files:
        ed, es = load_ucl(p, "ED"), load_ucl(p, "ES")
        for setting in SETTINGS:
            rows.append({"subject_id": p.stem, "setting": setting, **measures(ed, es, setting)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(files)} mice x {len(SETTINGS)} settings -> {args.out}")


if __name__ == "__main__":
    main()
