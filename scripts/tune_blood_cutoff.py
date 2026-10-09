"""Choose the blood-pool cutoff (autocontour.refine_blood) on UCL's 15 VALIDATION mice, never the test mice.

For each mouse the model runs once at the expert's ED and ES frames; each cutoff is then applied to the same
probabilities and the volumes compared with the expert's (same slice logic as the app).

Example:
    python scripts/tune_blood_cutoff.py --ucl-dir .../Data --split val --out results/blood_cutoff_val.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tracecmr import autocontour as ac
from tracecmr import labels as L
from tracecmr.app.session import Session

CUTOFFS = (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15)
MEASURES = ("edv_ul", "esv_ul", "ef_pct", "mass_ed_mg")


def split(ucl_dir: Path, which: str) -> list[str]:
    ids = sorted({p.stem for p in ucl_dir.glob("Mouse*.h5")})
    np.random.seed(42)
    s = np.array(ids)
    np.random.shuffle(s)
    return {"train": s[:90], "val": s[90:105], "test": s[105:130]}[which].tolist()


def dice(a, b):
    t = a.sum() + b.sum()
    return np.nan if t == 0 else 2 * np.logical_and(a, b).sum() / t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucl-dir", required=True, type=Path)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cutoffs", default=",".join(map(str, CUTOFFS)))
    ap.add_argument("--epi", action="store_true", help="tune the epicardial cutoff instead (blood cutoff 0.5)")
    args = ap.parse_args()
    cutoffs = [float(c) for c in args.cutoffs.split(",")]
    rows = []
    for m in split(args.ucl_dir, args.split):
        path = args.ucl_dir / f"{m}.h5"
        ex = Session.from_ucl(path)
        fe, _, _ = ex.compute()
        ref = fe.as_row()
        probs = {t: ac._ucl_probs(ex.image[..., t].astype(np.float32), ex.spacing) for t in (ex.ed_frame, ex.es_frame)}
        for c in cutoffs:
            s = Session.from_ucl(path, with_labels=False)
            s.papillary, s.ed_frame, s.es_frame = "myocardium", ex.ed_frame, ex.es_frame
            for t, p in probs.items():
                lab = ac.labels_from_probs(p, 0.5, c) if args.epi else ac.labels_from_probs(p, c)
                s.label[..., t] = ac._to_standard(lab, "ucl")
            s.auto_order()
            s.apply_suggestions()
            fn, _, _ = s.compute()
            r = {"mouse": m, "cutoff": c}
            for k in MEASURES:
                r[k], r[f"{k}_expert"] = fn.as_row()[k], ref[k]
            for ph, t in (("ed", ex.ed_frame), ("es", ex.es_frame)):
                r[f"dice_bp_{ph}"] = dice(s.label[..., t] == L.LV_BLOODPOOL, ex.label[..., t] == L.LV_BLOODPOOL)
                r[f"dice_myo_{ph}"] = dice(s.label[..., t] == L.LV_MYOCARDIUM, ex.label[..., t] == L.LV_MYOCARDIUM)
            rows.append(r)
        print(m, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{args.split} mice: TraceCMR minus expert, mean (SD); Dice mean")
    print("cutoff | EDV | ESV | EF | mass | Dice bp ED/ES | Dice myo ED/ES")
    for c in cutoffs:
        rr = [r for r in rows if r["cutoff"] == c]
        cells = []
        for k in MEASURES:
            d = np.array([r[k] - r[f"{k}_expert"] for r in rr])
            cells.append(f"{d.mean():+.1f} ({d.std(ddof=1):.1f})")
        dd = [np.nanmean([r[k] for r in rr]) for k in ("dice_bp_ed", "dice_bp_es", "dice_myo_ed", "dice_myo_es")]
        print(f"{c:.2f} | " + " | ".join(cells) + f" | {dd[0]:.3f}/{dd[1]:.3f} | {dd[2]:.3f}/{dd[3]:.3f}")


if __name__ == "__main__":
    main()
