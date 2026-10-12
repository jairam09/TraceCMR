"""Do the double-check flags (quality.py) point at the slices the model actually got wrong?

For UCL's validation mice (choose thresholds) or test mice (report): the model contours each mouse at the
expert's ED and ES frames, every check's raw score is recorded per slice, and each slice is compared with the
expert. Writes one row per slice; summarise with --summarise.

A slice counts as a real error when the model drew an LV the expert did not (e.g. atrium at the base), the whole-LV
overlap with the expert (Dice) is below 0.80, or the blood-pool area is off by more than 20% (and 10 pixels):
mistakes that change the volumes. (Plain overlap alone over-counts small apical and ES slices.) Slices the model left empty but the expert drew
are counted separately (a flag on a slice can't point at a slice that wasn't drawn).

    python scripts/validate_checks.py --ucl-dir .../Data --split val --out results/checks_val.csv
    python scripts/validate_checks.py --summarise results/checks_val.csv results/checks_test.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tracecmr import autocontour as ac
from tracecmr import labels as L
from tracecmr import quality as Q
from tracecmr.app.session import Session

DICE_OK = 0.80  # whole-LV overlap with the expert below this ...
AREA_OK = 0.20  # ... or blood-pool area off by more than this (and by more than 10 pixels): a real error


def split(ucl_dir: Path, which: str) -> list[str]:
    ids = sorted({p.stem for p in ucl_dir.glob("Mouse*.h5")})
    np.random.seed(42)
    s = np.array(ids)
    np.random.shuffle(s)
    return {"train": s[:90], "val": s[90:105], "test": s[105:130]}[which].tolist()


def dice(a, b):
    t = a.sum() + b.sum()
    return 1.0 if t == 0 else 2 * np.logical_and(a, b).sum() / t


def mouse_rows(path: Path) -> list[dict]:
    ex = Session.from_ucl(path)
    sp = ex.spacing
    nz = ex.image.shape[2]
    lab = np.zeros_like(ex.label)
    probs = {}
    for t in sorted({ex.ed_frame, ex.es_frame}):
        p = ac._ucl_probs(ex.image[..., t].astype(np.float32), sp)
        probs[t] = p
        lab[..., t] = ac._to_standard(ac.labels_from_probs(p, ac.blood_cutoff(), ac.epi_cutoff()), "ucl")
    rows = []
    pairs = {}
    for ph, t in (("ED", ex.ed_frame), ("ES", ex.es_frame)):
        zs = [z for z in range(nz) if np.isin(lab[:, :, z, t], (L.LV_BLOODPOOL, L.LV_MYOCARDIUM)).any()]
        pairs[ph] = [(t, z) for z in zs]
    unsure = {(t, z): Q.unsure_score(probs[t][:, :, z], lab[:, :, z, t]) for ph in pairs for t, z in pairs[ph]}
    reasons = Q.check_slices(lab, ex.image, sp, pairs, list(range(nz)), unsure={k: 0.0 for k in unsure})
    for ph, t in (("ED", ex.ed_frame), ("ES", ex.es_frame)):
        drawn = [z for _, z in pairs[ph]]
        for z in range(nz):
            m, e = lab[:, :, z, t], ex.label[:, :, z, t]
            mlv, elv = np.isin(m, (1, 2)), np.isin(e, (1, 2))
            if not mlv.any() and not elv.any():
                continue
            d_lv = dice(mlv, elv)
            d_bp = dice(m == L.LV_BLOODPOOL, e == L.LV_BLOODPOOL)
            a_m, a_e = int((m == L.LV_BLOODPOOL).sum()), int((e == L.LV_BLOODPOOL).sum())
            area_err = abs(a_m - a_e) / max(a_e, 1)
            if mlv.any() and not elv.any():
                kind = "extra"
            elif elv.any() and not mlv.any():
                kind = "missed"
            elif d_lv < DICE_OK or (area_err > AREA_OK and abs(a_m - a_e) > 10):
                kind = "error"
            else:
                kind = "ok"
            why = reasons.get((t, z), [])
            rows.append({
                "mouse": path.stem, "phase": ph, "frame": t, "slice": z, "kind": kind,
                "dice_lv": round(d_lv, 3), "dice_bp": round(d_bp, 3), "area_err": round(area_err, 3),
                "unsure": round(unsure.get((t, z), 0.0), 4),
                "image": round(Q.image_disagreement(m, ex.image[:, :, z, t], sp[:2]), 4) if mlv.any() else 0.0,
                "jump": int(any("jumps" in w for w in why)),
                "wall": int(any(("wall" in w) for w in why)),
                "es_gt_ed": int(any("ES cavity" in w for w in why)),
                "base_apex": int(bool(drawn) and z in (drawn[0], drawn[-1])),
            })
    return rows


def summarise(paths: list[str]) -> None:
    for p in paths:
        rows = list(csv.DictReader(open(p)))
        drawn = [r for r in rows if r["kind"] != "missed"]
        bad = [r for r in drawn if r["kind"] in ("error", "extra")]
        missed = sum(r["kind"] == "missed" for r in rows)
        checks = {
            "model unsure": lambda r: float(r["unsure"]) >= Q.UNSURE_FRAC,
            "image disagrees": lambda r: float(r["image"]) >= Q.IMAGE_FRAC,
            "anatomy": lambda r: r["jump"] == "1" or r["wall"] == "1" or r["es_gt_ed"] == "1",
            "base/apex": lambda r: r["base_apex"] == "1",
        }
        print(f"\n{p}: {len(drawn)} drawn slices, {len(bad)} real errors "
              f"({sum(r['kind'] == 'extra' for r in bad)} extra slices), {missed} slices the model left empty")
        print("check            | flags slices | catches errors | of flags, real errors")
        for name, f in list(checks.items()) + [("ALL COMBINED", lambda r: any(c(r) for c in checks.values()))]:
            fl = [r for r in drawn if f(r)]
            hit = [r for r in bad if f(r)]
            print(f"{name:16} | {len(fl) / max(1, len(drawn)):5.0%} ({len(fl):3}) | {len(hit) / max(1, len(bad)):5.0%} "
                  f"({len(hit):3}/{len(bad)}) | {len(hit) / max(1, len(fl)):5.0%}")
        ok_unflagged = [r for r in drawn if not any(c(r) for c in checks.values())]
        worst = sorted(ok_unflagged, key=lambda r: float(r["dice_lv"]))[:5]
        print("lowest-Dice slices with no flag:", [(r["mouse"], r["phase"], r["slice"], r["dice_lv"]) for r in worst])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucl-dir", type=Path)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summarise", nargs="*")
    args = ap.parse_args()
    if args.summarise:
        summarise(args.summarise)
        return
    rows = []
    for m in split(args.ucl_dir, args.split):
        rows += mouse_rows(args.ucl_dir / f"{m}.h5")
        print(m, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
