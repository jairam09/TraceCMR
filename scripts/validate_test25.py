"""Validate the whole TraceCMR pipeline on UCL's 25 held-out test mice, against the expert.

Each mouse is opened as a raw cine (expert contours hidden) and analysed twice, as a user would:

  A  "expert frames":  auto-contour at the expert's ED and ES frames, then the app's own slice order and
                       slice inclusion. Isolates contour accuracy.
  B  "end to end":     auto-contour every frame, ED/ES picked by the app from the volume curve, then the
                       app's own slice order and inclusion. The full pipeline, nothing from the expert.

The expert's numbers come from the expert contours at the expert's frames (same volume code).
The test split is reproduced from UCL's notebook (sorted mouse IDs, np.random.seed(42), last 25 of
90/15/25), so these mice were never used to train or tune the model.

Output (one folder per mouse, plus a summary):
  <out>/<MouseN>/contours_A.npz, contours_B.npz   TraceCMR contour files (open in the app)
  <out>/<MouseN>/comparison.csv                    expert vs A vs B
  <out>/<MouseN>/dice_A.csv                        per-slice Dice vs expert at ED and ES
  <out>/<MouseN>/images/                           raw | contoured PNGs and montages (A)
  <out>/summary.csv, summary.md, bland_altman.png

Example:
    TRACECMR_UCL_DIR=... python scripts/validate_test25.py --ucl-dir .../Data --out validation_test25
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np

from tracecmr import labels as L
from tracecmr.app.export_images import export_images
from tracecmr.app.session import Session

MEASURES = ("edv_ul", "esv_ul", "sv_ul", "ef_pct", "mass_ed_mg")


def test_mice(ucl_dir: Path) -> list[str]:
    ids = sorted({p.stem for p in ucl_dir.glob("Mouse*.h5")})
    if (ucl_dir / "ids.json").exists():  # only some mice downloaded: the full list of 130 IDs, for the split
        import json

        ids = sorted(json.loads((ucl_dir / "ids.json").read_text()))
    assert len(ids) == 130, f"expected 130 UCL mice, found {len(ids)}"
    np.random.seed(42)
    shuffled = np.array(ids)
    np.random.shuffle(shuffled)
    return shuffled[105:130].tolist()


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def finish(s: Session) -> dict:
    s.auto_order()
    s.apply_suggestions()
    fn, _, notes = s.compute()
    row = {k: fn.as_row()[k] for k in MEASURES}
    row.update(ed_frame=s.ed_frame, es_frame=s.es_frame, n_ed=len(s.included["ED"]), n_es=len(s.included["ES"]),
               notes="; ".join(notes))
    return row


def run_mouse(path: Path, out: Path) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    expert = Session.from_ucl(path)  # expert contours, expert frames, same slice logic
    fe, _, _ = expert.compute()
    ref = {k: fe.as_row()[k] for k in MEASURES}
    ted, tes = expert.ed_frame, expert.es_frame
    nz, nt = expert.image.shape[2], expert.image.shape[3]

    # A: model at the expert's frames
    a = Session.from_ucl(path, with_labels=False)
    a.subject_id, a.papillary = path.stem, "myocardium"  # UCL's convention
    a.ed_frame, a.es_frame = ted, tes
    a.autocontour_pairs([(z, t) for t in (ted, tes) for z in range(nz)])
    ra = finish(a)
    a.save_contours(out / "contours_A.npz")
    export_images(a, out / "images")
    with open(out / "dice_A.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "slice", "dice_bloodpool", "dice_myocardium"])
        for ph, t in (("ED", ted), ("ES", tes)):
            for z in range(nz):
                m, e = a.label[:, :, z, t], expert.label[:, :, z, t]
                w.writerow([ph, z, round(dice(m == L.LV_BLOODPOOL, e == L.LV_BLOODPOOL), 4),
                            round(dice(m == L.LV_MYOCARDIUM, e == L.LV_MYOCARDIUM), 4)])
    vol = {}
    for ph, t in (("ED", ted), ("ES", tes)):
        m, e = a.label[:, :, :, t], expert.label[:, :, :, t]
        vol[ph] = (dice(m == L.LV_BLOODPOOL, e == L.LV_BLOODPOOL), dice(m == L.LV_MYOCARDIUM, e == L.LV_MYOCARDIUM))

    # B: end to end, every frame contoured, ED/ES from the volume curve
    b = Session.from_ucl(path, with_labels=False)
    b.subject_id, b.papillary = path.stem, "myocardium"
    b.autocontour_pairs([(z, t) for t in range(nt) for z in range(nz)])
    b.auto_ed_es()
    rb = finish(b)
    b.save_contours(out / "contours_B.npz")

    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["measure", "expert", "A_expert_frames", "B_end_to_end", "A_minus_expert", "B_minus_expert"])
        for k in MEASURES:
            w.writerow([k, round(ref[k], 2), round(ra[k], 2), round(rb[k], 2), round(ra[k] - ref[k], 2),
                        round(rb[k] - ref[k], 2)])
        w.writerow(["ed_frame", ted, ra["ed_frame"], rb["ed_frame"], "", ""])
        w.writerow(["es_frame", tes, ra["es_frame"], rb["es_frame"], "", ""])
    row = {"mouse": path.stem, "seconds": round(time.time() - t0, 1),
           "dice_bp_ed": vol["ED"][0], "dice_bp_es": vol["ES"][0], "dice_myo_ed": vol["ED"][1],
           "dice_myo_es": vol["ES"][1], "expert_ed": ted, "expert_es": tes, "B_ed": rb["ed_frame"],
           "B_es": rb["es_frame"], "notes_A": ra["notes"], "notes_B": rb["notes"]}
    for k in MEASURES:
        row[f"{k}_expert"], row[f"{k}_A"], row[f"{k}_B"] = ref[k], ra[k], rb[k]
    return row


def summarise(rows: list[dict], out: Path) -> None:
    keys = list(rows[0].keys())
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    lines = [f"# TraceCMR validation on UCL's {len(rows)} held-out test mice", "",
             "Error = TraceCMR minus expert. LoA = 95% limits of agreement (bias ± 1.96 SD).", "",
             "| Measure | Mode | Bias | SD | LoA | Mean abs error | Worst |", "|---|---|---|---|---|---|---|"]
    names = {"edv_ul": "EDV (uL)", "esv_ul": "ESV (uL)", "sv_ul": "SV (uL)", "ef_pct": "EF (pts)",
             "mass_ed_mg": "LV mass ED (mg)"}
    for k in MEASURES:
        for mode, lab in (("A", "expert frames"), ("B", "end to end")):
            d = np.array([r[f"{k}_{mode}"] - r[f"{k}_expert"] for r in rows])
            bias, sd = d.mean(), d.std(ddof=1)
            worst = rows[int(np.argmax(np.abs(d)))]["mouse"]
            lines.append(f"| {names[k]} | {lab} | {bias:+.1f} | {sd:.1f} | {bias - 1.96 * sd:+.1f} to "
                         f"{bias + 1.96 * sd:+.1f} | {np.abs(d).mean():.1f} | {d[np.argmax(np.abs(d))]:+.1f} "
                         f"({worst}) |")
    dk = ("dice_bp_ed", "dice_bp_es", "dice_myo_ed", "dice_myo_es")
    lines += ["", "Dice (mode A, whole heart): " + ", ".join(
        f"{k.replace('dice_', '').replace('_', ' ')} {np.nanmean([r[k] for r in rows]):.3f}" for k in dk)]
    ed_ok = np.mean([abs(r["B_ed"] - r["expert_ed"]) <= 1 for r in rows])
    es_ok = np.mean([abs(r["B_es"] - r["expert_es"]) <= 1 for r in rows])
    lines += [f"End-to-end ED/ES frame within 1 frame of the expert: ED {ed_ok:.0%}, ES {es_ok:.0%}.", "",
              "Caveat: same lab, scanner and protocol the model was trained on (best case). External data is "
              "needed to show it generalises."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 6))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4)
    for col, k in enumerate(("edv_ul", "esv_ul", "ef_pct", "mass_ed_mg")):
        for row_i, (mode, lab) in enumerate((("A", "expert frames"), ("B", "end to end"))):
            ax = axes[row_i, col]
            x = np.array([r[f"{k}_expert"] for r in rows])
            y = np.array([r[f"{k}_{mode}"] for r in rows])
            mean, d = (x + y) / 2, y - x
            bias, sd = d.mean(), d.std(ddof=1)
            ax.scatter(mean, d, s=14, color="#2a6fdb")
            for v, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
                ax.axhline(v, color="#888", ls=ls, lw=1)
            ax.set_title(f"{names[k]} · {lab}", fontsize=9)
            ax.set_xlabel("mean of TraceCMR and expert", fontsize=8)
            ax.set_ylabel("TraceCMR − expert", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "bland_altman.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucl-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    mice = test_mice(args.ucl_dir)
    if args.limit:
        mice = mice[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, m in enumerate(sorted(mice, key=lambda s: int(re.sub(r"\D", "", s)))):
        try:
            r = run_mouse(args.ucl_dir / f"{m}.h5", args.out / m)
            rows.append(r)
            print(f"[{i + 1}/{len(mice)}] {m}: EF expert {r['ef_pct_expert']:.1f}, A {r['ef_pct_A']:.1f}, "
                  f"B {r['ef_pct_B']:.1f} ({r['seconds']} s)", flush=True)
        except Exception as e:
            print(f"[{i + 1}/{len(mice)}] {m}: FAILED {e!r}", flush=True)
    if rows:
        summarise(rows, args.out)


if __name__ == "__main__":
    main()
