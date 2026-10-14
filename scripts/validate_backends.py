"""Does Auto LV give the same answers when the UCL model runs with ONNX Runtime instead of TensorFlow, and
what does averaging over 8 orientations (test-time augmentation, TTA) do? On UCL's validation and test mice,
against the expert.

    TRACECMR_UCL_DIR=<authors' files> TRACECMR_UCL_ONNX=ucl_seg459.onnx \
        python scripts/validate_backends.py --ucl-dir <Data> --out results/backends

For every mouse: the model contours the expert's ED and ES frames exactly as the app does (Session.
autocontour_pairs, then the app's slice order and inclusion), and the volumes, EF, mass and Dice are compared
with the expert's own contours. Variants:

    tf          TensorFlow, one pass (what 0.3.4 does)
    onnx        ONNX Runtime, one pass (must match tf)
    onnx+tta@c  ONNX Runtime, average of 8 orientations, epicardium cutoff c (EPI_CUTOFF was tuned without TTA)

The cutoff for TTA is chosen on the 15 validation mice (smallest |mass bias| without moving EDV/ESV/EF), then
reported once on the 25 test mice. Splits as in UCL's notebook (seed 42; 90/15/25).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["TRACECMR_INPROCESS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from tracecmr import autocontour as ac  # noqa: E402
from tracecmr import labels as L  # noqa: E402
from tracecmr.app.session import Session  # noqa: E402
from validate_test25 import MEASURES, dice, finish  # noqa: E402

EPIS = (0.5, 0.6, 0.7, 0.8, 0.9)

_real_probs = ac._ucl_probs
_cache: dict = {}


def _cached_probs(vol, spacing):
    """The model runs once per (volume, backend, TTA); the cutoffs are applied to the cached probabilities."""
    key = (hashlib.sha1(np.ascontiguousarray(vol, np.float32).tobytes()).hexdigest(), tuple(spacing),
           os.environ.get("TRACECMR_UCL_BACKEND"), os.environ.get("TRACECMR_TTA"))
    if key not in _cache:
        _cache[key] = _real_probs(vol, spacing)
    return _cache[key]


ac._ucl_probs = _cached_probs


def use(backend: str, tta: bool, epi: float) -> None:
    if os.environ.get("TRACECMR_UCL_BACKEND") != backend:
        ac._ucl_model = None  # load the other backend
    os.environ["TRACECMR_UCL_BACKEND"] = backend
    os.environ["TRACECMR_TTA"] = "1" if tta else "0"
    os.environ["TRACECMR_EPI_CUTOFF"] = str(epi)


def split(ids: list[str], which: str) -> list[str]:
    np.random.seed(42)
    s = np.array(sorted(ids))
    np.random.shuffle(s)
    return (s[90:105] if which == "val" else s[105:130]).tolist()


def run(path: Path, variants: list[tuple[str, str, bool, float]]) -> dict:
    expert = Session.from_ucl(path)
    fe, _, _ = expert.compute()
    ref = {k: fe.as_row()[k] for k in MEASURES}
    ted, tes = expert.ed_frame, expert.es_frame
    nz = expert.image.shape[2]
    out = {"expert": ref}
    for name, backend, tta, epi in variants:
        use(backend, tta, epi)
        a = Session.from_ucl(path, with_labels=False)
        a.subject_id, a.papillary = path.stem, "myocardium"
        a.ed_frame, a.es_frame = ted, tes
        t0 = time.time()
        a.autocontour_pairs([(z, t) for t in (ted, tes) for z in range(nz)])
        sec = time.time() - t0
        row = {k: v for k, v in finish(a).items() if k in MEASURES}
        for ph, t in (("ed", ted), ("es", tes)):
            m, e = a.label[:, :, :, t], expert.label[:, :, :, t]
            row[f"dice_bp_{ph}"] = dice(m == L.LV_BLOODPOOL, e == L.LV_BLOODPOOL)
            row[f"dice_myo_{ph}"] = dice(m == L.LV_MYOCARDIUM, e == L.LV_MYOCARDIUM)
        row["seconds"] = sec
        out[name] = row
    return out


def table(results: dict[str, dict], names: list[str]) -> list[str]:
    lines = ["| Variant | EDV bias (SD) | ESV bias (SD) | EF bias (SD) | Mass bias (SD) | |EF err| max | Dice blood ED/ES "
             "| Dice myo ED/ES | s/mouse |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        rows = [r for r in results.values() if n in r]
        if not rows:
            continue

        def b(k):
            d = np.array([r[n][k] - r["expert"][k] for r in rows])
            return f"{d.mean():+.1f} ({d.std(ddof=1):.1f})"

        ef = np.array([abs(r[n]["ef_pct"] - r["expert"]["ef_pct"]) for r in rows])
        dsc = lambda k: np.nanmean([r[n][k] for r in rows])  # noqa: E731
        lines.append(f"| {n} | {b('edv_ul')} | {b('esv_ul')} | {b('ef_pct')} | {b('mass_ed_mg')} | {ef.max():.1f} | "
                     f"{dsc('dice_bp_ed'):.3f}/{dsc('dice_bp_es'):.3f} | {dsc('dice_myo_ed'):.3f}/"
                     f"{dsc('dice_myo_es'):.3f} | {np.mean([r[n]['seconds'] for r in rows]):.1f} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucl-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-tf", action="store_true", help="skip the TensorFlow baseline")
    ap.add_argument("--resume", action="store_true", help="keep the mice already in OUT/results.json")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = [p.stem for p in a.ucl_dir.glob("Mouse*.h5")]
    allids = json.loads((a.ucl_dir / "ids.json").read_text()) if (a.ucl_dir / "ids.json").exists() else ids
    val, test = split(allids, "val"), split(allids, "test")
    base = [] if a.no_tf else [("tf", "tensorflow", False, ac.EPI_CUTOFF)]
    base += [("onnx", "onnx", False, ac.EPI_CUTOFF)]
    tta = [(f"onnx+tta@{e}", "onnx", True, e) for e in EPIS]
    saved = a.out / "results.json"
    results: dict[str, dict] = json.loads(saved.read_text()) if a.resume and saved.exists() else {}
    for which, mice in (("val", val), ("test", test)):
        for m in mice:
            p = a.ucl_dir / f"{m}.h5"
            if f"{which}:{m}" in results:
                continue
            if not p.exists():
                print("missing", m, flush=True)
                continue
            t0 = time.time()
            _cache.clear()
            results[f"{which}:{m}"] = run(p, base + tta)
            r = results[f"{which}:{m}"]
            print(f"{which} {m}: EF expert {r['expert']['ef_pct']:.1f} | " + " | ".join(
                f"{n} {r[n]['ef_pct']:.1f}/{r[n]['mass_ed_mg']:.0f}mg" for n, *_ in base + tta[-2:]) +
                f" ({time.time() - t0:.0f}s)", flush=True)
            (a.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    names = [n for n, *_ in base + tta]
    val_res = {k: v for k, v in results.items() if k.startswith("val:")}
    test_res = {k: v for k, v in results.items() if k.startswith("test:")}
    # TTA cutoff from the validation mice only
    best = min(EPIS, key=lambda e: abs(np.mean([r[f"onnx+tta@{e}"]["mass_ed_mg"] - r["expert"]["mass_ed_mg"]
                                                for r in val_res.values()])))
    same = []
    if not a.no_tf:
        for r in results.values():
            same.append(max(abs(r["tf"][k] - r["onnx"][k]) for k in MEASURES))
    md = ["# Auto LV: TensorFlow vs ONNX Runtime, and 8-orientation averaging (TTA)", "",
          "Bias = TraceCMR minus expert, mean (SD), at the expert's ED/ES frames, through the app's own pipeline.",
          ""]
    if same:
        md += [f"TensorFlow vs ONNX, one pass: largest difference in any measure on any mouse = {max(same):.3g}.", ""]
    md += [f"## Validation mice ({len(val_res)})", ""] + table(val_res, names)
    md += ["", f"TTA epicardium cutoff chosen on validation: {best}", "",
           f"## Test mice ({len(test_res)}), reported once", ""]
    md += table(test_res, [n for n in names if not n.startswith("onnx+tta")] + [f"onnx+tta@{best}"])
    md += ["", "Caveat: UCL's own scanner and lab (best case for this model)."]
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
