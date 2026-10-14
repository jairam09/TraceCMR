"""Auto-contour on a clean machine, the way a tester gets it: run the one-click setup (download UCL's model,
install TensorFlow), then contour a UCL mouse through the app's model process and compare with the experts.

    python scripts/check_autocontour.py Mouse1.h5 [out.json]

Exit code 0 only if setup worked, the app still imports afterwards, and the contours agree with the
expert ones (Dice: blood pool >= 0.85, myocardium >= 0.70 at end-diastole).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def main() -> int:
    path = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else None
    from tracecmr import model_setup as ms

    report = {"python": sys.executable, "steps": []}
    t0 = time.time()
    last = None
    try:
        for st in ms.run_setup():
            if st.text != last:
                print(f"[{time.time() - t0:6.1f}s] {st.stage}: {st.text}", flush=True)
                report["steps"].append(st.text)
                last = st.text
    except ms.SetupError as e:
        report["error"] = str(e)
        print("SETUP FAILED:", e, flush=True)
        try:
            print(open(ms.log_path(), encoding="utf-8").read()[-4000:])
        except OSError:
            pass
        _write(out_json, report)
        return 1
    report["setup_seconds"] = round(time.time() - t0, 1)

    # the app must still start after TensorFlow was added to its Python
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    r = subprocess.run([sys.executable, "-c", "import napari, tracecmr.app._widget, h5py, numpy; "
                        "print(numpy.__version__, h5py.__version__)"], capture_output=True, text=True, env=env)
    report["app_imports_after_setup"] = r.returncode == 0
    report["numpy_h5py"] = r.stdout.strip()
    if r.returncode != 0:
        print("APP IMPORT FAILED AFTER SETUP:", r.stderr[-2000:], flush=True)

    os.environ["TRACECMR_UCL_DIR"] = str(ms.default_dir())
    from tracecmr import autocontour as ac
    from tracecmr import labels as L
    from tracecmr.app.session import Session

    report["model_status"] = ac.available()["ucl"]
    s = Session.from_ucl(path)  # expert contours, ED/ES from the dataset
    t = s.ed_frame
    zs = [z for z in range(s.image.shape[2]) if s.label[:, :, z, t].any()]
    t1 = time.time()
    pred = ac.run_pairs(s.image, s.spacing, [(z, t) for z in zs])  # through the app's model process
    report["run_seconds"] = round(time.time() - t1, 1)
    ac.reset_workers()
    gt = s.label[:, :, zs, t]
    report["dice_blood"] = round(dice(pred == L.LV_BLOODPOOL, gt == L.LV_BLOODPOOL), 3)
    report["dice_myo"] = round(dice(pred == L.LV_MYOCARDIUM, gt == L.LV_MYOCARDIUM), 3)
    ok = (report["app_imports_after_setup"] and report["dice_blood"] >= 0.85 and report["dice_myo"] >= 0.70)
    report["ok"] = bool(ok)
    print(json.dumps(report, indent=1), flush=True)
    _write(out_json, report)
    return 0 if ok else 1


def _write(path, report):
    if path:
        with open(path, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
