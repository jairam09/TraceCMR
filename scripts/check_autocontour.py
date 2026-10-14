"""Auto LV on a clean machine, the way a tester gets it.

    python scripts/check_autocontour.py Mouse1.h5 [out.json] [--expect-bundled]

1. Out of the box: how the installed app will run the model, and from where. With --expect-bundled (installers
   from 0.3.5 on) it must be the model the installer put in <TraceCMR>/models, run by ONNX Runtime.
2. Accuracy: contour a UCL mouse's end-diastole through the app's model process (with the 8-orientation average,
   as the app does) and compare with the experts.
3. The one-click setup that older installs get: the bundled model is moved aside and ONNX Runtime removed, then
   run_setup downloads the model, reinstalls ONNX Runtime and checks the model runs. The app must still import,
   must find the downloaded model by itself, and must give the same contours with it. The bundled model is put
   back at the end.

Exit code 0 only if every part worked and both runs agree with the experts (Dice: blood pool >= 0.85, myocardium
>= 0.70 at end-diastole).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BLOOD_MIN, MYO_MIN = 0.85, 0.70


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def contour(path: str) -> dict:
    """End-diastole of a UCL mouse through the app's model process, against the expert contours."""
    from tracecmr import autocontour as ac
    from tracecmr import labels as L
    from tracecmr.app.session import Session

    s = Session.from_ucl(path)
    t = s.ed_frame
    zs = [z for z in range(s.image.shape[2]) if s.label[:, :, z, t].any()]
    t1 = time.time()
    pred = ac.run_pairs(s.image, s.spacing, [(z, t) for z in zs])
    secs = round(time.time() - t1, 1)
    ac.reset_workers()  # let go of the model (and, on Windows, of ONNX Runtime's files)
    gt = s.label[:, :, zs, t]
    return {"seconds": secs, "slices": len(zs),
            "dice_blood": round(dice(pred == L.LV_BLOODPOOL, gt == L.LV_BLOODPOOL), 3),
            "dice_myo": round(dice(pred == L.LV_MYOCARDIUM, gt == L.LV_MYOCARDIUM), 3)}


def good(r: dict | None) -> bool:
    return bool(r) and r["dice_blood"] >= BLOOD_MIN and r["dice_myo"] >= MYO_MIN


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    expect_bundled = "--expect-bundled" in sys.argv
    path = args[0]
    out_json = args[1] if len(args) > 1 else None
    from tracecmr import autocontour as ac
    from tracecmr import model_setup as ms

    report: dict = {"python": sys.executable, "tta": ac.tta_enabled(), "steps": []}
    ok = True

    # 1. out of the box
    bundled = Path(sys.prefix) / "models" / ms.ONNX_NAME
    report["out_of_the_box"] = {"backend": ac.ucl_backend(), "model": str(ac.ucl_onnx_path()),
                                "bundled_model_present": bundled.is_file(),
                                "licence_beside_it": (bundled.parent / "UCL-LICENSE.txt").is_file()}
    print("out of the box:", report["out_of_the_box"], flush=True)
    if expect_bundled:
        oob = report["out_of_the_box"]
        if not (oob["backend"] == "onnx" and Path(oob["model"]) == bundled and oob["licence_beside_it"]):
            print("NOT READY OUT OF THE BOX: expected the bundled model, run by ONNX Runtime", flush=True)
            ok = False

    # 2. accuracy with what the installer gave us
    if ac.ucl_backend():
        report["bundled_run"] = contour(path)
        print("bundled model:", report["bundled_run"], flush=True)
        ok &= good(report["bundled_run"])

    # 3. the one-click setup, as an older install (no bundled model, no ONNX Runtime) gets it
    aside = bundled.with_name(bundled.name + ".aside")
    if bundled.is_file():
        os.replace(bundled, aside)
    shutil.rmtree(ms.default_dir(), ignore_errors=True)
    r = subprocess.run([ms.python_exe(), "-m", "pip", "uninstall", "-y", "onnxruntime"], capture_output=True,
                       text=True)
    report["runtime_removed"] = r.returncode == 0 and not ms.runtime_present()
    report["needs_setup_now"] = ac.ucl_backend() is None
    try:
        t0 = time.time()
        last = None
        try:
            for st in ms.run_setup():
                if st.text != last:
                    print(f"[{time.time() - t0:6.1f}s] {st.stage}: {st.text}", flush=True)
                    report["steps"].append(st.text)
                    last = st.text
            report["setup_seconds"] = round(time.time() - t0, 1)
        except ms.SetupError as e:
            report["setup_error"] = str(e)
            print("SETUP FAILED:", e, flush=True)
            try:
                print(open(ms.log_path(), encoding="utf-8").read()[-4000:])
            except OSError:
                pass
            ok = False
        if "setup_error" not in report:
            # the app must still start, and must find the downloaded model by itself
            env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
            r = subprocess.run([sys.executable, "-c", "import napari, tracecmr.app._widget, h5py, numpy, onnxruntime;"
                                " print(numpy.__version__, h5py.__version__, onnxruntime.__version__)"],
                               capture_output=True, text=True, env=env)
            report["app_imports_after_setup"] = r.returncode == 0
            report["numpy_h5py_onnxruntime"] = r.stdout.strip()
            if r.returncode != 0:
                print("APP IMPORT FAILED AFTER SETUP:", r.stderr[-2000:], flush=True)
            found = ac.ucl_onnx_path()
            report["after_setup"] = {"backend": ac.ucl_backend(), "model": str(found)}
            ok &= report["app_imports_after_setup"] and found == ms.default_dir() / ms.ONNX_NAME
            report["setup_run"] = contour(path)
            print("downloaded model:", report["setup_run"], flush=True)
            ok &= good(report["setup_run"])
            if report.get("bundled_run"):
                same = all(report["setup_run"][k] == report["bundled_run"][k] for k in ("dice_blood", "dice_myo"))
                report["same_as_bundled"] = same
                ok &= same
    finally:
        if aside.is_file():
            os.replace(aside, bundled)
    ok &= bool(report["runtime_removed"]) and bool(report["needs_setup_now"])
    report["ok"] = bool(ok)
    print(json.dumps(report, indent=1), flush=True)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
