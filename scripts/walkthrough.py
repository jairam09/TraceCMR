"""A new user's first session, start to finish, with a screenshot at every stage.

Starts TraceCMR the way the desktop shortcut does (splash, start screen, first-run tour), then acts like a person:
goes through the tour, opens a UCL mouse with "Add UCL .h5 file(s)", reads the results, turns on per-slice ED/ES
and presses Detect, opens ED | ES, saves a figure, looks at the Wall tab, exports everything and saves.
File dialogs are answered with their suggested path, as if the user clicked Save.

    python scripts/walkthrough.py Mouse1.h5 shots/

Writes shots/NN_name.png and shots/summary.json (every step: ok or the error, plus what the app said).
Exit code 0 only if every step worked and nothing raised.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

MOUSE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ.pop("TRACECMR_NO_TOUR", None)  # a new user sees the tour

steps: list[dict] = []
errors: list[str] = []
_shot_n = [0]


def _hook(etype, value, tb):  # anything the app raises inside Qt callbacks
    errors.append("".join(traceback.format_exception(etype, value, tb)))
    sys.__excepthook__(etype, value, tb)


sys.excepthook = _hook


def finish() -> None:
    ok = all(s["ok"] for s in steps) and not errors
    (OUT / "summary.json").write_text(json.dumps({"ok": ok, "steps": steps, "errors": errors}, indent=2))
    print(json.dumps({"ok": ok, "failed": [s["name"] for s in steps if not s["ok"]], "errors": len(errors)}))
    sys.stdout.flush()
    os._exit(0 if ok else 1)


def driver(viewer) -> None:
    from qtpy.QtWidgets import QApplication, QFileDialog, QTabWidget

    from tracecmr.app._widget import TraceCMRWidget

    app = QApplication.instance()
    win = viewer.window._qt_window

    def pump(seconds: float = 0.3) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def shot(name: str) -> str:
        pump(0.4)
        _shot_n[0] += 1
        fn = f"{_shot_n[0]:02d}_{name}.png"
        win.grab().save(str(OUT / fn))
        return fn

    def step(name: str, fn, take_shot: bool = True) -> None:
        rec = {"name": name, "ok": True}
        t0 = time.time()
        try:
            note = fn()
            if note:
                rec["note"] = str(note)
        except Exception:
            rec["ok"] = False
            rec["error"] = traceback.format_exc()
        rec["seconds"] = round(time.time() - t0, 2)
        if take_shot:
            try:
                rec["shot"] = shot(name)
            except Exception:
                rec["shot_error"] = traceback.format_exc()
        steps.append(rec)
        print(("OK   " if rec["ok"] else "FAIL ") + name, rec.get("note", ""), rec.get("error", "")[-700:], flush=True)

    panel = win.findChild(TraceCMRWidget)

    # 1. what the user sees when the app opens
    step("start_screen", lambda: f"{win.width()}x{win.height()}")

    # 2. the first-run tour: it should appear by itself; click Next through every stop
    def wait_tour():
        end = time.time() + 8
        while time.time() < end:
            t = getattr(panel, "_tour", None)
            if t is not None and t.isVisible():
                return "tour opened by itself"
            pump(0.2)
        raise RuntimeError("the first-run tour did not appear")

    step("tour_appears", wait_tour)
    tour = getattr(panel, "_tour", None)
    if tour is not None:
        for i in range(1, len(panel.tour_steps()) + 1):
            if not tour.isVisible():
                break
            step(f"tour_{i:02d}_{tour.btn_next.text().replace(' ', '_').lower()}",
                 lambda: tour.btn_next.click(), take_shot=False)
            pump(0.6)
            if tour.isVisible():
                steps[-1]["shot"] = shot(f"tour_{i:02d}")
        step("tour_closed", lambda: "closed" if not tour.isVisible() else (_ for _ in ()).throw(
            RuntimeError("tour still open after the last step")))

    # dialogs: answer with the suggested path, as if the user clicked Open / Save
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([str(MOUSE)], "UCL HDF5 (*.h5)"))
    QFileDialog.getSaveFileName = staticmethod(lambda parent=None, caption="", directory="", *a, **k:
                                               (str(directory), ""))

    # 3. open the mouse the way the page says: Open scan arrow -> Add UCL .h5 file(s)
    def open_mouse():
        panel._open_ucl()
        pump(1.5)
        s = panel.session
        if s is None:
            raise RuntimeError("no mouse opened")
        panel.fit_heart()
        return f"{s.subject_id}: {s.image.shape}, ED {s.ed_frame} ES {s.es_frame}"

    step("open_mouse", open_mouse)

    # 4. results
    def results():
        panel.compute()
        pump(0.5)
        fn = (panel.last or (None,))[0]
        if fn is None:
            raise RuntimeError("no results")
        vals = {k: round(v, 1) for k, v in vars(fn).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return {k: vals[k] for k in list(vals)[:10]}

    step("results_function", results)

    # 5. per-slice ED/ES + Detect
    def detect():
        panel.chk_per_slice.setChecked(True)
        pump(0.3)
        panel.btn_auto_frames.click()
        pump(1.5)
        return panel.lbl_order.text()

    step("per_slice_detect", detect)

    # 6. ED | ES side by side, then Save figure
    def compare():
        panel.show_compare()
        pump(1.0)
        if not panel._compare_open():
            raise RuntimeError("ED | ES view did not open")
        return "open"

    step("ed_es_view", compare)

    def save_fig():
        p = panel._compare.save_one()
        if p is None or not Path(p).exists() or Path(p).stat().st_size < 10_000:
            raise RuntimeError(f"figure not written: {p}")
        return f"{Path(p).name} ({Path(p).stat().st_size // 1024} KB)"

    step("save_figure", save_fig, take_shot=False)

    def close_compare():
        panel.show_compare()  # toggles it closed
        pump(0.5)
        return "closed" if not panel._compare_open() else "still open"

    step("ed_es_closed", close_compare, take_shot=False)

    # 7. Wall tab
    def wall():
        panel.results_tabs.show("wall")
        pump(0.8)
        return "wall tab shown"

    step("results_wall", wall)

    # 8. Export all results, then save
    def dismiss_dialogs_later(tag: str) -> None:
        """When a message box pops up, screenshot it, note its text and press its default button."""
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QMessageBox

        def check():
            box = app.activeModalWidget()
            if isinstance(box, QMessageBox):
                _shot_n[0] += 1
                fn = f"{_shot_n[0]:02d}_{tag}_dialog.png"
                box.grab().save(str(OUT / fn))
                steps.append({"name": f"{tag}_dialog", "ok": True, "note": box.text()[:500], "shot": fn})
                (box.defaultButton() or box.buttons()[-1]).click()
            else:
                QTimer.singleShot(300, check)

        QTimer.singleShot(300, check)

    def export():
        dismiss_dialogs_later("export")
        out = panel.export_all()
        if not out:
            raise RuntimeError(f"export wrote nothing: {panel.lbl_auto.text()}")
        written = []
        for v in (out.values() if isinstance(out, dict) else [out]):
            for p in (v if isinstance(v, (list, tuple)) else [v]):
                try:
                    if Path(p).exists():
                        written.append(Path(p).name)
                except TypeError:
                    pass
        return f"{len(written)} files: " + ", ".join(written[:8])

    step("export_all", export)
    step("save_contours", lambda: str(panel.save_contours(quiet=True)), take_shot=False)

    # 9. Auto LV the first time: ready straight away with the installers from 0.3.5 (TRACECMR_WALK_EXPECT_READY=1
    #    makes that a requirement); older copies offer the one-time setup, so say yes and wait for it. Then contour
    #    a frame that has no expert contours (set TRACECMR_WALK_SKIP_AUTO=1 to leave this out)
    if not os.environ.get("TRACECMR_WALK_SKIP_AUTO"):
        def setup_auto():
            if panel.model_ready():
                return "ready out of the box, no setup needed"
            if os.environ.get("TRACECMR_WALK_EXPECT_READY"):
                raise RuntimeError("Auto LV should work straight after installing, but it asks for a setup")
            dismiss_dialogs_later("auto_lv_offer")  # presses "Set up now"
            panel.btn_auto.click()
            pump(1.0)
            if not getattr(panel, "_setup_running", False) and not panel.model_ready():
                raise RuntimeError("Auto LV did not start the setup")
            t0 = time.time()
            while getattr(panel, "_setup_running", False):
                pump(1.0)
                if time.time() - t0 > 1800:
                    raise RuntimeError("setup still running after 30 minutes")
            dlg = panel.setup_dialog
            _shot_n[0] += 1
            fn = f"{_shot_n[0]:02d}_auto_lv_setup_dialog.png"
            dlg.grab().save(str(OUT / fn))
            steps.append({"name": "auto_lv_setup_dialog", "ok": True, "shot": fn,
                          "note": dlg.lbl_step.text() + " | " + dlg.lbl_text.text()})
            if not panel.model_ready():
                raise RuntimeError(f"setup ended without a working model: {dlg.lbl_text.text()}")
            dlg.hide()
            return f"set up in {time.time() - t0:.0f} s"

        step("auto_lv_setup", setup_auto)

        def run_auto():
            s = panel.session
            empty = [t for t in range(s.image.shape[3]) if not s.label[..., t].any()]
            if not empty:
                raise RuntimeError("no frame without contours to try Auto LV on")
            t = empty[len(empty) // 2]
            cur = list(panel.viewer.dims.current_step)
            cur[0] = t
            panel.viewer.dims.current_step = tuple(cur)
            pump(0.5)
            panel.btn_auto.click()
            t0 = time.time()
            pump(1.0)
            while not panel.btn_auto.isEnabled():
                pump(0.5)
                if time.time() - t0 > 600:
                    raise RuntimeError("Auto LV still running after 10 minutes")
            n = sum(1 for z in range(s.image.shape[2]) if s.label[:, :, z, t].any())
            if n == 0:
                raise RuntimeError(f"Auto LV drew nothing: {panel.lbl_auto.text()}")
            return f"frame {t}: {n} slices contoured in {time.time() - t0:.0f} s · {panel.lbl_auto.text()[:120]}"

        step("auto_lv_run", run_auto)

    try:
        from tracecmr.app.launch import log_path

        lp = log_path()
        if lp.exists():
            (OUT / "app_log_tail.txt").write_text(lp.read_text(errors="replace")[-6000:])
    except Exception:
        pass
    finish()


def main() -> None:
    from tracecmr.app import launch

    real_reveal = launch.reveal

    def reveal(viewer, splash, *a, **k):
        real_reveal(viewer, splash, *a, **k)
        from qtpy.QtCore import QTimer

        def run():
            try:
                driver(viewer)
            except Exception:
                errors.append(traceback.format_exc())
                finish()

        QTimer.singleShot(1500, run)

    launch.reveal = reveal

    def watchdog():  # never hang a CI machine
        time.sleep(2700)
        errors.append("walkthrough timed out after 45 minutes")
        finish()

    import threading

    threading.Thread(target=watchdog, daemon=True).start()
    launch.main([])


if __name__ == "__main__":
    main()
