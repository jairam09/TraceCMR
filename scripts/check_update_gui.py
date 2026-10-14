"""A patch update through the real window, the way a tester gets one (run with the installed app's Python).

    python scripts/check_update_gui.py patches_gui/latest-0.3.5.json Mouse1.h5 out/

Starts TraceCMR like its shortcut, pointed at a pretend release (TRACECMR_UPDATE_URL), opens a mouse, then waits
for the update card to appear by itself, clicks Update, then Restart now. TraceCMR saves, closes and hands the
update to the helper; the helper installs it and opens TraceCMR again. This script ends when TraceCMR closes:
the CI step after it checks the helper's result, that the reopened TraceCMR read it, and the new version.

Writes out/update_gui.json (each step) and screenshots of the card.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

MANIFEST = Path(sys.argv[1]).resolve()
MOUSE = Path(sys.argv[2]).resolve()
OUT = Path(sys.argv[3]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TRACECMR_UPDATE_URL"] = MANIFEST.as_uri()
os.environ["TRACECMR_NO_TOUR"] = "1"
steps: list[dict] = []


def write(ok: bool, extra: str = "") -> None:
    (OUT / "update_gui.json").write_text(json.dumps({"ok": ok, "steps": steps, "error": extra}, indent=2))


def fail(msg: str) -> None:
    steps.append({"name": "error", "ok": False, "error": msg})
    write(False, msg)
    print("FAIL", msg, flush=True)
    os._exit(1)


def driver(viewer) -> None:
    from qtpy.QtWidgets import QApplication

    from tracecmr.app._widget import TraceCMRWidget

    app = QApplication.instance()
    win = viewer.window._qt_window
    panel = win.findChild(TraceCMRWidget)

    def pump(s: float) -> None:
        end = time.time() + s
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def shot(widget, name: str) -> None:
        pump(0.3)
        widget.grab().save(str(OUT / f"{name}.png"))

    def wait_state(states: tuple[str, ...], seconds: float) -> str:
        end = time.time() + seconds
        while time.time() < end:
            card = panel.updates.card
            if card is not None and card.isVisible() and card.state in states:
                return card.state
            pump(0.2)
        card = panel.updates.card
        fail(f"waited {seconds:.0f} s for {states}; card: {None if card is None else (card.state, card.text.text())}")

    from tracecmr import __version__

    steps.append({"name": "started", "ok": True, "note": f"TraceCMR {__version__}, updates from {MANIFEST.name}"})
    from tracecmr.app.session import Session

    panel.load_session(Session.from_ucl(MOUSE))
    pump(1.0)
    steps.append({"name": "mouse_open", "ok": True, "note": panel.session.subject_id})

    state = wait_state(("available", "installer", "dev"), 90)
    shot(panel.updates.card, "update_1_offered")
    shot(win, "update_1_window")
    steps.append({"name": "card_appears_by_itself", "ok": state == "available",
                  "note": f"{state}: {panel.updates.card.title.text()} | {panel.updates.card.text.text()} | "
                          f"{panel.updates.card.notes.text()}"})
    if state != "available":
        fail(f"the card offered {state!r} instead of the patch")
    panel.updates.card.btn_first.click()  # Update
    wait_state(("ready",), 120)
    shot(panel.updates.card, "update_2_ready")
    steps.append({"name": "downloaded", "ok": True, "note": panel.updates.card.text.text()})
    steps.append({"name": "restart_now", "ok": True, "note": "clicked; TraceCMR should close and the helper run"})
    write(True)
    print(json.dumps({"ok": True, "steps": [s["name"] for s in steps]}), flush=True)
    panel.updates.card.btn_first.click()  # Restart now: the window closes, launch.main starts the helper


def main() -> None:
    from tracecmr.app import launch

    real_reveal = launch.reveal

    def reveal(viewer, splash, *a, **k):
        real_reveal(viewer, splash, *a, **k)
        from qtpy.QtCore import QTimer

        def run():
            try:
                driver(viewer)
            except SystemExit:
                raise
            except Exception:
                fail(traceback.format_exc())

        QTimer.singleShot(1000, run)

    launch.reveal = reveal

    def watchdog():
        time.sleep(600)
        fail("timed out after 10 minutes")

    threading.Thread(target=watchdog, daemon=True).start()
    launch.main([])


if __name__ == "__main__":
    main()
