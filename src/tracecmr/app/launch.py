"""Start TraceCMR (napari with the TraceCMR panel docked).

    tracecmr            from a terminal (messages go to the terminal)
    tracecmr-app        the desktop app: no console window, messages go to the log file (see LOG below)
    tracecmr Mouse40.h5 open a scan straight away
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .branding import ICON_PNG as ICON  # noqa: F401  (kept for older scripts/tests)
from .branding import Splash, animate_logo, create_app, prepare, set_icon, start_screen


def log_path() -> Path:
    """Where the desktop app writes its messages: TRACECMR_LOG if set, else Documents/TraceCMR results/logs."""
    env = os.environ.get("TRACECMR_LOG")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Documents" / "TraceCMR results" / "logs" / "tracecmr_log.txt"


def _windowless_output() -> Path | None:
    """With no console (pythonw / tracecmr-app), print() and errors would vanish or crash: send them to the log
    file instead, along with crash tracebacks. Returns the log path, or None when there is a console."""
    if sys.stdout is not None and sys.stderr is not None:
        return None
    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 5_000_000:  # keep it small: start over past 5 MB
            path.replace(path.with_suffix(".old.txt"))
        fh = open(path, "a", encoding="utf-8", buffering=1)
    except OSError:
        fh = open(os.devnull, "w")
    sys.stdout = sys.stdout or fh
    sys.stderr = sys.stderr or fh
    try:
        import datetime
        import faulthandler

        print(f"\n--- TraceCMR started {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ---", flush=True)
        faulthandler.enable(fh)  # a hard crash still leaves a traceback in the log
    except Exception:
        pass
    return path


def main(argv: list[str] | None = None) -> None:
    _windowless_output()
    argv = sys.argv[1:] if argv is None else argv
    create_app()  # fast: the splash appears before napari's slow import
    splash = Splash()
    with splash.track_imports(0.04, 0.82):  # the line glides as the program's parts load
        import napari

        prepare()
        from ._widget import TraceCMRWidget
        from .session import Session

    splash.progress(0.84, "Almost ready")
    viewer = napari.Viewer(title="TraceCMR", show=False)  # built hidden: no half-drawn window flashing up
    set_icon(viewer)
    panel = TraceCMRWidget(viewer)
    panel_dock = viewer.window.add_dock_widget(panel, name="TraceCMR", area="right")
    from ._widget import setup_dock

    setup_dock(panel_dock)
    add_panels_menu(viewer, [panel_dock] + list(getattr(panel, "docks", [])))
    splash.progress(0.92)
    start_screen(viewer, open_path=panel.open_path, animate=True)
    if argv:
        p = Path(argv[0])
        panel.load_session(Session.from_dicom(p) if p.is_dir() else Session.from_ucl(p))
    reveal(viewer, splash)
    animate_logo(viewer)
    from qtpy.QtCore import QTimer

    QTimer.singleShot(2400, panel.maybe_start_tour)  # first run only: once the logo has drawn itself in
    panel.updates.app_mode = True  # closing this window ends the program, so an update can install then
    QTimer.singleShot(3000, panel.updates.startup)  # "updated to X" note, then a quiet check for a new version
    napari.run()
    from .. import updater

    updater.apply_scheduled()  # an update waiting for TraceCMR to close: hand it to the helper and go
    # The window is closed and everything is saved: leave straight away. Letting Python tear down Qt, OpenGL and
    # napari piece by piece can crash at the very end on some systems (a pointless "stopped working" box).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(0)





def reveal(viewer, splash, settle_s: float = 0.6) -> None:
    """Show the finished window without a flash: it opens fully transparent, gets a moment to draw its theme and
    image canvas (which starts out white), then appears in one step as the splash closes."""
    import time

    from qtpy.QtWidgets import QApplication

    win = viewer.window._qt_window
    try:
        win.setWindowOpacity(0.0)
    except Exception:
        pass
    viewer.window.show()
    end = time.monotonic() + settle_s
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.015)
    try:
        win.setWindowOpacity(1.0)
    except Exception:
        pass
    win.raise_()
    win.activateWindow()
    splash.close(win)


def add_panels_menu(viewer, docks) -> None:
    """View → Panels: a tick for each TraceCMR panel (Scans, Results, Toolbar, Time bar) to hide or show it,
    and Show all panels (Ctrl+Shift+R). Closing a panel with its × says in the status bar where to get it back."""
    try:
        from qtpy.QtGui import QAction, QKeySequence
    except ImportError:  # Qt5
        from qtpy.QtGui import QKeySequence
        from qtpy.QtWidgets import QAction
    try:
        win = viewer.window._qt_window
        docks = [d for d in docks if d is not None]
        view = next((a.menu() for a in win.menuBar().actions()
                     if a.menu() is not None and a.text().replace("&", "").strip().lower() == "view"), None)
        order = {"Scans": 0, "Toolbar": 1, "Results": 2, "Time bar": 3}
        docks.sort(key=lambda d: order.get(d.windowTitle(), 9))
        show_all = QAction("Show all panels", win)
        show_all.setShortcut(QKeySequence("Ctrl+Shift+R"))
        show_all.setShortcutContext(show_all.shortcutContext().ApplicationShortcut)

        def restore():
            for d in docks:
                d.show()
                d.raise_()

        show_all.triggered.connect(restore)
        win.addAction(show_all)  # the shortcut works even with the menu bar hidden
        if view is not None:
            panels = view.addMenu("Panels")
            view.insertMenu(view.actions()[0], panels)
            view.insertSeparator(view.actions()[1])
            for d in docks:
                panels.addAction(d.toggleViewAction())  # ticked while the panel is showing
            panels.addSeparator()
            panels.addAction(show_all)
            viewer.window._tracecmr_panels_menu = panels

        def hidden_hint(dock):
            try:
                panel = next((w for w in win.findChildren(object) if type(w).__name__ == "TraceCMRWidget"), None)
                if getattr(panel, "_focusing", False):
                    return  # Focus says so itself
                if dock.isHidden() and win.isVisible():  # hidden on purpose (not just the window minimising)
                    viewer.status = (f"{dock.windowTitle()} panel hidden. Bring it back with the "
                                     f"{dock.windowTitle()} button (top right) or View → Panels.")
            except RuntimeError:  # the window is closing: its parts are already gone
                pass

        for d in docks:
            d.visibilityChanged.connect(lambda visible, dk=d: None if visible else hidden_hint(dk))
            d.toggleViewAction().setText(d.windowTitle())
        viewer.window._tracecmr_restore = show_all
    except Exception:
        pass


if __name__ == "__main__":
    main()
