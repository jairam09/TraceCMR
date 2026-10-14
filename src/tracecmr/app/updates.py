"""Updates in the app: a small card in the corner when a new version is out, Help → Check for updates, and a
note after an update saying what happened. The work itself is in tracecmr/updater.py.

    TraceCMR 0.3.5 is available            [Update]  [Later]
      -> downloads (a few seconds)         [Restart now]  [When I close TraceCMR]
      -> TraceCMR closes, the helper installs it and opens TraceCMR again
      -> "Updated to TraceCMR 0.3.5" with what's new

The check at start runs in the background and stays silent when offline or up to date. It is off for source
installs (git clone), in tests, with TRACECMR_NO_UPDATE_CHECK=1, or when unticked in Settings.
"""

from __future__ import annotations

import os
import threading

from qtpy.QtCore import QObject, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QProgressBar, QPushButton

from .. import updater
from .popover import Popover

EXTRA_STYLE = """
QProgressBar { background: #262B32; border: none; border-radius: 3px; height: 6px; }
QProgressBar::chunk { background: #8FA3B8; border-radius: 3px; }
"""


PRIMARY = ("QPushButton { background: #F1F2F4; color: #12151A; border: none; border-radius: 7px; padding: 6px 12px; "
           "font-size: 12px; font-weight: 600; } QPushButton:hover { background: #FFFFFF; }")


def _short(note: str, n: int = 150) -> str:
    """First sentence of a changelog bullet, at most n characters."""
    s = note.strip().replace("**", "")
    cut = s.find(". ")
    if 0 < cut < n:
        s = s[:cut + 1]
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


class UpdateCard(Popover):
    WIDTH = 360

    def __init__(self, parent):
        super().__init__(parent, "Update")
        self.setStyleSheet(self.styleSheet() + EXTRA_STYLE)
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setFixedWidth(self.WIDTH)
        self.notes = QLabel("")
        self.notes.setObjectName("note")
        self.notes.setWordWrap(True)
        self.notes.setFixedWidth(self.WIDTH)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.hide()
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_second = QPushButton("")
        self.btn_first = QPushButton("")
        self.btn_first.setStyleSheet(PRIMARY)  # light, like the tour's Next
        for b in (self.btn_second, self.btn_first):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(b)
        for w in (self.text, self.notes, self.bar):
            self.body.addWidget(w)
        self.body.addLayout(row)
        self._first_cb = self._second_cb = None
        self.btn_first.clicked.connect(lambda: self._first_cb and self._first_cb())
        self.btn_second.clicked.connect(lambda: self._second_cb and self._second_cb())
        self.state = ""

    def set(self, state: str, title: str, text: str, notes: list[str] | None = None,
            first: tuple[str, callable] | None = None, second: tuple[str, callable] | None = None,
            progress: bool = False) -> None:
        self.state = state
        self.title.setText(title)
        self.text.setText(text)
        if notes:
            self.notes.setText("What's new:\n" + "\n".join("•  " + _short(n) for n in notes[:4]))
            self.notes.show()
        else:
            self.notes.hide()
        self.bar.setVisible(progress)
        for btn, spec, attr in ((self.btn_first, first, "_first_cb"), (self.btn_second, second, "_second_cb")):
            btn.setVisible(spec is not None)
            if spec is not None:
                btn.setText(spec[0])
                setattr(self, attr, spec[1])
        self.adjustSize()

    def show_in_corner(self) -> None:
        """Bottom-right of the main window the first time; afterwards where it was left."""
        self.adjustSize()
        win = self.parentWidget()
        if not self._placed and win is not None:
            g = win.geometry()
            self.move(g.right() - self.width() - 28, g.bottom() - self.height() - 64)
            self._placed = True
        self.show()
        self.raise_()


class _Bridge(QObject):
    """Carries results from the background thread to the window's thread."""
    checked = Signal(object, bool)  # Update | None | Exception, manual
    progress = Signal(int, int)
    downloaded = Signal(object)  # Path | Exception


class UpdateController:
    """Owns the card and the update flow for one TraceCMR window."""

    START_DELAY_MS = 6000

    def __init__(self, panel):
        self.panel = panel
        self.app_mode = False  # set by launch.main: closing the window ends the program and runs the helper
        self.card: UpdateCard | None = None
        self.up: updater.Update | None = None
        self.wheel = None
        self._busy = False
        self.bridge = _Bridge()
        self.bridge.checked.connect(self._checked)
        self.bridge.progress.connect(self._progress)
        self.bridge.downloaded.connect(self._downloaded)

    # ---------- helpers ----------

    def _win(self):
        return self.panel.viewer.window._qt_window

    def _card(self) -> UpdateCard:
        if self.card is None:
            self.card = UpdateCard(self._win())
        return self.card

    def _status(self, msg: str) -> None:
        try:
            self.panel.viewer.status = msg
        except Exception:
            pass

    @staticmethod
    def check_on_start_enabled() -> bool:
        from ._widget import _settings

        return str(_settings().value("updates/check_on_start", True)).lower() in ("true", "1")

    @staticmethod
    def set_check_on_start(on: bool) -> None:
        from ._widget import _settings

        _settings().setValue("updates/check_on_start", bool(on))

    # ---------- start of the app ----------

    def startup(self) -> None:
        """After the window is up: say what an update just did, then check for a new one (quietly)."""
        res = updater.read_result()
        if res:
            self.show_result(res)
        if (os.environ.get("TRACECMR_NO_UPDATE_CHECK") or getattr(self.panel, "_testing", False)
                or not self.check_on_start_enabled() or updater.install_kind() == "dev"):
            return
        def when_free():  # not on top of the first-run tour
            tour = getattr(self.panel, "_tour", None)
            if tour is not None and tour.isVisible():
                QTimer.singleShot(3000, when_free)
            else:
                self.check(manual=False)

        QTimer.singleShot(self.START_DELAY_MS, when_free)

    def show_result(self, res: dict) -> None:
        print(f"update notice: {res.get('from')} -> {res.get('to')} ok={res.get('ok')}", flush=True)
        card = self._card()
        if res.get("ok"):
            card.set("updated", "Updated", f"TraceCMR is now version {res.get('to')}.", res.get("notes"),
                     first=("OK", card.hide))
        else:
            card.set("update_failed", "The update didn't install",
                     f"{res.get('message', '')} Details are in {updater.log_path().name} "
                     f"(Documents/TraceCMR results/logs). Help → Report a problem sends them to us.",
                     first=("OK", card.hide))
        card.show_in_corner()

    # ---------- checking ----------

    def check(self, manual: bool = True, sync: bool = False) -> None:
        """Help → Check for updates (manual) or the quiet check at start."""
        if self._busy:
            return
        pending = updater.scheduled()
        if manual and pending:
            card = self._card()
            card.set("scheduled", "Update ready",
                     f"TraceCMR {pending['version']} installs when you close TraceCMR.",
                     first=("Restart now", self.restart_now) if self.app_mode else ("OK", card.hide),
                     second=("OK", card.hide) if self.app_mode else None)
            card.show_in_corner()
            return
        if manual:
            card = self._card()
            card.set("checking", "Checking for updates", "Asking GitHub for the latest version…", progress=True)
            card.bar.setRange(0, 0)
            card.show_in_corner()
        self._busy = True

        def work():
            try:
                res = updater.check()
            except Exception as e:  # noqa: BLE001  (reported in the card or the log)
                res = e
            self.bridge.checked.emit(res, manual)

        if sync:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    def _checked(self, res, manual: bool) -> None:
        self._busy = False
        card = self._card()
        if isinstance(res, Exception):
            print(f"update check failed: {res}", flush=True)
            if manual:
                card.set("check_failed", "Couldn't check for updates", str(res),
                         first=("Try again", lambda: self.check(manual=True)), second=("Close", card.hide))
                card.show_in_corner()
            return
        if res is None:
            if manual:
                card.set("current", "You're up to date",
                         f"This is the latest version, TraceCMR {updater.current_version()}.",
                         first=("OK", card.hide))
                card.show_in_corner()
            return
        self.up = res
        kind = updater.install_kind()
        why = updater.can_patch(res)
        cur = updater.current_version()
        if kind == "dev":
            card.set("dev", f"TraceCMR {res.version} is out",
                     f"You have {cur}, running from a source folder. Update it with git pull (GitHub Desktop: "
                     f"Fetch origin, then Pull).", res.notes, first=("OK", card.hide))
        elif why or kind == "readonly" or not self.app_mode:
            reason = ("It adds new components, so it comes as a new installer." if why else
                      "This copy can't update itself, so use the new installer." if kind == "readonly" else
                      "Get it from the download page.")
            print(f"update {res.version} needs the installer: {why or kind}", flush=True)
            card.set("installer", f"TraceCMR {res.version} is available",
                     f"You have {cur}. {reason} Your results and settings are kept.", res.notes,
                     first=("Download", lambda: self._open(res.installer_url or res.page)),
                     second=("Later", card.hide))
        else:
            card.set("available", f"TraceCMR {res.version} is available",
                     f"You have {cur}. Updating takes under a minute and keeps your settings and results.",
                     res.notes, first=("Update", self.download), second=("Later", card.hide))
        card.show_in_corner()

    def _open(self, url: str) -> None:
        QDesktopServices.openUrl(QUrl(url))
        if self.card:
            self.card.hide()

    # ---------- downloading ----------

    def download(self, sync: bool = False) -> None:
        up = self.up
        if up is None or self._busy:
            return
        card = self._card()
        card.set("downloading", "Downloading", f"Downloading TraceCMR {up.version}…", progress=True)
        card.bar.setRange(0, 100)
        card.bar.setValue(0)
        self._busy = True

        def work():
            try:
                res = updater.download(up, progress=lambda d, t: self.bridge.progress.emit(d, t))
            except Exception as e:  # noqa: BLE001
                res = e
            self.bridge.downloaded.emit(res)

        if sync:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    def _progress(self, done: int, total: int) -> None:
        if self.card and total:
            self.card.bar.setValue(int(100 * done / total))

    def _downloaded(self, res) -> None:
        self._busy = False
        card = self._card()
        if isinstance(res, Exception):
            card.set("download_failed", "Update not downloaded", str(res),
                     first=("Try again", self.download), second=("Close", card.hide))
            return
        self.wheel = res
        updater.clean_downloads(keep=res.name)
        card.set("ready", f"TraceCMR {self.up.version} is ready",
                 "To finish, TraceCMR closes, installs it and opens again (about half a minute). "
                 "Your contours are saved first.",
                 first=("Restart now", self.restart_now), second=("When I close TraceCMR", self.when_closed))
        card.show_in_corner()

    # ---------- installing ----------

    def when_closed(self) -> None:
        if self.up is None or self.wheel is None:
            return
        updater.schedule(self.up, self.wheel, relaunch=False)
        self._status(f"TraceCMR {self.up.version} will install when you close TraceCMR.")
        if self.card:
            self.card.hide()

    def restart_now(self) -> None:
        pending = updater.scheduled()
        if self.up is not None and self.wheel is not None:
            updater.schedule(self.up, self.wheel, relaunch=True)
        elif pending:
            pending["relaunch"] = True
        else:
            return
        self._save_work()
        if self.card:
            self.card.hide()
        win = self._win()
        if win.close():
            QTimer.singleShot(0, QApplication.instance().quit)
        else:  # something kept the window open: install at the next close instead
            if updater.scheduled():
                updater.scheduled()["relaunch"] = False

    def _save_work(self) -> None:
        """Save the open mouse if it has changes (the same save as Ctrl+S, without the popup)."""
        p = self.panel
        try:
            if p.session is not None and p.autosave._current()[0] != p.autosave._last_hash:
                p.save_contours(quiet=True)
        except Exception as e:  # noqa: BLE001  (never block the restart; the recovery file is still there)
            print(f"save before update failed: {e!r}", flush=True)
