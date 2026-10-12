"""Help → Report a problem: one file with what's needed to fix a bug (versions, system, the end of the log),
copied to the clipboard and, when a support address is set, an email ready to send."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path


def support_email() -> str:
    from .branding import SUPPORT_EMAIL

    return os.environ.get("TRACECMR_SUPPORT_EMAIL", SUPPORT_EMAIL).strip()


def diagnostics(log_lines: int = 200) -> str:
    """Versions, system and the end of the log, as plain text."""
    from .. import __version__
    from .launch import log_path

    def version(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "?"

    try:
        from qtpy import API_NAME, QT_VERSION
    except Exception:
        API_NAME, QT_VERSION = "?", "?"
    try:
        from .. import autocontour as ac

        models = "; ".join(f"{k}: {v}" for k, v in ac.available().items())
    except Exception as e:
        models = f"? ({e})"
    lines = [f"TraceCMR {__version__}", f"Date: {datetime.now():%Y-%m-%d %H:%M}",
             f"System: {platform.platform()}", f"Python: {sys.version.split()[0]}",
             f"napari {version('napari')} · numpy {version('numpy')} · {API_NAME} (Qt {QT_VERSION})",
             f"Models: {models}", ""]
    log = log_path()
    if log.exists():
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-log_lines:]
        lines += [f"--- last {len(tail)} lines of {log.name} ---", *tail]
    else:
        lines.append("(no log file: TraceCMR was started from a terminal)")
    return "\n".join(lines)


def write_report() -> Path:
    """Save the diagnostics next to the log and return the file."""
    from .launch import log_path

    folder = log_path().parent
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"TraceCMR_problem_report_{datetime.now():%Y%m%d_%H%M}.txt"
    path.write_text(diagnostics(), encoding="utf-8")
    return path


def report_problem(parent) -> Path:
    """Write the report, copy it, open the folder, and open an email to support (if an address is set)."""
    from urllib.parse import quote

    from qtpy.QtCore import QUrl
    from qtpy.QtGui import QDesktopServices
    from qtpy.QtWidgets import QApplication, QMessageBox

    from .. import __version__

    path = write_report()
    QApplication.clipboard().setText(path.read_text(encoding="utf-8"))
    addr = support_email()
    testing = getattr(parent, "_testing", False)
    if addr and not testing:
        body = ("What happened, and what were you doing just before?\n\n\n"
                f"(Please attach {path.name}: it is in {path.parent}, and its text is also on your clipboard.)")
        QDesktopServices.openUrl(QUrl(f"mailto:{addr}?subject={quote(f'TraceCMR {__version__}: problem report')}"
                                      f"&body={quote(body)}"))
    if not testing:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        where = f"An email to {addr} is open: attach the report and describe what happened." if addr else \
            "Send it to the TraceCMR team with a short note on what happened."
        QMessageBox.information(parent, "Problem report ready",
                                f"Saved {path.name} (also copied to the clipboard).\n\n{where}")
    return path
