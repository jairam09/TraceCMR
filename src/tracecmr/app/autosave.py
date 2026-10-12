"""Auto-save and crash recovery.

Every minute, if anything changed since the last save, the open mouse's contours and choices are written to a
recovery file: <results folder>/.recovery/<scan id>.tracecmr.npz. A normal save removes it. If TraceCMR closes
before a save (a crash, a power cut, closing without saving), the next time that mouse is opened TraceCMR offers
to restore the changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

INTERVAL_MS = 60_000


class AutoSave:
    def __init__(self, panel):
        from qtpy.QtCore import QTimer

        self.panel = panel
        self._last_hash: str | None = None
        self.timer = QTimer(panel)
        self.timer.setInterval(INTERVAL_MS)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    # ---------- where and what ----------

    def path(self, session=None) -> Path | None:
        s = session or self.panel.session
        if s is None or not s.source:
            return None  # nothing to tie the file to (e.g. a test phantom)
        return self.panel.contours_dir() / ".recovery" / f"{self.panel._scan_tag(s)}.tracecmr.npz"

    def _current(self):
        """The labels as they are now (including edits not yet written back) and a fingerprint of everything
        that gets saved."""
        s = self.panel.session
        ed = self.panel.editor
        lab = ed.apply_to(s.label) if ed is not None and ed.edited else s.label
        h = hashlib.sha1(np.ascontiguousarray(lab).tobytes())
        extra = [s.ed_frame, s.es_frame, sorted(getattr(s, "es_by_slice", {}).items()),
                 sorted(getattr(s, "ed_by_slice", {}).items()), list(s.order),
                 {k: sorted(v) for k, v in s.included.items()}, s.rois, sorted(s.septum_manual.items())]
        h.update(json.dumps(extra, default=str, sort_keys=True).encode())
        return h.hexdigest(), lab

    # ---------- saving ----------

    def tick(self) -> Path | None:
        """Write the recovery file if something changed since the last save or recovery write."""
        s = self.panel.session
        target = self.path()
        if s is None or target is None:
            return None
        try:
            digest, lab = self._current()
            if digest == self._last_hash:
                return None
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part.npz")
            # Write the labels as they are now, and the editor's current points with them. (Writing the
            # points from the last normal save against the new labels made a restore show, and then write
            # back, the old contours: the labels came back but the picture didn't.)
            ed = self.panel.editor
            keep = (s.label, s.saved_curves, s.saved_hidden, s.saved_curve_hash)
            s.label = lab
            if ed is not None and ed.curves:
                s.saved_curves = {k: np.array(v, dtype=float) for k, v in ed.curves.items()}
                s.saved_hidden = {k for k in ed.hidden if k in s.saved_curves}
                s.saved_curve_hash = {k[:2]: s.slice_hash(*k[:2]) for k in s.saved_curves}
            try:
                written = s.save_contours(tmp)
            finally:
                s.label, s.saved_curves, s.saved_hidden, s.saved_curve_hash = keep
            os.replace(written, target)  # all at once: never a half-written recovery file
            self._last_hash = digest
            return target
        except Exception:  # auto-save must never get in the way
            return None

    def mark_saved(self) -> None:
        """A normal save happened: the recovery file is no longer needed."""
        target = self.path()
        if target is not None and target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        if self.panel.session is not None:
            try:
                self._last_hash = self._current()[0]
            except Exception:
                pass

    # ---------- recovery ----------

    def baseline(self) -> None:
        """The state as it is now counts as saved: nothing to write until something changes."""
        try:
            self._last_hash = self._current()[0]
        except Exception:
            self._last_hash = None

    def session_opened(self) -> Path | None:
        """A mouse was opened: its state now counts as saved. Returns a recovery file worth offering (newer than
        the last normal save), if there is one."""
        self.baseline()
        target = self.path()
        if target is None or not target.exists():
            return None
        saved = self.panel._remembered_path()
        if saved is not None and saved.exists() and saved.stat().st_mtime >= target.stat().st_mtime:
            target.unlink(missing_ok=True)  # an older leftover: the normal save is newer
            return None
        return target

    @staticmethod
    def when(path: Path) -> str:
        t = datetime.fromtimestamp(path.stat().st_mtime)
        return t.strftime("%H:%M") if t.date() == datetime.now().date() else t.strftime("%b %d, %H:%M")
