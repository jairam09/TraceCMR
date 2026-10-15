"""TraceCMR panel for napari: open a scan, set slice order, edit contours, compute.

The panel docks beside the napari viewer. napari shows the images and the
contours (as an editable labels layer: brush, eraser, fill), and the panel
holds the TraceCMR steps from top to bottom.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QSettings, QSize, Qt, QTimer
from qtpy.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QDialog,
    QToolButton,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QPushButton,
    QRadioButton,
    QCheckBox,
    QProgressBar,
    QComboBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.segmentation import find_boundaries

from .. import labels as L
from ..viz import WALL_CMAP_NAME, draw_bullseye
from .curve_editor import CurveEditor, curves_from_labels
from .session import Session
from . import _vispy_fixes

_vispy_fixes.apply()  # shapes redraws keep their compiled shader (smooth dot dragging)


def _settings() -> QSettings:
    """App settings. The first time after the rename from VentriMap, the old settings are copied over."""
    st = QSettings("TraceCMR", "TraceCMR")
    if not st.value("migrated_from_ventrimap"):
        old = QSettings("VentriMap", "VentriMap")
        for key in old.allKeys():
            if st.value(key) is None:
                st.setValue(key, old.value(key))
        st.setValue("migrated_from_ventrimap", "1")
    return st


RESULTS_EXT = (".tracecmr.npz", ".ventrimap.npz")  # saved contours: current name first, then the old one

BLOOD_RGB = (215, 25, 28)  # endocardium: deep red
MYO_RGB = (79, 195, 247)  # epicardium: light blue
RV_RGB = (255, 214, 0)  # right ventricle: yellow
ADD_TARGETS = ["endo", "epi", "rv"]
APP_MODELS = ("ucl",)  # auto-contour models offered in the app (the RV is drawn by hand)  # order of the 'New contours are' choices
WARN_STYLE = "color: #E3A23B;"
NOTE_STYLE = "color: #9AA1A8;"

TOOLBAR_STYLE = """
QToolButton { border: none; border-radius: 6px; padding: 2px 3px; color: #D6D9DD; font-size: 11px; }
QToolButton:hover { background: #3A4048; }
QToolButton:checked { background: #4A5562; }
QToolButton:disabled { color: #6B7178; }
QToolButton[popupMode="1"] { padding-right: 15px; }
QToolButton::menu-button { border: none; background: transparent; width: 14px; }
QToolButton::menu-button:hover { background: #3A4048; border-radius: 4px; }
QToolButton[popupMode="2"] { padding-right: 11px; }
QToolButton::menu-indicator { subcontrol-origin: padding; subcontrol-position: right bottom; right: 1px; bottom: 5px; }
QFrame#toolsep { background: #3A4048; border: none; }
"""


def _tool_separator() -> QFrame:
    """A thin vertical line between toolbar groups."""
    sep = QFrame()
    sep.setObjectName("toolsep")
    sep.setFixedSize(1, 34)
    return sep


def _icon(name: str, color: str = "#D6D9DD") -> QIcon:
    """Material Design icon via qtawesome; an empty icon (text only) if qtawesome is not installed."""
    try:
        import qtawesome as qta

        return qta.icon(name, color=color, color_disabled="#5C6168")
    except Exception:
        return QIcon()


CONVENTION_TEXT = {
    "myocardium": "Detailed: papillary muscles counted as muscle",
    "blood_pool": "Smooth: papillary muscles in the blood pool",
    None: "Unknown convention",
}


def _thumbnail(img2d: np.ndarray, lab2d: np.ndarray, size: int = 72, angle: float = 0.0,
               flip: bool = False) -> QPixmap:
    """Slice thumbnail with blood pool and myocardium outlines (x right, y down), turned like the view."""
    img = img2d.T.astype(np.float32)
    lab = lab2d.T
    ys, xs = np.nonzero(lab)
    if len(ys):  # crop around the heart so thumbnails are readable
        cy, cx = int(ys.mean()), int(xs.mean())
        half = max(24, int(max(np.ptp(ys), np.ptp(xs)) * 0.9))
    else:
        cy, cx, half = img.shape[0] // 2, img.shape[1] // 2, min(img.shape) // 2
    y0, y1 = max(cy - half, 0), min(cy + half, img.shape[0])
    x0, x1 = max(cx - half, 0), min(cx + half, img.shape[1])
    img, lab = img[y0:y1, x0:x1], lab[y0:y1, x0:x1]
    if angle or flip:
        from .orient import orient_image

        img, lab = orient_image(img, angle, flip, order=1), orient_image(lab, angle, flip, order=0)
    lo, hi = np.percentile(img, [1, 99]) if img.size else (0, 1)
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (np.dstack([g, g, g]) * 255).astype(np.uint8)
    for value, colour in ((L.LV_BLOODPOOL, BLOOD_RGB), (L.LV_MYOCARDIUM, MYO_RGB), (L.RV_BLOODPOOL, RV_RGB)):
        mask = lab == value
        if value == L.LV_MYOCARDIUM:
            mask = mask | (lab == L.LV_BLOODPOOL)  # outer edge of the LV = epicardium
        edge = find_boundaries(mask, mode="inner")
        rgb[edge] = colour
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)


PANEL_TITLES = {"Mouse": "Scans", "TraceCMR": "Results", "Tools": "Toolbar", "Time": "Time bar"}


class _PanelHeader(QWidget):
    """Slim header for a side panel: its name, and a × that hides it (View → Panels brings it back)."""

    def __init__(self, dock, title: str):
        super().__init__(dock)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 4, 2)
        lay.setSpacing(4)
        lbl = QLabel(title.upper())
        lbl.setStyleSheet("color: #7D8793; font-size: 10px; font-weight: 600; letter-spacing: 0.8px;")
        lay.addWidget(lbl)
        lay.addStretch(1)
        btn = QToolButton(self)
        btn.setText("✕")
        btn.setAutoRaise(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(f"Hide the {title} panel. The {title} button (top right) brings it back.")
        btn.setStyleSheet("QToolButton { color: #7D8793; border: none; padding: 0 4px; font-size: 11px; }"
                          "QToolButton:hover { color: #E6E8EB; }")
        btn.clicked.connect(dock.close)
        lay.addWidget(btn)
        self.close_btn = btn


def setup_dock(dock, header: bool = True) -> None:
    """Make a napari dock behave like a panel in a normal desktop app: stays where it is (no tearing off), named
    plainly ("Scans", "Results", ...) so it can be ticked on and off in the View menu, and, with `header`,
    a slim header with a × to hide it. Hiding never destroys the panel, so it always comes back intact."""
    from qtpy.QtWidgets import QDockWidget

    if dock is None:
        return
    try:  # napari rebuilds its own title strip whenever a panel moves; stop that for this one
        dock.dockLocationChanged.disconnect(dock._set_title_orientation)
    except (AttributeError, TypeError, RuntimeError):
        pass
    title = PANEL_TITLES.get(dock.objectName(), dock.objectName())
    dock.setWindowTitle(title)
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)  # close = hide; no floating
    dock.setTitleBarWidget(_PanelHeader(dock, title) if header else QWidget(dock))
    dock._tracecmr_panel = title


class SmoothCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas that keeps panel resizing smooth.

    A normal canvas re-renders its whole figure on every pixel of a drag (the bullseyes even rebuild), which made
    dragging a panel edge lag by a fifth of a second per step. This one shows a scaled snapshot while the size is
    changing and renders crisply once resizing pauses for `delay_ms`.
    """

    def __init__(self, figure, delay_ms: int = 120):
        super().__init__(figure)
        self._pending = None
        self._snapshot = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._apply_resize)
        self._delay = delay_ms

    def _take_snapshot(self):
        try:
            buf = np.asarray(self.buffer_rgba())
            h, w = buf.shape[:2]
            if w < 2 or h < 2:
                return None
            img = QImage(buf.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
            return QPixmap.fromImage(img.copy())
        except Exception:  # nothing drawn yet
            return None

    def _background(self) -> QColor:
        from matplotlib.colors import to_hex

        return QColor(to_hex(self.figure.get_facecolor()))

    def resizeEvent(self, event):  # noqa: N802 (Qt name)
        if not self.isVisible() or self._delay <= 0:
            return super().resizeEvent(event)
        if self._pending is None:
            self._snapshot = self._take_snapshot()
        from qtpy.QtGui import QResizeEvent

        self._pending = QResizeEvent(event.size(), event.oldSize() if self._pending is None
                                     else self._pending.oldSize())
        self._timer.start(self._delay)
        self.update()

    def _apply_resize(self):
        ev, self._pending = self._pending, None
        self._snapshot = None
        if ev is not None:
            super().resizeEvent(ev)
            self.update()

    def paintEvent(self, event):  # noqa: N802
        if self._pending is not None:
            p = QPainter(self)
            if self._snapshot is not None:  # the last picture, scaled evenly (never stretched) and centred
                p.fillRect(self.rect(), self._background())
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                snap = self._snapshot.size()
                k = min(self.width() / max(snap.width(), 1), self.height() / max(snap.height(), 1))
                w, h = snap.width() * k, snap.height() * k
                from qtpy.QtCore import QRectF

                p.drawPixmap(QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), self._snapshot,
                             QRectF(self._snapshot.rect()))
            else:  # nothing rendered yet: the figure's background, never a blank (white) widget
                p.fillRect(self.rect(), self._background())
            p.end()
            return
        super().paintEvent(event)


class _Track(QWidget):
    """The painted part of the time bar."""

    ED_COL, ES_COL, NOW_COL = QColor("#E8A33D"), QColor("#E8A33D"), QColor("#E6E6E6")

    def __init__(self, owner: "TraceCMRWidget"):
        super().__init__()
        self.owner = owner
        self.setMinimumHeight(56)
        self._drag = None

    def _track(self):
        return 12, max(self.width() - 16, 22)

    def _x(self, t: int, n: int) -> float:
        x0, x1 = self._track()
        return x0 + (x1 - x0) * (t / max(n - 1, 1))

    def _t(self, x: float, n: int) -> int:
        x0, x1 = self._track()
        return int(round(np.clip((x - x0) / max(x1 - x0, 1), 0, 1) * (n - 1)))

    def paintEvent(self, event):
        s = self.owner.session
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#262930"))
        if s is None:
            p.end()
            return
        n = s.image.shape[3]
        z = self.owner.current_slice()
        x0, x1 = self._track()
        top, bot = 8, self.height() - 18
        p.fillRect(int(x0), top, int(x1 - x0), bot - top, QColor("#414851"))
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        times = s.frame_times_ms
        step = max(1, n // 12)
        for t in range(n):
            x = self._x(t, n)
            p.setPen(QPen(QColor("#9AA1A8"), 1))
            p.drawLine(int(x), bot - (6 if t % step == 0 else 3), int(x), bot)
            if t % step == 0:
                p.drawText(int(x) - 10, bot + 13, f"{times[t]:.0f}" if times else str(t))
        for phase, col, y in (("ED", self.ED_COL, top + 11), ("ES", self.ES_COL, bot - 3)):
            x = self._x(s.t_for(z, phase), n)
            p.setPen(QPen(col, 3))
            p.drawLine(int(x), top, int(x), bot)
            p.drawText(int(x) + 4, y, phase)
        x = self._x(self.owner.current_frame(), n)
        p.setPen(QPen(self.NOW_COL, 1))
        p.drawLine(int(x), top - 4, int(x), bot + 2)
        p.end()

    def _xpos(self, event):
        return event.position().x() if hasattr(event, "position") else event.x()

    def mousePressEvent(self, event):
        s = self.owner.session
        if s is None:
            return
        n, z, xpos = s.image.shape[3], self.owner.current_slice(), self._xpos(event)
        d, ph = min((abs(self._x(s.t_for(z, ph), n) - xpos), ph) for ph in ("ED", "ES"))
        self._drag = ph if d <= 8 else "scrub"
        self._move(xpos)

    def mouseMoveEvent(self, event):
        if self._drag:
            self._move(self._xpos(event))

    def mouseReleaseEvent(self, event):
        if self._drag in ("ED", "ES"):
            self.owner.set_phase_frame(self._drag, self._t(self._xpos(event), self.owner.session.image.shape[3]))
        self._drag = None

    def _move(self, xpos):
        self.owner._show_frame(self._t(xpos, self.owner.session.image.shape[3]))
        self.update()


class TimeBar(QWidget):
    """Segment-style time bar under the image: play controls and ED/ES on the left, and a track with the
    frames (or ms), the current frame and draggable ED / ES markers for the slice being viewed."""

    def __init__(self, owner: "TraceCMRWidget"):
        super().__init__()
        self.owner = owner
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        left = QGridLayout()
        left.setHorizontalSpacing(4)
        left.setVerticalSpacing(2)

        def small(text, icon, tip):
            b = QToolButton()
            b.setIcon(_icon(icon))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            if not b.icon().isNull():
                b.setText("")
            else:
                b.setText(text)
            return b

        self.btn_play = small("▶", "mdi6.play", "Play / stop the cine")
        self.btn_prev = small("◀|", "mdi6.skip-previous", "Previous frame")
        self.btn_next = small("|▶", "mdi6.skip-next", "Next frame")
        row = QHBoxLayout()
        for b in (self.btn_play, self.btn_prev, self.btn_next):
            row.addWidget(b)
        row.addWidget(owner.lbl_frame)
        row.addStretch(1)
        left.addLayout(row, 0, 0, 1, 4)
        lab_ed, lab_es = QLabel("ED"), QLabel("ES")
        lab_ed.setStyleSheet("color: #E8A33D; font-weight: 600;")
        lab_es.setStyleSheet("color: #E8A33D; font-weight: 600;")
        left.addWidget(lab_ed, 1, 0)
        left.addWidget(owner.spin_ed, 1, 1)
        left.addWidget(owner.btn_set_ed, 1, 2)
        left.addWidget(lab_es, 2, 0)
        left.addWidget(owner.spin_es, 2, 1)
        left.addWidget(owner.btn_set_es, 2, 2)
        left.addWidget(owner.chk_per_slice, 1, 3)
        left.addWidget(owner.btn_auto_frames, 2, 3)
        holder = QWidget()
        holder.setLayout(left)
        holder.setFixedWidth(420)
        lay.addWidget(holder)
        self.track = _Track(owner)
        lay.addWidget(self.track, 1)
        self.btn_play.clicked.connect(owner._toggle_play)
        self.btn_prev.clicked.connect(lambda: owner.step_frame(-1))
        self.btn_next.clicked.connect(lambda: owner.step_frame(1))


class SeriesPanel(QWidget):
    """Left-hand list of the study's image series, like Segment's thumbnails. Click one to work on it."""

    def __init__(self, owner: "TraceCMRWidget"):
        super().__init__()
        self.owner = owner
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        cap = QLabel("Mice")
        cap.setStyleSheet("font-weight: 600;")
        row.addWidget(cap)
        row.addStretch(1)
        self.btn_root = QPushButton("Data folder…")
        self.btn_root.setToolTip("Choose the folder that holds your mice (one folder or .h5 file each).")
        row.addWidget(self.btn_root)
        lay.addLayout(row)
        self.mice = QListWidget()
        self.mice.setMaximumHeight(150)
        self.mice.setUniformItemSizes(True)
        self.mice.setToolTip("Click a mouse to open it. Saved contours load automatically.")
        lay.addWidget(self.mice)
        self.lbl_root = QLabel("")
        self.lbl_root.setStyleSheet(NOTE_STYLE)
        self.lbl_root.setWordWrap(True)
        lay.addWidget(self.lbl_root)
        self.btn_root.clicked.connect(self._choose_root)
        self.mice.itemClicked.connect(lambda it: owner.open_path(it.data(Qt.ItemDataRole.UserRole)))
        cap = QLabel("Series")
        cap.setStyleSheet("font-weight: 600;")
        lay.addWidget(cap)
        self.list = QListWidget()
        self.list.setIconSize(QSize(44, 44))
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setSpacing(1)
        self.list.setWordWrap(False)
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay.addWidget(self.list, 1)
        self.btn_combine = QPushButton("Combine selected into one stack")
        self.btn_combine.setToolTip("For one cine per slice: Ctrl-click the slices, then Combine.")
        lay.addWidget(self.btn_combine)
        self.btn_combine.setVisible(False)  # shown when there are separate single-slice series to combine
        self.lbl = QLabel("Open a scan to begin.")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet(NOTE_STYLE)
        lay.addWidget(self.lbl)
        self.btn_combine.clicked.connect(owner.combine_selected)
        self.list.itemClicked.connect(lambda it: owner.open_series(self._rows[self.list.row(it)]))
        self._rows: list[int] = []
        self.avail_height = None  # callable: the height the panel is shown in (set once it is docked)
        self._fit_lists()

    # ---------- data folder ----------

    def _choose_root(self):
        cur = _settings().value("folders/data") or ""
        d = QFileDialog.getExistingDirectory(self, "Folder that holds your mice", str(cur))
        if d:
            if self.owner._overlap(d, self.owner.contours_dir()):
                self.lbl_root.setText(f"⚠ Not changed: the raw scans and results folders ({self.owner.contours_dir()}) "
                                      "must be separate. Pick another folder, or move the results folder in Settings.")
                return
            _settings().setValue("folders/data", d)
            self.fill_mice(d)
            if hasattr(self.owner, "lbl_data_dir"):
                self.owner.lbl_data_dir.setText(d)

    def fill_mice(self, root=None) -> None:
        """List the mice in the data folder: each sub-folder (a ParaVision study or a mouse's DICOM series) and each
        .h5 file."""
        root = root or _settings().value("folders/data")
        self.mice.clear()
        if not root or not Path(root).is_dir():
            self.lbl_root.setText("Choose a data folder to list your mice.")
            return
        root = Path(root)
        res = self.owner.contours_dir()
        entries = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
                          and not self.owner._overlap(p, res)]
                         + list(root.glob("*.h5")), key=lambda p: p.name.lower())
        saved = set()
        if res.is_dir():  # per-mouse results folders, and older flat contour files
            saved = {p.name.split("__")[0] for p in res.iterdir()
                     if (p.is_dir() and any(f.name.endswith(RESULTS_EXT) for f in p.iterdir()))
                     or p.name.endswith(RESULTS_EXT)}
        for p in entries:
            stem = p.stem if p.suffix else p.name
            key = self.owner._safe(stem)
            folder = not p.suffix  # a DICOM folder's saved name is the folder name plus the series
            mark = "  ✓" if any(n == key or (folder and n.startswith(key + "_")) for n in saved) else ""
            it = QListWidgetItem(("📄 " if p.suffix.lower() == ".h5" else "📁 ") + p.name + mark)
            it.setData(Qt.ItemDataRole.UserRole, str(p))
            it.setToolTip(str(p) + ("\nHas saved contours" if mark else ""))
            self.mice.addItem(it)
        self.lbl_root.setText(f"{Path(root).name}  ·  {len(entries)} mice  ·  ✓ = saved contours")
        self.lbl_root.setToolTip(str(root))
        self._fit_lists()

    def refresh(self, study, current: int | None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        rows = []
        for i, ser in enumerate(study):
            if ser.stack is not None:
                continue  # its slices are in the slice list below
            rows.append(i)
        self._rows = rows
        for i in rows:
            ser = study[i]
            part = ser.stack is not None
            if part:
                pix = _thumbnail(ser.thumb, np.zeros(ser.thumb.shape, np.uint8), 44) if ser.thumb is not None \
                    else QPixmap(44, 44)
                item = QListWidgetItem(QIcon(pix), f"   slice {ser.slice_index}: {ser.name}")
                item.setSizeHint(QSize(0, 48))
                self.list.addItem(item)
                continue
            if ser.session is not None:
                se = ser.session
                z = se.image.shape[2] // 2
                pix = _thumbnail(se.image[:, :, z, se.ed_frame], se.label[:, :, z, se.ed_frame], 44)
            elif ser.thumb is not None:
                pix = _thumbnail(ser.thumb, np.zeros(ser.thumb.shape, np.uint8), 44)
            else:
                pix = QPixmap(44, 44)
            item = QListWidgetItem(QIcon(pix), f"{ser.name}\n{ser.summary()}")
            item.setToolTip(f"{ser.name}\n{ser.summary()}")
            item.setSizeHint(QSize(0, 52))
            if i == current:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.list.addItem(item)
        if current is not None and current in rows:
            self.list.setCurrentRow(rows.index(current))
        self.list.blockSignals(False)
        self.lbl.setText(f"{len(rows)} series. Click one to work on it." if study
                         else "Open a scan to begin.")
        # combining only makes sense for separate single-slice series
        self.btn_combine.setVisible(sum(1 for i in rows if study[i].n_slices == 1) > 1)
        self._fit_lists()

    MICE_ROW, SERIES_ROW, SLICE_ROW = 24, 54, 38  # list row heights (slices: with their thumbnail)

    def _fit_lists(self) -> None:
        """Size the mice and series lists to what they hold, but keep room for up to 6 rows of the slice list
        below: on a short screen the mice and series lists shrink first (they scroll), down to one row each."""
        tree = getattr(self, "tree", None)
        mice_rows, series_rows = min(max(self.mice.count(), 1), 5), min(max(self.list.count(), 1), 3)
        want = 0
        if tree is not None:
            want = min(max(tree.topLevelItemCount(), 3), 6) * self.SLICE_ROW + 30  # + its header
            avail = self.avail_height() if self.avail_height is not None else 0
            if avail > 0:
                lay = self.layout()
                lay.activate()
                overhead = (lay.minimumSize().height() - self.mice.minimumHeight() - self.list.minimumHeight()
                            - tree.minimumHeight())

                def total() -> int:
                    return overhead + mice_rows * self.MICE_ROW + 6 + series_rows * self.SERIES_ROW + 6 + want

                while total() > avail and (mice_rows > 1 or series_rows > 1):
                    if mice_rows > 2 or series_rows == 1:
                        mice_rows -= 1
                    else:
                        series_rows -= 1
        self.mice.setFixedHeight(mice_rows * self.MICE_ROW + 6)
        self.list.setFixedHeight(series_rows * self.SERIES_ROW + 6)
        if tree is not None:
            tree.setMinimumHeight(want)

    def add_slices(self, tree, buttons, note) -> None:
        """The slice list (order, ED/ES choices) of the open stack, under the series list."""
        lay = self.layout()
        cap = QLabel("Slices")
        cap.setStyleSheet("font-weight: 600; margin-top: 6px;")
        lay.addWidget(cap)
        self.tree = tree
        lay.addWidget(tree, 1)
        row = QHBoxLayout()
        for b in buttons:
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(note)
        lay.addStretch(0)

    def selected_rows(self) -> list[int]:
        return sorted(self._rows[self.list.row(it)] for it in self.list.selectedItems())



TABBAR_STYLE = """
QFrame#tabbar { background: #1E2227; border: 1px solid #3A4048; border-radius: 9px; }
QPushButton { background: transparent; color: #9AA1A9; border: 1px solid transparent; border-radius: 7px;
              padding: 5px 0px; font-size: 12px; font-weight: 600; }
QPushButton:hover { color: #E4E7EB; background: #282D33; }
QPushButton:checked { background: #3A414B; color: #F1F2F4; border: 1px solid #4A525D; }
"""

TAB_INFO = {  # hover card text for each Results tab
    "function": "Volumes, EF, mass, volume curve and per-slice table.",
    "wall": "Wall thickness and thickening bullseyes; wall lines on the image.",
    "regions": "Regions of interest: area, signal, CNR and SNR.",
}


class ResultsTabs:
    """The Results panel's tabs, one per kind of analysis, as a segmented control above the pages. Each page
    scrolls on its own. Ctrl+1, Ctrl+2, Ctrl+3 switch tabs; the Wall and ROI tools open their own tab."""

    def __init__(self, panel):
        from qtpy.QtWidgets import QButtonGroup

        self.panel = panel
        self.bar = QWidget()
        wrap = QHBoxLayout(self.bar)
        wrap.setContentsMargins(9, 2, 9, 0)
        frame = QFrame()
        frame.setObjectName("tabbar")
        frame.setStyleSheet(TABBAR_STYLE)
        self._row = QHBoxLayout(frame)
        self._row.setContentsMargins(3, 3, 3, 3)
        self._row.setSpacing(2)
        wrap.addWidget(frame)
        self.group = QButtonGroup(frame)
        self.group.setExclusive(True)
        self.index: dict[str, int] = {}
        self.buttons: dict[str, QPushButton] = {}
        self.scrolls: dict[str, QScrollArea] = {}

    def page(self, key: str, title: str) -> QVBoxLayout:
        """Add a tab; returns the layout to put its content in."""
        from .hints import set_hint

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(9, 6, 9, 9)
        lay.setSpacing(8)
        scroll.setWidget(body)
        idx = self.panel.results_stack.addWidget(scroll)
        btn = QPushButton(title)
        btn.setCheckable(True)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        set_hint(btn, title, TAB_INFO.get(key, ""), ["Ctrl", str(idx + 1)])
        self.group.addButton(btn, idx)
        self._row.addWidget(btn, 1)
        btn.clicked.connect(lambda _=False, k=key: self.show(k))
        self.index[key], self.buttons[key], self.scrolls[key] = idx, btn, scroll
        if idx == 0:
            btn.setChecked(True)
        return lay

    def current(self) -> str:
        i = self.panel.results_stack.currentIndex()
        return next((k for k, v in self.index.items() if v == i), "")

    def show(self, key: str) -> None:
        if key not in self.index:
            return
        changed = self.current() != key
        self.panel.results_stack.setCurrentIndex(self.index[key])
        self.buttons[key].setChecked(True)
        if changed:
            self.panel._results_tab_changed(key)


GROW_STEP_PX = 0.5  # Grow / Shrink: how far one press moves the contour, in image pixels

class TraceCMRWidget(QWidget):
    """Dock widget: Open, Frames, Slice order, Results (recomputed after every edit)."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.session: Session | None = None
        self.last = None  # (FunctionResult, WallResult | None)
        self.editor: CurveEditor | None = None
        self.study: list = []  # Series in the left panel
        self.current_series: int | None = None

        # ---------------- widgets (placed below: toolbar on top, series + slices on the left,
        # time bar at the bottom, results on the right, model options in Settings) ----------------
        def tool(text, icon, tip, checkable=False):
            b = QToolButton()
            b.setText(text)
            b.setIcon(_icon(icon))
            b.setIconSize(QSize(22, 22))
            b.setToolTip(tip)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            b.setCheckable(checkable)
            b.setMinimumWidth(40)
            b.setAutoRaise(True)
            return b

        self.btn_dcm = tool("Open scan", "mdi6.folder-open-outline",
                            "Open one mouse's scan folder: a Bruker ParaVision study, or a folder of DICOM files.")
        self.btn_ucl = tool("UCL file", "mdi6.file-plus-outline", "Add UCL .h5 files.")
        self.btn_save = tool("Save", "mdi6.content-save-outline",
                             "Save this mouse's contours and choices (Ctrl+S). They reload when you reopen it.")
        self.btn_auto = tool("Auto LV", "mdi6.auto-fix",
                             "Contour the LV on every slice at this frame with UCL's published model (Shah et al., "
                             "JCMR 2026; please cite it when you use Auto LV).")
        self.btn_endo = tool("Endo", "mdi6.circle-outline",
                             "Edit the endocardium (red).", True)
        self.btn_epi = tool("Epi", "mdi6.circle-outline",
                            "Edit the epicardium (light blue).", True)
        self.btn_rv = tool("RV", "mdi6.circle-outline",
                           "Edit the right ventricle (yellow).", True)
        self.btn_endo.setIcon(_icon("mdi6.circle-outline", "#D71A1C"))
        self.btn_epi.setIcon(_icon("mdi6.circle-outline", "#4FC3F7"))
        self.btn_rv.setIcon(_icon("mdi6.circle-outline", "#FFD600"))
        self.btn_endo.setChecked(True)
        self.btn_smooth = tool("Smooth", "mdi6.vector-curve", "Smooth the selected contour on this slice (S).")
        self.btn_snap = tool("Snap", "mdi6.magnet",
                             "Grow the endocardium into bright blood the model missed (stops at muscle).")
        self.btn_fit = tool("Fit", "mdi6.target",
                            "Redraw the endocardium around the bright blood: adds missed blood, returns dark "
                            "muscle to the wall. Best at ES.")
        self.btn_shrink = tool("Shrink", "mdi6.arrow-collapse-all",
                               "Shrink the selected contour by half a pixel ( [ ).")
        self.btn_grow = tool("Grow", "mdi6.arrow-expand-all",
                             "Grow the selected contour by half a pixel ( ] ). The endo never crosses the epi.")
        self.btn_points = tool("Points", "mdi6.dots-circle", "Show or hide the selected contour's points (P).", True)
        self.btn_points.setChecked(True)
        self.btn_clear = tool("Clear", "mdi6.trash-can-outline",
                              "Delete the selected contour on this slice and frame (Del). Undo restores it.")
        self.btn_undo = tool("Undo", "mdi6.undo", "Undo the last edit (Ctrl+Z)")
        self.btn_redo = tool("Redo", "mdi6.redo", "Redo the last undone edit (Ctrl+Y)")
        self.btn_brush = tool("Brush", "mdi6.brush-outline", "Paint pixels instead of editing curves", True)
        self.btn_zoom_out = tool("Zoom −", "mdi6.magnify-minus-outline", "Zoom out (key: −)")
        self.btn_zoom_in = tool("Zoom +", "mdi6.magnify-plus-outline", "Zoom in (key: +)")
        self.btn_zoom_fit = tool("Center", "mdi6.image-filter-center-focus", "Center and zoom on the heart (F).")
        self.btn_raw = tool("Raw", "mdi6.eye-off-outline",
                            "Hide the contours to see the raw image (H).", True)
        self.btn_pan = tool("Pan", "mdi6.hand-back-right-outline",
                            "Left-drag pans instead of editing (Space). Right-drag always pans.", True)
        self.btn_wall = tool("Wall", "mdi6.ruler",
                             "Wall lines on this slice, endo to epi (W). Arrow: thickness or thickening, septum.", True)
        self.btn_contrast = tool("Contrast", "mdi6.brightness-6",
                                 "Brightness and contrast (or Ctrl + drag the image).")
        self._checks = {}  # double-check flags (quality.py), filled by _run_checks
        self.btn_settings = tool("Settings", "mdi6.cog-outline", "Model, auto-contour, editing and folder options")
        # show / hide the side panels; Focus hides both so the scan gets the whole window (Segment style)
        self.btn_left = tool("Scans", "mdi6.dock-left", "Show or hide the Scans panel.", True)
        self.btn_right = tool("Results", "mdi6.dock-right", "Show or hide the Results panel.", True)
        self.btn_focus = tool("Focus", "mdi6.fullscreen",
                              "Hide both side panels so the image fills the window (Ctrl+Shift+F).", True)
        for b in (self.btn_left, self.btn_right):
            b.setChecked(True)
        # dropdowns that gather related tools, so the toolbar shows one button per job
        self.btn_refine = tool("Refine", "mdi6.tune-variant",
                               "Snap, fit, grow ( ] ) or shrink ( [ ) the selected contour.")
        self.btn_view = tool("View", "mdi6.eye-outline",
                             "Hide contours (H), smooth image (G), contrast, zoom.")
        self.btn_help = tool("Help", "mdi6.help-circle-outline", "Keyboard shortcuts (?) and the tour")
        self.btn_roi = tool("ROI", "mdi6.vector-rectangle",
                            "Measure area and signal in a drawn region; compare regions (R). Arrow: circle or "
                            "freehand.", True)
        self.smooth_image = str(_settings().value("view/smooth_image", False)).lower() in ("true", "1")

        # time bar controls
        self.lbl_frame = QLabel("Frame –")
        self.lbl_frame.setStyleSheet("font-weight: 600;")
        self.btn_play = QPushButton("Play cine", self)
        self.btn_play.hide()  # the time bar's ▶ plays the cine
        self.spin_ed, self.spin_es = QSpinBox(), QSpinBox()
        self.spin_ed.setToolTip("End-diastole frame of this slice")
        self.spin_es.setToolTip("End-systole frame of this slice")
        self.btn_set_ed = QPushButton("Set ED here")
        self.btn_set_es = QPushButton("Set ES here")
        self.btn_auto_frames = QPushButton("Detect")
        self.btn_auto_frames.setToolTip("Pick ED/ES from the contours (needs several contoured frames)")
        self.chk_per_slice = QCheckBox("Per-slice ED/ES")
        self.chk_per_slice.setToolTip("On: ED/ES apply to the slice in view (one cine per slice). Off: one "
                                      "ED/ES for the whole stack.")
        self.btn_show_ed, self.btn_show_es = QPushButton("Show", self), QPushButton("Show", self)  # for scripts
        self.btn_show_ed.hide()
        self.btn_show_es.hide()

        # slice list (left column, under the series)
        self.list = QTreeWidget()
        self.list.setColumnCount(4)
        self.list.setHeaderLabels(["Slice (base → apex)", "ED", "ES", "Note"])
        self.list.setRootIsDecorated(False)
        self.list.setIconSize(QSize(40, 40))
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setIconSize(QSize(34, 34))
        self.list.setColumnHidden(3, True)  # notes show as a tooltip and an orange name instead
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        hdr = self.list.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        self.list.setColumnWidth(1, 34)
        self.list.setColumnWidth(2, 34)
        self.list.setToolTip("Drag to order base → apex. Untick ED / ES to leave a slice out. Orange note: "
                             "check that slice.")
        self.btn_suggest = QPushButton("Suggest")
        self.btn_suggest.setToolTip("Re-apply the suggested slice choices")
        self.btn_flip = QPushButton("Flip")
        self.btn_flip.setToolTip("Reverse the order (base ↔ apex)")
        self.btn_auto_order = QPushButton("Auto-order")
        self.lbl_order = QLabel("")
        self.lbl_order.setWordWrap(True)

        # settings dialog contents
        self.cmb_model = QComboBox()
        self.btn_model_dir = QPushButton("Model folder…")
        self.btn_model_dir.setToolTip("Folder with the selected model (remembered).")
        self.chk_all_frames = QCheckBox("Contour every frame (slow; lets Detect find ED/ES)")
        self.chk_tta = QCheckBox("Steadier Auto LV: average 8 orientations of each slice (about 8x slower)")
        self.chk_tta.setToolTip("The model looks at each slice turned and flipped 8 ways and averages what it sees, "
                                "so the contours no longer depend on how the scanner oriented the image. Same "
                                "accuracy on UCL's test mice; much steadier on scans from other scanners.")
        self.chk_tta.setChecked(str(_settings().value("auto/tta", True)).lower() in ("true", "1"))
        os.environ["TRACECMR_TTA"] = "1" if self.chk_tta.isChecked() else "0"
        self.chk_tta.toggled.connect(self._tta_toggled)

        self.chk_raw = QCheckBox("Open UCL files as raw cines (hide the expert contours)")
        self.chk_smooth = QCheckBox("Smooth papillary muscles into the blood pool (experimental)")
        self.lbl_conv = QLabel("")
        self.lbl_conv.setStyleSheet(NOTE_STYLE)
        self.rb_curves = QRadioButton("Curves (interpolation points)")
        self.rb_pixels = QRadioButton("Pixel brush")
        self.rb_curves.setChecked(True)
        self.cmb_add = QComboBox()
        self.cmb_add.addItems(["endocardium (red)", "epicardium (light blue)", "right ventricle (yellow)"])
        self.settings = QDialog(self)
        self.settings.setWindowTitle("TraceCMR settings")
        sl = QVBoxLayout(self.settings)
        g = QGroupBox("Auto-contour")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel("Model"))
        row.addWidget(self.cmb_model, 1)
        row.addWidget(self.btn_model_dir)
        gl.addLayout(row)
        self.btn_setup_model = QPushButton("Set up auto-contouring…")
        from ..model_setup import DOWNLOAD_MB

        self.btn_setup_model.setToolTip(f"One-time download of UCL's model and the library that runs it (about "
                                         f"{DOWNLOAD_MB} MB). Not needed with the TraceCMR installers from 0.3.5 on.")
        self.btn_setup_model.clicked.connect(lambda: self.start_model_setup())
        gl.addWidget(self.btn_setup_model)
        for w in (self.chk_all_frames, self.chk_tta, self.chk_smooth, self.lbl_conv):
            gl.addWidget(w)
        sl.addWidget(g)
        g = QGroupBox("Editing")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(self.rb_curves)
        row.addWidget(self.rb_pixels)
        gl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("New contours are"))
        row.addWidget(self.cmb_add, 1)
        gl.addLayout(row)
        help_ = QLabel("Drag a point to move it · click the line to add one · right-click a point to delete it "
                       "· right-drag pans · scroll zooms · Ctrl+Z undoes. No contour? Click points around it, "
                       "right-click to close.")
        help_.setWordWrap(True)
        help_.setStyleSheet(NOTE_STYLE)
        gl.addWidget(help_)
        sl.addWidget(g)
        g = QGroupBox("Folders")
        gl = QVBoxLayout(g)
        note = QLabel("Raw scans are only read, never changed. Contours, tables and images go to the results "
                      "folder, one sub-folder per mouse.")
        note.setWordWrap(True)
        note.setStyleSheet(NOTE_STYLE)
        gl.addWidget(note)
        grid = QGridLayout()
        self.lbl_data_dir = QLabel("")
        self.lbl_contours_dir = QLabel("")
        for r, (cap, lbl, fn) in enumerate((("Raw scans (read only)", self.lbl_data_dir, self._choose_data_dir),
                                            ("Results", self.lbl_contours_dir, self._choose_contours_dir))):
            lbl.setWordWrap(True)
            lbl.setStyleSheet(NOTE_STYLE)
            btn = QPushButton("Change…")
            btn.clicked.connect(fn)
            grid.addWidget(QLabel(cap), r, 0)
            grid.addWidget(lbl, r, 1)
            grid.addWidget(btn, r, 2)
        grid.setColumnStretch(1, 1)
        gl.addLayout(grid)
        gl.addWidget(self.chk_raw)
        sl.addWidget(g)
        g = QGroupBox("Wall thickness")
        gl = QVBoxLayout(g)
        self.chk_mirror = QCheckBox("Mirror segment numbering (the scan is seen from the base, not the apex)")
        self.chk_mirror.setToolTip("Segments start at the septum (RV–LV junction) in AHA order, seen from the "
                                   "apex. Tick this if anterior and inferior come out swapped.")
        self.chk_mirror.setChecked(str(_settings().value("wall/mirror", False)).lower() in ("true", "1"))
        self.chk_mirror.toggled.connect(self._mirror_toggled)
        gl.addWidget(self.chk_mirror)
        sl.addWidget(g)
        g = QGroupBox("Updates")
        gl = QVBoxLayout(g)
        from .. import __version__
        from .updates import UpdateController

        row = QHBoxLayout()
        lbl_ver = QLabel(f"This is TraceCMR {__version__}.")
        lbl_ver.setStyleSheet(NOTE_STYLE)
        row.addWidget(lbl_ver, 1)
        btn_check = QPushButton("Check now")
        btn_check.clicked.connect(lambda: self.updates.check(manual=True))
        row.addWidget(btn_check)
        gl.addLayout(row)
        self.chk_updates = QCheckBox("Check for updates when TraceCMR starts")
        self.chk_updates.setToolTip("One small download from GitHub at start. Nothing about you or your scans "
                                    "is sent.")
        self.chk_updates.setChecked(UpdateController.check_on_start_enabled())
        self.chk_updates.toggled.connect(UpdateController.set_check_on_start)
        gl.addWidget(self.chk_updates)
        sl.addWidget(g)
        close = QPushButton("Close")
        close.clicked.connect(self.settings.hide)
        sl.addWidget(close)

        # ---------------- right panel: scan, status, results ----------------
        # Fixed header (scan, status), then tabs, one per analysis (Function, Wall, Regions), each page
        # scrolling on its own, and the export buttons fixed at the bottom.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        head = QWidget()
        self.main = QVBoxLayout(head)  # the header; the pages are built below
        self.main.setContentsMargins(9, 9, 9, 0)
        self.main.setSpacing(8)
        outer.addWidget(head)

        self.lbl_scan = QLabel("No scan open. Click <b>Open scan</b> (top left).")
        self.lbl_scan.setWordWrap(True)
        self.main.addWidget(self.lbl_scan)
        self.prog = QProgressBar()
        self.prog.setVisible(False)
        self.main.addWidget(self.prog)
        self.lbl_auto = QLabel("")
        self.lbl_auto.setWordWrap(True)
        self.lbl_auto.setStyleSheet(NOTE_STYLE)
        self.main.addWidget(self.lbl_auto)

        self.btn_compute = QPushButton("  Compute")
        self.btn_compute.setIcon(_icon("mdi6.calculator-variant-outline"))
        self.btn_compute.setMinimumHeight(34)
        self.btn_compute.setStyleSheet("font-weight: 600;")
        # Results recompute by themselves a moment after every edit, so the button is not shown; it is kept
        # (hidden) for scripts and older tests.
        self.btn_compute.setVisible(False)
        self.lbl_warn = QLabel("")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet(WARN_STYLE)
        self.lbl_warn.setVisible(False)
        self.main.addWidget(self.lbl_warn)

        from qtpy.QtWidgets import QStackedWidget

        self.results_tabs = ResultsTabs(self)
        outer.addWidget(self.results_tabs.bar)
        self.results_stack = QStackedWidget()
        outer.addWidget(self.results_stack, 1)
        page_function = self.results_tabs.page("function", "Function")
        page_wall = self.results_tabs.page("wall", "Wall")
        page_regions = self.results_tabs.page("regions", "Regions")

        box = QGroupBox("Left ventricle")
        lay = QVBoxLayout(box)
        self.grid_res = QGridLayout()
        self.grid_res.setHorizontalSpacing(6)
        self.grid_res.setVerticalSpacing(6)
        self.res_labels = {}
        for i, (key, name) in enumerate([("edv_ul", "EDV µL"), ("esv_ul", "ESV µL"), ("sv_ul", "SV µL"),
                                         ("ef_pct", "EF %"), ("mass_ed_mg", "Mass ED mg"),
                                         ("mass_es_mg", "Mass ES mg")]):
            tile = QFrame()
            tile.setStyleSheet("QFrame { background: #2F343C; border-radius: 6px; }")
            tl = QVBoxLayout(tile)
            tl.setContentsMargins(8, 6, 8, 6)
            tl.setSpacing(0)
            cap = QLabel(name)
            cap.setStyleSheet(NOTE_STYLE + " font-size: 11px; background: transparent;")
            val = QLabel("–")
            val.setStyleSheet("font-size: 18px; font-weight: 600; background: transparent;")
            tl.addWidget(cap)
            tl.addWidget(val)
            self.grid_res.addWidget(tile, i // 3, i % 3)
            self.res_labels[key] = val
        lay.addLayout(self.grid_res)
        self.vfig = Figure(figsize=(4.2, 1.4), facecolor="none")
        self.vcanvas = SmoothCanvas(self.vfig)
        self.vcanvas.setMinimumHeight(110)
        self.vcanvas.setStyleSheet("background: transparent;")
        cap = QLabel("Blood-pool volume over the cycle")
        cap.setStyleSheet(NOTE_STYLE)
        self.lbl_vcurve = cap
        lay.addWidget(cap)
        lay.addWidget(self.vcanvas)
        page_function.addWidget(box)

        self.box_rv = QGroupBox("Right ventricle")
        lay = QGridLayout(self.box_rv)
        lay.setHorizontalSpacing(6)
        self.rv_labels = {}
        for i, (key, name) in enumerate([("rvedv_ul", "RVEDV µL"), ("rvesv_ul", "RVESV µL"), ("rvsv_ul", "RVSV µL"),
                                         ("rvef_pct", "RVEF %")]):
            tile = QFrame()
            tile.setStyleSheet("QFrame { background: #2F343C; border-radius: 6px; }")
            tl = QVBoxLayout(tile)
            tl.setContentsMargins(8, 6, 8, 6)
            tl.setSpacing(0)
            cap = QLabel(name)
            cap.setStyleSheet(NOTE_STYLE + " font-size: 11px; background: transparent;")
            val = QLabel("–")
            val.setStyleSheet("font-size: 18px; font-weight: 600; background: transparent;")
            tl.addWidget(cap)
            tl.addWidget(val)
            lay.addWidget(tile, i // 2, i % 2)
            self.rv_labels[key] = val
        self.box_rv.setVisible(False)
        page_function.addWidget(self.box_rv)

        self.box_slices = QGroupBox("Per slice  (volumes µL · EF % · mass mg)")
        lay = QVBoxLayout(self.box_slices)
        self.tbl_slices = QTableWidget(0, 5)
        # short headers so all five columns fit even a narrow side panel; the units are in the box title
        self.tbl_slices.setHorizontalHeaderLabels(["Slice", "EDV", "ESV", "EF", "Mass"])
        for c, tip in enumerate(["Slice number, base to apex",
                                 "End-diastolic volume (µL)",
                                 "End-systolic volume (µL)",
                                 "Ejection fraction (%)",
                                 "Myocardial mass at end-diastole (milligrams)"]):
            self.tbl_slices.horizontalHeaderItem(c).setToolTip(tip)
        self.tbl_slices.horizontalHeaderItem(0).setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.tbl_slices.verticalHeader().setVisible(False)
        self.tbl_slices.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_slices.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_slices.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_slices.cellClicked.connect(self._slice_row_clicked)
        hdr_s = self.tbl_slices.horizontalHeader()
        # slice numbers as narrow as they can be; the four number columns share the rest equally
        hdr_s.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr_s.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr_s.setMinimumSectionSize(36)
        self.tbl_slices.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hdr_s.setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.tbl_slices.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.tbl_slices.setStyleSheet("QTableWidget::item { padding: 0 6px; } QHeaderView::section { padding: 2px 6px; }")
        self.tbl_slices.setToolTip("Slices with ED and ES contoured. Click a row to view it. Grey = left out; "
                                   "orange = ES not set; bottom row = whole heart.")
        lay.addWidget(self.tbl_slices)
        self.box_slices.setVisible(False)
        page_function.addWidget(self.box_slices)
        self._table_timer = QTimer(self)
        self._table_timer.setSingleShot(True)
        self._table_timer.timeout.connect(self._refresh_slice_table)

        box = QGroupBox("Wall")
        lay = QVBoxLayout(box)
        self.fig = Figure(figsize=(4.4, 2.3), facecolor="none")
        self.canvas = SmoothCanvas(self.fig)
        self.canvas.setMinimumHeight(230)
        self.canvas.mpl_connect("resize_event", self._draw_bullseyes)
        self.canvas.setStyleSheet("background: transparent;")
        lay.addWidget(self.canvas)
        self.lbl_notes = QLabel("")
        self.lbl_notes.setWordWrap(True)
        lay.addWidget(self.lbl_notes)
        page_wall.addWidget(box)
        page_wall.addWidget(self._wall_controls())
        from .roi import RoiTool

        self.roi = RoiTool(self)
        from .autosave import AutoSave

        self.autosave = AutoSave(self)
        from .updates import UpdateController

        self.updates = UpdateController(self)  # update card, Help → Check for updates
        try:  # every layer added later (curves, wall lines, regions) is turned like the image
            self.viewer.layers.events.inserted.connect(lambda e: self._orient_layer(e.value))
        except Exception:
            pass
        page_regions.addWidget(self.roi.box)
        for page in (page_function, page_wall, page_regions):
            page.addStretch(1)
        self.btn_export = QPushButton("  Export all results")
        self.btn_export.setIcon(_icon("mdi6.file-export-outline"))
        self.btn_export.setToolTip("Save, then write this mouse's workbook, bullseyes, slice images and "
                                   "study-table row to its results folder.")
        row = QHBoxLayout()
        row.addWidget(self.btn_export)
        self.btn_export_img = QPushButton("  Export images")
        self.btn_export_img.setIcon(_icon("mdi6.image-multiple-outline"))
        self.btn_export_img.setToolTip("Save raw and contoured PNGs of every contoured slice at ED and ES, "
                                       "plus a montage.")
        row.addWidget(self.btn_export_img)
        foot = QWidget()
        foot.setLayout(row)
        row.setContentsMargins(9, 0, 9, 9)
        outer.addWidget(foot)

        # ---------------- toolbar ----------------
        self.toolbar = QWidget()
        tb = QHBoxLayout(self.toolbar)
        tb.setContentsMargins(6, 2, 6, 2)
        tb.setSpacing(2)
        # one button per job, grouped left to right in the order of the work: file, contour, edit, view, analyse.
        # The tools folded into dropdowns (UCL file, Snap, Fit, Grow, Shrink, Raw, Contrast, Zoom) stay as
        # buttons behind the scenes, so their keys, tooltips and tests are unchanged.
        groups = [[self.btn_dcm, self.btn_save],
                  [self.btn_auto, self.btn_endo, self.btn_epi, self.btn_rv],
                  [self.btn_smooth, self.btn_refine, self.btn_points, self.btn_brush, self.btn_undo, self.btn_redo,
                   self.btn_clear],
                  [self.btn_pan, self.btn_zoom_fit, self.btn_view],
                  [self.btn_wall, self.btn_roi]]
        self.toolbar_groups = groups
        for k, grp in enumerate(groups):
            if k:
                tb.addSpacing(5)
                tb.addWidget(_tool_separator())
                tb.addSpacing(5)
            for b in grp:
                tb.addWidget(b)
        tb.addStretch(1)
        for b in (self.btn_left, self.btn_right, self.btn_focus):
            tb.addWidget(b)
        tb.addSpacing(5)
        tb.addWidget(_tool_separator())
        tb.addSpacing(5)
        tb.addWidget(self.btn_help)
        tb.addWidget(self.btn_settings)
        for b in (self.btn_ucl, self.btn_snap, self.btn_fit, self.btn_grow, self.btn_shrink, self.btn_raw,
                  self.btn_contrast, self.btn_zoom_in, self.btn_zoom_out):
            b.setParent(self.toolbar)
            b.hide()
        self.toolbar.setStyleSheet(TOOLBAR_STYLE)

        # wiring
        self.btn_ucl.clicked.connect(self._open_ucl)
        self.btn_dcm.clicked.connect(self._open_dicom)
        self.btn_show_ed.clicked.connect(lambda: self._show_frame(self.session.t_for(self.current_slice(), "ED")))
        self.btn_show_es.clicked.connect(lambda: self._show_frame(self.session.t_for(self.current_slice(), "ES")))
        self.spin_ed.valueChanged.connect(self._frames_changed)
        self.spin_es.valueChanged.connect(self._frames_changed)
        self.btn_auto_frames.clicked.connect(self._auto_frames)
        self.btn_set_ed.clicked.connect(lambda: self.set_phase_frame("ED", self.current_frame()))
        self.btn_set_es.clicked.connect(lambda: self.set_phase_frame("ES", self.current_frame()))
        self.chk_per_slice.toggled.connect(self._per_slice_toggled)
        self.btn_play.clicked.connect(self._toggle_play)
        self.viewer.dims.events.current_step.connect(self._update_frame_label)
        self.viewer.dims.events.current_step.connect(self._compare_changed)
        self.btn_flip.clicked.connect(self._flip)
        self.btn_auto_order.clicked.connect(self._auto_order)
        self.list.model().rowsInserted.connect(lambda *a: QTimer.singleShot(0, self._order_from_list))
        self.list.itemChanged.connect(self._item_changed)
        self.list.currentItemChanged.connect(self._goto_slice)
        self.btn_suggest.clicked.connect(self._apply_suggestions)
        self.btn_compute.clicked.connect(self.compute)
        self.btn_auto.clicked.connect(self.autocontour)
        self.btn_smooth.clicked.connect(self._smooth_points)
        self.btn_points.toggled.connect(self._points_toggled)
        self.btn_snap.clicked.connect(self._snap_slice)
        self.btn_fit.clicked.connect(self._fit_slice)
        self.btn_grow.clicked.connect(lambda: self.grow_contour(GROW_STEP_PX))
        self.btn_shrink.clicked.connect(lambda: self.grow_contour(-GROW_STEP_PX))
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_clear.clicked.connect(self.clear_slice)
        self.chk_smooth.toggled.connect(self._smooth_toggled)
        self.btn_model_dir.clicked.connect(self._choose_model_dir)
        self._load_model_dirs()
        self._fill_models()
        self.rb_curves.toggled.connect(self._mode_changed)
        self.cmb_add.currentIndexChanged.connect(self._add_target_changed)
        self.btn_export.clicked.connect(lambda: self.export_all())
        self.btn_export_img.clicked.connect(lambda: self.export_images())
        self.btn_settings.clicked.connect(lambda: (self.settings.show(), self.settings.raise_()))
        self.btn_save.clicked.connect(lambda: self._save_clicked())
        from qtpy.QtWidgets import QMenu

        from .menus import ToolMenu

        save_menu = ToolMenu(self.btn_save)
        save_menu.add("Save", lambda: self._save_clicked(), keys=["Ctrl", "S"], icon="mdi6.content-save-outline")
        save_menu.add("Save as…", lambda: self._save_clicked(save_as=True), keys=["Ctrl", "Shift", "S"],
                      icon="mdi6.content-save-edit-outline")
        save_menu.addSeparator()
        save_menu.add("Export all results", lambda: self.export_all(), icon="mdi6.file-export-outline",
                      tooltip="Workbook, images and study-table row")
        save_menu.add("Open the results folder", self._open_saved_folder, icon="mdi6.folder-outline")

        self.btn_save.setMenu(save_menu)
        try:
            from qtpy.QtGui import QActionGroup
        except ImportError:  # Qt5
            from qtpy.QtWidgets import QActionGroup

        self.wall_mode = "thickness"
        wall_menu = ToolMenu(self.btn_wall)
        self.act_wall_show = wall_menu.add("Show wall lines", lambda on: self.btn_wall.setChecked(on), keys="W",
                                           icon="mdi6.ruler", checkable=True)
        self.btn_wall.toggled.connect(lambda on: self.act_wall_show.isChecked() != on
                                      and self.act_wall_show.setChecked(on))
        wall_menu.addSeparator()
        grp = QActionGroup(wall_menu)
        self.act_wall = {}
        for mode, text in (("thickness", "Thickness (mm)"), ("thickening", "Thickening ED → ES (%)")):
            self.act_wall[mode] = wall_menu.add(text, lambda _=False, m=mode: self.set_wall_mode(m),
                                                checkable=True, checked=mode == self.wall_mode, group=grp)
        wall_menu.addSeparator()
        wall_menu.add("Set the septum on this slice", self.start_set_septum, icon="mdi6.cursor-default-click-outline",
                      tooltip="Then click the septum on the image")
        wall_menu.add("Reset the septum on this slice", lambda: self.reset_septum(all_slices=False),
                      icon="mdi6.restore")
        wall_menu.add("Reset the septum on all slices", lambda: self.reset_septum(all_slices=True),
                      icon="mdi6.restore")
        self.btn_wall.setMenu(wall_menu)
        self.btn_wall.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        roi_menu = ToolMenu(self.btn_roi)
        roi_grp = QActionGroup(roi_menu)
        self.act_roi_shape = {
            "circle": roi_menu.add("Circle", lambda _=False: (self.roi.set_kind("circle"), self.btn_roi.setChecked(True)),
                                   icon="mdi6.circle-outline", checkable=True, checked=True, group=roi_grp,
                                   tooltip="Drag from the centre outwards"),
            "freehand": roi_menu.add("Freehand", lambda _=False: (self.roi.set_kind("freehand"),
                                                                 self.btn_roi.setChecked(True)),
                                     icon="mdi6.draw", checkable=True, group=roi_grp,
                                     tooltip="Hold and drag around the region"),
            "line": roi_menu.add("Ruler (distance)", lambda _=False: (self.roi.set_kind("line"),
                                                                    self.btn_roi.setChecked(True)),
                                 icon="mdi6.ruler", checkable=True, group=roi_grp,
                                 tooltip="Drag between two points to measure the distance in mm"),
        }
        roi_menu.addSeparator()
        roi_menu.add("Remove the selected region", lambda: self.roi.remove_selected(), keys="Delete",
                     icon="mdi6.close")
        roi_menu.add("Clear all regions", lambda: self.roi.clear_all(), icon="mdi6.trash-can-outline")
        roi_menu.add("Export measurements (CSV)", lambda: self.roi.export_csv(), icon="mdi6.file-delimited-outline")
        self.btn_roi.setMenu(roi_menu)
        self.btn_roi.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_roi.toggled.connect(self._roi_toggled)
        self.btn_wall.toggled.connect(lambda on: on and self.results_tabs.show("wall"))
        self.btn_save.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._build_tool_menus(QMenu)
        self._set_hints()
        self.lbl_contours_dir.setText(str(self.contours_dir()))
        self.lbl_data_dir.setText(str(_settings().value("folders/data") or "not set"))
        self.btn_endo.clicked.connect(lambda: self.cmb_add.setCurrentIndex(0))
        self.btn_epi.clicked.connect(lambda: self.cmb_add.setCurrentIndex(1))
        self.btn_rv.clicked.connect(lambda: self.cmb_add.setCurrentIndex(2))
        self.cmb_add.currentIndexChanged.connect(
            lambda i: (self.btn_endo.setChecked(i == 0), self.btn_epi.setChecked(i == 1),
                       self.btn_rv.setChecked(i == 2)))
        self.btn_brush.toggled.connect(lambda on: (self.rb_pixels if on else self.rb_curves).setChecked(True))
        self.rb_pixels.toggled.connect(lambda on: self.btn_brush.setChecked(on))
        self.btn_zoom_in.clicked.connect(lambda: self.zoom(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self.zoom(0.8))
        self.btn_zoom_fit.clicked.connect(self.fit_heart)
        self.btn_raw.toggled.connect(self.show_raw)
        self.btn_wall.toggled.connect(lambda on: self.draw_wall())
        self.viewer.dims.events.current_step.connect(lambda e=None: self._wall_on() and self.draw_wall())
        self.btn_pan.toggled.connect(self._pan_toggled)
        self._build_contrast_popup()
        self.btn_contrast.clicked.connect(self._show_contrast_popup)
        for key, fn in (("f", lambda v: self.fit_heart()), ("h", lambda v: self.btn_raw.toggle()), ("w", lambda v: self.btn_wall.toggle()), ("Space", lambda v: self.btn_pan.toggle()), ("Control-z", lambda v: self.undo()), ("Control-y", lambda v: self.redo()),
                        ("Control-e", lambda v: self.show_compare()),
                        ("Control-Shift-z", lambda v: self.redo()), ("Escape", lambda v: self._escape()),
                        ("]", lambda v: self.grow_contour(GROW_STEP_PX)), ("[", lambda v: self.grow_contour(-GROW_STEP_PX)),
                        ("g", lambda v: self.set_smooth_image(not self.smooth_image)),
                        ("q", lambda v: self.cycle_structure()), ("k", lambda v: self._toggle_play()),
                        ("r", lambda v: self.btn_roi.toggle()),
                        ("o", lambda v: self.show_orientation()),
                        ("Control-s", lambda v: self._save_clicked()),
                        ("Control-Shift-s", lambda v: self._save_clicked(save_as=True)),
                        ("Up", lambda v: self.step_slice(-1)), ("Down", lambda v: self.step_slice(1)),
                        ("Left", lambda v: self.step_frame(-1)), ("Right", lambda v: self.step_frame(1))):
            try:
                self.viewer.bind_key(key, fn, overwrite=True)
            except Exception:  # keys are a convenience; the buttons always work
                pass
        self._image_shortcuts()
        self._set_enabled(False)
        self.series_panel = SeriesPanel(self)
        self.series_panel.add_slices(self.list, [self.btn_flip, self.btn_auto_order, self.btn_suggest],
                                     self.lbl_order)
        self.series_panel.fill_mice()
        # Clicking a list or table must not keep the keyboard: letter keys (S, P, Q, H, W…) stay shortcuts instead
        # of jumping to a row that starts with that letter. Up/Down still step through slices.
        for view in (self.list, self.series_panel.mice, self.series_panel.list, self.tbl_slices):
            view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.timebar = TimeBar(self)
        self.docks = []
        try:
            # the toolbar scrolls sideways when the window is narrower than it, instead of pushing buttons off-screen
            bar = QScrollArea()
            for b in (self.btn_dcm, self.btn_save, self.btn_wall, self.btn_roi):  # room for the menu arrow
                b.setMinimumWidth(58)
            self.toolbar.ensurePolished()
            self.toolbar.setMinimumWidth(self.toolbar.sizeHint().width())  # never squeezed into cut-off labels
            bar.setWidget(self.toolbar)
            bar.setWidgetResizable(True)
            bar.setFrameShape(QFrame.Shape.NoFrame)
            bar.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            bar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            bar.setFixedHeight(self.toolbar.sizeHint().height() + bar.horizontalScrollBar().sizeHint().height() + 2)
            bar.setMinimumWidth(200)
            self.toolbar_scroll = bar
            self.docks = [self.viewer.window.add_dock_widget(bar, area="top", name="Tools"),
                          self.viewer.window.add_dock_widget(self.timebar, area="bottom", name="Time")]
            left = QScrollArea()
            left.setWidget(self.series_panel)
            left.setWidgetResizable(True)
            self.series_panel.avail_height = lambda: left.viewport().height()
            from qtpy.QtCore import QEvent, QObject

            self._left_fit = QTimer(self)
            self._left_fit.setSingleShot(True)
            self._left_fit.timeout.connect(self.series_panel._fit_lists)
            fit = self._left_fit

            class _Resized(QObject):  # the panel got taller or shorter: share the room out again
                def eventFilter(self, obj, ev):  # noqa: N802 (Qt name)
                    if ev.type() == QEvent.Type.Resize and ev.oldSize().height() != ev.size().height():
                        fit.start(0)
                    return False

            self._left_filter = _Resized(left.viewport())
            left.viewport().installEventFilter(self._left_filter)
            left.setFrameShape(QFrame.Shape.NoFrame)
            left.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            left.setMinimumWidth(290)
            self.docks.append(self.viewer.window.add_dock_widget(left, area="left", name="Mouse"))
            for d in self.docks:
                setup_dock(d, header=d.objectName() == "Mouse")  # toolbar and time bar: no header
            QTimer.singleShot(0, self._wire_panel_buttons)  # the Results dock exists once the panel is docked
            self._scale_with_canvas()
            # Segment-style: the series column owns the left side; napari's layer panels come back with
            # the pixel brush (or from the Window menu)
            self._napari_docks(False)
            # no slice/time scrollbars under the image: slices are picked in the slice list (or Up/Down),
            # frames on the time bar (or Left/Right)
            qv = getattr(self.viewer.window, "_qt_viewer", None)
            if qv is not None and hasattr(qv, "dims"):
                qv.dims.setVisible(False)
        except Exception:  # headless viewers without a window
            pass

    # ---------- opening ----------

    @staticmethod
    def _start_folder(key: str) -> str:
        """Where a file dialog opens: the folder used last time for this, else the data folder, else home."""
        st = _settings()
        for d in (st.value(key, ""), st.value("folders/data", "")):
            if d and Path(str(d)).is_dir():
                return str(d)
        return str(Path.home())

    def _open_ucl(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open UCL mouse file(s)", self._start_folder("folders/last_ucl"),
                                                "UCL HDF5 (*.h5)")
        if paths:
            _settings().setValue("folders/last_ucl", str(Path(paths[0]).parent))
            self.add_ucl(paths, raw=self.chk_raw.isChecked())

    def add_ucl(self, paths, raw: bool = False) -> None:
        from .study import ucl_series

        first = len(self.study)
        self.study.extend(ucl_series(p, raw=raw) for p in paths)
        self.open_series(first)

    def _open_dicom(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose one mouse's scan folder (a ParaVision study folder, or a folder of DICOM files)",
            self._start_folder("folders/last_scan"))
        if folder:
            # next time, open in the folder that holds it, so the mouse's neighbours are one click away
            _settings().setValue("folders/last_scan", str(Path(folder).parent))
            self.open_path(folder)

    def open_study(self, folder) -> None:
        """List every series in a scan folder on the left and open the most likely short-axis cine. The folder
        can be a Bruker ParaVision study (read straight from the scanner's files) or a folder of DICOM files."""
        from .. import bruker
        from .study import scan_bruker_study, scan_dicom_folder

        folder = Path(folder)
        if folder.name.isdigit() and bruker.is_study(folder.parent):
            folder = folder.parent  # picked one scan of a ParaVision study: open the whole study
        if bruker.is_study(folder):
            found = scan_bruker_study(folder)
            if not found:
                self.series_panel.lbl.setText("This ParaVision study has no image scans TraceCMR can open.")
                return
            self._show_study(folder, found)
            return
        studies = [d for d in folder.iterdir() if d.is_dir() and bruker.is_study(d)] if folder.is_dir() else []
        if studies:  # a folder of ParaVision studies (one per mouse session): list them as mice
            self.series_panel.fill_mice(str(folder))
            self.series_panel.lbl.setText(f"This folder holds {len(studies)} ParaVision scan sessions. "
                                          f"Pick one in the Mice list.")
            return
        found = scan_dicom_folder(folder)
        if len(found) == 1 and found[0].n_slices == 1:  # picked one slice's folder: use the mouse folder above it
            parent = Path(folder).parent
            if sum(1 for _ in zip(range(5001), parent.rglob("*"))) <= 5000:
                above = scan_dicom_folder(parent)
                if len(above) > 1:
                    found, folder = above, parent
        if not found:
            self.series_panel.lbl.setText("No DICOM images or ParaVision scans found. Choose one mouse's folder.")
            return
        self._show_study(folder, found)

    def _show_study(self, folder, found) -> None:
        from .study import auto_stacks

        stacks = auto_stacks(found)  # one cine per slice -> one short-axis stack, parts listed under it
        for st in stacks:  # name stacks after the mouse folder, so saved contours are easy to find
            st.name = f"{Path(folder).name} SAX ({st.n_slices} slices)"
        first = len(self.study)
        self.study.extend(stacks + found)
        new = self.study[first:]
        sax = [i for i, s in enumerate(new) if s.kind == "Short axis" and s.n_slices > 1]
        self.open_series(first + (sax[0] if sax else 0))

    def open_series(self, i: int) -> None:
        """Switch to series i, keeping edits made to the one being left. A slice of a stack opens the
        stack at that slice."""
        ser = self.study[i]
        if ser.stack is not None:
            j = self.study.index(ser.stack)
            if j != self.current_series:
                self.open_series(j)
            step = list(self.viewer.dims.current_step)
            step[1] = ser.slice_index
            self.viewer.dims.current_step = tuple(step)
            return
        if self.session is not None and "contours" in self.viewer.layers:
            self._sync_labels()  # edits are stored in that series' session
        first_time = ser.session is None
        try:
            session = ser.load()
        except Exception as e:
            self.series_panel.lbl.setText(f"⚠ Could not open {ser.name}: {e}")
            return
        msg = self._load_saved(session) if first_time else ""
        self.current_series = i
        self.load_session(session)
        if msg:
            self.lbl_auto.setText(msg)
        self.series_panel.refresh(self.study, i)

    # ---------- folders, saving ----------

    def contours_dir(self) -> Path:
        v = _settings().value("folders/contours")
        if v:
            if Path(v).exists() or not Path(str(v).replace("VentriMap contours", "TraceCMR results")).exists():
                return Path(v)
            return Path(str(v).replace("VentriMap contours", "TraceCMR results"))  # folder renamed
        new, old = Path.home() / "Documents" / "TraceCMR results", Path.home() / "Documents" / "VentriMap contours"
        return old if old.exists() and not new.exists() else new

    @staticmethod
    def _overlap(a, b) -> bool:
        """True if folders a and b are the same or one is inside the other."""
        if not a or not b:
            return False
        a, b = (Path(os.path.normcase(os.path.abspath(x))) for x in (a, b))
        return a == b or a in b.parents or b in a.parents

    def _choose_contours_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder (contours, tables, images)",
                                             str(self.contours_dir()))
        if d:
            self.set_results_dir(d)

    def set_results_dir(self, d) -> bool:
        data = _settings().value("folders/data")
        if self._overlap(d, data):
            self.lbl_contours_dir.setText(f"⚠ Not changed: results must be outside the raw scans folder ({data}). "
                                          f"Still {self.contours_dir()}")
            return False
        _settings().setValue("folders/contours", str(d))
        self.lbl_contours_dir.setText(str(d))
        self.series_panel.fill_mice()
        return True

    def _choose_data_dir(self):
        self.series_panel._choose_root()
        self.lbl_data_dir.setText(str(_settings().value("folders/data") or "not set"))

    def _scan_tag(self, session=None) -> str:
        import hashlib

        session = session or self.session
        src = os.path.normcase(os.path.abspath(session.source)) if session.source else ""
        return hashlib.sha1(src.encode()).hexdigest()[:8]  # same file however the path was spelled

    def _remembered_path(self, session=None) -> Path | None:
        """Where you last saved this scan (with Save or Save as), if that file still exists."""
        session = session or self.session
        if session is None or not session.source:
            return None
        v = _settings().value(f"saved/{self._scan_tag(session)}")
        return Path(v) if v and Path(v).exists() else None

    def _save_target(self, session=None) -> Path:
        return self._remembered_path(session) or self._contours_path(session or self.session)

    def results_dir(self, session=None) -> Path:
        """This mouse's results folder: <results>/<mouse>__<id>/."""
        import hashlib

        session = session or self.session
        name = self._safe(session.subject_id) or "scan"
        src = os.path.normcase(os.path.abspath(session.source)) if session.source else ""
        tag = hashlib.sha1(src.encode()).hexdigest()[:8]  # same file however the path was spelled
        return self.contours_dir() / f"{name}__{tag}"

    @staticmethod
    def _safe(name: str) -> str:
        import re

        return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")

    def _contours_path(self, session) -> Path:
        d = self.results_dir(session)
        return d / f"{d.name.split('__')[0]}.tracecmr.npz"

    def _legacy_contours_path(self, session) -> Path:  # before results were kept per mouse
        return self.contours_dir() / f"{self.results_dir(session).name}.tracecmr.npz"

    def _find_saved(self, session) -> Path | None:
        """The saved contours for this scan: where you last saved it, the default place, then older
        names and places (VentriMap)."""
        mine = self._remembered_path(session)
        if mine is not None:
            return mine
        new, flat = self._contours_path(session), self._legacy_contours_path(session)
        for p in (new, new.with_name(new.name.replace(*RESULTS_EXT)),
                  flat, flat.with_name(flat.name.replace(*RESULTS_EXT))):
            if p.exists():
                return p
        return None

    @staticmethod
    def _stem(path: Path) -> str:
        n = path.name
        for ext in RESULTS_EXT:
            if n.endswith(ext):
                return n[: -len(ext)]
        return path.stem

    def _images_dir(self, path: Path) -> Path:
        """Images go in 'images' inside the mouse's own results folder, else '<name>_images' next to the file."""
        return path.parent / ("images" if path.parent == self.results_dir() else f"{self._stem(path)}_images")

    def save_contours(self, quiet: bool = False, path=None) -> Path | None:
        """Save the open mouse's contours, results table and images: to `path`, else where you last saved
        this scan, else its default results folder. Returns the file, or None if saving failed."""
        if self.session is None:
            return None
        self._sync_labels()
        if self.editor is not None and self.editor.curves:  # keep the exact points for the next time it's opened
            s_ = self.session
            s_.saved_curves = {k: np.array(v, dtype=float) for k, v in self.editor.curves.items()}
            s_.saved_hidden = {k for k in self.editor.hidden if k in s_.saved_curves}
            s_.saved_curve_hash = {k[:2]: s_.slice_hash(*k[:2]) for k in s_.saved_curves}
        self._last_save = None
        try:
            path = Path(path) if path else self._save_target()
            if not path.name.endswith(RESULTS_EXT):
                path = path.with_name(path.name + ".tracecmr.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            path = self.session.save_contours(path)
            csv_path = None
            if self.last:
                csv_path = path.with_name(self._stem(path) + "_results.csv")
                self._write_csv(csv_path)
            imgs = self.export_images(quiet=True, out=self._images_dir(path)) or []
        except Exception as e:
            self.lbl_auto.setText(f"⚠ Could not save: {e}")
            return None
        if self.session.source:
            _settings().setValue(f"saved/{self._scan_tag()}", str(path))
        self._last_save = {"file": path, "csv": csv_path, "images": len(imgs), "images_dir": self._images_dir(path)}
        self.autosave.mark_saved()
        if not quiet:
            self.lbl_auto.setText(f"Saved {path.name} in {path.parent}.")
        self.series_panel.fill_mice()
        return path

    def _save_clicked(self, save_as: bool = False) -> Path | None:
        """Save button / Ctrl+S: the first time (or Save as) a save window shows where it goes and lets you
        rename it; after that it saves to the same file. A confirmation says exactly what was written."""
        if self.session is None:
            return None
        target = self._save_target()
        if save_as or self._remembered_path() is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            chosen, _ = QFileDialog.getSaveFileName(self, "Save contours", str(target),
                                                    "TraceCMR contours (*.tracecmr.npz)")
            if not chosen:
                self.lbl_auto.setText("Not saved.")
                return None
            target = Path(chosen)
        path = self.save_contours(path=target)
        if path is not None:
            self._confirm_saved()
        return path

    def _confirm_saved(self) -> None:
        info = getattr(self, "_last_save", None)
        if not info or _settings().value("ui/hide_save_popup") == "1" or getattr(self, "_testing", False):
            return
        from qtpy.QtWidgets import QMessageBox

        parts = ["contours"] + (["results table"] if info["csv"] else []) + \
                ([f"{info['images']} images"] if info["images"] else [])
        box = QMessageBox(self)
        box.setWindowTitle("Saved")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"Saved {', '.join(parts)}.")
        box.setInformativeText(f"File: {info['file'].name}\nFolder: {info['file'].parent}\n\n"
                               "Ctrl+S now saves here (Save as… to change).")
        open_btn = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        chk = QCheckBox("Don't show this again (the status line shows where it saved)")
        box.setCheckBox(chk)
        box.exec()
        if chk.isChecked():
            _settings().setValue("ui/hide_save_popup", "1")
        if box.clickedButton() is open_btn:
            self._open_saved_folder()

    def _open_saved_folder(self) -> None:
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        if self.session is None:
            return
        folder = self._save_target().parent
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _load_saved(self, session) -> str:
        path = self._find_saved(session)
        if path is None:
            return ""
        try:
            session.load_contours(path)
            return f"Loaded your saved contours ({path.name})."
        except Exception as e:
            return f"⚠ Saved contours not loaded: {e}"

    def step_slice(self, d: int) -> None:
        if self.session is not None:
            nz = self.session.image.shape[2]
            step = list(self.viewer.dims.current_step)
            step[1] = int(np.clip(step[1] + d, 0, nz - 1))
            self.viewer.dims.current_step = tuple(step)

    def open_path(self, path) -> None:
        """Open one mouse from the data folder: a ParaVision study, a folder of DICOM series, or a UCL .h5 file. The previous
        mouse is saved first and the series list starts fresh."""
        path = Path(path)
        st = _settings()
        if not st.value("folders/data") and not self._overlap(path.parent, self.contours_dir()):
            # first mouse opened: its folder becomes the raw scans folder
            st.setValue("folders/data", str(path.parent))
            self.series_panel.fill_mice(str(path.parent))
            self.lbl_data_dir.setText(str(path.parent))
        if self.session is not None and self.session.label.any():
            self.save_contours(quiet=True)
        self.study.clear()
        self.current_series = None
        if path.suffix.lower() == ".h5":
            self.add_ucl([str(path)], raw=self.chk_raw.isChecked())
        else:
            self.open_study(path)

    def combine_selected(self) -> None:
        from .study import combine

        rows = self.series_panel.selected_rows()
        try:
            stack = combine([self.study[r] for r in rows])
        except ValueError as e:
            self.series_panel.lbl.setText(f"⚠ {e}")
            return
        self.study.append(stack)
        self.open_series(len(self.study) - 1)

    def _mirror_toggled(self, on: bool) -> None:
        _settings().setValue("wall/mirror", bool(on))
        if self.session is not None:
            self.session.mirror_segments = bool(on)
            self.compute(quiet=True)

    def load_session(self, session: Session, message: str = "") -> None:
        """Show a session in the viewer and fill the panel."""
        self.session = s = session
        s.mirror_segments = self.chk_mirror.isChecked()
        # Overlays first: removing the image while the outline layers remain shifts the frame/slice position,
        # the editor redraws for the new position, which shifts it again (an endless loop on some slices)
        if self.editor is not None:
            self.editor.detach()  # the old editor stops reacting to clicks and slice changes
        for name in ("rois", "control points", "curves", "wall thickness", "image", "contours"):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        dx, dy, dz = s.spacing
        # napari axes: (time, slice, y, x); our arrays are (x, y, slice, time)
        self.viewer.add_image(s.image.transpose(3, 2, 1, 0), name="image", scale=(1, dz, dy, dx),
                              colormap="gray", interpolation2d="cubic" if self.smooth_image else "nearest")
        self.viewer.layers["image"].events.contrast_limits.connect(self._compare_changed)
        lab = self.viewer.add_labels(s.label.transpose(3, 2, 1, 0), name="contours", scale=(1, dz, dy, dx),
                                     opacity=0.9)
        lab.contour = 1  # outlines, like Segment, rather than filled regions
        try:
            from napari.utils.colormaps import DirectLabelColormap

            lab.colormap = DirectLabelColormap(color_dict={
                None: (0, 0, 0, 0), 0: (0, 0, 0, 0),
                L.LV_BLOODPOOL: (*(c / 255 for c in BLOOD_RGB), 1.0),
                L.LV_MYOCARDIUM: (*(c / 255 for c in MYO_RGB), 1.0),
                L.RV_BLOODPOOL: (*(c / 255 for c in RV_RGB), 1.0),
            })
        except Exception:  # colours are cosmetic; older napari keeps its defaults
            pass
        self.viewer.dims.axis_labels = ("time", "slice", "y", "x")
        nt = s.image.shape[3]
        for spin in (self.spin_ed, self.spin_es):
            spin.blockSignals(True)
            spin.setRange(0, nt - 1)
            spin.blockSignals(False)
        for spin, v in ((self.spin_ed, s.ed_frame), (self.spin_es, s.es_frame)):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        self.chk_per_slice.blockSignals(True)
        self.chk_per_slice.setChecked(s.separate_slices)
        self.chk_per_slice.blockSignals(False)
        self.lbl_scan.setText(
            f"<b style='font-size:14px'>{s.subject_id}</b><br><span style='color:#9AA1A8'>{s.image.shape[2]} slices · "
            f"{nt} frames · {dx:.3g} × {dy:.3g} mm pixels · slices {dz:.3g} mm apart</span>"
        )
        self.lbl_conv.setText(CONVENTION_TEXT.get(s.papillary, CONVENTION_TEXT[None]))
        self._set_enabled(True)
        self.editor = CurveEditor(self.viewer, scale=(1, dz, dy, dx))
        self._shown_label = None  # labels the editor's curves were last built from (see _mode_changed)
        self._label_undo = None
        self.editor.on_change = self.lbl_auto.setText
        self.editor.load_frame = self._curves_for_frame
        self.editor.on_edit = lambda: self._table_timer.start(250)
        self.editor.on_window = self.adjust_contrast
        self.editor.pan_mode = self.btn_pan.isChecked()
        self.editor.show_dots = self.btn_points.isChecked()
        if s.view_rotation is None or s.view_flip is None:  # not chosen for this mouse: the last one used
            st = _settings()
            if s.view_rotation is None:
                s.view_rotation = float(st.value("view/rotation", 0.0) or 0.0)
            if s.view_flip is None:
                s.view_flip = str(st.value("view/flip", False)).lower() in ("true", "1")
        self._apply_orientation(recentre=False)
        self._rebuild_list()
        self._show_frame(s.ed_frame)
        self.viewer.reset_view()
        self._centre_on_heart()
        self._mode_changed()
        self._draw_volume_curve()
        self._reset_contrast()
        self._table_timer.start(0)  # results straight away when the mouse already has contours
        self.roi.session_loaded()  # its saved regions of interest
        self.autosave.baseline()
        QTimer.singleShot(0, self._prewarm_wall)  # not while the first click on Wall is waiting
        QTimer.singleShot(250, self._offer_recovery)  # once the window has settled on the new mouse
        if message:
            self.lbl_order.setText(message)
        if not getattr(self, "_testing", False):
            self._preload_model()

    # ---------- orientation: turn or mirror the picture (display only; see orient.py) ----------

    def orientation(self) -> tuple[float, bool]:
        s = self.session
        if s is None:
            return 0.0, False
        return float(s.view_rotation or 0.0), bool(s.view_flip)

    def _orient_affine(self):
        from .orient import napari_affine

        s = self.session
        dx, dy, _ = s.spacing
        nx, ny = s.image.shape[0], s.image.shape[1]
        angle, flip = self.orientation()
        return napari_affine(angle, flip, ((ny - 1) / 2 * dy, (nx - 1) / 2 * dx))

    def _orient_layer(self, layer) -> None:
        if self.session is None or getattr(layer, "ndim", 0) != 4:
            return
        try:
            layer.affine = self._orient_affine()
        except Exception:  # a layer that refuses a transform is shown unturned rather than breaking the panel
            pass

    def _to_world_yx(self, y_mm: float, x_mm: float) -> tuple[float, float]:
        """Where an unturned (y, x) position in mm appears in the turned view."""
        if self.session is None:
            return y_mm, x_mm
        w = self._orient_affine() @ np.array([0.0, 0.0, y_mm, x_mm, 1.0])
        return float(w[2]), float(w[3])

    def _apply_orientation(self, recentre: bool = True) -> None:
        if self.session is None:
            return
        for layer in list(self.viewer.layers):
            self._orient_layer(layer)
        if recentre:
            self._follow_heart()
            t = getattr(self, "_thumb_timer", None)
            if t is None:  # thumbnails catch up once the slider rests
                t = self._thumb_timer = QTimer(self)
                t.setSingleShot(True)
                t.timeout.connect(self._rebuild_list_safely)
            t.start(200)
            c = getattr(self, "_compare", None)
            if c is not None and c.is_open():
                c.schedule()
        self._sync_orientation_controls()

    def _rebuild_list_safely(self) -> None:
        try:
            if self.session is not None:
                self._rebuild_list()
        except RuntimeError:  # the window closed before the timer fired
            pass

    def _follow_heart(self) -> None:
        """Keep the heart in the middle of the view (same zoom) while the picture turns."""
        s = self.session
        lab = s.label[:, :, :, s.ed_frame]
        xs, ys, _ = np.nonzero(lab)
        dx, dy, _ = s.spacing
        if len(xs):
            y_mm, x_mm = float(ys.mean() * dy), float(xs.mean() * dx)
        else:
            y_mm, x_mm = (s.image.shape[1] - 1) / 2 * dy, (s.image.shape[0] - 1) / 2 * dx
        cam = self._camera
        centre = list(cam.center)
        centre[-2], centre[-1] = self._to_world_yx(y_mm, x_mm)
        cam.center = tuple(centre)

    def set_orientation(self, angle: float | None = None, flip: bool | None = None) -> None:
        """Turn (degrees, clockwise) and/or mirror the picture. Remembered for this mouse (saved with its
        contours) and as the starting orientation for the next mouse opened."""
        from .orient import normalise

        s = self.session
        if s is None:
            return
        if angle is not None:
            s.view_rotation = normalise(round(float(angle), 1))
        if flip is not None:
            s.view_flip = bool(flip)
        st = _settings()
        st.setValue("view/rotation", float(s.view_rotation or 0.0))
        st.setValue("view/flip", bool(s.view_flip))
        self._apply_orientation()

    def rotate_by(self, degrees: float) -> None:
        a, _ = self.orientation()
        self.set_orientation(angle=a + degrees)

    def _sync_orientation_controls(self) -> None:
        sld = getattr(self, "sld_rotation", None)
        if sld is None:
            return
        a, f = self.orientation()
        sld.blockSignals(True)
        sld.setValue(int(round(a)))
        sld.blockSignals(False)
        self.lbl_rotation.setText(f"{a:+.0f}°" if a else "0°")
        self.btn_mirror.blockSignals(True)
        self.btn_mirror.setChecked(f)
        self.btn_mirror.blockSignals(False)

    def _build_orientation_popup(self) -> None:
        """Rotate and mirror: a floating card like Brightness and contrast."""
        from qtpy.QtWidgets import QSlider

        from .hints import set_hint
        from .popover import Popover

        pop = Popover(self.viewer.window._qt_window if self._has_window() else self, "Rotate and mirror")
        self.orientation_popup = pop
        row = QHBoxLayout()
        row.setSpacing(12)
        self.sld_rotation = QSlider(Qt.Orientation.Horizontal)
        self.sld_rotation.setRange(-180, 180)
        self.sld_rotation.setMinimumWidth(220)
        self.sld_rotation.setPageStep(15)
        self.sld_rotation.valueChanged.connect(lambda v: self.set_orientation(angle=v))
        self.lbl_rotation = QLabel("0°")
        self.lbl_rotation.setObjectName("value")
        self.lbl_rotation.setFixedWidth(40)
        self.lbl_rotation.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(QLabel("Rotate"))
        row.addWidget(self.sld_rotation)
        row.addWidget(self.lbl_rotation)
        pop.body.addLayout(row)
        btns = QHBoxLayout()
        b_left, b_right = QPushButton("↺ 90°"), QPushButton("↻ 90°")
        self.btn_mirror = QPushButton("Mirror")
        self.btn_mirror.setCheckable(True)
        b_reset = QPushButton("Reset")
        set_hint(b_left, "Rotate left", "Turn the picture 90° anticlockwise.")
        set_hint(b_right, "Rotate right", "Turn the picture 90° clockwise.")
        set_hint(self.btn_mirror, "Mirror", "Flip the picture left to right.")
        set_hint(b_reset, "Reset", "Back to the scanner's orientation.")
        b_left.clicked.connect(lambda: self.rotate_by(-90))
        b_right.clicked.connect(lambda: self.rotate_by(90))
        self.btn_mirror.toggled.connect(lambda on: self.set_orientation(flip=on))
        b_reset.clicked.connect(lambda: self.set_orientation(angle=0.0, flip=False))
        for b in (b_left, b_right, self.btn_mirror, b_reset):
            btns.addWidget(b)
        btns.addStretch(1)
        pop.body.addLayout(btns)
        note = QLabel("Only the picture turns: contours and results are unchanged.\nSaved with this mouse; "
                      "the next mouse starts the same way.")
        note.setObjectName("note")
        pop.body.addWidget(note)
        self._sync_orientation_controls()

    def show_orientation(self) -> None:
        if self.session is None:
            return
        if getattr(self, "orientation_popup", None) is None:
            self._build_orientation_popup()
        pop = self.orientation_popup
        anchor = None
        try:
            anchor = self.viewer.window._qt_viewer.canvas.native
        except Exception:
            pass
        self._sync_orientation_controls()
        if anchor is not None:
            pop.show_near(anchor)
        else:
            pop.show()
        pop.raise_()

    @property
    def _camera(self):
        scene = getattr(self.viewer, "scene", None)
        return scene.camera if scene is not None and hasattr(scene, "camera") else self.viewer.camera

    def _compare_open(self):
        c = getattr(self, "_compare", None)
        return c if c is not None and c.is_open() else None

    def zoom(self, factor: float) -> None:
        """Zoom the image view in (>1) or out (<1) about its centre (the ED | ES pair while that view is up)."""
        c = self._compare_open()
        if c is not None:
            c.view.zoom_by(factor)
            return
        self._camera.zoom = self._camera.zoom * factor

    def fit_heart(self) -> None:
        """Reset the view, then centre and zoom on the heart if it has contours."""
        c = self._compare_open()
        if c is not None:
            c.view.reset_view()
            return
        self.viewer.reset_view()
        if self.session is not None:
            self._centre_on_heart(set_slice=False)

    def _centre_on_heart(self, set_slice: bool = True):
        """Zoom the view onto the heart (centroid of the ED contours), as in Segment."""
        s = self.session
        lab = s.label[:, :, :, s.ed_frame]
        xs, ys, zs = np.nonzero(lab)
        if not len(xs):
            return
        dx, dy, _ = s.spacing
        if set_slice:
            step = list(self.viewer.dims.current_step)
            step[1] = int(np.median(zs))
            self.viewer.dims.current_step = tuple(step)
        cam = self._camera
        centre = list(cam.center)
        wy, wx = self._to_world_yx(float(ys.mean() * dy), float(xs.mean() * dx))
        centre[-2], centre[-1] = wy, wx
        cam.center = tuple(centre)
        span = max(np.ptp(ys) * dy, np.ptp(xs) * dx, 1.0)
        full = max(lab.shape[0] * dx, lab.shape[1] * dy)
        cam.zoom = cam.zoom * 0.55 * full / span

    # ---------- frames ----------

    def _sync_labels(self):
        """Pull the user's edits (curves or pixel brush) back into the session."""
        if self.session is None or "contours" not in self.viewer.layers:
            return
        layer = self.viewer.layers["contours"]
        self.session.label = np.asarray(layer.data).transpose(3, 2, 1, 0)
        if self.editor is not None and self.editor.edited:  # pending curve edits
            edited = {k[:2] for k in self.editor.edited}
            self.session.label = self.editor.apply_to(self.session.label)
            self.editor.edited.clear()
            self._clear_checked_flags(edited)
            shown = getattr(self, "_shown_label", None)
            if shown is not None and shown.shape == self.session.label.shape:
                for t, z in edited:  # the curves are the truth on these slices; keep them on the next show
                    shown[:, :, z, t] = self.session.label[:, :, z, t]
            layer.data = self.session.label.transpose(3, 2, 1, 0).copy()

    def _clear_checked_flags(self, edited) -> None:
        """Once you've edited a slice, its 'double-check' note (base, apex, tiny cavity) goes away."""
        s, changed = self.session, False
        for t, z in edited:
            why = s.flags.get((t, z), "")
            if why.startswith(("base slice", "apex slice", "tiny cavity")):
                s.flags.pop((t, z))
                for ph in ("ED", "ES"):
                    if s.t_for(z, ph) == t and s.reasons.get(ph, {}).get(z, "").startswith("check: " + why[:10]):
                        s.reasons[ph][z] = ""
                changed = True
        if changed and hasattr(self, "list"):
            self._relabel_items()

    def _curves_for_frame(self, t: int) -> dict:
        """Contours of a frame that isn't ED or ES on any slice, built the first time it is viewed (Auto LV on
        every frame, or contours drawn there): what the user drew where nothing has changed it, else from the
        labels."""
        s = self.session
        if s is None or not 0 <= t < s.label.shape[3]:
            return {}
        curves = curves_from_labels(s.label, [t])
        saved = {k: v for k, v in s.curves_still_valid().items() if k[0] == t}
        if saved:
            slices = {k[:2] for k in saved}
            curves = {k: v for k, v in curves.items() if k[:2] not in slices}
            curves.update(saved)
        shown = getattr(self, "_shown_label", None)
        if shown is not None and shown.shape == s.label.shape:
            shown[:, :, :, t] = s.label[:, :, :, t]
            self._shown_frames.add(t)
        return curves

    def _mode_changed(self, *args):
        """Switch between curve editing and the pixel brush, carrying edits across."""
        if self.session is None or self.editor is None:
            return
        curves_mode = self.rb_curves.isChecked()
        self._sync_labels()  # carry edits from whichever mode was active
        self.viewer.layers["contours"].visible = not curves_mode
        if curves_mode:
            # Slices whose labels haven't changed since the curves were built keep their dots exactly as they
            # are (e.g. smoothed), e.g. when ED/ES is set on another slice.
            lab, frames = self.session.label, self.session.frames_in_use()
            shown = getattr(self, "_shown_label", None)
            keep = set()
            if shown is not None and shown.shape == lab.shape:
                old_frames = getattr(self, "_shown_frames", set())
                keep = {(t, z) for t in frames if t in old_frames for z in range(lab.shape[2])
                        if np.array_equal(lab[:, :, z, t], shown[:, :, z, t])}
            curves = curves_from_labels(lab, frames)
            saved = self.session.curves_still_valid()  # exactly what the user drew, where nothing has changed it
            saved = {k: v for k, v in saved.items() if k[0] in frames and k[:2] not in keep}
            if saved:
                slices = {k[:2] for k in saved}
                curves = {k: v for k, v in curves.items() if k[:2] not in slices}
                curves.update(saved)
            self._shown_label, self._shown_frames = np.array(lab, copy=True), set(frames)
            self.editor.show(curves, keep=keep, frames=set(frames))
            self.editor.set_focus(ADD_TARGETS[self.cmb_add.currentIndex()])
        else:
            self.editor.set_active(False)
            self.viewer.layers.selection.active = self.viewer.layers["contours"]
        self._napari_docks(not curves_mode)  # the brush needs napari's label controls (size, colour)
        self.cmb_add.setEnabled(curves_mode)
        if self.btn_raw.isChecked():  # contours are back on screen: leave the raw view
            self.btn_raw.blockSignals(True)
            self.btn_raw.setChecked(False)
            self.btn_raw.blockSignals(False)

    # ---------- side panels and focus ----------

    def _side_docks(self):
        """(Scans dock, Results dock): the left panel, and the dock holding this panel. None where missing."""
        from qtpy.QtWidgets import QDockWidget

        def alive(d):
            try:
                return d is not None and d.objectName() is not None
            except RuntimeError:  # deleted with its window
                return False

        left = next((d for d in getattr(self, "docks", []) if alive(d) and d.objectName() == "Mouse"), None)
        right, w = None, self.parentWidget()
        while w is not None and right is None:
            right = w if isinstance(w, QDockWidget) else None
            w = w.parentWidget()
        return left, right

    def _wire_panel_buttons(self, _tries: int = 0) -> None:
        """Connect the Scans / Results / Focus buttons to the side panels, both ways: a button lights up while its
        panel shows, whether the panel was hidden with its ×, the View menu or the button."""
        try:
            self._wire_panel_buttons_now(_tries)
        except RuntimeError:  # the window was closed before the panel was docked: nothing to wire
            pass

    def _wire_panel_buttons_now(self, _tries: int) -> None:
        left, right = self._side_docks()
        if right is None and _tries < 20:  # not docked yet: try again shortly (cancelled if this panel goes)
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(lambda: self._wire_panel_buttons(_tries + 1))
            t.start(50)
            return
        if getattr(self, "_panels_wired", False):
            return
        self._panels_wired = True
        for btn, dock in ((self.btn_left, left), (self.btn_right, right)):
            if dock is None:
                btn.setEnabled(False)
                continue
            btn.setChecked(not dock.isHidden())
            btn.toggled.connect(lambda on, d=dock: self._show_panel(d, on))
            dock.toggleViewAction().toggled.connect(lambda on, b=btn: self._panel_changed(b, on))
        self.btn_focus.toggled.connect(self._focus_toggled)
        try:
            from qtpy.QtGui import QKeySequence, QShortcut
        except ImportError:  # Qt5
            from qtpy.QtGui import QKeySequence
            from qtpy.QtWidgets import QShortcut
        try:
            win = self.viewer.window._qt_window
            self._focus_shortcut = QShortcut(QKeySequence("Ctrl+Shift+F"), win)
            self._focus_shortcut.activated.connect(self.btn_focus.toggle)
            self._help_shortcuts = []
            for keys in ("?", "F1"):
                sc = QShortcut(QKeySequence(keys), win)
                sc.activated.connect(self.show_shortcuts)
                self._help_shortcuts.append(sc)
        except Exception:
            pass

    # ---------- toolbar menus, view, shortcuts, tour ----------

    def _build_tool_menus(self, QMenu) -> None:
        """The dropdowns: Open (scan folder / UCL file), Refine, View and Help, each entry with its icon and its
        key drawn as a keycap."""
        from .menus import ToolMenu

        open_menu = ToolMenu(self.btn_dcm)
        open_menu.add("Open scan folder…", self._open_dicom, icon="mdi6.folder-open-outline")
        open_menu.add("Add UCL .h5 file(s)…", self._open_ucl, icon="mdi6.file-plus-outline")
        self.btn_dcm.setMenu(open_menu)
        self.btn_dcm.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        refine = ToolMenu(self.btn_refine)
        refine.add("Snap to the blood edge", self.btn_snap.click, icon="mdi6.magnet", tooltip=self.btn_snap.toolTip())
        refine.add("Fit to the blood pool", self.btn_fit.click, icon="mdi6.target", tooltip=self.btn_fit.toolTip())
        refine.addSeparator()
        refine.add("Grow half a pixel", self.btn_grow.click, keys="]", icon="mdi6.arrow-expand-all",
                   tooltip=self.btn_grow.toolTip())
        refine.add("Shrink half a pixel", self.btn_shrink.click, keys="[", icon="mdi6.arrow-collapse-all",
                   tooltip=self.btn_shrink.toolTip())
        self.btn_refine.setMenu(refine)
        self.btn_refine.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        view = ToolMenu(self.btn_view)
        self.act_raw = view.add("Hide contours", lambda on: self.btn_raw.setChecked(on), keys="H",
                                icon="mdi6.eye-off-outline", checkable=True, tooltip=self.btn_raw.toolTip())
        self.btn_raw.toggled.connect(lambda on: self.act_raw.isChecked() != on and self.act_raw.setChecked(on))
        self.act_smooth_image = view.add(
            "Smooth image", self.set_smooth_image, keys="G", icon="mdi6.blur", checkable=True,
            checked=self.smooth_image,
            tooltip="On: smooth (bicubic), like Segment. Off: the scanner's real pixels.")
        view.add("Brightness and contrast…", self.btn_contrast.click, icon="mdi6.brightness-6",
                 tooltip="Or hold Ctrl and drag the image")
        view.add("Rotate or mirror…", self.show_orientation, keys="O", icon="mdi6.rotate-right",
                 tooltip="Turn the picture to the orientation you're used to. Contours and results don't change.")
        view.addSeparator()
        view.add("Zoom in", self.btn_zoom_in.click, keys="+", icon="mdi6.magnify-plus-outline")
        view.add("Zoom out", self.btn_zoom_out.click, keys="−", icon="mdi6.magnify-minus-outline")
        view.addSeparator()
        view.add("ED | ES side by side", lambda: self.show_compare(), keys=["Ctrl", "E"],
                 icon="mdi6.compare-horizontal",
                 tooltip="This slice at end-diastole and end-systole in place of the image; saves figures")
        self.btn_view.setMenu(view)
        self.btn_view.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        help_menu = ToolMenu(self.btn_help)
        help_menu.add("Keyboard shortcuts", self.show_shortcuts, keys="?", icon="mdi6.keyboard-outline")
        help_menu.add("Take the tour", lambda: self.start_tour(), icon="mdi6.map-marker-path")
        help_menu.addSeparator()
        help_menu.add("Report a problem…", self._report_problem, icon="mdi6.lifebuoy",
                      tooltip="Saves a report with versions and the recent log, ready to send")
        help_menu.add("Check for updates…", lambda: self.updates.check(manual=True), icon="mdi6.update",
                      tooltip="Installs a newer TraceCMR in place, without a new installer")
        help_menu.addSeparator()
        help_menu.add("About TraceCMR and credits", self.show_about, icon="mdi6.information-outline",
                      tooltip="Version, licence, and the published work TraceCMR builds on")
        self.btn_help.setMenu(help_menu)
        self.btn_help.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

    def _wall_controls(self) -> QWidget:
        """The Wall tab's controls: wall lines on the image, what they show, and the septum."""
        box = QGroupBox("Wall lines on the image")
        lay = QVBoxLayout(box)
        self.chk_wall_lines = QCheckBox("Show on the image  (W)")
        self.chk_wall_lines.toggled.connect(lambda on: self.btn_wall.isChecked() != on and self.btn_wall.setChecked(on))
        self.btn_wall.toggled.connect(lambda on: self.chk_wall_lines.isChecked() != on
                                      and self.chk_wall_lines.setChecked(on))
        lay.addWidget(self.chk_wall_lines)
        row = QVBoxLayout()
        self.rb_wall_thickness = QRadioButton("Thickness (mm)")
        self.rb_wall_thickening = QRadioButton("Thickening, ED → ES (%)")
        self.rb_wall_thickness.setChecked(True)
        self.rb_wall_thickness.toggled.connect(lambda on: on and self.set_wall_mode("thickness"))
        self.rb_wall_thickening.toggled.connect(lambda on: on and self.set_wall_mode("thickening"))
        row.addWidget(self.rb_wall_thickness)
        row.addWidget(self.rb_wall_thickening)
        lay.addLayout(row)
        row = QHBoxLayout()
        b_set, b_reset = QPushButton("Set septum"), QPushButton("Reset septum")
        b_set.setToolTip("Then click the middle of the septum on this slice.")
        b_reset.setToolTip("Back to the automatic septum (from the RV).")
        b_set.clicked.connect(lambda: self.start_set_septum())
        b_reset.clicked.connect(lambda: self.reset_septum(all_slices=False))
        row.addWidget(b_set)
        row.addWidget(b_reset)
        row.addStretch(1)
        lay.addLayout(row)
        return box

    def _results_tab_changed(self, key: str) -> None:
        """Picking a tab brings up what it is about: Wall shows the wall lines; Regions says how to draw one."""
        if self.session is None:
            return
        if key == "wall" and not self.btn_wall.isChecked():
            self.btn_wall.setChecked(True)
        elif key == "regions" and not self.session.rois and not self.btn_roi.isChecked():
            self.lbl_auto.setText("Press R to draw a region of interest.")
        if key == "wall":
            QTimer.singleShot(0, self._draw_bullseyes)  # its canvas was hidden: draw at its real size

    def _set_hints(self) -> None:
        """Hover cards for the toolbar: what each button does in one line, and its key."""
        from .hints import install, set_hint

        install()
        for btn, title, body, keys in (
            (self.btn_dcm, "Open scan", "Open one mouse's scan folder: a Bruker ParaVision study or DICOM files. "
                                        "Arrow: add UCL .h5 files.", None),
            (self.btn_save, "Save", "Save contours and choices; they reload with the mouse. Arrow: Save as, Export.", ["Ctrl", "S"]),
            (self.btn_auto, "Auto LV", "Contour the LV on every slice at this frame with UCL's published model "
                                       "(Shah et al., JCMR 2026).", None),
            (self.btn_endo, "Endocardium", "Edit the inner LV contour (red).", ["Q"]),
            (self.btn_epi, "Epicardium", "Edit the outer LV contour (light blue).", ["Q"]),
            (self.btn_rv, "Right ventricle", "Edit the RV contour (yellow).", ["Q"]),
            (self.btn_smooth, "Smooth", "Smooth the selected contour on this slice. Repeat for more.", ["S"]),
            (self.btn_refine, "Refine", "Snap or fit to the blood edge, or grow or shrink by half a pixel.", ["[", "]"]),
            (self.btn_points, "Points", "Show the selected contour's points, or hide them for a clean view.", ["P"]),
            (self.btn_brush, "Brush", "Paint pixels instead of editing curves.", None),
            (self.btn_undo, "Undo", "", ["Ctrl", "Z"]),
            (self.btn_redo, "Redo", "", ["Ctrl", "Y"]),
            (self.btn_clear, "Clear", "Delete the selected contour on this slice and frame.",
             ["Del"]),
            (self.btn_pan, "Pan", "Left-drag pans instead of editing. Right-drag always pans.",
             ["Space"]),
            (self.btn_zoom_fit, "Center", "Center and zoom on the heart.", ["F"]),
            (self.btn_view, "View", "Hide contours, smooth image, contrast, zoom.", None),
            (self.btn_wall, "Wall", "Wall thickness lines on this slice. Arrow: thickening, septum.", ["W"]),
            (self.btn_roi, "Regions of interest", "Measure a drawn region's area and signal, or a distance with "
                                                  "the ruler. Compare two regions (CNR).", ["R"]),
            (self.btn_left, "Scans panel", "Mice, series and slices.", None),
            (self.btn_right, "Results panel", "Volumes, EF, wall and regions.", None),
            (self.btn_focus, "Focus", "Hide the side panels so the scan fills the window.", ["Ctrl", "Shift", "F"]),
            (self.btn_help, "Help", "Keyboard shortcuts, the tour and updates.", ["?"]),
            (self.btn_settings, "Settings", "Model, auto-contour, editing and folder options.", None),
        ):
            set_hint(btn, title, body, keys)

    def _image_shortcuts(self) -> None:
        """Keys that match every other app, working wherever the keyboard focus is (except while typing):
        + or = zooms in, − zooms out (also with Ctrl and on the keypad), Ctrl+0 centers, Delete / Backspace
        clears, S smooths, P shows or hides the points.

        They go through napari's key handling, which napari's window feeds every key no other widget used, so
        they work after clicking a button or a list as well as on the image. "+" can't be bound in napari by
        name, so it is caught from the key events directly."""
        bindings = (
            ("s", lambda: self.btn_smooth.isEnabled() and self._smooth_points()),
            ("p", lambda: self.btn_points.isEnabled() and self.btn_points.toggle()),
            ("Delete", self._delete_key), ("Backspace", self._delete_key),
            ("=", lambda: self.zoom(1.25)), ("Control-=", lambda: self.zoom(1.25)),
            ("-", lambda: self.zoom(0.8)), ("Control--", lambda: self.zoom(0.8)),
            ("Control-0", self.fit_heart),
            ("Control-1", lambda: self.results_tabs.show("function")),
            ("Control-2", lambda: self.results_tabs.show("wall")),
            ("Control-3", lambda: self.results_tabs.show("regions")),
        )
        for key, fn in bindings:
            try:
                self.viewer.bind_key(key, lambda v, f=fn: f(), overwrite=True)
            except Exception:
                pass
        self._key_shortcuts = [key for key, _ in bindings] + ["+"]
        try:
            self.viewer.window._qt_viewer.canvas._scene_canvas.events.key_press.connect(self._plus_key)
        except Exception:  # headless
            pass

    def _plus_key(self, event) -> None:
        if getattr(event, "text", "") == "+":  # Shift+= on most keyboards, or the keypad +
            self.zoom(1.25)

    def set_smooth_image(self, on: bool) -> None:
        """Draw the image smoothly (bicubic, as Segment does) or as its real pixels. Remembered between runs;
        the contours are not affected."""
        on = bool(on)
        self.smooth_image = on
        _settings().setValue("view/smooth_image", on)
        act = getattr(self, "act_smooth_image", None)
        if act is not None and act.isChecked() != on:
            act.blockSignals(True)
            act.setChecked(on)
            act.blockSignals(False)
        if "image" in self.viewer.layers:
            self.viewer.layers["image"].interpolation2d = "cubic" if on else "nearest"
        self.viewer.status = "Smooth image" if on else "Pixel image: the scanner's real pixels"
        self._compare_changed()

    def cycle_structure(self) -> None:
        """Q: switch the contour being edited, Endo → Epi → RV → Endo."""
        if not self.cmb_add.isEnabled():
            return
        self.cmb_add.setCurrentIndex((self.cmb_add.currentIndex() + 1) % 3)
        self.viewer.status = f"Editing: {('Endo', 'Epi', 'RV')[self.cmb_add.currentIndex()]}"

    def show_shortcuts(self) -> None:
        """? or F1: every keyboard shortcut on one card."""
        from .help import ShortcutSheet

        sheet = getattr(self, "_shortcut_sheet", None)
        if sheet is not None and sheet.isVisible():
            sheet.close()
            return
        self._shortcut_sheet = ShortcutSheet(self.viewer.window._qt_window)
        self._shortcut_sheet.popup()

    def tour_steps(self) -> list:
        from .help import TourStep

        left, right = self._side_docks()
        time = next((d for d in self.docks if d.objectName() == "Time"), None)
        return [
            TourStep("Welcome",
                     "A one-minute tour: opening a scan, contouring and reading the results."),
            TourStep("Open a scan",
                     "Open one mouse's folder; its slices load on the left. The arrow adds UCL .h5 files.", lambda: [self.btn_dcm]),
            TourStep("Your mice and slices",
                     "Pick a mouse, then a series. Drag slices into base-to-apex order; untick one to leave it "
                     "out of ED or ES.", lambda: [left]),
            TourStep("Contour, then choose what to edit",
                     "Auto LV contours every slice in about a second. Endo, Epi and RV pick the contour to "
                     "edit (Q cycles).",
                     lambda: [self.btn_auto, self.btn_endo, self.btn_epi, self.btn_rv]),
            TourStep("Edit by hand",
                     "Drag a point to move it, click the line to add one, right-click to delete. S smooths, "
                     "Refine nudges, P hides the points.",
                     lambda: [self.btn_smooth, self.btn_refine, self.btn_points, self.btn_brush, self.btn_undo,
                              self.btn_redo, self.btn_clear]),
            TourStep("Look closely",
                     "Pan, center on the heart, and View: hide contours, smooth image, contrast, rotate, zoom.", lambda: [self.btn_pan, self.btn_zoom_fit, self.btn_view]),
            TourStep("Measure",
                     "Wall shows thickness on the slice. ROI measures a region's area and signal and compares "
                     "regions (CNR), e.g. scar vs remote muscle on LGE.", lambda: [self.btn_wall, self.btn_roi]),
            TourStep("Set end-diastole and end-systole",
                     "Play the cine and set ED and ES for each slice.", lambda: [time]),
            TourStep("Read the results",
                     "Results update as you edit, in three tabs: Function, Wall, Regions (Ctrl+1/2/3). Check "
                     "flagged slices, then export below.", lambda: [right]),
            TourStep("Make room for the scan",
                     "Focus hides the side panels; Scans and Results bring each back.", lambda: [self.btn_left, self.btn_right, self.btn_focus]),
            TourStep("Shortcuts and help",
                     "Press ? for every shortcut. Replay this tour from Help.",
                     lambda: [self.btn_help]),
        ]

    def start_tour(self) -> None:
        """Show the tour now (Help → Take the tour, or the first time the app opens)."""
        from .help import Tour

        old = getattr(self, "_tour", None)
        try:
            if old is not None and old.isVisible():
                return
        except RuntimeError:
            pass
        if self.btn_focus.isChecked():
            self.btn_focus.setChecked(False)
        restore = getattr(self.viewer.window, "_tracecmr_restore", None)
        if restore is not None:
            restore.trigger()
        logo = None
        try:
            from .branding import _logo_pixmap

            logo = _logo_pixmap("dark", 200)
        except Exception:
            pass

        def done(completed: bool) -> None:
            _settings().setValue("tour/seen", True)
            self.viewer.status = ("Tour finished. Press ? for shortcuts." if completed
                                  else "Tour skipped. Replay it from Help.")

        self._tour = Tour(self.viewer.window._qt_window, self.tour_steps(), on_finish=done, logo=logo)
        self._tour.start()

    def maybe_start_tour(self) -> None:
        """The first time TraceCMR opens: offer the tour (once; TRACECMR_NO_TOUR=1 turns it off)."""
        import os

        if os.environ.get("TRACECMR_NO_TOUR") or str(_settings().value("tour/seen", False)).lower() in ("true", "1"):
            return
        self.start_tour()

    def _show_panel(self, dock, on: bool) -> None:
        if dock.isHidden() == on:
            dock.setVisible(on)
            if on:
                dock.raise_()

    def _panel_changed(self, btn, on: bool) -> None:
        if btn.isChecked() != on:
            btn.blockSignals(True)
            btn.setChecked(on)
            btn.blockSignals(False)
        left, right = self._side_docks()
        both_hidden = all(d is None or d.isHidden() for d in (left, right))
        if self.btn_focus.isChecked() != both_hidden:
            self.btn_focus.blockSignals(True)
            self.btn_focus.setChecked(both_hidden)
            self.btn_focus.blockSignals(False)

    def _focus_toggled(self, on: bool) -> None:
        """Focus: hide both side panels and the time bar (remembering which were open) so the scan fills the
        window; frames still change with Left / Right. Off again: bring back what was open."""
        left, right = self._side_docks()
        time_bar = next((d for d in getattr(self, "docks", []) if d is not None and d.objectName() == "Time"), None)
        docks = [d for d in (left, right, time_bar) if d is not None]
        self._focusing = True
        try:
            if on:
                self._focus_restore = [d for d in docks if not d.isHidden()] or docks
                for d in docks:
                    d.hide()
                self.viewer.status = "Focus: panels hidden. Ctrl+Shift+F brings them back."
            else:
                for d in getattr(self, "_focus_restore", None) or docks:
                    d.show()
                    d.raise_()
                self.viewer.status = "Ready"
        finally:
            self._focusing = False

    def _scale_with_canvas(self) -> None:
        """Make the image grow and shrink with the viewing area (hiding a panel, Focus, maximising the window),
        keeping the user's framing. napari otherwise keeps the zoom fixed, so extra room only added black space."""
        try:
            native = self.viewer.window._qt_viewer.canvas.native
        except Exception:
            return
        from qtpy.QtCore import QEvent, QObject

        panel = self

        class _Filter(QObject):
            def eventFilter(self, obj, ev):  # noqa: N802 (Qt name)
                if ev.type() == QEvent.Type.Resize:
                    panel._canvas_resized(ev.oldSize(), ev.size())
                return False

        self._canvas_filter = _Filter(native)
        native.installEventFilter(self._canvas_filter)

    def _canvas_resized(self, old, new) -> None:
        if self.session is None or not len(self.viewer.layers):
            return
        if old.width() < 50 or old.height() < 50 or new.width() < 50 or new.height() < 50:
            return
        try:
            ext = np.asarray(self.viewer.layers.extent.world)
            rows, cols = self.viewer.dims.displayed[-2:]
            h, w = float(ext[1, rows] - ext[0, rows]), float(ext[1, cols] - ext[0, cols])
        except Exception:
            return
        if h <= 0 or w <= 0:
            return
        fit_old = min(old.width() / w, old.height() / h)
        fit_new = min(new.width() / w, new.height() / h)
        if fit_old > 0 and abs(fit_new / fit_old - 1) > 1e-3:
            self._camera.zoom = self._camera.zoom * fit_new / fit_old

    def _napari_docks(self, show: bool) -> None:
        qv = getattr(self.viewer.window, "_qt_viewer", None)
        for name in ("dockLayerList", "dockLayerControls"):
            dock = getattr(qv, name, None)
            if dock is not None:
                dock.setVisible(show)

    def _add_target_changed(self, idx):
        if self.editor is not None:
            self.editor.set_focus(ADD_TARGETS[idx])
        if not self.btn_points.isChecked():  # picking a contour means editing it: its points come back
            self.btn_points.setChecked(True)

    def _points_toggled(self, on: bool) -> None:
        if self.editor is not None:
            self.editor.set_show_dots(on)
        if self.session is not None:
            self.lbl_auto.setText("" if on else "Points hidden (left-drag pans). P or Endo / Epi / RV shows them.")

    def current_frame(self) -> int:
        return int(self.viewer.dims.current_step[0]) if self.session is not None else 0

    def _update_frame_label(self, event=None):
        if self.session is None or self.viewer.dims.ndim < 4:
            return
        s = self.session
        t, z = self.current_frame(), self.current_slice()
        ted, tes = s.t_for(z, "ED"), s.t_for(z, "ES")
        tag = " · ED" if t == ted else " · ES" if t == tes else ""
        self.lbl_frame.setText(f"Frame {t} of {s.image.shape[3] - 1}{tag}")
        for spin, v in ((self.spin_ed, ted), (self.spin_es, tes)):  # spins show this slice's frames
            if spin.value() != v:
                spin.blockSignals(True)
                spin.setValue(v)
                spin.blockSignals(False)
        if hasattr(self, "timebar"):
            self.timebar.track.update()
        if hasattr(self, "tbl_slices"):
            self._select_table_row()
        # Segment-style text in the image corner
        name = s.slice_names[z] if s.slice_names and z < len(s.slice_names) else ""
        ms = f" · {s.frame_times_ms[t]:.0f} ms" if s.frame_times_ms and t < len(s.frame_times_ms) else ""
        text = (f"{s.subject_id}\nSlice {z}{(' · ' + name) if name else ''}\n"
                f"Frame {t} of {s.image.shape[3] - 1}{ms}{tag}")  # counted from 0, like slices and the frame bar
        try:
            canvas = getattr(self.viewer, "canvas", None)
            ov = canvas.overlays["text"] if canvas is not None and hasattr(canvas, "overlays") \
                else self.viewer.text_overlay
            ov.visible, ov.text, ov.color, ov.font_size = True, text, "white", 10
            ov.position = "top_left"
        except Exception:
            pass

    def _toggle_play(self):
        """Play/stop the cine along the time axis (napari's own player)."""
        try:
            qt_dims = self.viewer.window._qt_viewer.dims
        except Exception:
            return
        if getattr(qt_dims, "is_playing", False):
            qt_dims.stop()
            self.btn_play.setText("Play cine")
            self.timebar.btn_play.setText("▶")
        else:
            qt_dims.play(axis=0, fps=12)
            self.btn_play.setText("Stop")
            self.timebar.btn_play.setText("■")

    def _show_frame(self, t: int):
        step = list(self.viewer.dims.current_step)
        step[0] = t
        self.viewer.dims.current_step = tuple(step)

    def _frames_changed(self):
        """The ED/ES spin boxes were edited."""
        if self.session is None:
            return
        z = self.current_slice()
        for ph, spin in (("ED", self.spin_ed), ("ES", self.spin_es)):
            if spin.value() != self.session.t_for(z, ph):
                self.set_phase_frame(ph, spin.value())

    def set_phase_frame(self, phase: str, t: int) -> None:
        """Set ED or ES to frame t: on the slice being viewed (per-slice mode) or on every slice."""
        if self.session is None:
            return
        self._sync_labels()
        z = self.current_slice() if self.chk_per_slice.isChecked() else None
        self.session.set_phase_frame(phase, int(t), z)
        self._phases_changed()

    def _phases_changed(self):
        self._table_timer.start(0)
        self._rebuild_list()
        self._update_frame_label()
        self._draw_volume_curve()
        if self.rb_curves.isChecked():
            self._mode_changed()
        self.timebar.track.update()

    def _per_slice_toggled(self, on: bool):
        if self.session is None:
            return
        self.session.separate_slices = on
        if not on and (self.session.ed_by_slice or self.session.es_by_slice):
            self._sync_labels()
            self.session.ed_by_slice.clear()
            self.session.es_by_slice.clear()
            self._phases_changed()

    def current_slice(self) -> int:
        return int(self.viewer.dims.current_step[1]) if self.session is not None and self.viewer.dims.ndim >= 4 else 0

    def step_frame(self, d: int) -> None:
        if self.session is not None:
            self._show_frame((self.current_frame() + d) % self.session.image.shape[3])

    def _auto_frames(self):
        self._sync_labels()
        msg = self.session.auto_ed_es()
        self._phases_changed()
        if self.session.separate_slices:  # how much picking ED/ES per slice changed the result
            from ..drift import phase_timing

            try:
                timing = phase_timing(self.session)
                if len(timing.slices) >= 2:
                    msg = f"{msg} {timing.summary()}"
            except Exception:
                pass
        self.lbl_order.setText(msg)

    # ---------- slice order ----------

    def _rebuild_list(self):
        s = self.session
        self.list.blockSignals(True)
        self.list.clear()
        for z in s.order:
            t = s.t_for(z, "ED")
            item = QTreeWidgetItem(["", "", "", ""])
            item.setIcon(0, QIcon(_thumbnail(s.image[:, :, z, t], s.label[:, :, z, t], angle=s.view_rotation or 0.0,
                                             flip=bool(s.view_flip))))
            item.setData(0, Qt.ItemDataRole.UserRole, z)
            item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
                          & ~Qt.ItemFlag.ItemIsDropEnabled)
            for col, ph in ((1, "ED"), (2, "ES")):
                item.setCheckState(col, Qt.CheckState.Checked if z in s.included[ph] else Qt.CheckState.Unchecked)
            self.list.addTopLevelItem(item)
        self.list.blockSignals(False)
        self._relabel_items()
        self._show_warnings()
        if hasattr(self, "series_panel"):
            self.series_panel._fit_lists()  # room for this stack's slices

    def _items(self):
        return [self.list.topLevelItem(i) for i in range(self.list.topLevelItemCount())]

    def _untouched(self, t: int, z: int) -> bool:
        """True while slice z at frame t still has exactly the LV outline Auto LV drew (any edit, by points, brush,
        Fit, Grow or Snap, makes it the user's). Hand-drawn slices are never 'untouched'."""
        s = self.session
        ml, ran = getattr(s, "model_label", None), getattr(s, "model_ran", None)
        if ml is None or ran is None or ml.shape != s.label.shape or not ran[z, t]:
            return False
        cur, mod = s.label[:, :, z, t], ml[:, :, z, t]
        lv = (L.LV_BLOODPOOL, L.LV_MYOCARDIUM)
        return np.array_equal(np.where(np.isin(cur, lv), cur, 0), np.where(np.isin(mod, lv), mod, 0))

    def _run_checks(self) -> None:
        """Which ED/ES slices to double-check (quality.py), shown as ⚠ in the slice list with the reasons. Only
        slices still exactly as Auto LV drew them are checked: once you've edited a slice, your outline stands."""
        from .. import quality

        s = self.session
        self._checks = {}
        if s is None:
            return
        try:
            pairs = {}
            for ph in ("ED", "ES"):
                pairs[ph] = [(s.t_for(z, ph), z) for z in s.order if z in s.included[ph]
                             and np.isin(s.label[:, :, z, s.t_for(z, ph)], (1, 2)).any()]
            found = quality.check_slices(s.label, s.image, s.spacing, pairs, list(s.order), s.valid_unsure())
            self._checks = {k: v for k, v in found.items() if self._untouched(*k)}
        except Exception:  # checks are a help; never let them break the results
            self._checks = {}
        if hasattr(self, "list"):
            try:
                self._relabel_items()
            except RuntimeError:  # the window is closing
                pass

    def _relabel_items(self):
        s = self.session
        self.list.blockSignals(True)
        items = self._items()
        zs = [it.data(0, Qt.ItemDataRole.UserRole) for it in items]
        drawn = [z for z in zs if z in s.included["ED"] and np.any(s.label[:, :, z, s.t_for(z, "ED")])]
        for it, z in zip(items, zs):
            tag = " · base" if drawn and z == drawn[0] else " · apex" if drawn and z == drawn[-1] else ""
            name = s.slice_names[z] if s.slice_names and z < len(s.slice_names) else ""
            it.setText(0, f"Slice {z}{tag}")  # the series name is in the tooltip
            r_ed, r_es = (s.reasons[ph].get(z, "") for ph in ("ED", "ES"))
            r_ed, r_es = ("" if r.startswith("wall covers") else r for r in (r_ed, r_es))
            # Auto LV's own notes ("check: base slice…") only while the slice is still the model's outline
            def _from_auto_lv(r, ph):
                why = s.flags.get((s.t_for(z, ph), z))
                return bool(why) and r.startswith("check") and why[:20] in r
            r_ed, r_es = ("" if _from_auto_lv(r, ph) and not self._untouched(s.t_for(z, ph), z) else r
                          for r, ph in ((r_ed, "ED"), (r_es, "ES")))
            if r_ed == r_es:
                why = [r_ed] if r_ed else []
            else:
                why = [f"{ph}: {r}" for ph, r in (("ED", r_ed), ("ES", r_es)) if r]
            checks = getattr(self, "_checks", {}) or {}
            for ph in ("ED", "ES"):
                for c in checks.get((s.t_for(z, ph), z), []):
                    why.append(f"check {ph}: {c}")
            it.setText(3, "; ".join(why))
            flagged = any("check" in w for w in why)
            it.setForeground(3, QColor("#E3A23B") if flagged else QColor("#9AA1A8"))
            it.setToolTip(3, it.text(3))
            if flagged:
                it.setText(0, "⚠ " + it.text(0))
            it.setForeground(0, QColor("#E3A23B") if flagged else it.foreground(1))
            lines = []
            for w in why:  # one tidy line per reason: "ED: …" / "ES: …"
                w = w.replace("check ED: ", "ED: ").replace("check ES: ", "ES: ").replace("ED: check: ", "ED: ")
                w = w.replace("ES: check: ", "ES: ").replace("check: ", "")
                if w not in lines:
                    lines.append(w)
            tip = f"Slice {z}" + (f": {name}" if name else "")
            if lines:
                tip += ("\nDouble-check:" if flagged else "") + "".join(f"\n• {ln}" for ln in lines)
            it.setToolTip(0, tip)
        self.list.blockSignals(False)

    def _order_from_list(self, *args):
        try:
            zs = [it.data(0, Qt.ItemDataRole.UserRole) for it in self._items()]
        except RuntimeError:  # the window closed before this deferred call ran
            return
        if self.session is None:
            return
        if sorted(zs) != sorted(self.session.order):  # mid-drag; wait for the drop to finish
            return
        self.session.order = zs
        self._relabel_items()
        self._show_warnings()

    def _item_changed(self, item, column):
        if column not in (1, 2):
            return
        z = item.data(0, Qt.ItemDataRole.UserRole)
        ph = "ED" if column == 1 else "ES"
        self.session.set_included(z, item.checkState(column) == Qt.CheckState.Checked, ph)
        self._table_timer.start(0)
        self._relabel_items()
        self._show_warnings()

    def _apply_suggestions(self):
        self._sync_labels()
        self.session.apply_suggestions()
        self._rebuild_list()

    def _goto_slice(self, item, _prev=None):
        if item is None:
            return
        step = list(self.viewer.dims.current_step)
        step[1] = item.data(0, Qt.ItemDataRole.UserRole)
        self.viewer.dims.current_step = tuple(step)

    def _flip(self):
        self.session.flip()
        self._rebuild_list()

    def _auto_order(self):
        self._sync_labels()
        msg = self.session.auto_order()
        self._rebuild_list()
        self.lbl_order.setText(msg + ("\n" + self.lbl_order.text() if self.lbl_order.text().startswith("⚠") else ""))

    def _show_warnings(self):
        w = self.session.order_warnings() if self.session else []
        if w:
            self.lbl_order.setStyleSheet(WARN_STYLE)
            self.lbl_order.setText("⚠ " + "\n⚠ ".join(w))
        elif self.lbl_order.text().startswith("⚠"):
            self.lbl_order.setText("")
        if not w:
            self.lbl_order.setStyleSheet(NOTE_STYLE)

    # ---------- auto-contour ----------

    MODEL_ENV = {"ucl": "TRACECMR_UCL_DIR"}

    def _load_model_dirs(self):
        """Use model folders the user chose before, unless an environment variable already says where."""
        st = _settings()
        for key, env in self.MODEL_ENV.items():
            v = st.value(f"models/{key}")
            if v and not os.environ.get(env):
                os.environ[env] = str(v)
            elif not v and os.environ.get(env):  # remember it, so the desktop app (no launcher script) finds it too
                st.setValue(f"models/{key}", os.environ[env])

    def set_model_dir(self, key: str, path: str, remember: bool = True) -> str:
        """Use `path` for model `key`; returns the model's status ('ready' or why not)."""
        from .. import autocontour as ac

        os.environ[self.MODEL_ENV[key]] = path
        if remember:
            _settings().setValue(f"models/{key}", path)
        self._fill_models()
        if key in self._model_keys:
            self.cmb_model.setCurrentIndex(self._model_keys.index(key))
        status = ac.available()[key]
        self.lbl_auto.setText("Model found." if status == "ready" else f"Model not found there: {status}")
        return status

    def _choose_model_dir(self):
        key = self._model_keys[self.cmb_model.currentIndex()]
        folder = QFileDialog.getExistingDirectory(self, f"Folder with the {key.upper()} model")
        if folder:
            self.set_model_dir(key, folder)

    def _fill_models(self):
        from .. import autocontour as ac

        self.cmb_model.clear()
        self._model_keys = []
        for key, status in ac.available().items():
            if key not in APP_MODELS:
                continue
            info = ac.MODELS[key]
            self.cmb_model.addItem(info.label + ("" if status == "ready" else "  (not installed)"))
            self._model_keys.append(key)
        ready = [i for i, k in enumerate(self._model_keys) if ac.available().get(k) == "ready"]
        if ready:
            self.cmb_model.setCurrentIndex(ready[0])
        ok = bool(ready)
        self.lbl_auto.setText("" if ok else "Auto-contouring needs a one-time download. Click Auto LV to set it up.")
        if hasattr(self, "btn_setup_model"):
            self.btn_setup_model.setText("Auto-contouring is set up" if ok else "Set up auto-contouring…")
            self.btn_setup_model.setEnabled(not ok)

    # ---------- one-click setup of auto-contouring (model_setup.py) ----------

    def _tta_toggled(self, on: bool) -> None:
        from .. import autocontour as ac

        _settings().setValue("auto/tta", bool(on))
        os.environ["TRACECMR_TTA"] = "1" if on else "0"
        ac.reset_workers()  # the model process reads this when it starts

    def model_ready(self, key: str = "ucl") -> bool:
        from .. import autocontour as ac

        return ac.available().get(key) == "ready"

    def _dark_box(self, title: str, text: str, info: str = ""):
        from qtpy.QtWidgets import QMessageBox

        box = QMessageBox(self.viewer.window._qt_window if self._has_window() else self)
        box.setWindowTitle(title)
        box.setText(text)
        if info:
            box.setInformativeText(info)
        box.setIcon(QMessageBox.Icon.NoIcon)
        pal = box.palette()
        pal.setColor(pal.ColorRole.Window, QColor("#1B1F25"))
        box.setPalette(pal)
        box.setAutoFillBackground(True)
        return box

    def offer_model_setup(self) -> None:
        """Auto LV clicked before the model is installed: explain the one-time download and offer it."""
        from qtpy.QtWidgets import QMessageBox

        from .. import model_setup

        if getattr(self, "_testing", False):
            if getattr(self, "_setup_answer", False):
                self.start_model_setup(sync=True)
            else:
                self.lbl_auto.setText("Auto-contouring needs a one-time download. Click Auto LV to set it up.")
            return
        box = self._dark_box(
            "Set up auto-contouring",
            "Auto LV uses UCL's published mouse heart model (Shah et al., JCMR 2026).",
            f"Setting it up is a one-time download of about {model_setup.DOWNLOAD_MB} MB: the model, and ONNX "
            "Runtime, the small library that runs it. It takes a minute or two and needs an internet connection. "
            "Your scans are not uploaded, and you can keep working while it downloads.")
        go = box.addButton("Set up now", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(go)
        box.exec()
        if box.clickedButton() is go:
            self.start_model_setup()

    def _build_setup_dialog(self):
        from qtpy.QtWidgets import QProgressBar

        dlg = QDialog(self.viewer.window._qt_window if self._has_window() else self)
        dlg.setWindowTitle("Setting up auto-contouring")
        dlg.setModal(False)
        dlg.setMinimumWidth(440)
        pal = dlg.palette()
        pal.setColor(pal.ColorRole.Window, QColor("#1B1F25"))
        dlg.setPalette(pal)
        dlg.setAutoFillBackground(True)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        dlg.lbl_step = QLabel("")
        dlg.lbl_step.setStyleSheet("font-weight: 600; font-size: 13px;")
        dlg.lbl_text = QLabel("")
        dlg.lbl_text.setWordWrap(True)
        dlg.bar = QProgressBar()
        dlg.bar.setTextVisible(False)
        dlg.bar.setFixedHeight(8)
        dlg.lbl_note = QLabel("You can keep working while this runs. It only happens once.")
        dlg.lbl_note.setStyleSheet(NOTE_STYLE)
        dlg.lbl_note.setWordWrap(True)
        row = QHBoxLayout()
        row.addStretch(1)
        dlg.btn_run = QPushButton("Run Auto LV now")
        dlg.btn_retry = QPushButton("Try again")
        dlg.btn_cancel = QPushButton("Cancel")
        for b in (dlg.btn_run, dlg.btn_retry, dlg.btn_cancel):
            row.addWidget(b)
        dlg.btn_run.clicked.connect(lambda: (dlg.hide(), self.autocontour()))
        dlg.btn_retry.clicked.connect(lambda: self.start_model_setup())
        dlg.btn_cancel.clicked.connect(self._cancel_model_setup)
        for w in (dlg.lbl_step, dlg.lbl_text, dlg.bar, dlg.lbl_note):
            lay.addWidget(w)
        lay.addLayout(row)
        return dlg

    def _cancel_model_setup(self) -> None:
        if getattr(self, "_setup_running", False):
            self._setup_cancel = True
            self.setup_dialog.lbl_text.setText("Cancelling…")
        else:
            self.setup_dialog.hide()

    def _setup_step(self, step) -> None:
        dlg = self.setup_dialog
        names = {"model": "Step 1 of 2: the model", "runtime": "Step 2 of 2: ONNX Runtime",
                 "check": "Checking", "done": "Done"}
        dlg.lbl_step.setText(names.get(step.stage, ""))
        if step.total > 0:
            pct = step.done / step.total
            dlg.bar.setRange(0, 1000)
            dlg.bar.setValue(int(1000 * pct))
            mb = f" ({step.done / 1e6:.0f} of {step.total / 1e6:.0f} MB)" if step.total > 2_000_000 else ""
            dlg.lbl_text.setText(step.text + mb)
        else:
            dlg.bar.setRange(0, 0)  # busy
            dlg.lbl_text.setText(step.text)

    def _setup_finished(self, ok: bool, message: str) -> None:
        from .. import autocontour as ac
        from .. import model_setup

        self._setup_running = False
        dlg = self.setup_dialog
        dlg.btn_cancel.setText("Close")
        dlg.bar.setVisible(False)
        if ok:
            ac.reset_workers()  # a fresh model process that sees the new library and folder
            self.set_model_dir("ucl", str(model_setup.default_dir()))
            self._preloaded = False
            if not getattr(self, "_testing", False):
                self._preload_model()  # the first Auto LV click is then quick
            dlg.lbl_step.setText("Auto-contouring is ready")
            dlg.lbl_text.setText("Click Auto LV to contour every slice at the frame you're viewing.")
            dlg.lbl_note.setText("")
            dlg.btn_run.setVisible(self.session is not None)
            dlg.btn_retry.setVisible(False)
            self.lbl_auto.setText("Auto-contouring is ready. Click Auto LV.")
        else:
            dlg.lbl_step.setText("Setup stopped" if message != "Cancelled." else "Cancelled")
            dlg.lbl_text.setText(message if message != "Cancelled." else
                                 "Nothing half-finished was kept. You can set it up any time from Auto LV.")
            dlg.btn_run.setVisible(False)
            dlg.btn_retry.setVisible(True)
            self.lbl_auto.setText("Auto-contouring isn't set up yet. Click Auto LV to try again.")

    def start_model_setup(self, sync: bool = False) -> None:
        """Download the model and ONNX Runtime with progress (non-blocking; sync=True for tests)."""
        from .. import model_setup

        if getattr(self, "_setup_running", False):
            self.setup_dialog.show()
            self.setup_dialog.raise_()
            return
        if getattr(self, "setup_dialog", None) is None:
            self.setup_dialog = self._build_setup_dialog()
        dlg = self.setup_dialog
        dlg.btn_run.setVisible(False)
        dlg.btn_retry.setVisible(False)
        dlg.btn_cancel.setText("Cancel")
        dlg.bar.setVisible(True)
        dlg.lbl_note.setText("You can keep working while this runs. It only happens once.")
        self._setup_step(model_setup.Step("model", 0, 0, "Starting…"))
        self._setup_cancel = False
        self._setup_running = True
        if not getattr(self, "_testing", False):
            dlg.show()
            dlg.raise_()

        def steps():
            yield from model_setup.run_setup(model_setup.default_dir(), lambda: self._setup_cancel)

        if sync:
            try:
                for st in steps():
                    self._setup_step(st)
                self._setup_finished(True, "")
            except model_setup.SetupError as e:
                self._setup_finished(False, str(e))
            except Exception as e:  # never leave the dialog spinning
                self._setup_finished(False, f"Something went wrong: {e}. Details are in {model_setup.log_path()}.")
            return
        from napari.qt.threading import create_worker

        worker = create_worker(steps)
        worker.yielded.connect(self._setup_step)
        worker.returned.connect(lambda _r: self._setup_finished(True, ""))

        def failed(e):
            msg = str(e) if isinstance(e, model_setup.SetupError) else \
                f"Something went wrong: {e}. Details are in {model_setup.log_path()}."
            self._setup_finished(False, msg)

        worker.errored.connect(failed)
        self._setup_worker = worker
        worker.start()

    def autocontour_pairs(self) -> list[tuple[int, int]]:
        """Which (slice, frame) images Auto-contour does: the frame being viewed on every slice. When that
        frame is the slice's ED (or ES) and slices have their own ED/ES, each slice's own ED (or ES)."""
        s = self.session
        t0, z0 = self.current_frame(), self.current_slice()
        nz = s.image.shape[2]
        if self.chk_per_slice.isChecked():
            for ph in ("ED", "ES"):
                if t0 == s.t_for(z0, ph):
                    return [(z, s.t_for(z, ph)) for z in range(nz)]
        return [(z, t0) for z in range(nz)]

    def _has_lv(self, z: int, t: int) -> bool:
        return bool(np.isin(self.session.label[:, :, z, t], (L.LV_BLOODPOOL, L.LV_MYOCARDIUM)).any())

    def autocontour(self, sync: bool = False, redo_current: bool | None = None):
        """Run the chosen model in the background (sync=True for tests and scripts).

        Only slices with no LV contour at that frame are contoured, so finished slices are never overwritten.
        When every slice already has one, it offers to redo just the slice being viewed (redo_current answers
        that question without a dialog). Undo (Ctrl+Z) reverts the whole Auto LV run."""
        import time

        from .. import autocontour as ac

        if self.session is None:
            return
        key = self._model_keys[self.cmb_model.currentIndex()]
        if not self.model_ready(key):
            self.offer_model_setup()
            return
        self._sync_labels()
        other = None
        all_frames = self.chk_all_frames.isChecked()
        pairs = self.autocontour_pairs()
        here = self.current_frame()
        kept = 0
        if not all_frames:
            empty = [(z, t) for z, t in pairs if not self._has_lv(z, t)]
            kept = len(pairs) - len(empty)
            if empty:
                pairs = empty
            else:
                z0 = self.current_slice()
                if redo_current is None:
                    from qtpy.QtWidgets import QMessageBox

                    ans = QMessageBox.question(
                        self, "Redo this slice?",
                        f"Every slice at this frame already has contours.\n\n"
                        f"Redo slice {z0}? Its endo and epi are replaced (your RV is kept). Ctrl+Z undoes.")
                    redo_current = ans == QMessageBox.StandardButton.Yes
                if not redo_current:
                    self.lbl_auto.setText("Nothing changed: every slice at this frame already has contours.")
                    return
                pairs = [p for p in pairs if p[0] == z0] or [(z0, here)]
                kept = 0
        self._label_undo = self.session.label.copy()  # Ctrl+Z reverts the whole run
        self.btn_auto.setEnabled(False)
        self.prog.setVisible(True)
        self.prog.setRange(0, 0)
        self.lbl_auto.setText(f"Running the {ac.MODELS[key].label}…")
        t_start = time.time()

        def work():
            if all_frames:
                return self.session.run_autocontour(key, all_frames=True, compare_with=other)
            return self.session.autocontour_pairs(pairs, key)

        def done(msg):
            self.prog.setVisible(False)
            self.btn_auto.setEnabled(True)
            self.chk_smooth.blockSignals(True)
            self.chk_smooth.setChecked(False)
            self.chk_smooth.blockSignals(False)
            extra = f" Kept your contours on {kept} slice(s)." if kept else ""
            self.lbl_auto.setText(f"{msg} ({time.time() - t_start:.0f} s).{extra} Ctrl+Z to undo.")
            self._after_labels_changed(keep_frame=None if all_frames else here)

        def failed(err):
            self.prog.setVisible(False)
            self.btn_auto.setEnabled(True)
            self.lbl_auto.setText(f"⚠ Auto-contour failed: {err}")

        if sync:
            try:
                done(work())
            except Exception as e:
                failed(e)
            return
        from napari.qt.threading import create_worker

        worker = create_worker(work)
        worker.returned.connect(done)
        worker.errored.connect(failed)
        worker.start()

    def _preload_model(self):
        """Load the selected model in the background so the first Auto-contour click is quick."""
        if getattr(self, "_preloaded", False) or not self._model_keys:
            return
        from .. import autocontour as ac

        key = self._model_keys[self.cmb_model.currentIndex()]
        if ac.available().get(key) != "ready":
            return
        self._preloaded = True
        try:
            from napari.qt.threading import create_worker

            w = create_worker(ac.preload, key)
            w.errored.connect(lambda e: setattr(self, "_preloaded", False))
            w.start()
        except Exception:
            self._preloaded = False

    def _smooth_points(self):
        """Smooth the selected contour on the slice being viewed (key S). Its points stay for adjusting."""
        if self.editor is None or self.editor.points is None or not self.rb_curves.isChecked():
            self.lbl_auto.setText("Switch to Curves to smooth contours.")
            return
        step = self.viewer.dims.current_step
        n = self.editor.smooth(int(step[0]), int(step[1]))
        name = {"endo": "endocardium", "epi": "epicardium", "rv": "RV"}.get(self.editor.focus, "contours")
        self.lbl_auto.setText(f"Smoothed the {name}. S again for more; Ctrl+Z undoes."
                              if n else f"No {name} on this slice and frame.")

    def _snap_slice(self):
        """Grow the endocardium on the current slice to the bright-blood edge in the image (refine.py)."""
        from ..contours import mask_to_curve
        from ..refine import snap_slice

        if self.session is None or self.editor is None:
            return
        self._sync_labels()
        t, z = (int(v) for v in self.viewer.dims.current_step[:2])
        s = self.session
        before = s.label[:, :, z, t].copy()
        after = snap_slice(before, s.image[:, :, z, t], s.spacing[:2])
        grown = int((after == 1).sum() - (before == 1).sum())
        if grown == 0:
            self.lbl_auto.setText("Snap: nothing to add on this slice.")
            return
        self.editor.push_undo()  # the change lives in the curves, so Undo reverts it
        c = mask_to_curve(after == 1)
        if c is not None:
            self.editor.curves[(t, z, "endo")] = c.control
            self.editor.edited.add((t, z, "endo"))
            self.editor._refresh_layers()
        dx, dy, _ = s.spacing
        self.lbl_auto.setText(f"Snap: added {grown} pixels ({grown * dx * dy:.2f} mm²) to the blood pool. "
                              "Ctrl+Z undoes.")

    def _fit_slice(self):
        """Redraw the endocardium on the current slice around the bright blood (refine.fit_slice)."""
        from ..contours import mask_to_curve
        from ..refine import fit_slice

        if self.session is None or self.editor is None:
            return
        self._sync_labels()
        t, z = (int(v) for v in self.viewer.dims.current_step[:2])
        s = self.session
        before = s.label[:, :, z, t].copy()
        after = fit_slice(before, s.image[:, :, z, t], s.spacing[:2])
        if np.array_equal(after == 1, before == 1):
            self.lbl_auto.setText("Fit: no clear blood edge on this slice (needs endo, epi and bright blood). "
                                  "Drag the points instead.")
            return
        c = mask_to_curve(after == 1)
        if c is None:
            return
        self.editor.push_undo()
        self.editor.curves[(t, z, "endo")] = c.control
        self.editor.edited.add((t, z, "endo"))
        self.editor.hidden.discard((t, z, "endo"))
        self.editor._refresh_layers()
        self.editor._notify()
        added = int(((after == 1) & (before != 1)).sum())
        removed = int(((before == 1) & (after != 1)).sum())
        self.lbl_auto.setText(f"Fit: added {added} blood pixels and gave {removed} back to the wall. Check it; "
                              "Ctrl+Z undoes.")

    def grow_contour(self, d: float):
        """Grow (d > 0) or shrink (d < 0) the selected contour on the slice and frame being viewed, by d pixels."""
        if self.session is None or self.editor is None or not self.rb_curves.isChecked():
            return
        t, z = (int(v) for v in self.viewer.dims.current_step[:2])
        name, change = self.editor.offset(t, z, d)
        word = {"endo": "endocardium", "epi": "epicardium", "rv": "RV"}.get(name or self.editor.focus or "endo")
        if name is None:
            self.lbl_auto.setText(f"No {word} on this slice and frame.")
        elif change == 0:
            self.lbl_auto.setText(f"Can't {'grow' if d > 0 else 'shrink'} the {word} any further here"
                                  + (" (it would cross the other contour)." if d > 0 or name == "epi" else "."))
        else:
            dx, dy, _ = self.session.spacing
            self.lbl_auto.setText(f"{'Grew' if d > 0 else 'Shrank'} the {word} ({change * dx * dy:+.3f} mm²).")

    def undo(self):
        """Undo the last contour edit (Ctrl+Z)."""
        if self.editor is not None and self.rb_curves.isChecked() and self.editor.undo():
            self.lbl_auto.setText("Undone.")
        elif getattr(self, "_label_undo", None) is not None and self.session is not None \
                and self._label_undo.shape == self.session.label.shape:
            here = self.current_frame()
            self.session.label = self._label_undo
            self.session.label_detailed = self._label_undo.copy()
            self._label_undo = None
            self.viewer.layers["contours"].data = self.session.label.transpose(3, 2, 1, 0).copy()
            self._after_labels_changed(keep_frame=here)
            self.lbl_auto.setText("Auto LV undone: previous contours restored.")
        else:
            self.lbl_auto.setText("Nothing to undo.")

    def _offer_recovery(self) -> None:
        """If TraceCMR closed before this mouse's last changes were saved, offer to bring them back."""
        if self.session is None:
            return
        rec = self.autosave.session_opened()
        if rec is None:
            return
        from qtpy.QtWidgets import QMessageBox

        if getattr(self, "_testing", False):
            choice = getattr(self, "_recovery_answer", QMessageBox.StandardButton.No)
        else:
            box = QMessageBox(self.viewer.window._qt_window if self._has_window() else self)
            box.setWindowTitle("Recover unsaved changes")
            box.setText(f"TraceCMR closed before your last changes to {self.session.subject_id} were saved "
                        f"(autosaved at {self.autosave.when(rec)}).")
            box.setInformativeText("Restore them?")
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            box.setDefaultButton(QMessageBox.StandardButton.Yes)
            box.setIcon(QMessageBox.Icon.NoIcon)
            pal = box.palette()  # dark from its first frame: no light flash before the theme applies
            pal.setColor(pal.ColorRole.Window, QColor("#1B1F25"))
            box.setPalette(pal)
            box.setAutoFillBackground(True)
            box.exec()
            choice = box.standardButton(box.clickedButton())
        if choice != QMessageBox.StandardButton.Yes:
            rec.unlink(missing_ok=True)
            return
        try:
            self.session.load_contours(rec)
        except Exception as e:
            self.lbl_auto.setText(f"⚠ Could not restore: {e}")
            return
        here = self.current_frame()
        self.viewer.layers["contours"].data = self.session.label.transpose(3, 2, 1, 0).copy()
        self._after_labels_changed(keep_frame=here)
        self.roi.refresh()
        self.autosave.session_opened()  # the restored state is the new baseline (the file stays until you save)
        self.lbl_auto.setText("Unsaved changes restored. Save to keep them.")

    def show_compare(self) -> None:
        """ED | ES side by side in place of the image (Ctrl+E shows and hides it); it follows the slice, edits
        and contrast."""
        if self.session is None:
            self.lbl_auto.setText("Open a scan first.")
            return
        if getattr(self, "_compare", None) is None:
            from .compare import CompareWindow

            self._compare = CompareWindow(self)
            self._compare.show()
        else:
            self._compare.toggle()

    def _compare_changed(self, *_) -> None:
        c = getattr(self, "_compare", None)
        if c is not None:
            c.schedule()

    def about_text(self) -> tuple[str, str]:
        from .. import __version__
        from .. import autocontour as ac

        head = f"TraceCMR {__version__}: free, open-source analysis of mouse cardiac MRI (MIT licence)."
        body = ("Auto LV uses the mouse heart segmentation model published by Shah, Stuckey, Steeden et al. at "
                "UCL, unchanged (converted to run without TensorFlow), under the authors' licence (GPL-3.0). If you "
                "use Auto LV, please cite their paper:\n\n" + ac.UCL_CITATION + "\n\nModel, code and data: "
                + ac.UCL_SOURCE + "\n\nThe UCL team's open mouse dataset was also used to test TraceCMR. "
                "TraceCMR is built on napari and uses the Inter typeface (SIL Open Font Licence).\n\n"
                "https://github.com/jairam09/TraceCMR")
        return head, body

    def show_about(self) -> None:
        head, body = self.about_text()
        box = self._dark_box("About TraceCMR", head, body)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()

    def _report_problem(self) -> None:
        from .support import report_problem

        path = report_problem(self)
        self.lbl_auto.setText(f"Problem report saved: {path.name} (copied to the clipboard).")

    def redo(self):
        """Redo the last undone contour edit (Ctrl+Y or Ctrl+Shift+Z)."""
        if self.editor is not None and self.rb_curves.isChecked() and self.editor.redo():
            self.lbl_auto.setText("Redone.")
        else:
            self.lbl_auto.setText("Nothing to redo.")

    def clear_slice(self):
        """The trash can: delete both contours on the slice and frame being viewed."""
        if self.editor is None or not self.rb_curves.isChecked():
            return
        t, z = (int(v) for v in self.viewer.dims.current_step[:2])
        n = self.editor.clear_slice(t, z)
        self.lbl_auto.setText(f"Cleared slice {z}, frame {t}. Ctrl+Z undoes." if n
                              else "No contours on this slice and frame.")

    def _edit_points(self):  # older name: show the points
        self.btn_points.setChecked(True)

    def _smooth_toggled(self, on: bool):
        if self.session is None:
            return
        self._sync_labels()
        self.session.set_smooth(on)
        self._after_labels_changed()

    def _after_labels_changed(self, keep_frame: int | None = None):
        """Push session labels to the viewer and refresh everything that depends on them."""
        s = self.session
        self.viewer.layers["contours"].data = s.label.transpose(3, 2, 1, 0).copy()
        for spin, v in ((self.spin_ed, s.ed_frame), (self.spin_es, s.es_frame)):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        self.lbl_conv.setText(CONVENTION_TEXT.get(s.papillary, CONVENTION_TEXT[None]))
        self._rebuild_list()
        self._update_frame_label()
        self._draw_volume_curve()
        if self.rb_curves.isChecked():
            self._mode_changed()
        self._show_frame(s.ed_frame if keep_frame is None else keep_frame)
        self._refresh_slice_table()
        if self.study:
            self.series_panel.refresh(self.study, self.current_series)

    def _refresh_slice_table(self):
        """Per-slice volumes, EF and mass (Segment style), plus the total over the selected slices."""
        s = self.session
        if s is None:
            return
        self._sync_labels()
        rows = s.per_slice()
        self.box_slices.setVisible(bool(rows))
        if not rows:
            return
        sel_ed = [r for r in rows if r["slice"] in s.included["ED"]]
        sel_es = [r for r in rows if r["slice"] in s.included["ES"]]
        edv, esv = sum(r["edv_ul"] for r in sel_ed), sum(r["esv_ul"] for r in sel_es)
        total = {"slice": "Total", "edv_ul": edv, "esv_ul": esv,
                 "ef_pct": 100 * (edv - esv) / edv if edv else float("nan"),
                 "mass_ed_mg": sum(r["mass_ed_mg"] for r in sel_ed), "included": True}
        self.tbl_slices.setRowCount(len(rows) + 1)
        grey, warn, bold = QColor("#6B7178"), QColor("#E3A23B"), QFont()
        bold.setBold(True)
        self._table_slices = [r["slice"] for r in rows]
        for i, r in enumerate(rows + [total]):
            z = r["slice"]
            unset = r.get("es_not_set", False)
            label = str(z)
            vals = [label, f"{r['edv_ul']:.1f}", "ES not set" if unset else f"{r['esv_ul']:.1f}",
                    "–" if unset or not np.isfinite(r["ef_pct"]) else f"{r['ef_pct']:.0f}", f"{r['mass_ed_mg']:.1f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c > 0:  # numbers right-aligned so decimals line up
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if not r["included"]:
                    it.setForeground(grey)
                elif unset and c in (0, 2, 3):
                    it.setForeground(warn)
                    it.setToolTip("ES equals ED on this slice. Play its cine and press Set ES here on the "
                                  "smallest frame.")
                if i == len(rows):
                    it.setFont(bold)
                self.tbl_slices.setItem(i, c, it)
        head = self.tbl_slices.horizontalHeader().sizeHint().height()
        self.tbl_slices.setFixedHeight(min(head + 4 + 26 * (len(rows) + 1), 340))
        self._select_table_row()
        if not getattr(self, "_computing", False):  # keep the results live, like Segment
            self.compute(quiet=True)

    def _slice_row_clicked(self, row: int, _col: int) -> None:
        """Click a per-slice row to view that slice."""
        zs = getattr(self, "_table_slices", [])
        if 0 <= row < len(zs):
            step = list(self.viewer.dims.current_step)
            step[1] = zs[row]
            self.viewer.dims.current_step = tuple(step)

    def _select_table_row(self) -> None:
        """Highlight the per-slice row of the slice being viewed."""
        zs = getattr(self, "_table_slices", [])
        z = self.current_slice()
        self.tbl_slices.blockSignals(True)
        if z in zs:
            self.tbl_slices.selectRow(zs.index(z))
        else:
            self.tbl_slices.clearSelection()
        self.tbl_slices.blockSignals(False)

    def _draw_volume_curve(self):
        s = self.session
        self.vfig.clear()
        v = s.volume_curve() if s is not None else []
        drawn = [i for i, x in enumerate(v) if x > 0]
        show = len(drawn) >= 3  # only worth a plot once several frames are contoured
        self.vcanvas.setVisible(show)
        self.lbl_vcurve.setVisible(show)
        if not show:
            self.vcanvas.draw()
            return
        ax = self.vfig.add_axes((0.14, 0.24, 0.84, 0.7))
        ax.plot(drawn, [v[i] for i in drawn], color="#9AA1A8", lw=1.5)
        ax.plot([s.ed_frame], [v[s.ed_frame]], "o", color="#E58A94")
        ax.plot([s.es_frame], [v[s.es_frame]], "o", color="#7FB2E5")
        ax.set_xlabel("frame", fontsize=8, color="#9AA1A8")
        ax.set_ylabel("µL", fontsize=8, color="#9AA1A8")
        ax.tick_params(labelsize=7, colors="#9AA1A8")
        for sp in ax.spines.values():
            sp.set_color("#4A5866")
        ax.set_facecolor("none")
        self.vcanvas.draw()

    # ---------- compute ----------

    def compute(self, quiet: bool = False):
        """Volumes, EF, mass and wall thickness. Runs by itself after every edit once ED and ES both have
        contours (quiet=True: no warning while the contours are still incomplete)."""
        if self.session is None:
            return
        self._sync_labels()
        s = self.session
        try:
            fn, wall, notes = s.compute()
        except Exception as e:
            if quiet:  # still incomplete: say what is missing, without a warning colour
                for lab in self.res_labels.values():
                    lab.setText("–")
                self.lbl_notes.setStyleSheet(NOTE_STYLE)
                self.lbl_notes.setText(f"Results appear once ED and ES both have contours. Missing: {e}")
                return
            self.lbl_notes.setStyleSheet(WARN_STYLE)
            self.lbl_notes.setText(f"⚠ {e}")
            return
        self._computing = True
        self.last = (fn, wall)
        row = fn.as_row()
        for key, lab in self.res_labels.items():
            v = row[key]
            lab.setText(f"{v:.1f}" if np.isfinite(v) else "–")
        rv = s.rv_volumes()
        self.last_rv = rv
        self.box_rv.setVisible(rv is not None)
        for key, lab in self.rv_labels.items():
            lab.setText(f"{rv[key]:.1f}" if rv else "–")
        self._bullseye_wall = wall
        self._draw_bullseyes()
        self._compare_changed()
        self._run_checks()
        self._refresh_slice_table()
        self._computing = False
        if self._wall_on():
            self.draw_wall()
        all_notes = notes + fn.flags + (wall.flags if wall else [])
        if self._checks:
            per = {}
            for (t, z), why in self._checks.items():
                ph = "ED" if t == s.t_for(z, "ED") else "ES"
                per.setdefault(z, []).append(ph)
            listed = ", ".join(f"{z} ({'/'.join(sorted(set(phs)))})" for z, phs in
                               sorted(per.items(), key=lambda kv: s.order.index(kv[0]) if kv[0] in s.order else 0))
            all_notes.insert(0, f"Double-check slice(s) {listed}: marked ⚠ in the slice list (hover for why). "
                                "Unmarked slices are less likely, not certain, to be right.")
        missing = s.es_not_set()
        self.lbl_warn.setText(f"⚠ ES not set on slice(s) {', '.join(map(str, missing))}, so EF reads low. On each, "
                              "play the cine and press Set ES here at the smallest frame." if missing else "")
        self.lbl_warn.setVisible(bool(missing))
        self.lbl_notes.setStyleSheet(NOTE_STYLE)
        self.lbl_notes.setText("\n".join(f"• {n}" for n in all_notes))

    def _bullseye_key(self, wall, w: int, h: int):
        """What the bullseyes depend on: the numbers, the layout and the panel size (to the nearest 8 px, so
        the odd pixel of difference between a hidden and a shown page is not a reason to redraw)."""
        if wall is None:
            return None
        parts = [str(wall.ed.reference), str(sorted((lv, np.asarray(r.segments_mm).tobytes())
                                                    for lv, r in wall.ed.levels.items())),
                 str(sorted((lv, np.asarray(v).tobytes()) for lv, v in wall.thickening_pct.items())),
                 str(self.THICKENING_RANGE)]
        return (w // 8, h // 8, hash("".join(parts)))

    def _draw_bullseyes(self, event=None) -> None:
        """The two bullseyes, laid out for the panel's current size: square, side by side, colour bar underneath,
        numbers sized to fit (hidden when there's no room). Redrawn when the panel is resized.

        Building them costs about a fifth of a second (two polar plots, 32 segments), so they are rebuilt only
        when the numbers or the size actually change: picking the Wall tab then shows what is already drawn."""
        wall = getattr(self, "_bullseye_wall", None)
        key = self._bullseye_key(wall, self.canvas.width() or 400, self.canvas.height() or 230)
        if key is not None and key == getattr(self, "_bullseye_drawn", None) and self.fig.axes:
            return
        self._bullseye_drawn = key
        self.fig.clear()
        if wall is not None:
            w, h = self.canvas.width() or 400, self.canvas.height() or 230
            side_px = max(40.0, min(w / 2 * 0.80, h * 0.60))  # each bullseye is a square this big
            fs = float(np.clip(side_px / 24.0, 0, 9))
            fs = fs if fs >= 4.5 else 0
            sw, sh = side_px / w, side_px / h
            ed_vals = {lv: r.segments_mm for lv, r in wall.ed.levels.items()}
            aha = wall.ed.reference.startswith("septum")
            for k, (vals, title, lo, hi, cmap, fmt) in enumerate((
                    (ed_vals, "ED wall (mm)", 0.4, 1.6, WALL_CMAP_NAME, "{:.2f}"),
                    (wall.thickening_pct, "Thickening (%)", *self.THICKENING_RANGE, "RdYlGn", "{:.0f}"))):
                cx = 0.25 + 0.5 * k
                bottom = 0.20  # room underneath for the colour bar and its numbers; above for the title
                draw_bullseye(self.fig, vals, "", "", lo, hi, cmap=cmap, fmt=fmt, aha=aha, fontsize=fs,
                              rect=(cx - sw / 2, bottom, sw, sh), cbar="horizontal",
                              cax_rect=(cx - sw * 0.4, 0.115, sw * 0.8, 0.03))
                self.fig.text(cx, 0.985, title, ha="center", va="top", color="#9AA1A8",
                              fontsize=float(np.clip(side_px / 16.0, 7, 11)))
            for ax in self.fig.axes:
                ax.title.set_color("#9AA1A8")
                ax.tick_params(colors="#9AA1A8")
        self.canvas.draw_idle()

    def export_images(self, quiet: bool = False, out=None) -> list[Path]:
        """Raw | contoured PNGs (each contoured slice at ED and ES) and montages into <mouse results>/images."""
        from .export_images import export_images

        if self.session is None:
            return []
        self._sync_labels()
        if not self.session.label.any():
            if not quiet:
                self.lbl_auto.setText("No contours to export yet.")
            return []
        out = Path(out) if out else self._images_dir(self._save_target())
        files = export_images(self.session, out)
        if not quiet:
            self.lbl_auto.setText(f"Wrote {len(files)} images to {out}.")
        return files

    # ---------- brightness / contrast (Segment style) ----------

    def _build_contrast_popup(self) -> None:
        """Brightness and contrast: a floating card over the image (drag it by its title, Esc closes)."""
        from qtpy.QtWidgets import QSlider

        from .hints import set_hint
        from .popover import Popover, keycap

        pop = Popover(self.viewer.window._qt_window if self._has_window() else self, "Brightness and contrast")
        self.contrast_popup = pop
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        self.sld_brightness, self.sld_contrast = QSlider(Qt.Orientation.Horizontal), QSlider(Qt.Orientation.Horizontal)
        self._contrast_values = []
        for r, (name, sld) in enumerate((("Brightness", self.sld_brightness), ("Contrast", self.sld_contrast))):
            sld.setRange(-100, 100)
            sld.setValue(0)
            sld.setMinimumWidth(220)
            sld.valueChanged.connect(self._apply_contrast)
            val = QLabel("0")
            val.setObjectName("value")
            val.setFixedWidth(34)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            sld.valueChanged.connect(lambda v, lbl=val: lbl.setText(f"{v:+d}" if v else "0"))
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(sld, r, 1)
            grid.addWidget(val, r, 2)
        pop.body.addLayout(grid)
        row = QHBoxLayout()
        b_auto, b_reset = QPushButton("Auto for this slice"), QPushButton("Reset")
        set_hint(b_auto, "Auto", "Stretch the contrast to this slice and frame.")
        set_hint(b_reset, "Reset", "Back to the default for the whole scan.")
        b_auto.clicked.connect(self._auto_contrast)
        b_reset.clicked.connect(lambda: self._reset_contrast())
        row.addWidget(b_auto)
        row.addWidget(b_reset)
        row.addStretch(1)
        pop.body.addLayout(row)
        tip = QHBoxLayout()
        tip.setSpacing(6)
        tip.addWidget(keycap("Ctrl"))
        note = QLabel("+ drag on the image:  ↔ contrast,  ↕ brightness")
        note.setObjectName("note")
        tip.addWidget(note)
        tip.addStretch(1)
        pop.body.addLayout(tip)
        self._contrast_base = None

    def _has_window(self) -> bool:
        try:
            return self.viewer.window._qt_window is not None
        except Exception:
            return False

    def _show_contrast_popup(self) -> None:
        anchor = None
        try:
            anchor = self.viewer.window._qt_viewer.canvas.native
        except Exception:
            pass
        self.contrast_popup.show_near(anchor)

    def _reset_contrast(self, base=None) -> None:
        """Back to the default window: 0.5th to 99.5th percentile of the whole scan."""
        if self.session is None or "image" not in self.viewer.layers:
            return
        img = self.session.image
        lo, hi = (float(v) for v in np.percentile(img[::2, ::2], [0.5, 99.5])) if base is None else base
        if hi <= lo:
            hi = lo + 1.0
        self._contrast_base = (lo, hi)
        layer = self.viewer.layers["image"]
        span = hi - lo
        layer.contrast_limits_range = (float(img.min()) - 3 * span, float(img.max()) + 3 * span)
        for sld in (self.sld_brightness, self.sld_contrast):
            sld.blockSignals(True)
            sld.setValue(0)
            sld.blockSignals(False)
        self._apply_contrast()

    def _auto_contrast(self) -> None:
        if self.session is None:
            return
        t, z = self.current_frame(), self.current_slice()
        self._reset_contrast(tuple(float(v) for v in np.percentile(self.session.image[:, :, z, t], [1, 99.5])))

    def _apply_contrast(self, *args) -> None:
        if self._contrast_base is None or "image" not in self.viewer.layers:
            return
        lo, hi = self._contrast_base
        width0, centre0 = hi - lo, (hi + lo) / 2
        width = width0 * 2 ** (-self.sld_contrast.value() / 40)  # +100 = about 6x more contrast
        centre = centre0 - self.sld_brightness.value() / 100 * width0
        self.viewer.layers["image"].contrast_limits = (centre - width / 2, centre + width / 2)

    def adjust_contrast(self, d_contrast: int, d_brightness: int) -> None:
        """Nudge the sliders (used by Ctrl + drag on the image)."""
        self.sld_contrast.setValue(self.sld_contrast.value() + d_contrast)
        self.sld_brightness.setValue(self.sld_brightness.value() + d_brightness)

    def _pan_toggled(self, on: bool) -> None:
        if self.editor is not None:
            self.editor.pan_mode = on
        self.lbl_auto.setText("Pan on: drag moves the image. Space to edit again." if on else "")

    WALL_RANGE_MM = (0.4, 1.6)  # colour scale of the wall lines, same as the ED wall bullseye

    def start_set_septum(self) -> None:
        """Wall menu: the next click on the image sets the septum (the middle of where the RV meets the LV)."""
        if self.session is None or self.editor is None:
            return
        if not self.btn_wall.isChecked():
            self.btn_wall.setChecked(True)
        self.editor.click_hook = self._septum_clicked
        self.lbl_auto.setText("Click the middle of the septum (the wall shared with the RV). Esc cancels.")

    def _roi_toggled(self, on: bool) -> None:
        if on and self.session is None:
            self.btn_roi.setChecked(False)
            return
        self.roi.set_active(on)

    def _escape(self) -> None:
        """Esc: leave the ED | ES view, else stop drawing regions, else cancel a pending click."""
        c = getattr(self, "_compare", None)
        if c is not None and c.is_open():
            c.hide()
        elif self.btn_roi.isChecked():
            self.btn_roi.setChecked(False)
        else:
            self.cancel_click()

    def _delete_key(self) -> None:
        """Delete / Backspace: remove the selected region in ROI mode, else clear the contour on this slice."""
        if self.btn_roi.isChecked():
            self.roi.remove_selected()
        elif self.btn_clear.isEnabled():
            self.clear_slice()

    def cancel_click(self) -> None:
        if self.editor is not None and self.editor.click_hook is not None:
            self.editor.click_hook = None
            self.lbl_auto.setText("Cancelled.")

    def _septum_clicked(self, t: int, z: int, p) -> None:
        from scipy import ndimage as _ndi

        s = self.session
        self.editor.click_hook = None  # one click only
        self._sync_labels()
        blood = s.label[:, :, z, t] == 1
        if not blood.any():
            blood = s.label[:, :, z, s.t_for(z, "ED")] == 1
        if not blood.any():
            self.lbl_auto.setText(f"Slice {z} has no endocardium: draw it, then set the septum.")
            return
        cx, cy = _ndi.center_of_mass(blood)
        dx, dy, _ = s.spacing
        s.septum_manual[z] = float(np.rad2deg(np.arctan2((p[1] - cy) * dy, (p[0] - cx) * dx)) % 360.0)
        self.compute(quiet=True)
        self.draw_wall()
        self.lbl_auto.setText(f"Septum set by hand on slice {z} (magenta). Reset septum restores the automatic one.")

    def reset_septum(self, all_slices: bool) -> None:
        s = self.session
        if s is None:
            return
        z = self.current_slice()
        if all_slices:
            n = len(s.septum_manual)
            s.septum_manual.clear()
        else:
            n = int(s.septum_manual.pop(z, None) is not None)
        self.compute(quiet=True)
        self.draw_wall()
        self.lbl_auto.setText(("Septum back to automatic" + (" on all slices." if all_slices else f" on slice {z}."))
                              if n else "No septum was set by hand" + ("." if all_slices else " on this slice."))

    def _wall_on(self) -> bool:
        try:
            return self.btn_wall.isChecked()
        except RuntimeError:  # the window is closing
            return False

    THICKENING_RANGE = (-40.0, 80.0)  # colour scale of thickening (%): yellow ~20%, healthy (40%+) green, thinning red; same as the bullseye
    WALL_SECTORS = 40  # thickening is compared ED vs ES in sectors of 9 degrees around the slice

    def set_wall_mode(self, mode: str) -> None:
        """'thickness' or 'thickening': what the Wall lines show (and turns Wall on)."""
        self.wall_mode = mode
        self.act_wall[mode].setChecked(True)
        rb = getattr(self, "rb_wall_thickness" if mode == "thickness" else "rb_wall_thickening", None)
        if rb is not None and not rb.isChecked():
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)
        if self.btn_wall.isChecked():
            self.draw_wall()
        else:
            self.btn_wall.setChecked(True)

    def _wall_lines(self, st, t, z, values, lo, hi, cmap_name):
        from matplotlib import colormaps

        dx, dy, _ = self.session.spacing
        cmap = colormaps[cmap_name]
        lines, colours = [], []
        for k in range(0, len(st.chords_mm), 3):  # every 3rd: about 40 lines
            v = values[k]
            if not np.isfinite(v):
                continue
            a, b = st.chords_mm[k]
            lines.append(np.array([[t, z, a[1] / dy, a[0] / dx], [t, z, b[1] / dy, b[0] / dx]]))
            colours.append(cmap(float(np.clip((v - lo) / (hi - lo), 0, 1))))
        return lines, colours

    def draw_wall(self) -> None:
        """Wall button: draw thickness lines (endo to epi, through the centerline) on the slice and frame being viewed."""
        from ..wall import slice_thickness

        name = "wall thickness"
        layer = self.viewer.layers[name] if name in self.viewer.layers else None
        if self.session is None or not self._wall_on():
            if layer is not None:
                layer.visible = False
            return
        if getattr(self, "_drawing_wall", False):
            return
        self._drawing_wall = True
        try:
            s = self.session
            t, z = (int(v) for v in self.viewer.dims.current_step[:2])
            if not (0 <= t < s.label.shape[3] and 0 <= z < s.label.shape[2]) or "image" not in self.viewer.layers:
                return  # mid-way through opening a mouse: nothing to draw yet
            self._sync_labels()
            dx, dy, dz = s.spacing
            msg = "Wall: no endo and epi on this slice and frame."
            lines, colours, widths = [], [], []
            if getattr(self, "wall_mode", "thickness") == "thickening":
                t_ed, t_es = s.t_for(z, "ED"), s.t_for(z, "ES")
                st_ed = slice_thickness(s.label[:, :, z, t_ed], (dx, dy))
                st_es = slice_thickness(s.label[:, :, z, t_es], (dx, dy)) if t_es != t_ed else None
                if st_ed is None or st_es is None or st_ed.chords_mm is None or st_es.chords_mm is None:
                    msg = (f"Thickening needs ED (frame {t_ed}) and ES (frame {t_es}) contours on slice {z}."
                           if t_es != t_ed else f"Thickening: ES not set on slice {z}.")
                else:
                    n = self.WALL_SECTORS
                    edges = np.linspace(0, 360, n + 1)

                    def sector_means(st):
                        idx = np.clip(np.digitize(st.angles_deg, edges) - 1, 0, n - 1)
                        return np.array([np.nanmean(st.thickness_mm[idx == k]) if (idx == k).any() else np.nan
                                         for k in range(n)]), idx

                    with np.errstate(all="ignore"):
                        m_ed, _ = sector_means(st_ed)
                        m_es, _ = sector_means(st_es)
                        pct = 100.0 * (m_es - m_ed) / m_ed
                    show = st_es if t == t_es else st_ed  # draw on the frame being viewed
                    _, idx = sector_means(show)
                    lo, hi = self.THICKENING_RANGE
                    lines, colours = self._wall_lines(show, t, z, pct[idx], lo, hi, "RdYlGn")
                    ok = np.isfinite(pct)
                    thin = int((pct[ok] < 0).sum())
                    msg = (f"Thickening, slice {z}: mean {np.nanmean(pct):.0f}% (wall {np.nanmean(m_ed):.2f} → "
                           f"{np.nanmean(m_es):.2f} mm). Green ≥40%, yellow ≤20%, red thins"
                           + (f"; thinning in {thin} of {ok.sum()} sectors." if thin else "."))
                    if t not in (t_ed, t_es):
                        msg += " Shown on the ED outline."
            else:
                st = slice_thickness(s.label[:, :, z, t], (dx, dy))
                if st is not None and st.chords_mm is not None:
                    lo, hi = self.WALL_RANGE_MM
                    lines, colours = self._wall_lines(st, t, z, st.thickness_mm, lo, hi, WALL_CMAP_NAME)  # same scale as the ED bullseye
                    th = st.thickness_mm[np.isfinite(st.thickness_mm)]
                    msg = (f"Wall, slice {z}: mean {th.mean():.2f} mm (range {th.min():.2f}–{th.max():.2f}). "
                           f"Blue thin → red thick ({lo}–{hi} mm).")
            from ..wall import septal_angle

            by_hand = z in s.septum_manual
            if by_hand:
                sep = (s.septum_manual[z], ())  # set by hand: the septum only, no insertion points
            else:
                sep = septal_angle(s.label[:, :, z, t], (dx, dy))
                if sep is None and t != s.t_for(z, "ED"):
                    sep = septal_angle(s.label[:, :, z, s.t_for(z, "ED")], (dx, dy))  # RV usually traced at ED
            blood = s.label[:, :, z, t] == 1
            if sep is not None and blood.any():
                from scipy import ndimage as _ndi

                cx, cy = _ndi.center_of_mass(blood)
                lvm = np.isin(s.label[:, :, z, t], (1, 2))

                def exit_px(mask, a):
                    """Distance (pixels) from the centroid to where a ray at angle a leaves the mask."""
                    ar = np.deg2rad(a)
                    r = np.arange(0, 3 * max(mask.shape), 0.25)
                    x, y = cx + r * np.cos(ar) / dx * min(dx, dy), cy + r * np.sin(ar) / dy * min(dx, dy)
                    ok = (x >= 0) & (y >= 0) & (x < mask.shape[0] - 1) & (y < mask.shape[1] - 1)
                    v = _ndi.map_coordinates(mask.astype(float), [x[ok], y[ok]], order=1) >= 0.5
                    out = np.nonzero(~v)[0]
                    return float(r[ok][out[0]]) if len(out) else float(r[ok][-1])

                def point(a, rr):
                    ar = np.deg2rad(a)
                    f = min(dx, dy)
                    return cy + rr * np.sin(ar) / dy * f, cx + rr * np.cos(ar) / dx * f  # (row y, col x)

                tick = 0.35 / min(dx, dy)  # half-length of the insertion ticks: 0.35 mm
                for k, a in enumerate((sep[0], *sep[1])):  # septum, then the two RV insertion points
                    r_epi = exit_px(lvm, a)
                    if k == 0:  # from the endocardium to a little past the epicardium
                        a0, a1 = point(a, exit_px(blood, a)), point(a, r_epi + tick)
                    else:  # a short tick across the epicardium
                        a0, a1 = point(a, r_epi - tick), point(a, r_epi + tick)
                    lines.append(np.array([[t, z, *a0], [t, z, *a1]]))
                    colours.append((1.0, 0.0, 1.0, 1.0))  # magenta: no contour or scale uses it
                    widths.append(1.1 if k == 0 else 0.9)
                msg += (" Magenta line: septum, set by hand." if by_hand else
                        " Magenta: septum (from the RV); short lines: RV insertion points.")
            elif blood.any():
                msg += " No septum here (no RV): Wall menu → Set the septum."
            layer = self._wall_layer()
            layer.data = []
            if lines:
                widths = [0.6] * (len(lines) - len(widths)) + widths  # wall lines thin, septum markers thicker
                layer.add(lines, shape_type="line", edge_color=colours, edge_width=widths)
            layer.visible = True
            layer.editable = False
            self.lbl_auto.setText(msg)
        finally:
            self._drawing_wall = False

    def _wall_layer(self):
        """The layer the wall lines are drawn into, created the first time it is needed. Making it costs about
        a quarter of a second, so `_prewarm_wall` does it quietly after a mouse is opened rather than leaving
        it for the first click on Wall."""
        name = "wall thickness"
        if name in self.viewer.layers:
            return self.viewer.layers[name]
        dx, dy, dz = self.session.spacing
        prev = self.viewer.layers.selection.active  # adding a layer selects it: give the selection back
        layer = self.viewer.add_shapes(name=name, ndim=4, scale=(1, dz, dy, dx))
        layer.visible = False
        layer.editable = False
        if prev is not None:
            self.viewer.layers.selection.active = prev
        if self.editor is not None and self.editor.points is not None:  # keep the points on top
            i = self.viewer.layers.index(layer)
            self.viewer.layers.move(i, self.viewer.layers.index(self.editor.points))
        return layer

    def _prewarm_wall(self) -> None:
        """Get the slow one-off work for the Wall tab out of the way while the mouse is being opened."""
        if self.session is None or not self._has_window():
            return
        try:
            self._wall_layer()
        except Exception:  # never let this stop a mouse from opening
            pass

    def show_raw(self, on: bool) -> None:
        """Hide every contour (on) to look at the raw image, or bring them back. Nothing is deleted."""
        if self.session is None:
            return
        # Only visibility changes: nothing is rebuilt, so the contours come back the instant H is released.
        curves_mode = self.rb_curves.isChecked() and self.editor is not None
        if on:
            for name in ("contours", "curves", "control points"):
                if name in self.viewer.layers:
                    self.viewer.layers[name].visible = False
            if self.editor is not None:
                self.editor.set_active(False)
            self.lbl_auto.setText("Contours hidden (H to show).")
        else:
            if curves_mode:
                self.editor.set_active(True)
                if self.roi.active:  # drawing regions: keep napari from panning on left-drag
                    self.roi.set_active(True)
            elif "contours" in self.viewer.layers:
                self.viewer.layers["contours"].visible = True
            self.lbl_auto.setText("")

    def _export(self):
        if not self.last:
            return
        d = self.results_dir()
        d.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Export results", str(d / f"{d.name.split('__')[0]}_results.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        self._write_csv(path)
        self.lbl_notes.setText(self.lbl_notes.text() + f"\n• Saved {Path(path).name}")

    def study_table_path(self) -> Path:
        return self.contours_dir() / "study_summary.xlsx"

    def export_all(self, quiet: bool = False) -> dict | None:
        """One click: save, then the mouse's workbook, bullseyes and slice images in its results folder, and its row
        in the study table. Returns what was written (None if nothing could be)."""
        from . import report
        from ..viz import save_bullseye

        if self.session is None:
            return None
        self.compute(quiet=True)
        if not self.last:
            self.lbl_auto.setText("Export needs ED and ES contours first.")
            return None
        path = self.save_contours(quiet=True)
        if path is None:
            return None
        fn, wall = self.last
        s = self.session
        try:
            _, _, notes = s.compute()
        except Exception:
            notes = []
        rep = report.build(s, fn, wall, getattr(self, "last_rv", None), notes)
        stem = self._stem(path)
        out = {"folder": path.parent}
        try:
            out["workbook"] = report.write_mouse(path.parent / f"{stem}_TraceCMR_results.xlsx", rep)
            if s.rois:
                out["rois"] = self.roi.export_csv(path.parent / f"{stem}_ROIs.csv")
            from ..drift import phase_timing, write_csv as write_timing

            timing = phase_timing(s)
            if len(timing.slices) >= 2:  # contoured on enough frames to compare global and per-slice ED/ES
                out["timing"] = write_timing(timing, path.parent / f"{stem}_ED_ES_timing.csv", s.subject_id)
            img_dir = self._images_dir(path)
            img_dir.mkdir(parents=True, exist_ok=True)
            if wall is not None:
                aha = wall.ed.reference.startswith("septum")
                save_bullseye({lv: r.segments_mm for lv, r in wall.ed.levels.items()}, img_dir / "bullseye_ED_wall.png",
                              "ED wall thickness (mm)", "mm", 0.4, 1.6, cmap=WALL_CMAP_NAME, aha=aha)
                save_bullseye(wall.thickening_pct, img_dir / "bullseye_thickening.png", "Systolic thickening (%)",
                              "%", *self.THICKENING_RANGE, cmap="RdYlGn", fmt="{:.0f}", aha=aha)
            out["images"] = len(list(img_dir.glob("*.png")))
        except Exception as e:
            self.lbl_auto.setText(f"⚠ Export failed: {e}")
            if not quiet and not getattr(self, "_testing", False):
                from qtpy.QtWidgets import QMessageBox

                QMessageBox.warning(self, "Export failed", f"The export stopped with this error:\n\n{e}\n\n"
                                    "Your contours were saved.")
            return None
        try:
            out["study"] = report.update_study(self.study_table_path(), rep["study"], key=str(path.parent.name or path))
        except PermissionError:
            out["study"] = None
            self.lbl_auto.setText(f"Exported. {self.study_table_path().name} is open in Excel: close it and export "
                                  "again to add this mouse's row.")
        if out.get("study") is not None:
            xl = Path(out["workbook"]).suffix == ".xlsx"
            self.lbl_auto.setText(f"Exported {Path(out['workbook']).name}, {out['images']} images and the study row "
                                  f"in {Path(out['study']).name}." + ("" if xl else " (CSV: run TraceCMR update for "
                                                                     "Excel files.)"))
        self._last_export = out
        if not quiet and not getattr(self, "_testing", False):
            from qtpy.QtWidgets import QMessageBox

            box = QMessageBox(self)
            box.setWindowTitle("Exported")
            box.setText(f"Everything for {s.subject_id} is in\n{path.parent}\n\n• {Path(out['workbook']).name}\n"
                        f"• {out['images']} images in images{os.sep}\n"
                        + (f"• its row in {Path(out['study']).name} ({Path(out['study']).parent.name})"
                           if out.get("study") else "• study table not updated (open in Excel?)"))
            opened = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Ok)
            box.exec()
            if box.clickedButton() is opened:
                self._open_saved_folder()
        return out

    def _write_csv(self, path):
        fn, wall = self.last
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["measure", "value"])
            for k, v in fn.as_row().items():
                w.writerow([k, v])
            for k, v in (getattr(self, "last_rv", None) or {}).items():
                w.writerow([k, v])
            w.writerow(["slice_order_base_to_apex", " ".join(str(z) for z in self.session.order)])
            for ph in ("ED", "ES"):
                left = sorted(set(self.session.order) - self.session.included[ph])
                w.writerow([f"slices_left_out_{ph}", " ".join(str(z) for z in left)])
            if wall:
                w.writerow([])
                w.writerow(["level", "segment", "edwt_mm", "eswt_mm", "thickening_pct"])
                for r in wall.rows():
                    w.writerow([r["level"], r["segment"], r["edwt_mm"], r["eswt_mm"], r["thickening_pct"]])
            rows = self.session.per_slice()
            if rows:
                w.writerow([])
                w.writerow(["slice", "edv_ul", "esv_ul", "ef_pct", "mass_ed_mg", "included"])
                for r in rows:
                    w.writerow([r["slice"], r["edv_ul"], r["esv_ul"], r["ef_pct"], r["mass_ed_mg"], r["included"]])

    def _set_enabled(self, on: bool):
        for wdg in (self.spin_ed, self.spin_es, self.btn_show_ed, self.btn_show_es, self.btn_auto_frames,
                    self.btn_play, self.btn_set_ed, self.btn_set_es,
                    self.btn_flip, self.btn_auto_order, self.list, self.btn_suggest, self.btn_compute, self.btn_export,
                    self.rb_curves, self.rb_pixels, self.cmb_add, self.btn_auto, self.chk_smooth, self.btn_smooth, self.btn_undo, self.btn_redo, self.btn_points, self.btn_snap, self.btn_fit, self.btn_grow, self.btn_shrink, self.btn_clear,
                    self.btn_endo, self.btn_epi, self.btn_rv, self.btn_brush, self.btn_zoom_in, self.btn_zoom_out,
                    self.btn_zoom_fit, self.chk_per_slice, self.btn_raw, self.btn_wall, self.btn_contrast, self.btn_pan, self.btn_export_img,
                    self.btn_refine, self.btn_view, self.btn_roi):
            wdg.setEnabled(on)
