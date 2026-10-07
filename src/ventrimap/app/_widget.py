"""VentriMap panel for napari: open a scan, set slice order, edit contours, compute.

The panel docks beside the napari viewer. napari shows the images and the
contours (as an editable labels layer: brush, eraser, fill), and the panel
holds the VentriMap steps from top to bottom.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QColor, QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
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
    QComboBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.segmentation import find_boundaries

from .. import labels as L
from ..viz import draw_bullseye
from .curve_editor import CurveEditor, curves_from_labels
from .session import Session

BLOOD_RGB = (230, 75, 75)
MYO_RGB = (60, 200, 110)
WARN_STYLE = "color: #E3A23B;"
NOTE_STYLE = "color: #9AA1A8;"

CONVENTION_TEXT = {
    "myocardium": "Detailed: papillary muscles counted as muscle",
    "blood_pool": "Smooth: papillary muscles in the blood pool",
    None: "Unknown convention",
}


def _thumbnail(img2d: np.ndarray, lab2d: np.ndarray, size: int = 72) -> QPixmap:
    """Slice thumbnail with blood pool and myocardium outlines (x right, y down)."""
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
    lo, hi = np.percentile(img, [1, 99]) if img.size else (0, 1)
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (np.dstack([g, g, g]) * 255).astype(np.uint8)
    for value, colour in ((L.LV_BLOODPOOL, BLOOD_RGB), (L.LV_MYOCARDIUM, MYO_RGB)):
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


class VentriMapWidget(QWidget):
    """Dock widget: Open, Frames, Slice order, Compute, Results."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.session: Session | None = None
        self.last = None  # (FunctionResult, WallResult | None)
        self.editor: CurveEditor | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        self.main = QVBoxLayout(body)
        self.main.setSpacing(10)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        title = QLabel("<b style='font-size:15px'>ventrimap</b> &nbsp;<span style='color:#9AA1A8'>v0 prototype</span>")
        self.main.addWidget(title)

        # 1. Open
        box = QGroupBox("1  Open a scan")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.btn_ucl = QPushButton("UCL file (.h5)…")
        self.btn_dcm = QPushButton("DICOM folder…")
        row.addWidget(self.btn_ucl)
        row.addWidget(self.btn_dcm)
        lay.addLayout(row)
        self.lbl_scan = QLabel("No scan open.")
        self.lbl_scan.setWordWrap(True)
        self.lbl_scan.setStyleSheet(NOTE_STYLE)
        lay.addWidget(self.lbl_scan)
        self.main.addWidget(box)

        # 2. Frames
        box = QGroupBox("2  End-diastole and end-systole")
        grid = QGridLayout(box)
        self.spin_ed, self.spin_es = QSpinBox(), QSpinBox()
        grid.addWidget(QLabel("ED frame"), 0, 0)
        grid.addWidget(self.spin_ed, 0, 1)
        self.btn_show_ed = QPushButton("Show")
        grid.addWidget(self.btn_show_ed, 0, 2)
        grid.addWidget(QLabel("ES frame"), 1, 0)
        grid.addWidget(self.spin_es, 1, 1)
        self.btn_show_es = QPushButton("Show")
        grid.addWidget(self.btn_show_es, 1, 2)
        self.btn_auto_frames = QPushButton("Detect from contours")
        grid.addWidget(self.btn_auto_frames, 2, 0, 1, 3)
        self.main.addWidget(box)

        # 3. Slice order
        box = QGroupBox("3  Slice order  (base at top, apex at bottom)")
        lay = QVBoxLayout(box)
        hint = QLabel("Drag to reorder. Tick ED / ES to choose which slices count in each phase "
                      "(suggested automatically; orange = please check). Click a slice to view it.")
        hint.setWordWrap(True)
        hint.setStyleSheet(NOTE_STYLE)
        lay.addWidget(hint)
        self.list = QTreeWidget()
        self.list.setColumnCount(4)
        self.list.setHeaderLabels(["Slice", "ED", "ES", "Why"])
        self.list.setRootIsDecorated(False)
        self.list.setIconSize(QSize(44, 44))
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setMinimumHeight(320)
        self.list.setColumnWidth(0, 120)
        self.list.setColumnWidth(1, 36)
        self.list.setColumnWidth(2, 36)
        lay.addWidget(self.list)
        self.btn_suggest = QPushButton("Re-apply suggested slices")
        lay.addWidget(self.btn_suggest)
        row = QHBoxLayout()
        self.btn_flip = QPushButton("Flip order")
        self.btn_auto_order = QPushButton("Auto-order")
        row.addWidget(self.btn_flip)
        row.addWidget(self.btn_auto_order)
        lay.addLayout(row)
        self.lbl_order = QLabel("")
        self.lbl_order.setWordWrap(True)
        lay.addWidget(self.lbl_order)
        self.main.addWidget(box)

        # 4. Contours
        box = QGroupBox("4  Contours")
        lay = QVBoxLayout(box)
        self.lbl_conv = QLabel("")
        self.lbl_conv.setWordWrap(True)
        lay.addWidget(self.lbl_conv)
        row = QHBoxLayout()
        self.rb_curves = QRadioButton("Curves (drag points)")
        self.rb_pixels = QRadioButton("Pixel brush")
        self.rb_curves.setChecked(True)
        row.addWidget(self.rb_curves)
        row.addWidget(self.rb_pixels)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("New points go on"))
        self.cmb_add = QComboBox()
        self.cmb_add.addItems(["endocardium (red)", "epicardium (green)"])
        row.addWidget(self.cmb_add)
        lay.addLayout(row)
        tip = QLabel("Curves: drag a white point to reshape the contour. To add a point, press 2 (add mode) and "
                     "click near the contour; to delete, select it and press Delete. Pixel brush: paint in the "
                     "'contours' layer (1 = blood pool, 2 = myocardium).")
        tip.setWordWrap(True)
        tip.setStyleSheet(NOTE_STYLE)
        lay.addWidget(tip)
        self.main.addWidget(box)

        # 5. Compute + results
        self.btn_compute = QPushButton("Compute")
        self.btn_compute.setMinimumHeight(34)
        self.btn_compute.setStyleSheet("font-weight: 600;")
        self.main.addWidget(self.btn_compute)

        box = QGroupBox("Results")
        lay = QVBoxLayout(box)
        self.grid_res = QGridLayout()
        self.res_labels = {}
        for i, (key, name) in enumerate([("edv_ul", "EDV (µL)"), ("esv_ul", "ESV (µL)"), ("sv_ul", "SV (µL)"),
                                         ("ef_pct", "EF (%)"), ("mass_ed_mg", "Mass ED (mg)"),
                                         ("mass_es_mg", "Mass ES (mg)")]):
            cap = QLabel(name)
            cap.setStyleSheet(NOTE_STYLE)
            val = QLabel("–")
            val.setStyleSheet("font-size: 16px; font-weight: 600;")
            self.grid_res.addWidget(cap, (i // 3) * 2, i % 3)
            self.grid_res.addWidget(val, (i // 3) * 2 + 1, i % 3)
            self.res_labels[key] = val
        lay.addLayout(self.grid_res)
        self.fig = Figure(figsize=(4.4, 2.3), facecolor="none")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(250)
        self.canvas.setStyleSheet("background: transparent;")
        lay.addWidget(self.canvas)
        self.lbl_notes = QLabel("")
        self.lbl_notes.setWordWrap(True)
        lay.addWidget(self.lbl_notes)
        self.btn_export = QPushButton("Export CSV…")
        lay.addWidget(self.btn_export)
        self.main.addWidget(box)
        self.main.addStretch(1)

        # wiring
        self.btn_ucl.clicked.connect(self._open_ucl)
        self.btn_dcm.clicked.connect(self._open_dicom)
        self.btn_show_ed.clicked.connect(lambda: self._show_frame(self.spin_ed.value()))
        self.btn_show_es.clicked.connect(lambda: self._show_frame(self.spin_es.value()))
        self.spin_ed.valueChanged.connect(self._frames_changed)
        self.spin_es.valueChanged.connect(self._frames_changed)
        self.btn_auto_frames.clicked.connect(self._auto_frames)
        self.btn_flip.clicked.connect(self._flip)
        self.btn_auto_order.clicked.connect(self._auto_order)
        self.list.model().rowsInserted.connect(lambda *a: QTimer.singleShot(0, self._order_from_list))
        self.list.itemChanged.connect(self._item_changed)
        self.list.currentItemChanged.connect(self._goto_slice)
        self.btn_suggest.clicked.connect(self._apply_suggestions)
        self.btn_compute.clicked.connect(self.compute)
        self.rb_curves.toggled.connect(self._mode_changed)
        self.cmb_add.currentIndexChanged.connect(self._add_target_changed)
        self.btn_export.clicked.connect(self._export)
        self._set_enabled(False)

    # ---------- opening ----------

    def _open_ucl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open UCL mouse file", "", "UCL HDF5 (*.h5)")
        if path:
            self.load_session(Session.from_ucl(path))

    def _open_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM cine folder")
        if folder:
            self.load_session(Session.from_dicom(folder))

    def load_session(self, session: Session, message: str = "") -> None:
        """Show a session in the viewer and fill the panel."""
        self.session = s = session
        for name in ("image", "contours"):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        dx, dy, dz = s.spacing
        # napari axes: (time, slice, y, x); our arrays are (x, y, slice, time)
        self.viewer.add_image(s.image.transpose(3, 2, 1, 0), name="image", scale=(1, dz, dy, dx),
                              colormap="gray")
        lab = self.viewer.add_labels(s.label.transpose(3, 2, 1, 0), name="contours", scale=(1, dz, dy, dx),
                                     opacity=0.9)
        lab.contour = 1  # outlines, like Segment, rather than filled regions
        try:
            from napari.utils.colormaps import DirectLabelColormap

            lab.colormap = DirectLabelColormap(color_dict={
                None: (0, 0, 0, 0), 0: (0, 0, 0, 0),
                L.LV_BLOODPOOL: (*(c / 255 for c in BLOOD_RGB), 1.0),
                L.LV_MYOCARDIUM: (*(c / 255 for c in MYO_RGB), 1.0),
            })
        except Exception:  # colours are cosmetic; older napari keeps its defaults
            pass
        self.viewer.dims.axis_labels = ("time", "slice", "y", "x")
        nt = s.image.shape[3]
        for spin in (self.spin_ed, self.spin_es):
            spin.blockSignals(True)
            spin.setRange(0, nt - 1)
            spin.blockSignals(False)
        self.spin_ed.setValue(s.ed_frame)
        self.spin_es.setValue(s.es_frame)
        self.lbl_scan.setText(
            f"<b>{s.subject_id}</b> · {s.image.shape[2]} slices · {nt} frames · "
            f"pixel {dx:.3g} × {dy:.3g} mm · slices {dz:.3g} mm apart"
        )
        self.lbl_conv.setText(CONVENTION_TEXT.get(s.papillary, CONVENTION_TEXT[None]))
        self._set_enabled(True)
        self.editor = CurveEditor(self.viewer, scale=(1, dz, dy, dx))
        self._rebuild_list()
        self._show_frame(s.ed_frame)
        self.viewer.reset_view()
        self._centre_on_heart()
        self._mode_changed()
        if message:
            self.lbl_order.setText(message)

    def _centre_on_heart(self):
        """Zoom the view onto the heart (centroid of the ED contours), as in Segment."""
        s = self.session
        lab = s.label[:, :, :, s.ed_frame]
        xs, ys, zs = np.nonzero(lab)
        if not len(xs):
            return
        dx, dy, _ = s.spacing
        step = list(self.viewer.dims.current_step)
        step[1] = int(np.median(zs))
        self.viewer.dims.current_step = tuple(step)
        cam = self.viewer.camera
        centre = list(cam.center)
        centre[-2], centre[-1] = float(ys.mean() * dy), float(xs.mean() * dx)
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
            self.session.label = self.editor.apply_to(self.session.label)
            self.editor.edited.clear()
            layer.data = self.session.label.transpose(3, 2, 1, 0).copy()

    def _mode_changed(self, *args):
        """Switch between curve editing and the pixel brush, carrying edits across."""
        if self.session is None or self.editor is None:
            return
        curves_mode = self.rb_curves.isChecked()
        self._sync_labels()  # carry edits from whichever mode was active
        if curves_mode:
            frames = sorted({self.session.ed_frame, self.session.es_frame})
            self.editor.show(curves_from_labels(self.session.label, frames))
            self.editor.add_to = "endo" if self.cmb_add.currentIndex() == 0 else "epi"
        self.viewer.layers["contours"].visible = not curves_mode
        for name in ("curves", "control points"):
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = curves_mode
        self.viewer.layers.selection.active = (self.viewer.layers["control points"] if curves_mode
                                               else self.viewer.layers["contours"])
        self.cmb_add.setEnabled(curves_mode)

    def _add_target_changed(self, idx):
        if self.editor is not None:
            self.editor.add_to = "endo" if idx == 0 else "epi"

    def _show_frame(self, t: int):
        step = list(self.viewer.dims.current_step)
        step[0] = t
        self.viewer.dims.current_step = tuple(step)

    def _frames_changed(self):
        if self.session is None:
            return
        self._sync_labels()
        self.session.ed_frame = self.spin_ed.value()
        self.session.es_frame = self.spin_es.value()
        self._rebuild_list()
        if self.rb_curves.isChecked():
            self._mode_changed()

    def _auto_frames(self):
        self._sync_labels()
        msg = self.session.auto_ed_es()
        self.spin_ed.setValue(self.session.ed_frame)
        self.spin_es.setValue(self.session.es_frame)
        self.lbl_order.setText(msg)

    # ---------- slice order ----------

    def _rebuild_list(self):
        s = self.session
        self.list.blockSignals(True)
        self.list.clear()
        t = s.ed_frame
        for z in s.order:
            item = QTreeWidgetItem(["", "", "", ""])
            item.setIcon(0, QIcon(_thumbnail(s.image[:, :, z, t], s.label[:, :, z, t])))
            item.setData(0, Qt.ItemDataRole.UserRole, z)
            item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
                          & ~Qt.ItemFlag.ItemIsDropEnabled)
            for col, ph in ((1, "ED"), (2, "ES")):
                item.setCheckState(col, Qt.CheckState.Checked if z in s.included[ph] else Qt.CheckState.Unchecked)
            self.list.addTopLevelItem(item)
        self.list.blockSignals(False)
        self._relabel_items()
        self._show_warnings()

    def _items(self):
        return [self.list.topLevelItem(i) for i in range(self.list.topLevelItemCount())]

    def _relabel_items(self):
        s = self.session
        self.list.blockSignals(True)
        items = self._items()
        zs = [it.data(0, Qt.ItemDataRole.UserRole) for it in items]
        drawn = [z for z in zs if z in s.included["ED"] and np.any(s.label[:, :, z, s.ed_frame])]
        for it, z in zip(items, zs):
            tag = " · base" if drawn and z == drawn[0] else " · apex" if drawn and z == drawn[-1] else ""
            it.setText(0, f"{z}{tag}")
            r_ed, r_es = (s.reasons[ph].get(z, "") for ph in ("ED", "ES"))
            r_ed, r_es = ("" if r.startswith("wall covers") else r for r in (r_ed, r_es))
            if r_ed == r_es:
                why = [r_ed] if r_ed else []
            else:
                why = [f"{ph}: {r}" for ph, r in (("ED", r_ed), ("ES", r_es)) if r]
            it.setText(3, "; ".join(why))
            flagged = any("check" in w for w in why)
            it.setForeground(3, QColor("#E3A23B") if flagged else QColor("#9AA1A8"))
            it.setToolTip(3, it.text(3))
        self.list.blockSignals(False)

    def _order_from_list(self, *args):
        zs = [it.data(0, Qt.ItemDataRole.UserRole) for it in self._items()]
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

    # ---------- compute ----------

    def compute(self):
        self._sync_labels()
        s = self.session
        try:
            fn, wall, notes = s.compute()
        except Exception as e:
            self.lbl_notes.setStyleSheet(WARN_STYLE)
            self.lbl_notes.setText(f"⚠ {e}")
            return
        self.last = (fn, wall)
        row = fn.as_row()
        for key, lab in self.res_labels.items():
            v = row[key]
            lab.setText(f"{v:.1f}" if np.isfinite(v) else "–")
        self.fig.clear()
        if wall is not None:
            ed_vals = {lv: r.segments_mm for lv, r in wall.ed.levels.items()}
            draw_bullseye(self.fig, ed_vals, "ED wall (mm)", "", 0.4, 1.6, rect=(0.02, 0.02, 0.40, 0.74))
            draw_bullseye(self.fig, wall.thickening_pct, "Thickening (%)", "", -20, 100, cmap="RdYlGn",
                          fmt="{:.0f}", rect=(0.52, 0.02, 0.40, 0.74))
            for ax in self.fig.axes:
                ax.title.set_color("#9AA1A8")
                ax.tick_params(colors="#9AA1A8", labelsize=7)
        self.canvas.draw()
        all_notes = notes + fn.flags + (wall.flags if wall else [])
        self.lbl_notes.setStyleSheet(NOTE_STYLE)
        self.lbl_notes.setText("\n".join(f"• {n}" for n in all_notes))

    def _export(self):
        if not self.last:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", f"{self.session.subject_id}_ventrimap.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        fn, wall = self.last
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["measure", "value"])
            for k, v in fn.as_row().items():
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
        self.lbl_notes.setText(self.lbl_notes.text() + f"\n• Saved {Path(path).name}")

    def _set_enabled(self, on: bool):
        for wdg in (self.spin_ed, self.spin_es, self.btn_show_ed, self.btn_show_es, self.btn_auto_frames,
                    self.btn_flip, self.btn_auto_order, self.list, self.btn_suggest, self.btn_compute, self.btn_export,
                    self.rb_curves, self.rb_pixels, self.cmb_add):
            wdg.setEnabled(on)
