"""Regions of interest (ROI): draw a circle or a freehand outline on the image and read what is inside it (area,
mean signal, spread, range), then compare two regions: contrast-to-noise and signal-to-noise.

Typical use: one ROI in scar (bright on LGE), one in healthy "remote" muscle, one in the air outside the body
for the noise.

    CNR = |mean A − mean B| / noise SD          SNR = mean A / noise SD

ROIs are stored with the mouse (Session.rois) as plain dicts, in image pixels (x, y), so they reload with its
contours:

    {"name": "ROI 1", "kind": "circle", "t": 0, "z": 3, "cx": 40.2, "cy": 51.0, "r_mm": 0.3}
    {"name": "scar", "kind": "freehand", "t": 0, "z": 3, "pts": [[x, y], ...]}
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# ROI outline colours: none of them is used by a contour (red endo, light blue epi, yellow RV) or the septum
# (magenta)
ROI_COLOURS = ["#FF9F40", "#A78BFA", "#34D399", "#F472B6", "#A3E635", "#22D3EE"]  # no white: it vanishes on bright blood
MIN_RADIUS_PX = 1.0
MIN_POINTS = 3


# ---------------------------------------------------------------- geometry and numbers (no Qt)

def roi_mask(roi: dict, shape: tuple[int, int], spacing: tuple[float, float]) -> np.ndarray:
    """Pixels whose centres fall inside the ROI, on an (x, y) image of `shape` with pixel size `spacing` (mm)."""
    dx, dy = spacing
    xs, ys = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    if roi["kind"] == "circle":  # a circle in millimetres (an ellipse in pixels if they are not square)
        return ((xs - roi["cx"]) * dx) ** 2 + ((ys - roi["cy"]) * dy) ** 2 <= roi["r_mm"] ** 2
    if roi["kind"] == "line":  # the pixels along a ruler line (its signal is a line profile)
        from skimage.draw import line as _line

        (x0, y0), (x1, y1) = np.rint(np.asarray(roi["pts"], dtype=float)[[0, -1]]).astype(int)
        rr, cc = _line(x0, y0, x1, y1)
        ok = (rr >= 0) & (cc >= 0) & (rr < shape[0]) & (cc < shape[1])
        m = np.zeros(shape, bool)
        m[rr[ok], cc[ok]] = True
        return m
    from matplotlib.path import Path as _Path

    pts = np.asarray(roi["pts"], dtype=float)
    if len(pts) < MIN_POINTS:
        return np.zeros(shape, bool)
    inside = _Path(pts).contains_points(np.column_stack([xs.ravel(), ys.ravel()]))
    return inside.reshape(shape)


def roi_stats(image2d: np.ndarray, mask: np.ndarray, spacing: tuple[float, float]) -> dict:
    """Area (mm²), pixel count, mean, SD (sample), min and max of the image inside the mask."""
    v = np.asarray(image2d, dtype=float)[mask]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "area_mm2": 0.0, "mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan}
    return {"n": n, "area_mm2": float(n * spacing[0] * spacing[1]), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if n > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}


def length_mm(roi: dict, spacing: tuple[float, float]) -> float:
    """A ruler's length in millimetres (0 for regions)."""
    if roi["kind"] != "line":
        return 0.0
    (x0, y0), (x1, y1) = np.asarray(roi["pts"], dtype=float)[[0, -1]]
    return float(np.hypot((x1 - x0) * spacing[0], (y1 - y0) * spacing[1]))


def cnr(mean_a: float, mean_b: float, noise_sd: float) -> float:
    """Contrast-to-noise ratio: how far apart two regions are, in units of the image noise."""
    if not noise_sd or not np.isfinite(noise_sd):
        return float("nan")
    return abs(mean_a - mean_b) / noise_sd


def snr(mean: float, noise_sd: float) -> float:
    if not noise_sd or not np.isfinite(noise_sd):
        return float("nan")
    return mean / noise_sd


def outline(roi: dict, spacing: tuple[float, float], n: int = 64) -> np.ndarray:
    """The ROI's outline as (n, 2) image pixels (x, y), for drawing."""
    if roi["kind"] == "circle":
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([roi["cx"] + roi["r_mm"] / spacing[0] * np.cos(a),
                                roi["cy"] + roi["r_mm"] / spacing[1] * np.sin(a)])
    return np.asarray(roi["pts"], dtype=float)


def fmt(v: float, digits: int | None = None) -> str:
    """A number for the table: 3 significant digits for small values (image units vary from 0–1 to thousands),
    whole numbers with thousands separators for large ones."""
    if v is None or not np.isfinite(v):
        return "–"
    a = abs(v)
    if digits is None:
        digits = 0 if a >= 100 else 1 if a >= 10 else 2 if a >= 1 else 3
    if a >= 1000:
        return f"{v:,.0f}"
    text = f"{v:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text  # no "-0.000"


# ---------------------------------------------------------------- the tool (Qt)

class RoiTool:
    """The ROI button's behaviour and the Regions box in the Results panel. Owned by the TraceCMR panel."""

    LAYER = "rois"

    def __init__(self, panel):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (QAbstractItemView, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
                                    QHeaderView, QLabel, QPushButton, QTableWidget, QVBoxLayout)

        self.panel = panel
        self.viewer = panel.viewer
        self.kind = "circle"
        self.active = False
        self._attached = False
        self._drawing = None  # the ROI being drawn: dict
        self._stats: list[dict] = []

        self.box = QGroupBox("Regions of interest")
        lay = QVBoxLayout(self.box)
        self.hint = QLabel("Draw a region with ROI (R) to measure area and signal.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #9AA1A9;")
        lay.addWidget(self.hint)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Size", "Mean", "SD"])
        self.table.horizontalHeaderItem(1).setToolTip("Area in mm² (regions) or length in mm (rulers)")
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setMinimumSectionSize(36)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setToolTip("Double-click a name to rename it (e.g. scar, remote, air). Click a row to view "
                              "its slice.")
        self.table.itemChanged.connect(self._renamed)
        self.table.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # for renaming; a plain click hands the keys back
        self.table.cellClicked.connect(self._row_clicked)
        self.table.cellDoubleClicked.connect(lambda r, c: c == 0 and self.table.editItem(self.table.item(r, 0)))
        lay.addWidget(self.table)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        self.cmb_a, self.cmb_b, self.cmb_noise = QComboBox(), QComboBox(), QComboBox()
        for c in (self.cmb_a, self.cmb_b, self.cmb_noise):
            c.currentIndexChanged.connect(self._update_ratios)
        self._cmp_labels = [QLabel("Compare"), QLabel("with"), QLabel("Noise from")]
        for r, (lbl, cmb) in enumerate(zip(self._cmp_labels, (self.cmb_a, self.cmb_b, self.cmb_noise))):
            grid.addWidget(lbl, r, 0)
            grid.addWidget(cmb, r, 1)
        grid.setColumnStretch(1, 1)
        self.cmb_noise.setToolTip("Noise = SD of this region, ideally air outside the body. Air noise reads "
                                  "low in magnitude images; some papers divide it by 0.655.")
        lay.addLayout(grid)
        self.lbl_ratio = QLabel("")
        self.lbl_ratio.setStyleSheet("font-weight: 600;")
        self.lbl_ratio.setWordWrap(True)
        self.lbl_ratio.setToolTip("CNR = |mean A − mean B| ÷ noise SD. Above 3–5: clearly different.\nSNR = "
                                  "mean ÷ noise SD.")
        lay.addWidget(self.lbl_ratio)
        row = QHBoxLayout()
        self.btn_remove = QPushButton("Remove")
        self.btn_clear = QPushButton("Clear all")
        self.btn_csv = QPushButton("Export CSV")
        for b in (self.btn_remove, self.btn_clear, self.btn_csv):
            row.addWidget(b)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_csv.clicked.connect(lambda: self.export_csv())
        lay.addLayout(row)
        self.refresh()
        self.viewer.dims.events.current_step.connect(lambda e=None: self.redraw())

    # ---------- state ----------

    @property
    def rois(self) -> list:
        s = self.panel.session
        return s.rois if s is not None else []

    def _spacing(self):
        s = self.panel.session
        return float(s.spacing[0]), float(s.spacing[1])

    def _here(self):
        t, z = (int(v) for v in self.viewer.dims.current_step[:2])
        return t, z

    def set_kind(self, kind: str) -> None:
        self.kind = kind
        if self.active:
            self._say_how()

    def _say_how(self) -> None:
        how = {"circle": "drag from the centre out for a circle", "line": "drag to measure a distance"}.get(
            self.kind, "drag around the region to outline it")
        self.panel.lbl_auto.setText(f"ROI: {how}. Esc stops; Delete removes the selected region.")

    def set_active(self, on: bool) -> None:
        """ROI mode: left-drag draws regions (contour editing waits until it is off)."""
        on = bool(on) and self.panel.session is not None
        self.active = on
        ed = self.panel.editor
        if ed is not None:
            ed.suspended = on
        if on and self.panel.btn_brush.isChecked():
            self.panel.btn_brush.setChecked(False)
        for layer in self.viewer.layers:  # napari must not pan on left-drag while drawing
            if hasattr(layer, "mouse_pan"):
                layer.mouse_pan = False if on else not (ed is not None and ed.active)
        if on and not self._attached:
            self.viewer.mouse_drag_callbacks.append(self._on_mouse)
            self._attached = True
        elif not on and self._attached:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._on_mouse)
            except ValueError:
                pass
            self._attached = False
        if on:
            self._say_how()
            self.reveal()
        else:
            self.panel.lbl_auto.setText("")
        self.redraw()

    def reveal(self) -> None:
        """Bring the regions into view: open the Results panel if it is hidden, on the Regions tab."""
        btn = getattr(self.panel, "btn_right", None)
        if btn is not None and btn.isEnabled() and not btn.isChecked():
            btn.setChecked(True)
        tabs = getattr(self.panel, "results_tabs", None)
        if tabs is not None:
            tabs.show("regions")

    # ---------- drawing ----------

    def _to_px(self, world) -> tuple[float, float]:
        s = self.panel.session
        dx, dy = float(s.spacing[0]), float(s.spacing[1])
        return float(world[-1]) / dx, float(world[-2]) / dy

    def _on_mouse(self, viewer, event):
        if not self.active or self.panel.session is None or event.button != 1 or len(event.position) < 4:
            return
        t, z = self._here()
        x0, y0 = self._to_px(event.position)
        dx, dy = self._spacing()
        if self.kind == "circle":
            roi = {"kind": "circle", "t": t, "z": z, "cx": x0, "cy": y0, "r_mm": 0.0}
        elif self.kind == "line":
            roi = {"kind": "line", "t": t, "z": z, "pts": [[x0, y0], [x0, y0]]}
        else:
            roi = {"kind": "freehand", "t": t, "z": z, "pts": [[x0, y0]]}
        self._drawing = roi
        yield
        while event.type == "mouse_move":
            x, y = self._to_px(event.position)
            if roi["kind"] == "circle":
                roi["r_mm"] = float(np.hypot((x - x0) * dx, (y - y0) * dy))
            elif roi["kind"] == "line":
                roi["pts"][1] = [x, y]
            else:
                last = roi["pts"][-1]
                if np.hypot(x - last[0], y - last[1]) >= 0.4:  # a point every ~half pixel
                    roi["pts"].append([x, y])
            self.redraw()
            yield
        self._drawing = None
        if roi["kind"] == "circle":
            big_enough = roi["r_mm"] >= MIN_RADIUS_PX * min(dx, dy)
        elif roi["kind"] == "line":
            big_enough = length_mm(roi, (dx, dy)) >= MIN_RADIUS_PX * min(dx, dy)
        else:
            big_enough = len(roi["pts"]) >= MIN_POINTS
        if big_enough:
            self.add(roi)
        else:
            self.redraw()

    def add(self, roi: dict) -> dict:
        roi = dict(roi)
        used = {r.get("name") for r in self.rois}
        prefix = "Ruler" if roi["kind"] == "line" else "ROI"
        k = 1
        while f"{prefix} {k}" in used:
            k += 1
        roi.setdefault("name", f"{prefix} {k}")
        roi["colour"] = ROI_COLOURS[(len(self.rois)) % len(ROI_COLOURS)]
        self.rois.append(roi)
        self.refresh()
        self.table.selectRow(len(self.rois) - 1)
        self.reveal()
        st = self._stats[-1]
        if roi["kind"] == "line":
            self.panel.lbl_auto.setText(f"{roi['name']}: {st['length_mm']:.2f} mm (mean signal along it "
                                        f"{fmt(st['mean'])}).")
        else:
            self.panel.lbl_auto.setText(f"{roi['name']}: {fmt(st['area_mm2'], 2)} mm², mean {fmt(st['mean'])}, "
                                        f"SD {fmt(st['sd'])} ({st['n']} px). Double-click the name to rename.")
        return roi

    def redraw(self) -> None:
        """Show the ROIs of the slice and frame being viewed (and the one being drawn)."""
        s = self.panel.session
        layer = self.viewer.layers[self.LAYER] if self.LAYER in self.viewer.layers else None
        if s is None:
            if layer is not None:
                layer.data = []
            return
        t, z = self._here()
        here = [r for r in self.rois if r["t"] == t and r["z"] == z]
        if self._drawing is not None:
            here = here + [dict(self._drawing, colour="#FFFFFF", name="")]
        shapes, colours, names, kinds = [], [], [], []
        sp = self._spacing()
        for r in here:
            if r["kind"] == "circle" and r["r_mm"] <= 0:
                continue
            if r["kind"] == "freehand" and len(r["pts"]) < 2:
                continue
            if r["kind"] == "line" and length_mm(r, sp) <= 0:
                continue
            o = outline(r, sp)
            shapes.append(np.column_stack([np.full(len(o), t), np.full(len(o), z), o[:, 1], o[:, 0]]))
            colours.append(r.get("colour", "#FFFFFF"))
            label = r.get("name", "")
            if r["kind"] == "line":
                label = f"{label}  {length_mm(r, sp):.2f} mm".strip()
            names.append(label)
            kinds.append(r["kind"])
        if layer is None:
            if not shapes:
                return
            dx, dy, dz = (float(v) for v in s.spacing)
            prev = self.viewer.layers.selection.active  # adding a layer selects it: give the selection back
            layer = self.viewer.add_shapes(name=self.LAYER, ndim=4, scale=(1, dz, dy, dx))
            if prev is not None:
                self.viewer.layers.selection.active = prev
            layer.editable = False
        layer.data = []
        if shapes:
            drawing_free = self._drawing is not None and self._drawing["kind"] == "freehand"
            types = ["path" if k == "line" or (drawing_free and i == len(shapes) - 1) else "polygon"
                     for i, k in enumerate(kinds)]
            layer.add(shapes, shape_type=types, edge_color=colours, face_color="transparent", edge_width=0.6)
            try:
                layer.features = {"name": names, "colour": colours}
                layer.text = {"string": "{name}", "color": {"feature": "colour"}, "size": 10,
                              "anchor": "upper_left"}
            except Exception:  # labels are a nicety
                pass
        layer.visible = True

    # ---------- table and ratios ----------

    def _compute(self) -> list[dict]:
        s = self.panel.session
        out = []
        if s is None:
            return out
        sp = self._spacing()
        for r in self.rois:
            img = s.image[:, :, r["z"], r["t"]]
            st = roi_stats(img, roi_mask(r, img.shape, sp), sp)
            st["length_mm"] = length_mm(r, sp)
            if r["kind"] == "line":
                st["area_mm2"] = 0.0
            out.append(st)
        return out

    def refresh(self) -> None:
        """Recompute every ROI's numbers and refill the table and the comparison."""
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QColor, QPixmap, QIcon
        from qtpy.QtWidgets import QTableWidgetItem

        self._stats = self._compute()
        rois = self.rois
        self.table.blockSignals(True)
        self.table.setRowCount(len(rois))
        for i, (r, st) in enumerate(zip(rois, self._stats)):
            pm = QPixmap(10, 10)
            pm.fill(QColor(r.get("colour", "#FFFFFF")))
            name = QTableWidgetItem(QIcon(pm), r.get("name", f"ROI {i + 1}"))
            name.setFlags(name.flags() | Qt.ItemFlag.ItemIsEditable)
            tip = (f"{r['kind'].capitalize()} on slice {r['z']}, frame {r['t']} · {st['n']} pixels · "
                   f"range {fmt(st['min'])} – {fmt(st['max'])}")
            name.setToolTip(tip)
            self.table.setItem(i, 0, name)
            size = f"{st['length_mm']:.2f} mm" if r["kind"] == "line" else fmt(st["area_mm2"], 2)
            for j, text in enumerate([size, fmt(st["mean"]), fmt(st["sd"])], start=1):
                it = QTableWidgetItem(text)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                it.setToolTip(tip)
                self.table.setItem(i, j, it)
        self.table.blockSignals(False)
        row_h = self.table.verticalHeader().defaultSectionSize()
        self.table.setFixedHeight(min(6, max(len(rois), 1)) * row_h + self.table.horizontalHeader().sizeHint().height() + 4)
        have = bool(rois)
        self.table.setVisible(have)
        for w in (self.btn_remove, self.btn_clear, self.btn_csv):
            w.setEnabled(have)
        self.hint.setVisible(not have)
        self._fill_combos()
        self.redraw()

    def _fill_combos(self) -> None:
        self._areas = [i for i, r in enumerate(self.rois) if r["kind"] != "line"]  # rulers aren't compared
        names = [self.rois[i].get("name", f"ROI {i + 1}") for i in self._areas]
        had = self.cmb_a.count()
        keep = [c.currentText() for c in (self.cmb_a, self.cmb_b, self.cmb_noise)] if had >= 2 else ["", "", ""]
        for c in (self.cmb_a, self.cmb_b, self.cmb_noise):
            c.blockSignals(True)
            c.clear()
        self.cmb_a.addItems(names)
        self.cmb_b.addItems(names)
        self.cmb_noise.addItems([f"SD inside {n}" for n in names])
        defaults = (0, 1, 2 if len(names) > 2 else 1)
        for c, old, d in zip((self.cmb_a, self.cmb_b, self.cmb_noise), keep, defaults):
            i = c.findText(old)
            c.setCurrentIndex(i if i >= 0 else min(d, c.count() - 1))
            c.blockSignals(False)
        show = len(names) >= 2
        for w in (self.cmb_a, self.cmb_b, self.cmb_noise, self.lbl_ratio, *self._cmp_labels):
            w.setVisible(show)
        self._update_ratios()

    def ratios(self) -> dict | None:
        """CNR of A vs B and SNR of A and B, with the noise SD of the chosen region; None with < 2 ROIs."""
        areas = getattr(self, "_areas", [])
        if len(areas) < 2:
            return None
        a, b, n = self.cmb_a.currentIndex(), self.cmb_b.currentIndex(), self.cmb_noise.currentIndex()
        if min(a, b, n) < 0:
            return None
        a, b, n = areas[a], areas[b], areas[n]  # combo positions → ROI numbers
        sa, sb, sn = self._stats[a], self._stats[b], self._stats[n]
        return {"a": a, "b": b, "noise": n, "noise_sd": sn["sd"], "cnr": cnr(sa["mean"], sb["mean"], sn["sd"]),
                "snr_a": snr(sa["mean"], sn["sd"]), "snr_b": snr(sb["mean"], sn["sd"])}

    def _update_ratios(self) -> None:
        r = self.ratios()
        if r is None:
            self.lbl_ratio.setText("")
            return
        na, nb = self.cmb_a.currentText(), self.cmb_b.currentText()
        if r["a"] == r["b"]:
            self.lbl_ratio.setText("Pick two different regions to compare.")
            return
        self.lbl_ratio.setText(f"CNR {fmt(r['cnr'])}   ·   SNR {na} {fmt(r['snr_a'])}, {nb} {fmt(r['snr_b'])}")

    def _renamed(self, item) -> None:
        if item.column() != 0 or not (0 <= item.row() < len(self.rois)):
            return
        new = item.text().strip() or f"ROI {item.row() + 1}"
        self.rois[item.row()]["name"] = new
        self._fill_combos()
        self.redraw()
        self._give_keys_back()

    def _give_keys_back(self) -> None:
        """After a click in the table, letter keys go back to being shortcuts (not a search in the table)."""
        try:
            self.viewer.window._qt_viewer.canvas.native.setFocus()
        except Exception:
            pass

    def _row_clicked(self, row: int, _col: int) -> None:
        self._give_keys_back()
        if 0 <= row < len(self.rois):
            r = self.rois[row]
            step = list(self.viewer.dims.current_step)
            if (step[0], step[1]) != (r["t"], r["z"]):
                step[0], step[1] = r["t"], r["z"]
                self.viewer.dims.current_step = tuple(step)

    # ---------- removing and exporting ----------

    def remove_selected(self) -> bool:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:  # nothing picked: the newest region on this slice
            t, z = self._here()
            here = [i for i, r in enumerate(self.rois) if (r["t"], r["z"]) == (t, z)]
            rows = here[-1:]
        if not rows:
            return False
        for i in reversed(rows):
            del self.rois[i]
        self.refresh()
        return True

    def clear_all(self) -> None:
        self.rois.clear()
        self.refresh()

    def rows(self) -> list[dict]:
        """One row per ROI for files: name, slice, frame, area, pixels, mean, SD, min, max."""
        out = []
        for r, st in zip(self.rois, self._stats):
            out.append({"name": r.get("name", ""), "shape": r["kind"], "slice": r["z"], "frame": r["t"],
                        "area_mm2": st["area_mm2"], "length_mm": st.get("length_mm", 0.0), "pixels": st["n"], "mean": st["mean"], "sd": st["sd"],
                        "min": st["min"], "max": st["max"]})
        return out

    def export_csv(self, path: str | Path | None = None) -> Path | None:
        """Write the ROI table (and the CNR/SNR shown) to a CSV in the mouse's results folder."""
        if not self.rois:
            return None
        if path is None:
            folder = self.panel.results_dir()
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{folder.name.split('__')[0]}_ROIs.csv"
        path = Path(path)
        rows = self.rows()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
            r = self.ratios()
            if r is not None and r["a"] != r["b"]:
                f.write("\n")
                f.write(f"CNR ({self.cmb_a.currentText()} vs {self.cmb_b.currentText()}, noise = SD of "
                        f"{self.rois[r['noise']].get('name')}),{r['cnr']:.3f}\n")
                f.write(f"SNR {self.cmb_a.currentText()},{r['snr_a']:.3f}\n")
                f.write(f"SNR {self.cmb_b.currentText()},{r['snr_b']:.3f}\n")
        self.panel.lbl_auto.setText(f"ROI measurements written to {path}")
        return path

    def session_loaded(self) -> None:
        """A new mouse: its saved ROIs (if any) come back; ROI mode ends."""
        if self.active:
            self.set_active(False)
            self.panel.btn_roi.setChecked(False)
        self._drawing = None
        self.refresh()
