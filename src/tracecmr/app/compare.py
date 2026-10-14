"""ED | ES side by side: the slice being viewed at end-diastole and end-systole, next to each other, with the
contours. Ctrl+E swaps the image for the pair, filling the image area; Ctrl+E or Esc brings the image back.
It follows the slice, edits and brightness/contrast live, and saves publication figures (one slice, or every
slice as a grid) as PNG (300 dpi), SVG or PDF.

Read-only: editing stays in the main view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import labels as L

COLOURS = {"endo": "#D71A1C", "epi": "#4FC3F7", "rv": "#FFD600"}
FIGURE_DPI = 300


# ---------------------------------------------------------------- drawing (no Qt): reused by the window and export

def slice_contours(panel, t: int, z: int) -> dict[str, np.ndarray]:
    """The contours on one slice and frame as (n, 2) pixel arrays (x, y): the editor's curves where it has
    them (exactly what's on screen, edits included), else traced from the labels."""
    from ..contours import spline_through

    out = {}
    ed = panel.editor
    if ed is not None:
        for name in ("endo", "epi", "rv"):
            ctrl = ed.curves.get((t, z, name))
            if ctrl is not None and len(ctrl) >= 3:
                out[name] = spline_through(np.asarray(ctrl, dtype=float), 300)
    if out:
        return out
    from skimage import measure

    lab = panel.session.label[:, :, z, t]
    masks = {"endo": lab == L.LV_BLOODPOOL,
             "epi": np.isin(lab, [L.LV_BLOODPOOL, L.LV_MYOCARDIUM, L.SCAR, L.MVO]),
             "rv": lab == L.RV_BLOODPOOL}
    for name, m in masks.items():
        if m.any():
            cs = measure.find_contours(m.astype(float), 0.5)
            if cs:
                c = max(cs, key=len)
                out[name] = c  # find_contours gives (row, col) = (x, y) for an (x, y) array
    return out


def heart_box(contours: list[dict], shape: tuple[int, int], margin: float = 0.45) -> tuple[int, int, int, int]:
    """A square crop (x0, x1, y0, y1) around every contour given, with a margin; the whole image if none."""
    pts = [c for d in contours for c in d.values()]
    if not pts:
        return 0, shape[0], 0, shape[1]
    p = np.vstack(pts)
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (0.5 + margin)
    half = max(half, 8)
    xa, xb = int(max(0, cx - half)), int(min(shape[0], cx + half))
    ya, yb = int(max(0, cy - half)), int(min(shape[1], cy + half))
    return xa, xb, ya, yb


def pair_rects(w: float, h: float, aspect: float, gap_px: float = 8, pad_px: float = 4) -> list[tuple]:
    """Where two images of height/width `aspect` go in a w x h area, as large as possible: side by side when
    it is wide, ED above ES when it is tall. Centred, small gap. Pixel (x, y, w, h), y from the top."""
    row_w = min((w - 2 * pad_px - gap_px) / 2, (h - 2 * pad_px) / aspect)
    col_w = min(w - 2 * pad_px, (h - 2 * pad_px - gap_px) / 2 / aspect)
    iw = max(row_w, col_w, 1.0)
    ih = iw * aspect
    if row_w >= col_w:
        x0, y0 = (w - 2 * iw - gap_px) / 2, (h - ih) / 2
        corners = [(x0, y0), (x0 + iw + gap_px, y0)]
    else:
        x0, y0 = (w - iw) / 2, (h - 2 * ih - gap_px) / 2
        corners = [(x0, y0 + ih + gap_px), (x0, y0)]
    return [(x, h - y - ih, iw, ih) for x, y in corners]


def crop_box(panel, z: int, cont: dict, crop) -> tuple[int, int, int, int]:
    """The crop for slice z: `crop` False = the whole image; True = around this slice's heart; "all" = one box
    around the heart on every slice in the analysis (the widest one sets it), so every slice is shown at the
    same scale and place and a base-to-apex series reads as one."""
    s = panel.session
    shape = s.image.shape[:2]
    if not crop:
        return 0, shape[0], 0, shape[1]
    if isinstance(crop, tuple):  # a crop the user locked: the same window on every slice
        return crop
    if crop == "all":
        zs = [zz for zz in s.order if zz in s.included.get("ED", set()) | s.included.get("ES", set())] or list(s.order)
        allc = [slice_contours(panel, s.t_for(zz, ph), zz) for zz in zs for ph in ("ED", "ES")]
        return heart_box(allc, shape)
    return heart_box(list(cont.values()), shape)


def panel_orientation(panel) -> tuple[float, bool]:
    """How the main view turns the picture (orient.py): (degrees clockwise, mirrored)."""
    try:
        return panel.orientation()
    except Exception:
        return 0.0, False


def pair_frames(panel, z: int, crop=True) -> dict:
    """Everything needed to show slice z at ED and ES: the crop box (the window shown), pixel spacing, contrast
    window, how the picture is turned and, per phase, the frame number, image and contours. When the picture is
    turned, the image covers a larger source box ("src") so the turned window has no empty corners. Shared by
    the live view and the figures."""
    from .orient import grow_box

    s = panel.session
    t_ed, t_es = s.t_for(z, "ED"), s.t_for(z, "ES")
    lo, hi = panel_contrast(panel)
    cont = {ph: slice_contours(panel, t, z) for ph, t in (("ED", t_ed), ("ES", t_es))}
    box = crop_box(panel, z, cont, crop)
    angle, flip = panel_orientation(panel)
    src = grow_box(box, s.image.shape[:2], angle, flip)
    xa, xb, ya, yb = src
    return {"box": box, "src": src, "angle": angle, "flip": flip, "shape": s.image.shape[:2],
            "dx": float(s.spacing[0]), "dy": float(s.spacing[1]), "lo": lo, "hi": hi,
            "phases": [(ph, t, s.image[xa:xb, ya:yb, z, t].T, cont[ph]) for ph, t in (("ED", t_ed), ("ES", t_es))]}


def draw_pair(fig, panel, z: int, contours_on=True, scale_bar=True, crop=True, labels=True, axes=None):
    """Draw slice z at ED and ES into `fig` (or into the two given axes). Returns the axes."""
    s = panel.session
    t_ed, t_es = s.t_for(z, "ED"), s.t_for(z, "ES")
    lo, hi = panel_contrast(panel)
    cont = {ph: slice_contours(panel, t, z) for ph, t in (("ED", t_ed), ("ES", t_es))}
    box = crop_box(panel, z, cont, crop)
    xa, xb, ya, yb = box
    dx, dy = float(s.spacing[0]), float(s.spacing[1])
    angle, flip = panel_orientation(panel)
    from .orient import grow_box, is_identity, pixel_matrix_xy

    turned = not is_identity(angle, flip)
    sxa, sxb, sya, syb = grow_box(box, s.image.shape[:2], angle, flip) if turned else box
    if axes is None:
        fig.clf()
        axes = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)
    for ax, (ph, t) in zip(axes, (("ED", t_ed), ("ES", t_es))):
        ax.clear()
        img = s.image[sxa:sxb, sya:syb, z, t].T  # rows = y, columns = x, like the viewer
        tr = ax.transData
        if turned:  # turn about the window's centre, exactly as the main view does
            from matplotlib.transforms import Affine2D

            m = pixel_matrix_xy(angle, flip, (dx, dy))
            cx, cy = (xa + xb - 1) / 2.0, (ya + yb - 1) / 2.0
            off = np.array([cx, cy]) - m @ np.array([cx, cy])
            tr = Affine2D(np.array([[m[0, 0], m[0, 1], off[0]], [m[1, 0], m[1, 1], off[1]], [0, 0, 1]])) + ax.transData
        im = ax.imshow(img, cmap="gray", vmin=lo, vmax=hi,
                       interpolation="nearest" if not panel.smooth_image else "bicubic",
                       extent=(sxa - 0.5, sxb - 0.5, syb - 0.5, sya - 0.5), aspect=dy / dx)
        if turned:
            im.set_transform(tr)
        if contours_on:
            for name, c in cont[ph].items():
                c = np.vstack([c, c[:1]])
                ax.plot(c[:, 0], c[:, 1], color=COLOURS[name], linewidth=1.2, transform=tr)
        ax.set_xlim(xa - 0.5, xb - 0.5)
        ax.set_ylim(yb - 0.5, ya - 0.5)
        ax.set_axis_off()
        ax.set_facecolor("black")
        if labels:
            # fixed offsets in points from the corner, so the two lines never overlap however small the image
            ax.annotate(ph, (0, 1), xycoords="axes fraction", xytext=(6, -5), textcoords="offset points",
                        color="white", fontsize=11, fontweight="bold", va="top", ha="left")
            ax.annotate(f"frame {t + 1}", (0, 1), xycoords="axes fraction", xytext=(6, -20),
                        textcoords="offset points", color="#B8BEC6", fontsize=8, va="top", ha="left")
        if scale_bar and ph == "ES":
            length = _nice_bar_mm((xb - xa) * dx)
            px = length / dx
            x1 = xb - 0.5 - 0.06 * (xb - xa)
            y = yb - 0.5 - 0.07 * (yb - ya)
            ax.plot([x1 - px, x1], [y, y], color="white", linewidth=2.5, solid_capstyle="butt")
            ax.text(x1 - px / 2, y - 0.03 * (yb - ya), f"{length:g} mm", color="white", fontsize=8,
                    ha="center", va="bottom")
    return axes


def _nice_bar_mm(width_mm: float) -> float:
    """A round scale-bar length about a fifth of the image width."""
    target = width_mm / 5
    for v in (0.25, 0.5, 1, 2, 5, 10, 20, 50):
        if v >= target * 0.7:
            return v
    return 50.0


def panel_contrast(panel) -> tuple[float, float]:
    try:
        lo, hi = panel.viewer.layers["image"].contrast_limits
        return float(lo), float(hi)
    except Exception:
        img = panel.session.image
        lo, hi = np.percentile(img[::2, ::2], [0.5, 99.5])
        return float(lo), float(hi)


def save_slice_figure(panel, z: int, path: Path, **opts) -> Path:
    """One slice, ED | ES, as a publication figure."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 3), facecolor="black")
    draw_pair(fig, panel, z, **opts)
    path = Path(path)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor="black")
    return path


def save_all_slices_figure(panel, path: Path, **opts) -> Path:
    """Every slice in the analysis (base to apex), one row each: ED on the left, ES on the right."""
    from matplotlib.figure import Figure

    s = panel.session
    zs = [z for z in s.order if z in s.included.get("ED", set()) | s.included.get("ES", set())] or list(s.order)
    fig = Figure(figsize=(4.4, 2.2 * len(zs)), facecolor="black")
    axes = fig.subplots(len(zs), 2, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.995, bottom=0.005, wspace=0.02, hspace=0.04)
    for row, z in zip(axes, zs):
        draw_pair(fig, panel, z, axes=row, **opts)
        row[0].text(-0.04, 0.5, f"{z}", transform=row[0].transAxes, color="white", fontsize=9,
                    ha="right", va="center")
    path = Path(path)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor="black")
    return path


# ---------------------------------------------------------------- the live view (plain Qt painting)

class PairView:
    """The ED | ES picture as an ordinary Qt widget. Painting two small images and a few lines takes about a
    millisecond, so it redraws at every step of a resize or pop-out with nothing stretched and no lag. The
    figures are still rendered by matplotlib (draw_pair) when saved."""

    def __new__(cls, *a, **k):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QWidget

        class _W(QWidget):
            def __init__(self):
                super().__init__()
                self.frames: dict | None = None
                self.opts = {"contours_on": True, "scale_bar": True}
                self.smooth = False
                self.zoom = 1.0  # 1 = fit; both images zoom and pan together
                self.pan = [0.0, 0.0]  # screen pixels, applied inside each image's frame
                self._drag = None
                self.setMinimumSize(80, 80)
                self.setAutoFillBackground(False)
                self.setCursor(Qt.CursorShape.OpenHandCursor)

            # ----- zoom and pan: wheel or +/- zoom, drag pans, double-click or 0 fits -----

            def reset_view(self):
                self.zoom, self.pan = 1.0, [0.0, 0.0]
                self.update()

            def zoom_by(self, factor, at=None):
                old = self.zoom
                self.zoom = max(1.0, min(40.0, self.zoom * factor))
                if self.zoom == old:
                    return
                k = self.zoom / old
                if at is not None:  # keep the point under the cursor still (relative to its frame's centre)
                    cx, cy = at
                    self.pan = [cx - (cx - self.pan[0]) * k, cy - (cy - self.pan[1]) * k]
                else:
                    self.pan = [self.pan[0] * k, self.pan[1] * k]
                if self.zoom == 1.0:
                    self.pan = [0.0, 0.0]
                self.update()

            def _frame_at(self, pos):
                """Offset of pos from the centre of whichever frame it is in (or the nearest)."""
                for x, y, w, h in self._rects():
                    if x <= pos.x() <= x + w and y <= pos.y() <= y + h:
                        return pos.x() - (x + w / 2), pos.y() - (y + h / 2)
                x, y, w, h = self._rects()[0]
                return pos.x() - (x + w / 2), pos.y() - (y + h / 2)

            def wheelEvent(self, e):  # noqa: N802
                d = e.angleDelta().y() or e.pixelDelta().y()
                if d:
                    self.zoom_by(1.0015 ** d, self._frame_at(e.position()))

            def mousePressEvent(self, e):  # noqa: N802
                if e.button() == Qt.MouseButton.LeftButton:
                    self._drag = (e.position().x() - self.pan[0], e.position().y() - self.pan[1])
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)

            def mouseMoveEvent(self, e):  # noqa: N802
                if self._drag is not None and self.zoom > 1.0:
                    self.pan = [e.position().x() - self._drag[0], e.position().y() - self._drag[1]]
                    self.update()

            def mouseReleaseEvent(self, e):  # noqa: N802
                self._drag = None
                self.setCursor(Qt.CursorShape.OpenHandCursor)

            def mouseDoubleClickEvent(self, e):  # noqa: N802
                self.reset_view()

            def visible_box(self):
                """The part of the image (xa, xb, ya, yb) showing in the ED frame right now, zoom and pan
                included: what Lock crop keeps for every slice."""
                f = self.frames
                if not f:
                    return None
                from .orient import matrix_xy

                xa, xb, ya, yb = f["box"]
                nx, ny = max(xb - xa, 1), max(yb - ya, 1)
                fx, fy, fw, fh = self._rects()[0]
                w, h = fw * self.zoom, fh * self.zoom
                sx, sy = w / nx, h / ny
                # the frame's centre and half-size, taken back through the turn into image pixels
                inv = np.linalg.inv(matrix_xy(f.get("angle", 0.0), f.get("flip", False)))
                off = inv @ np.array([-self.pan[0], -self.pan[1]])
                half = np.abs(inv) @ np.array([fw / 2, fh / 2])
                cx, cy = (xa + xb) / 2 + off[0] / sx, (ya + yb) / 2 + off[1] / sy
                x0, x1 = cx - half[0] / sx, cx + half[0] / sx
                y0, y1 = cy - half[1] / sy, cy + half[1] / sy
                shape = f["shape"]
                box = (int(max(0, round(x0))), int(min(shape[0], round(x1))),
                       int(max(0, round(y0))), int(min(shape[1], round(y1))))
                return box if box[1] - box[0] >= 4 and box[3] - box[2] >= 4 else None

            def _rects(self):
                f = self.frames
                xa, xb, ya, yb = f["box"]
                aspect = max(yb - ya, 1) * f["dy"] / max(max(xb - xa, 1) * f["dx"], 1e-9)
                return pair_rects(self.width(), self.height(), aspect)

            def set_frames(self, frames, opts, smooth):
                self.frames, self.opts, self.smooth = frames, opts, smooth
                self._images = [self._to_image(img) for _, _, img, _ in frames["phases"]] if frames else []
                self.update()

            def _to_image(self, img):
                from qtpy.QtGui import QImage

                lo, hi = self.frames["lo"], self.frames["hi"]
                a = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)
                g = np.ascontiguousarray((a * 255).astype(np.uint8))
                h, w = g.shape
                qi = QImage(g.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
                return qi

            def paintEvent(self, _e):  # noqa: N802
                from qtpy.QtCore import QPointF, QRectF
                from qtpy.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

                p = QPainter(self)
                p.fillRect(self.rect(), QColor("black"))
                f = self.frames
                if not f or not self._images:
                    p.end()
                    return
                xa, xb, ya, yb = f["box"]
                nx, ny = max(xb - xa, 1), max(yb - ya, 1)
                aspect = ny * f["dy"] / max(nx * f["dx"], 1e-9)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.smooth)
                for (ph, t, _img, cont), qi, (fx, fy, fw, fh) in zip(f["phases"], self._images,
                                                                  pair_rects(self.width(), self.height(), aspect)):
                    p.save()
                    p.setClipRect(QRectF(fx, fy, fw, fh))  # the zoomed image stays inside its own frame
                    # zoom about the frame's centre, then pan; the same for both frames
                    w, h = fw * self.zoom, fh * self.zoom
                    x = fx + (fw - w) / 2 + self.pan[0]
                    y = fy + (fh - h) / 2 + self.pan[1]
                    sx, sy = w / nx, h / ny  # pixel -> screen
                    angle, flip = f.get("angle", 0.0), f.get("flip", False)
                    if angle or flip:  # turn about the window's centre, as the main view does
                        from qtpy.QtGui import QTransform

                        from .orient import matrix_xy

                        m = matrix_xy(angle, flip)
                        c = np.array([x + w / 2, y + h / 2])
                        o = c - m @ c
                        p.setTransform(QTransform(m[0, 0], m[1, 0], m[0, 1], m[1, 1], o[0], o[1]))
                    sxa, sxb, sya, syb = f.get("src", f["box"])
                    p.drawImage(QRectF(x + (sxa - xa) * sx, y + (sya - ya) * sy,
                                       (sxb - sxa) * sx, (syb - sya) * sy), qi)
                    if self.opts.get("contours_on", True):
                        for name, c in cont.items():
                            path = QPainterPath()
                            pts = [QPointF(x + (cx - xa + 0.5) * sx, y + (cy - ya + 0.5) * sy) for cx, cy in c]
                            path.moveTo(pts[0])
                            for q in pts[1:]:
                                path.lineTo(q)
                            path.closeSubpath()
                            p.setPen(QPen(QColor(COLOURS[name]), 1.6))
                            p.drawPath(path)
                    p.resetTransform()  # labels and the scale bar stay upright on the frame
                    p.setPen(QColor("white"))
                    font = QFont(p.font())
                    font.setPointSize(11)
                    font.setBold(True)
                    p.setFont(font)
                    x, y, w, h = fx, fy, fw, fh  # labels and the scale bar sit on the frame
                    p.drawText(QPointF(x + 8, y + 20), ph)
                    font.setPointSize(8)
                    font.setBold(False)
                    p.setFont(font)
                    p.setPen(QColor("#B8BEC6"))
                    p.drawText(QPointF(x + 8, y + 35), f"frame {t + 1}")
                    if self.opts.get("scale_bar", True) and ph == "ES":
                        length = _nice_bar_mm(nx * f["dx"])
                        bar = length / f["dx"] * sx
                        x1 = x + w - 0.06 * w
                        yb_ = y + h - 0.07 * h
                        p.setPen(QPen(QColor("white"), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                        p.drawLine(QPointF(x1 - bar, yb_), QPointF(x1, yb_))
                        p.setPen(QColor("white"))
                        p.drawText(QRectF(x1 - bar - 40, yb_ - 20, bar + 80, 16), Qt.AlignmentFlag.AlignCenter,
                                   f"{length:g} mm")
                    p.restore()
                p.end()

        return _W()


# ---------------------------------------------------------------- the window

class CompareWindow:
    """ED | ES as a view over the image: Ctrl+E swaps the image for the pair (as big as the image area), with
    a slim bar on top for the switches and the figure buttons; Ctrl+E or Esc brings the image back. The pair
    zooms and pans together: wheel or + / −, drag, and double-click, F or Ctrl+0 to fit. Nothing
    new to dock, pop out or resize. Owned by the panel; hiding never destroys it."""

    def __init__(self, panel):
        from qtpy.QtCore import QEvent, QObject, Qt, QTimer
        from qtpy.QtGui import QColor, QPalette
        from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

        self.panel = panel
        self.dock = None
        self._host = None  # the widget the image canvas lives in; the view covers the canvas's rectangle
        self._canvas = None
        self.win = QWidget(None if not panel._has_window() else None)
        self.win.setObjectName("tracecmr_compare")
        pal = self.win.palette()  # dark from the first frame, never a white flash
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            pal.setColor(role, QColor("#000000"))
        self.win.setPalette(pal)
        self.win.setAutoFillBackground(True)
        self.win.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.win.setStyleSheet("#tracecmr_compare { background: #000000; } "
                               "#bar { background: #1B1F25; } "
                               "QLabel, QCheckBox { color: #D5D9DE; font-size: 12px; background: transparent; }"
                               "QPushButton { background: #2A3038; color: #D5D9DE; border: none; border-radius: 7px; "
                               "padding: 4px 12px; font-size: 12px; font-weight: 600; } "
                               "QPushButton:hover { background: #343B44; }"
                               "QToolButton { color: #9AA3AD; border: none; font-size: 14px; padding: 0 6px; }"
                               "QToolButton:hover { color: #F1F2F4; }")
        lay = QVBoxLayout(self.win)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("bar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 5, 8, 5)
        row.setSpacing(8)
        self.title = QLabel("")  # the mouse and slice: drawn in the view's corner, so the bar never squeezes it
        self.title.setStyleSheet("font-size: 13px; font-weight: 600; color: #F1F2F4; background: transparent;")
        self.title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addStretch(1)
        self.chk_contours, self.chk_scale = QCheckBox("Contours"), QCheckBox("Scale bar")
        for c in (self.chk_contours, self.chk_scale):
            c.setChecked(True)
            c.setMinimumWidth(c.sizeHint().width())  # the controls never get squeezed into cut-off text
            c.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keys stay shortcuts
            c.toggled.connect(self.schedule)
            row.addWidget(c)
        from qtpy.QtWidgets import QComboBox

        self.cmb_crop = QComboBox()
        self.cmb_crop.addItems(["Crop: this slice", "Crop: same for all", "Whole image"])
        self.cmb_crop.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cmb_crop.setToolTip("This slice: tight around the heart on this slice.\n"
                                 "Same for all slices: one crop around the heart on every slice, so every slice "
                                 "is shown at the same scale and place (also in the All slices figure).")
        self.cmb_crop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cmb_crop.setStyleSheet("QComboBox { background: #2A3038; color: #D5D9DE; border: none; "
                                    "border-radius: 7px; padding: 3px 10px; font-size: 12px; } "
                                    "QComboBox::drop-down { border: none; width: 18px; } "
                                    "QComboBox QAbstractItemView { background: #1B1F25; color: #D5D9DE; "
                                    "border: 1px solid #3A4048; outline: 0; padding: 4px; } "
                                    "QComboBox QAbstractItemView::item { padding: 6px 10px; border: none; "
                                    "min-height: 20px; } "
                                    "QComboBox QAbstractItemView::item:selected, "
                                    "QComboBox QAbstractItemView::item:hover { background: #343B44; "
                                    "color: #F1F2F4; border-radius: 5px; }")
        self.cmb_crop.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        self.cmb_crop.currentIndexChanged.connect(self._crop_changed)
        row.addWidget(self.cmb_crop)
        self.btn_lock = QPushButton("Lock")
        self.btn_lock.setCheckable(True)
        self.btn_lock.setToolTip("Keep exactly this window (crop, zoom and pan) on every slice and in the figures.\n"
                                 "Set it up on the slice with the widest ventricle, then lock. Click again to unlock.")
        self.btn_lock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_lock.setStyleSheet("QPushButton:checked { background: #D71A1C; color: white; }")
        self.btn_lock.toggled.connect(self._lock_toggled)
        self.locked_box = None
        row.addWidget(self.btn_lock)
        row.addSpacing(8)
        b_one, b_all = QPushButton("Save…"), QPushButton("All slices…")
        b_one.setToolTip("Save this slice, ED | ES, as PNG (300 dpi), SVG or PDF")
        b_all.setToolTip("Every slice in the analysis, base to apex, ED | ES per row")
        for b in (b_one, b_all):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setMinimumWidth(b.sizeHint().width())
            row.addWidget(b)
        b_one.clicked.connect(self.save_one)
        b_all.clicked.connect(self.save_all)
        close = QToolButton()
        close.setText("✕")
        close.setToolTip("Back to the image (Ctrl+E or Esc)")
        close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(self.hide)
        row.addWidget(close)
        lay.addWidget(bar)
        self.view = PairView()
        lay.addWidget(self.view, 1)
        self.title.setParent(self.view)
        self.title.move(12, 8)
        self.title.show()
        self._timer = QTimer(self.win)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)  # cheap to redraw: follows edits almost live
        self._timer.timeout.connect(self.refresh)
        self.win.keyPressEvent = self._keys
        self.win.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keys keep going to the image shortcuts
        self.win.hide()
        if panel._has_window():
            self._cover_canvas()
        else:  # no main window (scripts, some tests): a plain window
            self.win.setWindowTitle("ED | ES")
            self.win.resize(820, 470)

        class _Follow(QObject):
            """Keeps the view exactly over the image canvas as the window or panels change size."""

            def eventFilter(_self, obj, event):  # noqa: N805
                if event.type() in (QEvent.Type.Resize, QEvent.Type.Move) and self.is_open():
                    self._fit()
                return False

        self._follow = _Follow(self.win)
        if self._canvas is not None:
            self._canvas.installEventFilter(self._follow)

    def _cover_canvas(self) -> None:
        try:
            self._canvas = self.panel.viewer.window._qt_viewer.canvas.native
            self._host = self._canvas.parentWidget()
            self.win.setParent(self._host)
        except Exception:
            self._canvas = self._host = None

    def _fit(self) -> None:
        if self._canvas is not None and self._host is not None:
            try:
                self.win.setGeometry(self._canvas.geometry())
            except RuntimeError:
                pass

    def show(self) -> None:
        self._fit()
        self.win.show()
        self.win.raise_()
        self.refresh(now=True)

    def hide(self) -> None:
        self.win.hide()

    def toggle(self) -> None:
        self.hide() if self.is_open() else self.show()

    def is_open(self) -> bool:
        try:
            return not self.win.isHidden()
        except RuntimeError:
            return False

    def schedule(self, *_) -> None:
        if self.is_open():
            self._timer.start()

    def refresh(self, now: bool = False) -> None:
        p = self.panel
        if p.session is None:
            return
        z = p.current_slice()
        s = p.session
        self.title.setText(f"{s.subject_id}  ·  slice {z}")
        self.title.adjustSize()
        self.title.raise_()
        o = self.opts()
        self.view.set_frames(pair_frames(p, z, crop=o["crop"]), o, p.smooth_image)
        if now:
            self.view.repaint()

    def opts(self) -> dict:
        crop = self.locked_box if self.locked_box is not None else (True, "all", False)[self.cmb_crop.currentIndex()]
        return {"contours_on": self.chk_contours.isChecked(), "scale_bar": self.chk_scale.isChecked(),
                "crop": crop}

    def _crop_changed(self, *_) -> None:
        if self.btn_lock.isChecked():  # a new crop choice replaces the lock
            self.btn_lock.setChecked(False)
        self.schedule()

    def _lock_toggled(self, on: bool) -> None:
        """Lock: the window showing now (crop, zoom, pan) becomes the crop for every slice; the zoom resets
        because the window itself is now the crop. Unlock: back to the crop menu's setting."""
        if on:
            self.locked_box = self.view.visible_box()
            if self.locked_box is None:
                self.btn_lock.setChecked(False)
                return
            self.view.reset_view()
            self.btn_lock.setText("Locked")
        else:
            self.locked_box = None
            self.btn_lock.setText("Lock")
        self.refresh(now=True)

    def _keys(self, event) -> None:
        from qtpy.QtCore import Qt

        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self.panel.step_slice(-1 if event.key() == Qt.Key.Key_Up else 1)
        elif event.key() == Qt.Key.Key_Escape:
            self.hide()

    # ---------- figures ----------

    def _ask_path(self, default_name: str) -> Path | None:
        from qtpy.QtWidgets import QFileDialog

        folder = self.panel.results_dir() / "images"
        folder.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self.win, "Save figure", str(folder / default_name),
                                              "PNG image, 300 dpi (*.png);;SVG (*.svg);;PDF (*.pdf)")
        return Path(path) if path else None

    def save_one(self) -> Path | None:
        z = self.panel.current_slice()
        path = self._ask_path(f"{self.panel._safe(self.panel.session.subject_id)}_slice{z}_ED_ES.png")
        if path is None:
            return None
        save_slice_figure(self.panel, z, path, **self.opts())
        self.panel.lbl_auto.setText(f"Figure saved: {path.name}")
        return path

    def save_all(self) -> Path | None:
        path = self._ask_path(f"{self.panel._safe(self.panel.session.subject_id)}_all_slices_ED_ES.png")
        if path is None:
            return None
        save_all_slices_figure(self.panel, path, **self.opts())
        self.panel.lbl_auto.setText(f"Figure saved: {path.name}")
        return path
