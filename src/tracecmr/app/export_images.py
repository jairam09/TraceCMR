"""Write contoured images (PNG) for one mouse: each slice at ED and ES, raw beside contoured, plus a
montage per phase. Only ever writes into the results folder; the raw scans are never touched."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .. import labels as L

ENDO = "#D7191C"  # red
EPI = "#4FC3F7"  # light blue
RV = "#FFD600"  # yellow


def _crop(session, z: int, t: int, half_min: int = 24):
    img = session.image[:, :, z, t].T.astype(np.float32)  # x right, y down (as in the viewer)
    lab = session.label[:, :, z, t].T
    ys, xs = np.nonzero(lab)
    if len(ys):
        cy, cx = int(ys.mean()), int(xs.mean())
        half = max(half_min, int(max(np.ptp(ys), np.ptp(xs)) * 0.9))
    else:
        cy, cx, half = img.shape[0] // 2, img.shape[1] // 2, min(img.shape) // 2
    angle, flip = float(getattr(session, "view_rotation", 0.0) or 0.0), bool(getattr(session, "view_flip", False))
    turned = bool(angle % 360) or flip
    grow = int(np.ceil(half * 1.45)) if turned else half  # room for the corners once turned
    y0, y1 = max(cy - grow, 0), min(cy + grow, img.shape[0])
    x0, x1 = max(cx - grow, 0), min(cx + grow, img.shape[1])
    img, lab = img[y0:y1, x0:x1], lab[y0:y1, x0:x1]
    if turned:  # the same turn as the viewer (orient.py), then back to the usual window around the heart
        from .orient import orient_image

        img, lab = orient_image(img, angle, flip, order=1), orient_image(lab, angle, flip, order=0)
        my, mx = (img.shape[0] - 1) / 2, (img.shape[1] - 1) / 2
        ya, xa = int(max(0, round(my - half))), int(max(0, round(mx - half)))
        img, lab = img[ya:ya + 2 * half, xa:xa + 2 * half], lab[ya:ya + 2 * half, xa:xa + 2 * half]
    lo, hi = np.percentile(img, [1, 99]) if img.size else (0, 1)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), lab


def _draw(ax, img, lab, contours: bool, title: str = "") -> None:
    ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    if contours:
        for mask, colour in ((lab == L.LV_BLOODPOOL, ENDO),
                             ((lab == L.LV_BLOODPOOL) | (lab == L.LV_MYOCARDIUM), EPI),
                             (lab == L.RV_BLOODPOOL, RV)):
            if mask.any():
                ax.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=1.2)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8, color="white")


def _save(fig: Figure, path: Path) -> None:
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=150, facecolor="black")


def export_images(session, out_dir) -> list[Path]:
    """Write PNGs for every contoured slice at ED and ES into out_dir; returns the files written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    name = session.subject_id or "scan"
    for ph in ("ED", "ES"):
        done = []
        for z in session.order:
            t = session.t_for(z, ph)
            if not np.any(session.label[:, :, z, t]):
                continue
            img, lab = _crop(session, z, t)
            fig = Figure(figsize=(5, 2.6), facecolor="black")
            a, b = fig.subplots(1, 2)
            _draw(a, img, lab, False, "raw")
            _draw(b, img, lab, True, "contoured")
            fig.suptitle(f"{name} · slice {z} · {ph} (frame {t})", fontsize=9, color="white")
            p = out / f"slice{z:02d}_{ph}.png"
            _save(fig, p)
            written.append(p)
            done.append((z, t, img, lab))
        if done:  # montage, base to apex
            n = len(done)
            cols = min(n, 5)
            rows = int(np.ceil(n / cols))
            fig = Figure(figsize=(2 * cols, 2.1 * rows + 0.3), facecolor="black")
            axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
            for ax, (z, t, img, lab) in zip(axes, done):
                _draw(ax, img, lab, True, f"slice {z} · frame {t}")
            for ax in axes[n:]:
                ax.set_axis_off()
            fig.suptitle(f"{name} · {ph} · base → apex", fontsize=10, color="white")
            p = out / f"montage_{ph}.png"
            _save(fig, p)
            written.append(p)
    return written
