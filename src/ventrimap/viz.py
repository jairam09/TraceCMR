"""Overlay figures: the single most useful debugging tool in this project.

Colors follow the lab-figure convention: red = LV endocardium,
green = LV epicardium, magenta = RV endocardium, on a black background.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; works without a screen
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from . import labels as L
from .io import CardiacImage

CONTOUR_STYLE = {
    "LV endo": "red",
    "LV epi": "lime",
    "RV endo": "magenta",
    "Scar": "yellow",
}


def _masks_for_contours(label2d: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "LV endo": label2d == L.LV_BLOODPOOL,
        "LV epi": np.isin(label2d, [L.LV_BLOODPOOL, L.LV_MYOCARDIUM, L.SCAR, L.MVO]),
        "RV endo": label2d == L.RV_BLOODPOOL,
        "Scar": np.isin(label2d, [L.SCAR, L.MVO]),
    }


def save_bullseye(
    values: dict[str, np.ndarray],
    out_path: str | Path,
    title: str,
    label: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    fmt: str = "{:.2f}",
) -> Path:
    """Bullseye plot: basal ring outside, mid ring, apical ring inside (AHA layout).

    `values` maps level -> per-segment values (6 basal, 6 mid, 4 apical).
    Segment 1 starts at the reference angle (3 o'clock = image x-axis) and runs
    clockwise, matching the image (y points down), so a segment sits where it
    appears on the scan. Missing levels are drawn grey.
    """
    out_path = Path(out_path)
    rings = [("basal", 3.0, 2.0), ("mid", 2.0, 1.0), ("apical", 1.0, 0.25)]
    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)  # clockwise: image y axis points down
    cm = plt.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for level, outer, inner in rings:
        n = {"basal": 6, "mid": 6, "apical": 4}[level]
        vals = values.get(level)
        width = 2 * np.pi / n
        for s in range(n):
            v = None if vals is None else vals[s]
            colour = "#cccccc" if v is None or not np.isfinite(v) else cm(norm(v))
            ax.bar(s * width + width / 2, outer - inner, width=width, bottom=inner,
                   color=colour, edgecolor="white", linewidth=1.5)
            if v is not None and np.isfinite(v):
                r, g, b, _ = colour
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text = fmt.format(v)
                if text.lstrip("-").strip("0.") == "":  # avoid "-0"
                    text = text.lstrip("-")
                ax.text(s * width + width / 2, (outer + inner) / 2, text,
                        ha="center", va="center", fontsize=7,
                        color="black" if luminance > 0.5 else "white")
    ax.set_axis_off()
    ax.set_ylim(0, 3.0)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.08, label=label)
    ax.set_title(title, fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def save_overlay(
    ci: CardiacImage,
    out_path: str | Path,
    slices: list[int] | None = None,
    max_cols: int = 5,
) -> Path:
    """Save a grid of slices with label contours drawn on top.

    By default shows every slice that contains any label (or all slices if
    there is no label). Each panel is titled with its slice index.
    """
    out_path = Path(out_path)
    nz = ci.image.shape[2]
    if slices is None:
        if ci.label is not None and np.any(ci.label):
            slices = [z for z in range(nz) if np.any(ci.label[:, :, z])]
        else:
            slices = list(range(nz))
    if not slices:
        raise ValueError("no slices to draw")

    ncols = min(max_cols, len(slices))
    nrows = int(np.ceil(len(slices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), facecolor="black", squeeze=False)

    lo, hi = np.percentile(ci.image, [1, 99])
    used: set[str] = set()
    for ax, z in zip(axes.flat, slices):
        # Transpose so x runs left-right and y runs top-bottom, like most viewers.
        ax.imshow(ci.image[:, :, z].T, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        if ci.label is not None:
            for name, mask in _masks_for_contours(ci.label[:, :, z]).items():
                if not mask.any():
                    continue
                for c in measure.find_contours(mask.T.astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], color=CONTOUR_STYLE[name], linewidth=1.0)
                used.add(name)
        ax.set_title(f"slice {z}", color="white", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(slices):]:
        ax.axis("off")

    if used:
        handles = [plt.Line2D([], [], color=CONTOUR_STYLE[n], label=n) for n in CONTOUR_STYLE if n in used]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, labelcolor="white", fontsize=9)
    dx, dy, dz = ci.spacing
    fig.suptitle(
        f"{ci.subject_id}{' ' + ci.phase if ci.phase else ''}   spacing {dx:.3g} x {dy:.3g} x {dz:.3g} mm",
        color="white",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=120, facecolor="black")
    plt.close(fig)
    return out_path
