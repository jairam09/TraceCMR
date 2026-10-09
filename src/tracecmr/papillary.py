"""Convert contours between papillary-muscle conventions.

Models and the UCL dataset label papillary muscles and trabeculae as myocardium
("detailed"): the endocardial border dips inward wherever they sit. Many labs,
including the author's Segment workflow, draw a smooth endocardium with the
papillary muscles inside the blood pool. `smooth_papillary` converts detailed
labels to that convention:

1. From the blood-pool centre, cast rays and find the endocardial radius r(theta)
   (the same sub-pixel rays as wall.py).
2. Papillary muscles and trabeculae appear as narrow spikes in wall thickness
   (epicardial minus endocardial radius); the compact wall varies smoothly. A
   circular morphological opening of thickness (min filter then max filter over
   `window_deg`) removes spikes narrower than the window.
3. The new border sits that compact thickness inside the epicardium (never
   thinner than `min_wall_mm`, never shrinking the original blood pool), so it
   follows the shape of the wall rather than assuming a round cavity.
4. Blood pool = inside the new border; myocardium = the rest of the LV. The
   epicardium is unchanged, so LV size is unchanged and only the split between
   blood and muscle moves.

The conversion is one-way (detailed -> smooth). Keep the detailed labels to
switch back.

STATUS: EXPERIMENTAL, off by default (2026-10-07). On 30 UCL mice it raised EDV
by +14% and ESV by +19% on average (EF -1.9 pts, up to -4.6). Visual checks show
it fills papillary slits well at end-diastole but also pushes the border into
real wall where wall thickness varies regionally, mostly at end-systole, where
papillary muscles press against the wall and cannot be told apart from it in the
labels. It must be tuned against hand-drawn smooth contours (the author's Segment
convention) before it becomes the default.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import draw

from . import labels as L
from .wall import slice_thickness


def _circular(f, r: np.ndarray, size: int) -> np.ndarray:
    pad = size
    ext = np.concatenate([r[-pad:], r, r[:pad]])
    return f(ext, size=size)[pad:-pad]


def smooth_slice(label2d: np.ndarray, spacing_xy, window_deg: float = 90.0, min_wall_mm: float = 0.25,
                 n_rays: int = 180) -> np.ndarray:
    """One slice of standard labels -> same slice with papillary muscles moved into the blood pool."""
    blood = label2d == L.LV_BLOODPOOL
    wall = np.isin(label2d, (L.LV_MYOCARDIUM, L.SCAR, L.MVO))
    if not blood.any() or not wall.any():
        return label2d.copy()
    st = slice_thickness(label2d, spacing_xy, n_rays=n_rays, method="rays")
    if st is None:
        return label2d.copy()
    r, e = st.endo_r_mm.copy(), st.epi_r_mm.copy()
    ok = np.isfinite(r) & np.isfinite(e)
    if ok.sum() < n_rays // 2:  # too irregular to trust: leave unchanged
        return label2d.copy()
    ang = np.arange(n_rays)
    r = np.interp(ang, ang[ok], r[ok], period=n_rays)
    e = np.interp(ang, ang[ok], e[ok], period=n_rays)
    size = max(3, int(round(window_deg / (360.0 / n_rays))) | 1)
    # Papillary muscles and trabeculae are narrow spikes in wall thickness; the compact wall
    # varies smoothly. A circular opening (min then max filter) removes spikes narrower than
    # the window, and the new border sits that compact thickness inside the epicardium.
    thick = e - r
    compact = _circular(ndimage.maximum_filter1d, _circular(ndimage.minimum_filter1d, thick, size), size)
    compact = np.maximum(compact, min_wall_mm)
    # Only move the border where the dip is clearly larger than pixel staircase noise;
    # otherwise the lower envelope of a jagged thickness profile inflates every cavity.
    noise = 1.5 * max(spacing_xy)
    closed = np.where(thick - compact > noise, e - compact, r)
    closed = np.maximum(closed, r)  # never shrink the blood pool
    dx, dy = spacing_xy
    cx, cy = st.centre_px
    theta = np.deg2rad(st.angles_deg)
    px = cx + closed * np.cos(theta) / dx
    py = cy + closed * np.sin(theta) / dy
    rr, cc = draw.polygon(px, py, label2d.shape)
    new_blood = np.zeros_like(blood)
    new_blood[rr, cc] = True
    lv = blood | wall
    new_blood = (new_blood & lv) | blood
    out = label2d.copy()
    out[lv & ~new_blood & (label2d == L.LV_BLOODPOOL)] = L.LV_MYOCARDIUM  # (not expected; kept for safety)
    moved = new_blood & (label2d == L.LV_MYOCARDIUM)
    out[moved] = L.LV_BLOODPOOL
    return out


def smooth_papillary(label: np.ndarray, spacing, **kw) -> np.ndarray:
    """Apply `smooth_slice` to every slice (and frame, for 4D (x, y, slice, time) arrays)."""
    out = label.copy()
    sxy = spacing[:2]
    if label.ndim == 3:
        for z in range(label.shape[2]):
            out[:, :, z] = smooth_slice(label[:, :, z], sxy, **kw)
    elif label.ndim == 4:
        for t in range(label.shape[3]):
            if not label[..., t].any():
                continue
            for z in range(label.shape[2]):
                out[:, :, z, t] = smooth_slice(label[:, :, z, t], sxy, **kw)
    else:
        raise ValueError("label must be (x, y, slice) or (x, y, slice, time)")
    return out
