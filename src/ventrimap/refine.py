"""Snap the endocardium to the image edge after automatic contouring.

Segmentation networks tend to round off thin, bright tips of the blood pool
(for example between papillary muscles). This step recovers them with a fixed,
deterministic rule, so the same scan always gives the same result:

1. On each slice, estimate blood brightness (median of the blood pool's core)
   and muscle brightness (median of the wall's middle), and set a cut-off
   halfway between them.
2. Grow the blood pool into wall pixels brighter than the cut-off that are
   connected to it, within `max_grow_mm` of the original border, and never
   within `min_wall_mm` of the epicardium.
3. The epicardium does not move; myocardium = LV minus the new blood pool.

Only growth is applied (the blood pool never shrinks), so the papillary
convention of the input is kept.

Validation (UCL model output on UCL's 25 held-out test mice, vs expert; 2026-10-07):
applied at ED and ES it improved EDV (-2.6% -> +0.4%) and mass (+6.4% -> +4.0%)
but over-grew at ES (ESV -0.6% -> +5.2%); EF error barely changed. So it is a
per-slice tool the user applies where a tip is visibly cut, not an automatic step.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import labels as L


def snap_slice(label2d: np.ndarray, image2d: np.ndarray, spacing_xy, max_grow_mm: float = 0.4,
               min_wall_mm: float = 0.15, cut_frac: float = 0.5) -> np.ndarray:
    blood = label2d == L.LV_BLOODPOOL
    wall = label2d == L.LV_MYOCARDIUM
    if blood.sum() < 10 or wall.sum() < 10:
        return label2d.copy()
    px = float(min(spacing_xy))
    core = ndimage.binary_erosion(blood, iterations=2)
    if core.sum() < 5:
        core = blood
    lv = blood | wall | (label2d == L.SCAR)
    wall_mid = wall & ndimage.binary_erosion(lv, iterations=2) & ~ndimage.binary_dilation(blood, iterations=2)
    if wall_mid.sum() < 5:
        wall_mid = wall
    mu_b, mu_m = float(np.median(image2d[core])), float(np.median(image2d[wall_mid]))
    if mu_b <= mu_m:  # not bright-blood contrast: leave unchanged
        return label2d.copy()
    cut = mu_m + cut_frac * (mu_b - mu_m)  # 0.5 = halfway; higher = stricter (only clearly bright pixels)
    reach = ndimage.distance_transform_edt(~blood) * px <= max_grow_mm
    keep_off_epi = ndimage.distance_transform_edt(lv) * px > min_wall_mm
    cand = wall & (image2d > cut) & reach & keep_off_epi
    if not cand.any():
        return label2d.copy()
    comps, n = ndimage.label(cand | blood)
    touching = set(np.unique(comps[blood])) - {0}
    grown = np.isin(comps, list(touching)) & cand
    out = label2d.copy()
    out[grown] = L.LV_BLOODPOOL
    return out


def snap_endocardium(label: np.ndarray, image: np.ndarray, spacing, **kw) -> np.ndarray:
    """Apply `snap_slice` to every slice of (x, y, slice) or every contoured frame of (x, y, slice, time)."""
    out = label.copy()
    sxy = spacing[:2]
    if label.ndim == 3:
        for z in range(label.shape[2]):
            out[:, :, z] = snap_slice(label[:, :, z], image[:, :, z], sxy, **kw)
    elif label.ndim == 4:
        for t in range(label.shape[3]):
            if not label[..., t].any():
                continue
            for z in range(label.shape[2]):
                out[:, :, z, t] = snap_slice(label[:, :, z, t], image[:, :, z, t], sxy, **kw)
    else:
        raise ValueError("label must be (x, y, slice) or (x, y, slice, time)")
    return out
