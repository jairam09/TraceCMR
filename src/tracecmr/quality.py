"""Which slices to double-check: four independent checks on each contoured slice.

1. model unsure     the model could not decide between classes along a large part of the outline
                    (top two class probabilities within MARGIN of each other)
2. image disagrees  redrawing the endocardium around the bright blood in the image (refine.fit_slice) would
                    change it a lot: dark muscle inside the outline or bright blood outside it. Independent of
                    the model, so it also catches outlines the model was confidently wrong about.
3. anatomy          the slice does not fit with the rest of the heart: LV area jumps versus both neighbours,
                    ES cavity larger than ED, endo touching epi (no wall), or a wall implausibly thin or thick
4. base/apex        the first and last contoured slices (flagged by Auto LV, see session.autocontour_pairs)

Thresholds were chosen on UCL's 15 validation mice and the hit rates reported on the 25 held-out test mice
(scripts/validate_checks.py, docs/methods.md). The flags say where to look first; an unflagged slice is not
guaranteed correct.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import labels as L
from .refine import fit_slice

MARGIN = 0.4  # a pixel is "undecided" when its top two class probabilities differ by less than this
UNSURE_FRAC = 0.10  # flag when this share of the outline band is undecided (chosen on the 15 validation mice)
IMAGE_FRAC = 0.30  # flag when fitting to the image would change this share of the blood pool (validation mice)
AREA_JUMP = 1.6  # flag when the LV area differs from both neighbours by more than this factor
WALL_TOUCH_MM = 0.12  # endo this close to epi counts as touching
WALL_TOUCH_FRAC = 0.08  # ... flagged when it happens along this share of the wall
WALL_RANGE_MM = (0.3, 2.4)  # a slice's mean wall outside this is implausible for a mouse


def unsure_score(prob2d: np.ndarray, lab2d: np.ndarray) -> float:
    """Share of the outline band (2 px either side of the endo and epi borders) where the model was undecided.
    prob2d: (x, y, n_classes) probabilities; lab2d: the standard labels drawn from them."""
    lv = np.isin(lab2d, (L.LV_BLOODPOOL, L.LV_MYOCARDIUM))
    blood = lab2d == L.LV_BLOODPOOL
    if not lv.any():
        return 0.0
    band = np.zeros_like(lv)
    for m in (lv, blood):
        if m.any():
            band |= ndi.binary_dilation(m, iterations=2) & ~ndi.binary_erosion(m, iterations=2)
    p = np.sort(prob2d[band], axis=-1)
    margin = p[:, -1] - p[:, -2]
    return float((margin < MARGIN).mean()) if len(margin) else 0.0


def image_disagreement(label2d: np.ndarray, image2d: np.ndarray, spacing_xy) -> float:
    """Share of the blood pool that fitting to the image would add or remove (0 = the outline matches)."""
    blood = label2d == L.LV_BLOODPOOL
    if blood.sum() < 10:
        return 0.0
    fitted = fit_slice(label2d, image2d, spacing_xy) == L.LV_BLOODPOOL
    return float((fitted ^ blood).sum() / blood.sum())


def wall_problems(label2d: np.ndarray, spacing_xy) -> list[str]:
    from .wall import slice_thickness

    st = slice_thickness(label2d, spacing_xy)
    if st is None:
        return []
    th = st.thickness_mm[np.isfinite(st.thickness_mm)]
    if not len(th):
        return []
    out = []
    if (th < WALL_TOUCH_MM).mean() >= WALL_TOUCH_FRAC:
        out.append("endo touches the epi (no wall left in places)")
    m = float(th.mean())
    if m < WALL_RANGE_MM[0]:
        out.append(f"wall implausibly thin ({m:.2f} mm)")
    elif m > WALL_RANGE_MM[1]:
        out.append(f"wall implausibly thick ({m:.2f} mm)")
    return out


def _slice_checks(sl: np.ndarray, image2d: np.ndarray, sp) -> list[str]:
    """Checks 2 and 3 (wall) on one slice: the slow part, about 10 ms a slice."""
    out = []
    d = image_disagreement(sl, image2d, sp)
    if d >= IMAGE_FRAC:
        out.append(f"outline off the blood edge by {d:.0%}")
    return out + wall_problems(sl, sp)


def check_slices(label4d: np.ndarray, image4d: np.ndarray, spacing, pairs: dict[str, list[tuple[int, int]]],
                 order: list[int], unsure: dict[tuple[int, int], float] | None = None,
                 only: set[tuple[int, int]] | None = None, cache: dict | None = None) -> dict[tuple[int, int], list[str]]:
    """{(frame, slice): [reasons]} for the ED and ES slices that need a second look.

    pairs: {"ED": [(frame, slice), ...], "ES": [...]} the contoured slices of each phase; order: slices base to
    apex (for neighbours); unsure: model unsure scores from the last Auto LV (slices edited since are left out).
    only: run the per-slice image and wall checks just on these (frame, slice) (the others still count as
    neighbours); cache: a dict kept between calls on the same scan, so slices that haven't changed are not
    checked again (after an edit, only the edited slice is).
    """
    import hashlib
    sp = tuple(spacing[:2])
    out: dict[tuple[int, int], list[str]] = {}

    def add(key, why):
        out.setdefault(key, [])
        if why not in out[key]:
            out[key].append(why)

    for (t, z), score in (unsure or {}).items():
        if score >= UNSURE_FRAC:
            add((t, z), f"model unsure along {score:.0%} of the outline")
    area = {}
    for ph, lst in pairs.items():
        for t, z in lst:
            sl = label4d[:, :, z, t]
            if not np.isin(sl, (L.LV_BLOODPOOL, L.LV_MYOCARDIUM)).any():
                continue
            area[(ph, z)] = (t, int(np.isin(sl, (L.LV_BLOODPOOL, L.LV_MYOCARDIUM)).sum()),
                             int((sl == L.LV_BLOODPOOL).sum()))
            if only is not None and (t, z) not in only:
                continue
            if cache is None:
                found = _slice_checks(sl, image4d[:, :, z, t], sp)
            else:
                key = (t, z, sp, hashlib.blake2b(np.ascontiguousarray(sl).tobytes(), digest_size=16).digest())
                found = cache.get(key)
                if found is None:
                    found = cache[key] = _slice_checks(sl, image4d[:, :, z, t], sp)
            for why in found:
                add((t, z), why)
    for ph in pairs:
        zs = [z for z in order if (ph, z) in area]
        for i in range(1, len(zs) - 1):
            t, a, _ = area[(ph, zs[i])]
            lo, hi = area[(ph, zs[i - 1])][1], area[(ph, zs[i + 1])][1]
            if a > AREA_JUMP * max(lo, hi) or a * AREA_JUMP < min(lo, hi):
                add((t, zs[i]), "LV size jumps vs neighbouring slices")
    for z in order:
        if ("ED", z) in area and ("ES", z) in area:
            t_es, _, b_es = area[("ES", z)]
            _, _, b_ed = area[("ED", z)]
            if b_ed and b_es > 1.05 * b_ed and area[("ES", z)][0] != area[("ED", z)][0]:
                add((t_es, z), "ES cavity is bigger than ED on this slice")
    return out
