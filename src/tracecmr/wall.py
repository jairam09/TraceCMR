"""Regional LV wall thickness and systolic thickening.

Default method (version 2): the centerline method (Sheehan et al., Circulation 1986), as used by
Segment and most cardiac MRI papers. Per short-axis slice:
  1. Endocardial and epicardial borders are traced with sub-pixel precision (iso-line at 0.5).
  2. The centerline is the line midway between them: the zero line of
     (signed distance to the endocardium) + (signed distance to the epicardium).
  3. At evenly spaced points along the centerline, wall thickness = distance to the endocardium +
     distance to the epicardium. On the centerline these are equal, so this is the diameter of the
     largest circle that fits in the wall there: thickness measured perpendicular to the wall,
     whatever its shape or angle.
  4. Each point's angle (around the blood-pool centroid) puts it in a segment.

Version 1 (method="rays", still used by papillary.py): rays from the blood-pool centroid; thickness =
where a ray leaves the myocardium minus where it leaves the blood pool. Rays overestimate thickness where
the wall is oblique to the ray (non-round ventricles), which is why the centerline method is the default.

Known limitations (documented in docs/methods.md):
  - With labels that count papillary muscles as myocardium (e.g. UCL), an
    attached papillary muscle is measured as wall and inflates thickness there.
  - Angles are measured from the image x-axis unless a reference angle is given.
    Segment numbers therefore describe positions in the image, not AHA names,
    until a septal reference (RV insertion) is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from . import labels as L
from .io import CardiacImage
from .orientation import assign_levels, detect_for_heart, to_base_first

BLOOD = (L.LV_BLOODPOOL,)
WALL = (L.LV_MYOCARDIUM, L.SCAR, L.MVO)

SEGMENTS_PER_LEVEL = {"basal": 6, "mid": 6, "apical": 4}  # AHA layout


@dataclass
class SliceThickness:
    """Wall thickness around one slice."""

    angles_deg: np.ndarray  # ray angles, degrees, counter-clockwise from reference
    thickness_mm: np.ndarray  # NaN where a ray found no valid wall
    endo_r_mm: np.ndarray
    epi_r_mm: np.ndarray
    centre_px: tuple[float, float]
    chords_mm: np.ndarray | None = None  # centerline method: (n, 2, 2) endo point and epi point (x, y mm) per sample


def _first_exit(profile: np.ndarray, radii: np.ndarray, start: int = 0) -> float:
    """Radius where an interpolated mask profile first drops below 0.5 after `start`.

    Linear interpolation between the two samples either side of the crossing
    gives sub-sample precision. Returns NaN if the profile never exits.
    """
    inside = profile >= 0.5
    for i in range(max(start, 1), len(profile)):
        if inside[i - 1] and not inside[i]:
            p0, p1 = profile[i - 1], profile[i]
            frac = (p0 - 0.5) / (p0 - p1) if p0 != p1 else 0.0
            return float(radii[i - 1] + frac * (radii[i] - radii[i - 1]))
    return float("nan")


def _border_mm(mask: np.ndarray, dx: float, dy: float, step_mm: float) -> np.ndarray | None:
    """Outer border of a filled mask as densely resampled (x, y) points in mm (sub-pixel, iso-line 0.5)."""
    from skimage import measure

    from .contours import _smooth_outline, resample_closed

    padded = np.pad(mask.astype(float), 1)
    cs = measure.find_contours(padded, 0.5)
    if not cs:
        return None
    c = max(cs, key=len)[:-1] - 1.0  # (x index, y index), undo the padding; drop the repeated end point
    if len(c) < 4:
        return None
    # the iso-line follows the pixel staircase; distances would catch its corners, so smooth it (area kept)
    c = _smooth_outline(resample_closed(c, 4 * len(c)), 1.0)
    pts = np.column_stack([c[:, 0] * dx, c[:, 1] * dy])
    perim = float(np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T).sum())
    return resample_closed(pts, max(64, int(perim / step_mm)))


def slice_thickness_centerline(
    label2d: np.ndarray,
    spacing_xy: tuple[float, float],
    n_points: int = 120,
    reference_deg: float = 0.0,
    upsample: int = 4,
) -> SliceThickness | None:
    """Wall thickness at `n_points` evenly spaced points along the centerline (see module docstring).

    Returns None if the slice has no blood pool or no wall around it.
    """
    from scipy.spatial import cKDTree
    from skimage import measure

    from .contours import resample_closed

    blood = np.isin(label2d, BLOOD)
    if not blood.any():
        return None
    comps, n = ndi.label(blood)
    if n > 1:  # keep the main cavity
        blood = comps == (1 + int(np.argmax(ndi.sum(blood, comps, range(1, n + 1)))))
    blood = ndi.binary_fill_holes(blood)  # enclosed papillary muscles are not wall
    lv = ndi.binary_fill_holes(np.isin(label2d, BLOOD + WALL) | blood)
    if (lv & ~blood).sum() < 4:
        return None
    dx, dy = spacing_xy
    px_mm = min(dx, dy)
    endo = _border_mm(blood, dx, dy, px_mm / 6)
    epi = _border_mm(lv, dx, dy, px_mm / 6)
    if endo is None or epi is None:
        return None

    # Fine grid over the heart (a few pixels of margin so the centerline closes inside it). The signed
    # distances are only needed near the wall: elsewhere the sign alone places the zero line correctly.
    lo = epi.min(axis=0) - 3 * px_mm
    hi = epi.max(axis=0) + 3 * px_mm
    h = px_mm / upsample
    gx = np.arange(lo[0], hi[0], h)
    gy = np.arange(lo[1], hi[1], h)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    coords = np.vstack([GX.ravel() / dx, GY.ravel() / dy])  # in pixel indices
    in_blood = ndi.map_coordinates(blood.astype(float), coords, order=1) >= 0.5
    in_lv = ndi.map_coordinates(lv.astype(float), coords, order=1) >= 0.5
    f = np.where(in_lv, -1.0, 1.0)  # cavity and wall: negative; outside: positive
    band_px = ndi.binary_dilation(lv & ~blood, iterations=2)
    near = ndi.map_coordinates(band_px.astype(float), coords, order=0) > 0
    pts = np.column_stack([GX.ravel()[near], GY.ravel()[near]])
    d_en, _ = cKDTree(endo).query(pts)
    d_ep, _ = cKDTree(epi).query(pts)
    f[near] = np.where(in_blood[near], -d_en, d_en) + np.where(in_lv[near], -d_ep, d_ep)
    f = f.reshape(GX.shape)
    cs = measure.find_contours(f, 0.0)
    if not cs:
        return None
    c = max(cs, key=len)
    centre_line = np.column_stack([lo[0] + c[:, 0] * h, lo[1] + c[:, 1] * h])
    if len(centre_line) < 8:
        return None
    centre_line = resample_closed(centre_line, n_points)

    d_endo, i_endo = cKDTree(endo).query(centre_line)
    d_epi, i_epi = cKDTree(epi).query(centre_line)
    thickness = d_endo + d_epi

    cx, cy = ndi.center_of_mass(blood)
    cmm = np.array([cx * dx, cy * dy])
    rel = centre_line - cmm
    angles = np.mod(np.rad2deg(np.arctan2(rel[:, 1], rel[:, 0])) - reference_deg, 360.0)
    order = np.argsort(angles)
    return SliceThickness(
        angles_deg=angles[order],
        thickness_mm=thickness[order],
        endo_r_mm=np.hypot(*(endo[i_endo] - cmm).T)[order],
        epi_r_mm=np.hypot(*(epi[i_epi] - cmm).T)[order],
        centre_px=(float(cx), float(cy)),
        chords_mm=np.stack([endo[i_endo], epi[i_epi]], axis=1)[order],
    )


_CACHE: dict = {}  # slice_thickness results by (label hash, spacing, n, reference); results are not mutated


def slice_thickness(
    label2d: np.ndarray,
    spacing_xy: tuple[float, float],
    n_rays: int = 120,
    reference_deg: float = 0.0,
    step_px: float = 0.25,
    method: str = "centerline",
) -> SliceThickness | None:
    """Wall thickness around one slice: centerline method (default) or `method="rays"` (version 1,
    `n_rays` evenly spaced rays from the blood-pool centroid).

    Returns None if the slice has no blood pool (e.g. the apical cap).
    """
    if method == "centerline":
        # the app recomputes after every edit, and an edit changes one slice: reuse the others
        import hashlib

        key = (hashlib.sha1(np.ascontiguousarray(label2d).tobytes()).hexdigest(), label2d.shape,
               tuple(float(v) for v in spacing_xy), n_rays, float(reference_deg))
        if key not in _CACHE:
            if len(_CACHE) > 512:
                _CACHE.clear()
            _CACHE[key] = slice_thickness_centerline(label2d, spacing_xy, n_rays, reference_deg)
        return _CACHE[key]
    if method != "rays":
        raise ValueError("method must be 'centerline' or 'rays'")
    blood = np.isin(label2d, BLOOD).astype(float)
    epi = np.isin(label2d, BLOOD + WALL).astype(float)
    if blood.sum() == 0:
        return None
    dx, dy = spacing_xy
    cx, cy = ndi.center_of_mass(blood)

    # Longest possible ray: from centre to the farthest epicardial pixel, plus margin.
    ix, iy = np.nonzero(epi)  # arrays are indexed (x, y)
    max_r_mm = float(np.max(np.hypot((ix - cx) * dx, (iy - cy) * dy))) + 2 * max(dx, dy)
    step_mm = step_px * min(dx, dy)
    radii = np.arange(0.0, max_r_mm, step_mm)

    angles = reference_deg + np.arange(n_rays) * 360.0 / n_rays
    endo_r = np.full(n_rays, np.nan)
    epi_r = np.full(n_rays, np.nan)
    for k, a in enumerate(np.deg2rad(angles)):
        px = cx + radii * np.cos(a) / dx
        py = cy + radii * np.sin(a) / dy
        coords = np.vstack([px, py])
        bprof = ndi.map_coordinates(blood, coords, order=1, mode="constant", cval=0.0)
        eprof = ndi.map_coordinates(epi, coords, order=1, mode="constant", cval=0.0)
        if bprof[0] < 0.5:  # centre not inside the blood pool (very irregular cavity)
            continue
        r_endo = _first_exit(bprof, radii)
        if np.isnan(r_endo):
            continue
        start = int(np.searchsorted(radii, r_endo))
        r_epi = _first_exit(eprof, radii, start=start)
        if np.isnan(r_epi) or r_epi <= r_endo:
            continue
        endo_r[k], epi_r[k] = r_endo, r_epi

    return SliceThickness(
        angles_deg=np.mod(angles - reference_deg, 360.0),
        thickness_mm=epi_r - endo_r,
        endo_r_mm=endo_r,
        epi_r_mm=epi_r,
        centre_px=(float(cx), float(cy)),
    )


def segment_means(st: SliceThickness, n_segments: int) -> np.ndarray:
    """Mean thickness in each of `n_segments` equal angular sectors (sector 1 starts at 0 deg)."""
    edges = np.linspace(0.0, 360.0, n_segments + 1)
    out = np.full(n_segments, np.nan)
    for s in range(n_segments):
        sel = (st.angles_deg >= edges[s]) & (st.angles_deg < edges[s + 1])
        vals = st.thickness_mm[sel]
        if np.any(np.isfinite(vals)):
            out[s] = float(np.nanmean(vals))
    return out


def septal_angle(label2d: np.ndarray, spacing_xy: tuple[float, float], max_gap_mm: float = 0.6):
    """Where the RV meets the LV on one slice: (septum angle, (insertion angle 1, insertion angle 2)) in degrees,
    measured like `slice_thickness` angles (around the blood-pool centroid, reference 0). None without an RV.

    The contact arc is the part of the epicardial border within `max_gap_mm` of the RV (traced RVs usually leave
    a small gap); its two ends are the RV insertion points and its middle is the septum.
    """
    blood = np.isin(label2d, BLOOD)
    rv = label2d == L.RV_BLOODPOOL
    if not blood.any() or rv.sum() < 10:
        return None
    lv = ndi.binary_fill_holes(np.isin(label2d, BLOOD + WALL) | blood)
    border = lv & ~ndi.binary_erosion(lv)
    dx, dy = spacing_xy
    gap = ndi.distance_transform_edt(~rv, sampling=(dx, dy))
    for limit in (max_gap_mm / 2, max_gap_mm):
        contact = border & (gap <= limit)
        if contact.sum() >= 3:
            break
    else:
        return None
    cx, cy = ndi.center_of_mass(blood)
    ix, iy = np.nonzero(contact)
    ang = np.rad2deg(np.arctan2((iy - cy) * dy, (ix - cx) * dx))
    # circular mean = septum; the insertion points are the ends of the arc around it
    mean = np.rad2deg(np.arctan2(np.sin(np.deg2rad(ang)).mean(), np.cos(np.deg2rad(ang)).mean()))
    rel = (ang - mean + 180.0) % 360.0 - 180.0
    return float(mean % 360.0), (float((mean + rel.min()) % 360.0), float((mean + rel.max()) % 360.0))


# AHA order (Cerqueira et al., Circulation 2002), viewed from the apex, counterclockwise on the image:
# basal/mid 1 anterior, 2 anteroseptal, 3 inferoseptal, 4 inferior, 5 inferolateral, 6 anterolateral;
# apical 1 anterior, 2 septal, 3 inferior, 4 lateral. The septum's middle is the 2/3 boundary (6 segments) or the
# middle of segment 2 (4 segments): the start of segment 1 lies this far before it.
SEPTUM_OFFSET_DEG = {6: 120.0, 4: 135.0}


def _circmean(deg) -> float:
    r = np.deg2rad(np.asarray(deg, float))
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0)


@dataclass
class LevelResult:
    level: str
    slices: list[int]  # base-first slice indices that make up this level
    segments_mm: np.ndarray  # mean thickness per segment, averaged over the level's slices


@dataclass
class HeartWall:
    """Regional wall thickness for one phase of one heart."""

    subject_id: str
    phase: str | None
    levels: dict[str, LevelResult]
    per_slice: dict[int, SliceThickness] = field(repr=False, default_factory=dict)
    base_end: str = "first"
    papillary: str | None = None
    reference: str = "image x-axis"


def septum_by_level(ci: CardiacImage, base_end: str, manual: dict[int, float] | None = None) -> dict[str, float]:
    """Septum angle per level: circular mean over the level's slices.

    `manual` maps slice index (in `ci`, before reordering) to a septum angle the user set by hand; in a level with
    any such slice only those are used. Otherwise slices with an RV traced give it (`septal_angle`).
    """
    bf, order = to_base_first(ci, base_end)
    levels = assign_levels(bf)
    found: dict[str, list[float]] = {}
    by_hand: dict[str, list[float]] = {}
    manual = manual or {}
    for z, level in levels.items():
        if order[z] in manual:
            by_hand.setdefault(level, []).append(float(manual[order[z]]))
            continue
        r = septal_angle(bf.label[:, :, z], bf.spacing[:2])
        if r is not None:
            found.setdefault(level, []).append(r[0])
    found.update(by_hand)
    return {level: _circmean(a) for level, a in found.items()}


def heart_wall(
    ci: CardiacImage,
    base_end: str,
    n_rays: int = 120,
    reference_deg: float = 0.0,
    septum: dict[str, float] | None = None,
    mirror: bool = False,
) -> HeartWall:
    """Wall thickness per AHA-style level and segment for one phase.

    `base_end` ("first"/"last") comes from orientation detection for the whole heart,
    so ED and ES are split into levels the same way.

    `septum` (degrees per level, from `septum_by_level`) numbers the segments anatomically (AHA order,
    counterclockwise on the image as seen from the apex; `mirror=True` if the slices are seen from the base).
    Levels without a septum use the septum of the nearest level that has one; with none at all the segments
    start at `reference_deg` from the image x-axis, as before.
    """
    from dataclasses import replace

    bf, _order = to_base_first(ci, base_end)
    levels = assign_levels(bf)
    per_slice: dict[int, SliceThickness] = {}
    for z in levels:
        st = slice_thickness(bf.label[:, :, z], bf.spacing[:2], n_rays, 0.0)
        if st is not None:
            per_slice[z] = st

    order = list(SEGMENTS_PER_LEVEL)
    sept = dict(septum or {})
    for level in order:  # borrow from the nearest level (the RV is often not traced at the apex)
        if level not in sept and sept:
            near = min(sept, key=lambda lv: abs(order.index(lv) - order.index(level)))
            sept[level] = sept[near]
    sign = 1.0 if mirror else -1.0  # raw angles run clockwise on the image (y points down); AHA runs counterclockwise

    def aligned(st: SliceThickness, level: str, nseg: int) -> SliceThickness:
        if level in sept:
            start = sept[level] - sign * SEPTUM_OFFSET_DEG[nseg]
            return replace(st, angles_deg=np.mod(sign * (st.angles_deg - start), 360.0))
        return replace(st, angles_deg=np.mod(st.angles_deg - reference_deg, 360.0))

    results: dict[str, LevelResult] = {}
    for level, nseg in SEGMENTS_PER_LEVEL.items():
        zs = [z for z, lv in levels.items() if lv == level and z in per_slice]
        if not zs:
            continue
        stack = np.vstack([segment_means(aligned(per_slice[z], level, nseg), nseg) for z in zs])
        with np.errstate(all="ignore"):
            seg = np.nanmean(stack, axis=0)
        results[level] = LevelResult(level=level, slices=zs, segments_mm=seg)

    return HeartWall(
        subject_id=ci.subject_id,
        phase=ci.phase,
        levels=results,
        per_slice=per_slice,
        base_end=base_end,
        papillary=ci.papillary,
        reference=("septum (RV insertion), AHA order" + (", mirrored" if mirror else "")) if sept else
                  "image x-axis",
    )


@dataclass
class WallResult:
    """Regional end-diastolic thickness, end-systolic thickness and systolic thickening."""

    subject_id: str
    ed: HeartWall
    es: HeartWall
    thickening_pct: dict[str, np.ndarray]  # per level, per segment
    flags: list[str] = field(default_factory=list)

    def rows(self) -> list[dict]:
        """One row per level and segment, for a CSV."""
        out = []
        for level, nseg in SEGMENTS_PER_LEVEL.items():
            if level not in self.ed.levels or level not in self.es.levels:
                continue
            for s in range(nseg):
                out.append(
                    {
                        "subject_id": self.subject_id,
                        "level": level,
                        "segment": s + 1,
                        "edwt_mm": float(self.ed.levels[level].segments_mm[s]),
                        "eswt_mm": float(self.es.levels[level].segments_mm[s]),
                        "thickening_pct": float(self.thickening_pct[level][s]),
                        "papillary": self.ed.papillary,
                        "method": "centerline",
                        "reference": self.ed.reference,
                    }
                )
        return out


def compute_wall(
    ed: CardiacImage, es: CardiacImage, n_rays: int = 120, base_end: str | None = None, mirror: bool = False,
    manual_septum: dict[str, dict[int, float]] | None = None,
) -> WallResult:
    """Regional wall thickness at ED and ES, and systolic thickening per segment.

    Thickening = (ESWT - EDWT) / EDWT x 100%, compared level by level, because the
    base moves toward the apex in systole and ED/ES slices are not the same tissue.

    `base_end` ("first"/"last") sets the orientation by hand, e.g. from the slice
    order the user chose in the app; if None it is detected (and may refuse).
    """
    if ed.papillary != es.papillary:
        raise ValueError("ED and ES use different papillary conventions")
    if base_end is None:
        base_end = detect_for_heart(ed, es).base_end
    # septum from the RV at ED (the RV is largest there), else ES; the same reference for both phases
    man = manual_septum or {}
    septum = septum_by_level(ed, base_end, man.get("ED")) or septum_by_level(es, base_end, man.get("ES"))
    ed_w = heart_wall(ed, base_end, n_rays, septum=septum, mirror=mirror)
    es_w = heart_wall(es, base_end, n_rays, septum=septum, mirror=mirror)

    thick: dict[str, np.ndarray] = {}
    for level in SEGMENTS_PER_LEVEL:
        if level in ed_w.levels and level in es_w.levels:
            e = ed_w.levels[level].segments_mm
            s = es_w.levels[level].segments_mm
            with np.errstate(all="ignore"):
                thick[level] = 100.0 * (s - e) / e

    flags = []
    for level, vals in thick.items():
        bad = [i + 1 for i, v in enumerate(vals) if np.isfinite(v) and v < 0]
        if bad:
            note = " (basal through-plane motion / outflow tract likely)" if level == "basal" else ""
            flags.append(f"negative thickening in {level} segment(s) {bad}: check contours{note}")
    if ed.papillary == "myocardium":
        flags.append("papillary muscles labeled as myocardium: thickness inflated where they attach")
    if man.get("ED") or man.get("ES"):
        flags.append("septum set by hand on some slices (Wall menu → Reset septum to go back to automatic)")
    if septum:
        missing = [lv for lv in SEGMENTS_PER_LEVEL if lv not in septum and lv in ed_w.levels]
        flags.append("segments placed from the RV insertion (septum = segments 2-3, apical 2) in AHA order, assuming "
                     "the slices are seen from the apex" + (" (mirrored)" if mirror else "")
                     + "; if the scan is seen from the base, tick Settings → Mirror segment numbering"
                     + (f"; {', '.join(missing)} level borrows the septum from the nearest level" if missing else ""))
    else:
        flags.append("no RV traced: segments numbered from the image x-axis, not anatomical (trace the RV on a few "
                     "slices to place them from the septum)")
    return WallResult(subject_id=ed.subject_id, ed=ed_w, es=es_w, thickening_pct=thick, flags=flags)
