"""Regional LV wall thickness and systolic thickening (Phase 3, version 1: radial rays).

Method, per short-axis slice:
  1. Centre = centroid of the LV blood pool.
  2. Cast rays outward from the centre at evenly spaced angles.
  3. Along each ray, find where it leaves the blood pool (endocardium) and,
     beyond that, where it leaves the myocardium (epicardium). Borders are found
     with sub-pixel precision by interpolating the masks, so a 1 mm mouse wall
     (~10 pixels) is not limited to whole-pixel steps.
  4. Wall thickness at that angle = epicardial radius - endocardial radius (mm).

Known limitations of version 1 (documented in docs/methods.md):
  - Rays overestimate thickness where the wall is oblique to the ray (non-round
    ventricles). Version 2 will use centreline chords.
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


def slice_thickness(
    label2d: np.ndarray,
    spacing_xy: tuple[float, float],
    n_rays: int = 120,
    reference_deg: float = 0.0,
    step_px: float = 0.25,
) -> SliceThickness | None:
    """Wall thickness along `n_rays` evenly spaced rays on one slice.

    Returns None if the slice has no blood pool (e.g. the apical cap).
    """
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


def heart_wall(
    ci: CardiacImage,
    base_end: str,
    n_rays: int = 120,
    reference_deg: float = 0.0,
) -> HeartWall:
    """Wall thickness per AHA-style level and segment for one phase.

    `base_end` ("first"/"last") comes from orientation detection for the whole heart,
    so ED and ES are split into levels the same way.
    """
    bf, _order = to_base_first(ci, base_end)
    levels = assign_levels(bf)
    per_slice: dict[int, SliceThickness] = {}
    for z in levels:
        st = slice_thickness(bf.label[:, :, z], bf.spacing[:2], n_rays, reference_deg)
        if st is not None:
            per_slice[z] = st

    results: dict[str, LevelResult] = {}
    for level, nseg in SEGMENTS_PER_LEVEL.items():
        zs = [z for z, lv in levels.items() if lv == level and z in per_slice]
        if not zs:
            continue
        stack = np.vstack([segment_means(per_slice[z], nseg) for z in zs])
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
                        "reference": self.ed.reference,
                    }
                )
        return out


def compute_wall(
    ed: CardiacImage, es: CardiacImage, n_rays: int = 120, base_end: str | None = None
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
    ed_w = heart_wall(ed, base_end, n_rays)
    es_w = heart_wall(es, base_end, n_rays)

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
    flags.append("segments numbered from the image x-axis (no septal reference yet)")
    return WallResult(subject_id=ed.subject_id, ed=ed_w, es=es_w, thickening_pct=thick, flags=flags)
