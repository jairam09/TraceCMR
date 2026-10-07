"""Which end of a short-axis stack is the base, and which slices are basal / mid / apical.

Datasets do not store slices in a consistent order (in the UCL mouse data,
13 of 16 checked mice were base-first and 3 apex-first), so this is detected
from the labels rather than assumed.

Rules, from anatomy:
  1. Area profile (primary): the LV narrows toward the apex, so the blood pool
     is smaller in the apical third of the stack than in the basal third.
  2. Apical cap (confirmation): slices with myocardium but no blood pool
     occur only at the apex.
  3. End slices (confirmation): the last blood-pool slice at the apex end is
     smaller than the one at the base end.

Rule 1 must be decisive, and no rule may disagree with it; otherwise the
detector refuses and asks for the orientation to be given by hand.

Validated against visual (anatomical) labels on 16 UCL mice: 16/16 correct.
That is a small set with only 3 apex-first stacks; re-check on new datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import labels as L
from .io import CardiacImage

# The basal third's blood pool must be at least this many times the apical third's.
MIN_THIRDS_RATIO = 1.25


class OrientationUncertain(ValueError):
    """Raised when the base/apex rules disagree or are not decisive."""


@dataclass
class Orientation:
    base_end: str  # "first" (slice 0 side) or "last"
    thirds_ratio: float  # basal-third / apical-third blood pool area (>1)
    evidence: dict[str, str] = field(default_factory=dict)


def _heart_slices(ci: CardiacImage) -> list[int]:
    if ci.label is None:
        raise ValueError(f"{ci.subject_id}: orientation needs a label")
    has = np.isin(ci.label, [L.LV_BLOODPOOL, L.LV_MYOCARDIUM, L.SCAR, L.MVO]).any(axis=(0, 1))
    zs = [int(z) for z in np.flatnonzero(has)]
    if len(zs) < 3:
        raise OrientationUncertain(f"{ci.subject_id}: only {len(zs)} labeled slices")
    return zs


def detect_base_apex(ci: CardiacImage) -> Orientation:
    """Decide whether the base is at the first or last labeled slice.

    Raises OrientationUncertain instead of guessing.
    """
    zs = _heart_slices(ci)
    dx, dy, _ = ci.spacing
    blood = np.array([np.count_nonzero(ci.label[:, :, z] == L.LV_BLOODPOOL) for z in zs]) * dx * dy

    # Rule 1: thirds of the area profile.
    n = len(blood)
    t = max(1, n // 3)
    first_third, last_third = blood[:t].sum(), blood[-t:].sum()
    if min(first_third, last_third) <= 0:
        big, small = max(first_third, last_third), 1e-9
    else:
        big, small = max(first_third, last_third), min(first_third, last_third)
    ratio = float(big / small)
    vote_thirds = "first" if first_third > last_third else "last"
    evidence = {"thirds": f"base={vote_thirds} (ratio {ratio:.2f})"}
    if ratio < MIN_THIRDS_RATIO:
        raise OrientationUncertain(
            f"{ci.subject_id}: blood pool areas at the two ends are too similar "
            f"(ratio {ratio:.2f} < {MIN_THIRDS_RATIO}); give the orientation by hand"
        )

    # Rule 2: apical cap at exactly one end.
    cap_first, cap_last = blood[0] == 0, blood[-1] == 0
    if cap_first != cap_last:
        vote_cap = "last" if cap_first else "first"  # base is the end WITHOUT the cap
        evidence["apical_cap"] = f"base={vote_cap}"
        if vote_cap != vote_thirds:
            raise OrientationUncertain(f"{ci.subject_id}: apical cap contradicts the area profile")
    else:
        evidence["apical_cap"] = "not decisive"

    # Rule 3: end slices with blood pool.
    nz = blood[blood > 0]
    if nz[0] != nz[-1]:
        vote_ends = "first" if nz[0] > nz[-1] else "last"
        evidence["end_slices"] = f"base={vote_ends}"
        if vote_ends != vote_thirds:
            raise OrientationUncertain(f"{ci.subject_id}: end-slice areas contradict the area profile")

    return Orientation(base_end=vote_thirds, thirds_ratio=ratio, evidence=evidence)


def detect_for_heart(ed: CardiacImage, es: CardiacImage | None = None) -> Orientation:
    """Orientation for a whole heart (it does not change between ED and ES).

    ED is used first because the taper is clearer at end-diastole (UCL: 16/16
    decided at ED vs 14/16 at ES, none wrong). If ED is uncertain, ES is tried.
    If both are decisive and disagree, the detector refuses.
    """
    ed_result = es_result = None
    try:
        ed_result = detect_base_apex(ed)
    except OrientationUncertain:
        pass
    if es is not None:
        try:
            es_result = detect_base_apex(es)
        except OrientationUncertain:
            pass
    if ed_result and es_result and ed_result.base_end != es_result.base_end:
        raise OrientationUncertain(f"{ed.subject_id}: ED and ES give opposite orientations")
    result = ed_result or es_result
    if result is None:
        raise OrientationUncertain(f"{ed.subject_id}: orientation unclear at ED and ES; give it by hand")
    result.evidence["decided_from"] = "ED" if ed_result else "ES"
    if es is not None:
        result.evidence["ES_check"] = "agrees" if es_result else "not decisive"
    return result


def to_base_first(ci: CardiacImage, base_end: str | None = None) -> tuple[CardiacImage, list[int]]:
    """Return a copy with slices ordered base -> apex, plus the original index of each slice.

    Pass base_end="first"/"last" to override detection (e.g. when it is uncertain).
    """
    if base_end is None:
        base_end = detect_base_apex(ci).base_end
    if base_end not in ("first", "last"):
        raise ValueError(f"base_end must be 'first' or 'last', got {base_end!r}")
    nz = ci.image.shape[2]
    order = list(range(nz)) if base_end == "first" else list(range(nz - 1, -1, -1))
    out = replace(
        ci,
        image=ci.image[:, :, order],
        label=None if ci.label is None else ci.label[:, :, order],
    )
    return out, order


def assign_levels(ci_base_first: CardiacImage) -> dict[int, str]:
    """Label each heart slice of a base-first stack as basal / mid / apical / apical cap.

    Slices with blood pool are split into thirds from base to apex (any extra slice
    goes to the basal and then mid levels). Myocardium-only slices at the apex are
    the apical cap, as in the AHA 17-segment model.
    """
    zs = _heart_slices(ci_base_first)
    has_blood = {z: bool(np.any(ci_base_first.label[:, :, z] == L.LV_BLOODPOOL)) for z in zs}
    cavity = [z for z in zs if has_blood[z]]
    levels: dict[int, str] = {}
    for name, chunk in zip(("basal", "mid", "apical"), np.array_split(np.array(cavity), 3)):
        for z in chunk:
            levels[int(z)] = name
    for z in zs:
        if not has_blood[z]:
            levels[z] = "apical cap" if z > max(cavity) else "unassigned"
    return dict(sorted(levels.items()))
