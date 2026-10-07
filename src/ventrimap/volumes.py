"""Global LV function from labeled ED and ES stacks (Phase 2).

Volumes use Simpson's method of discs: for each slice, area x slice spacing,
summed over slices. With voxel masks this is the same as counting voxels and
multiplying by the voxel volume. Units: mm^3 = microliters (uL); mass in mg.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import labels as L
from .io import CardiacImage

MYOCARDIAL_DENSITY_G_PER_ML = 1.05  # standard assumption; 1 uL of myocardium -> 1.05 mg

BLOOD = (L.LV_BLOODPOOL,)
MYOCARDIUM = (L.LV_MYOCARDIUM, L.SCAR, L.MVO)  # scar is still myocardial tissue

# QC thresholds (flags, not errors).
MASS_MISMATCH_FLAG = 0.15  # LV mass should be nearly identical at ED and ES


def slice_areas_mm2(ci: CardiacImage, values: tuple[int, ...]) -> np.ndarray:
    """Area of the given label(s) on each slice, in mm^2."""
    if ci.label is None:
        raise ValueError(f"{ci.subject_id} {ci.phase}: no label")
    dx, dy, _ = ci.spacing
    counts = np.isin(ci.label, values).sum(axis=(0, 1))
    return counts * dx * dy


def volume_ul(ci: CardiacImage, values: tuple[int, ...]) -> float:
    """Volume of the given label(s) in uL: sum of slice areas x slice spacing."""
    return float(slice_areas_mm2(ci, values).sum() * ci.spacing[2])


@dataclass
class FunctionResult:
    subject_id: str
    edv_ul: float
    esv_ul: float
    sv_ul: float
    ef_pct: float
    mass_ed_mg: float
    mass_es_mg: float
    papillary: str | None
    n_slices_ed: int
    n_slices_es: int
    flags: list[str] = field(default_factory=list)
    ed_blood_area_mm2: list[float] = field(default_factory=list, repr=False)
    es_blood_area_mm2: list[float] = field(default_factory=list, repr=False)

    def as_row(self) -> dict:
        """Flat dict for a CSV row (per-slice areas left out)."""
        row = asdict(self)
        row.pop("ed_blood_area_mm2")
        row.pop("es_blood_area_mm2")
        row["flags"] = "; ".join(self.flags)
        return row


def compute_function(ed: CardiacImage, es: CardiacImage) -> FunctionResult:
    """EDV, ESV, SV, EF and LV mass from an ED stack and an ES stack of the same heart.

    Refuses to combine stacks with different spacing or papillary conventions.
    Problems that may be real (or may be bugs) are returned as flags.
    """
    for ci in (ed, es):
        if ci.label is None:
            raise ValueError(f"{ci.subject_id} {ci.phase}: no label")
    if not np.allclose(ed.spacing, es.spacing):
        raise ValueError(f"ED spacing {ed.spacing} != ES spacing {es.spacing}")
    if ed.papillary != es.papillary:
        raise ValueError(
            f"ED and ES use different papillary conventions ({ed.papillary} vs {es.papillary})"
        )

    ed_area = slice_areas_mm2(ed, BLOOD)
    es_area = slice_areas_mm2(es, BLOOD)
    dz = ed.spacing[2]
    edv = float(ed_area.sum() * dz)
    esv = float(es_area.sum() * dz)
    sv = edv - esv
    ef = 100.0 * sv / edv if edv > 0 else float("nan")
    mass_ed = volume_ul(ed, MYOCARDIUM) * MYOCARDIAL_DENSITY_G_PER_ML
    mass_es = volume_ul(es, MYOCARDIUM) * MYOCARDIAL_DENSITY_G_PER_ML

    flags: list[str] = []
    if edv <= 0:
        flags.append("no blood pool at ED")
    if esv >= edv:
        flags.append("ESV >= EDV: ED and ES may be swapped")
    if mass_ed > 0 and abs(mass_es - mass_ed) / mass_ed > MASS_MISMATCH_FLAG:
        flags.append(
            f"LV mass differs {100 * (mass_es - mass_ed) / mass_ed:+.0f}% between ED and ES "
            "(should be nearly equal; check contours)"
        )
    if ed.papillary is None:
        flags.append("papillary convention unknown")

    return FunctionResult(
        subject_id=ed.subject_id,
        edv_ul=edv,
        esv_ul=esv,
        sv_ul=sv,
        ef_pct=ef,
        mass_ed_mg=mass_ed,
        mass_es_mg=mass_es,
        papillary=ed.papillary,
        n_slices_ed=int(np.count_nonzero(ed_area)),
        n_slices_es=int(np.count_nonzero(es_area)),
        flags=flags,
        ed_blood_area_mm2=ed_area.tolist(),
        es_blood_area_mm2=es_area.tolist(),
    )


def check_comparable(a: FunctionResult, b: FunctionResult, allow_mismatch: bool = False) -> None:
    """Raise if two results used different papillary conventions.

    Comparing across conventions mixes a systematic bias into what looks like
    method disagreement. Pass allow_mismatch=True only when that bias is the
    thing you are measuring.
    """
    if a.papillary != b.papillary and not allow_mismatch:
        raise ValueError(
            f"{a.subject_id}: papillary conventions differ ({a.papillary} vs {b.papillary}); "
            "results are not directly comparable"
        )
