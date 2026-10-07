"""The data behind one open scan in the app, independent of the user interface.

A Session holds a cine stack (x, y, slice, time), its labels, the ED/ES frames,
and the slice order the user chose (base first, with excluded slices). Keeping
this free of Qt means it can be tested without a screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import labels as L
from ..io import CardiacImage
from ..orientation import OrientationUncertain, detect_base_apex
from ..volumes import FunctionResult, compute_function
from ..wall import WallResult, compute_wall


def wall_cover(img: np.ndarray, blood: np.ndarray, wall: np.ndarray, ref_ratio: float, n_sectors: int = 36) -> float:
    """Fraction of the way around the slice where the drawn wall looks like muscle.

    In bright-blood cine, muscle is dark and blood bright. Intensities are taken
    relative to this slice's own blood pool (slice brightness varies, e.g. inflow
    at the base). `ref_ratio` is wall/blood brightness in this heart's mid slices;
    a sector counts as muscle if its ratio is below midway between ref_ratio and 1.
    A drawn ring that crosses the outflow tract or atrium has blood-bright sectors.
    """
    xs, ys = np.nonzero(blood)
    cx, cy = xs.mean(), ys.mean()
    wx, wy = np.nonzero(wall)
    ang = (np.arctan2(wy - cy, wx - cx) + np.pi) / (2 * np.pi) * n_sectors
    sector = np.minimum(ang.astype(int), n_sectors - 1)
    vals = img[wx, wy] / max(float(np.median(img[blood])), 1e-6)
    threshold = (ref_ratio + 1.0) / 2
    muscle, seen = 0, 0
    for k in range(n_sectors):
        v = vals[sector == k]
        if v.size:
            seen += 1
            muscle += int(np.median(v) < threshold)
    return muscle / seen if seen else 0.0


@dataclass
class Session:
    """One open scan.

    Attributes:
        image: (x, y, slice, time) intensities.
        label: (x, y, slice, time) standard labels (0 where nothing is drawn).
        spacing: (dx, dy, dz) in mm.
        order: slice indices from base to apex, as the user arranged them.
        included: per phase ("ED", "ES"), the slice indices that count toward results.
            ED and ES are separate because the base moves toward the apex in systole: the
            top slice can be LV at ED but atrium/outflow tract at ES.
        reasons: per phase, a short note for each slice explaining the suggestion.
        slice_positions_mm: physical slice positions when the file has them (DICOM),
            used to warn if the user's order contradicts the scanner geometry.
    """

    image: np.ndarray
    label: np.ndarray
    spacing: tuple[float, float, float]
    subject_id: str = ""
    papillary: str | None = None
    ed_frame: int = 0
    es_frame: int = 0
    order: list[int] = field(default_factory=list)
    included: dict[str, set[int]] = field(default_factory=dict)
    reasons: dict[str, dict[int, str]] = field(default_factory=dict)
    slice_positions_mm: list[float] | None = None
    source: str = ""
    flags: dict = field(default_factory=dict)  # (frame, slice) -> reason from auto-contour checks
    label_detailed: np.ndarray | None = None  # model output before optional papillary smoothing
    contoured_by: str = ""

    def __post_init__(self) -> None:
        if self.image.ndim != 4 or self.label.shape != self.image.shape:
            raise ValueError("image and label must both be (x, y, slice, time) with the same shape")
        nz = self.image.shape[2]
        if not self.order:
            self.order = list(range(nz))
        if not self.included:
            self.included = {"ED": set(range(nz)), "ES": set(range(nz))}
        if not self.reasons:
            self.reasons = {"ED": {}, "ES": {}}

    # ---------- loading ----------

    @classmethod
    def from_ucl(cls, path: str | Path, with_labels: bool = True) -> "Session":
        """Open a UCL .h5 file. with_labels=False opens it as a raw cine (no expert contours),
        which mimics a fresh scanner export for trying auto-contour."""
        import h5py

        from ..datasets.ucl import UCL_SPACING, find_ed_es
        from ..labels import LABEL_MAPS

        with h5py.File(path, "r") as f:
            image = np.asarray(f["Images"], dtype=np.float32)
            raw = np.asarray(f["Masks"])
        lut = np.zeros(int(raw.max()) + 1, dtype=np.uint8)
        for src, dst in LABEL_MAPS["ucl"].items():
            if src < lut.size:
                lut[src] = dst
        label = lut[raw]
        if not with_labels:
            return cls(image=image, label=np.zeros(image.shape, np.uint8), spacing=UCL_SPACING,
                       subject_id=Path(path).stem + " (raw)", source=str(path))
        frames = find_ed_es(path)
        s = cls(image=image, label=label, spacing=UCL_SPACING, subject_id=Path(path).stem,
                papillary="myocardium", ed_frame=frames["ED"], es_frame=frames["ES"], source=str(path))
        s.auto_order()
        s.apply_suggestions()
        return s

    @classmethod
    def from_dicom(cls, folder: str | Path) -> "Session":
        """Open a DICOM cine folder. No contours yet: they are drawn by hand or by auto-contour."""
        from ..dicom import load_dicom_cine

        cine = load_dicom_cine(folder)
        image = np.asarray(cine.data, dtype=np.float32)
        return cls(image=image, label=np.zeros(image.shape, dtype=np.uint8), spacing=cine.spacing,
                   subject_id=cine.subject_id, ed_frame=0, es_frame=0,
                   slice_positions_mm=list(cine.slice_positions_mm), source=str(folder))

    # ---------- slice order ----------

    def flip(self) -> None:
        self.order = self.order[::-1]

    def move(self, slice_index: int, new_position: int) -> None:
        self.order.remove(slice_index)
        self.order.insert(new_position, slice_index)

    def set_included(self, slice_index: int, include: bool, phase: str | None = None) -> None:
        """Include/leave out a slice for one phase, or both when phase is None."""
        for ph in ([phase] if phase else ["ED", "ES"]):
            (self.included[ph].add if include else self.included[ph].discard)(slice_index)
            self.reasons[ph][slice_index] = "set by user"

    def suggest_slices(self, min_wall_cover: float = 0.5) -> dict[str, dict[int, tuple[bool, str]]]:
        """Suggested inclusion for every slice and phase, with the reason.

        Rules (common clinical practice, adapted to mouse SAX stacks):
          - no contour at this phase -> leave out
          - blood pool and wall present -> include, unless it is one of the two most
            basal contoured slices and less than `min_wall_cover` of the drawn ring
            looks like muscle in the image: then it is flagged "check" but still
            included. Contours alone cannot show a partial ring (experts close it
            anyway), and the image check is not reliable enough to exclude slices on
            its own: on 130 UCL mice it flagged 5 basal slices, of which flow
            artifacts or bright slices explained most (2026-10-07).
          - wall only (no blood pool) at the apical end -> include (apical cap)
          - wall only elsewhere -> leave out (check the contours)
        """
        out: dict[str, dict[int, tuple[bool, str]]] = {}
        for ph, t in (("ED", self.ed_frame), ("ES", self.es_frame)):
            lab = self.label[:, :, :, t]
            img = self.image[:, :, :, t]
            drawn = [z for z in self.order if np.any(lab[:, :, z])]
            mid = drawn[len(drawn) // 3: max(len(drawn) // 3 + 1, 2 * len(drawn) // 3)]
            wall_ref = np.median(img[:, :, mid][np.isin(lab[:, :, mid], (L.LV_MYOCARDIUM, L.SCAR))]) if mid else 0
            blood_ref = np.median(img[:, :, mid][lab[:, :, mid] == L.LV_BLOODPOOL]) if mid else 0
            ref_ratio = float(wall_ref / blood_ref) if blood_ref else 0.5
            res: dict[int, tuple[bool, str]] = {}
            for z in self.order:
                sl = lab[:, :, z]
                blood = sl == L.LV_BLOODPOOL
                wall = (sl == L.LV_MYOCARDIUM) | (sl == L.SCAR)
                if not blood.any() and not wall.any():
                    res[z] = (False, "no contour")
                elif not blood.any():
                    apical = drawn and drawn.index(z) >= len(drawn) - 2
                    res[z] = (True, "apical cap (wall only)") if apical else (False, "wall but no blood pool: check")
                elif not wall.any():
                    res[z] = (False, "blood pool but no wall: check")
                else:
                    cover = wall_cover(img[:, :, z], blood, wall, ref_ratio)
                    basal = drawn.index(z) < 2
                    if basal and cover < min_wall_cover:
                        res[z] = (True, f"check: basal, only {cover:.0%} of the ring looks like muscle "
                                        "(outflow tract/atrium?)")
                    else:
                        res[z] = (True, f"wall covers {cover:.0%}")
            out[ph] = res
        return out

    def apply_suggestions(self) -> None:
        for ph, res in self.suggest_slices().items():
            self.included[ph] = {z for z, (inc, _) in res.items() if inc}
            self.reasons[ph] = {z: why for z, (_, why) in res.items()}
            t = self.ed_frame if ph == "ED" else self.es_frame
            for (ft, z), why in self.flags.items():  # auto-contour checks override "wall covers"
                if ft == t and z in self.included[ph]:
                    self.reasons[ph][z] = f"check: {why}"

    # ---------- auto-contour ----------

    def run_autocontour(self, model: str = "ucl", all_frames: bool = True, compare_with: str | None = None,
                        progress=None) -> str:
        """Contour with a published model; flags slices to review; suggests ED/ES. Returns a status line."""
        from .. import autocontour as ac

        frames = list(range(self.image.shape[3])) if all_frames else sorted({self.ed_frame, self.es_frame})
        lab = ac.run(self.image, self.spacing, model, frames=frames, progress=progress)
        flags = {k: v for k, v in ac.sanity_flags(lab, frames).items()}
        if compare_with:
            other = ac.run(self.image, self.spacing, compare_with, frames=frames)
            for k, d in ac.disagreement(lab, other, frames).items():
                flags.setdefault(k, f"models disagree (overlap {d:.0%})")
        self.label = lab
        self.label_detailed = lab.copy()
        self.papillary = ac.MODELS[model].papillary
        self.contoured_by = ac.MODELS[model].label
        self.flags = flags
        msg = f"Contoured {len(frames)} frame(s) with the {ac.MODELS[model].label}."
        if all_frames:
            msg += " " + self.auto_ed_es()
        self.auto_order()
        self.apply_suggestions()
        n = len({z for (t, z) in flags if t in (self.ed_frame, self.es_frame)})
        msg += f" {n} slice(s) flagged for review." if n else " No slices flagged."
        return msg

    def set_smooth(self, on: bool) -> None:
        """Experimental: move papillary muscles into the blood pool (see papillary.py)."""
        from ..papillary import smooth_papillary

        if self.label_detailed is None:
            self.label_detailed = self.label.copy()
        if on:
            self.label = smooth_papillary(self.label_detailed, self.spacing)
            self.papillary = "blood_pool"
        else:
            self.label = self.label_detailed.copy()
            self.papillary = "myocardium" if self.contoured_by.startswith("UCL") or self.source.endswith(".h5") else None

    def volume_curve(self) -> list[float]:
        """LV blood-pool volume (uL) per frame over contoured frames (0 where a frame has no contours)."""
        dx, dy, dz = self.spacing
        blood = (self.label == L.LV_BLOODPOOL).reshape(-1, self.label.shape[3])
        return [float(v) * dx * dy * dz for v in blood.sum(axis=0)]

    def auto_order(self) -> str:
        """Order slices base-first using the blood-pool taper. Returns a short status message."""
        try:
            o = detect_base_apex(self.frame("ED", reorder=False))
        except (OrientationUncertain, ValueError) as e:
            return f"Could not tell base from apex automatically ({e}). Set the order by hand."
        nz = self.image.shape[2]
        self.order = list(range(nz)) if o.base_end == "first" else list(range(nz - 1, -1, -1))
        return f"Ordered base to apex automatically (basal third {o.thirds_ratio:.2f}x the apical)."

    def order_warnings(self) -> list[str]:
        """Checks on the user's order. Warnings only: the user decides."""
        out = []
        active = [z for z in self.order if z in self.included["ED"]]
        ed = self.label[:, :, :, self.ed_frame]
        areas = [int(np.count_nonzero(ed[:, :, z] == L.LV_BLOODPOOL)) for z in active]
        labeled = [a for a in areas if a > 0]
        if len(labeled) >= 3:
            k = max(1, len(labeled) // 3)
            base, apex = np.mean(labeled[:k]), np.mean(labeled[-k:])
            if apex > 1.25 * base:
                out.append("The apex end has a larger blood pool than the base end. The order may be reversed.")
        if self.slice_positions_mm is not None and len(active) > 2:
            pos = [self.slice_positions_mm[z] for z in active]
            steps = np.sign(np.diff(pos))
            if not (np.all(steps > 0) or np.all(steps < 0)):
                out.append("This order does not follow the slice positions recorded by the scanner.")
        return out

    # ---------- results ----------

    def frame(self, phase: str, reorder: bool = True) -> CardiacImage:
        """One phase as a CardiacImage; by default in the user's order, excluded slices removed."""
        t = {"ED": self.ed_frame, "ES": self.es_frame}[phase]
        zs = [z for z in self.order if z in self.included[phase]] if reorder else list(range(self.image.shape[2]))
        return CardiacImage(
            image=self.image[:, :, zs, t],
            spacing=self.spacing,
            label=self.label[:, :, zs, t],
            subject_id=self.subject_id,
            phase=phase,
            papillary=self.papillary,
        )

    def auto_ed_es(self) -> str:
        """ED/ES = frames with the largest/smallest drawn blood pool among frames contoured on (nearly) all slices."""
        blood = (self.label == L.LV_BLOODPOOL).reshape(-1, *self.label.shape[2:])
        drawn = (self.label > 0).reshape(-1, *self.label.shape[2:])
        n_slices = drawn.any(axis=0).sum(axis=0)  # per frame
        if n_slices.max() == 0:
            return "No contours yet: draw them or run auto-contour first."
        cand = np.flatnonzero(n_slices >= n_slices.max() - 1)
        vol = blood.sum(axis=(0, 1))
        if len(cand) < 2:
            return "Only one contoured frame: contour both end-diastole and end-systole."
        self.ed_frame = int(cand[np.argmax(vol[cand])])
        self.es_frame = int(cand[np.argmin(vol[cand])])
        return f"End-diastole = frame {self.ed_frame}, end-systole = frame {self.es_frame}."

    def compute(self) -> tuple[FunctionResult, WallResult | None, list[str]]:
        """Volumes and wall thickness using the user's order (base first). Returns notes too."""
        notes = self.order_warnings()
        ed, es = self.frame("ED"), self.frame("ES")
        if not np.any(ed.label == L.LV_BLOODPOOL) or not np.any(es.label == L.LV_BLOODPOOL):
            raise ValueError("Both end-diastole and end-systole need a drawn blood pool.")
        fn = compute_function(ed, es)
        try:
            wall = compute_wall(ed, es, base_end="first")  # the user's order is base-first by definition
        except Exception as e:  # wall needs myocardium; volumes are still useful without it
            wall = None
            notes.append(f"Wall thickness not computed: {e}")
        return fn, wall, notes
