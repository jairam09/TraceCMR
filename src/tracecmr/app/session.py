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


def _model_settings() -> dict:
    try:
        from .. import autocontour as ac

        return {"blood_cutoff": ac.blood_cutoff(), "epi_cutoff": ac.epi_cutoff()}
    except Exception:
        return {}


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
    mirror_segments: bool = False  # wall segments: slices seen from the base, not the apex (Settings)
    septum_manual: dict[int, float] = field(default_factory=dict)  # slice -> septum angle set by hand (degrees)
    # the editor's interpolation points as the user left them, so reopening shows exactly what was drawn (rebuilding
    # them from pixels smooths sharp tips a little every time); used only on slices whose pixels still match
    saved_curves: dict = field(default_factory=dict)  # (frame, slice, "endo"/"epi"/"rv") -> (n, 2) points
    saved_hidden: set = field(default_factory=set)  # keys smoothed with their points hidden
    saved_curve_hash: dict = field(default_factory=dict)  # (frame, slice) -> hash of that slice's labels
    # Per-slice ED/ES (Segment-style): when each short-axis slice is its own cine acquisition, the user
    # picks end-diastole and end-systole in each slice's cine. Slices not listed use ed_frame/es_frame.
    separate_slices: bool = False
    ed_by_slice: dict[int, int] = field(default_factory=dict)
    es_by_slice: dict[int, int] = field(default_factory=dict)
    frame_times_ms: list[float] | None = None
    slice_names: list[str] | None = None  # e.g. the series each slice came from
    # The model's own contours, kept apart from the user's corrected ones (training data for fine-tuning):
    # model_label holds the last auto-contour of each image, model_ran[z, t] says which images have one.
    model_label: np.ndarray | None = None
    model_ran: np.ndarray | None = None

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

    @classmethod
    def from_dicom_files(cls, paths, name: str = "series", separate_slices: bool = False,
                         slice_names: list[str] | None = None) -> "Session":
        """Open a cine from an explicit list of DICOM files (one series, or several single-slice series)."""
        from ..dicom import load_dicom_cine

        paths = [Path(p) for p in paths]
        cine = load_dicom_cine(paths[0].parent, subject_id=name, paths=paths)
        image = np.asarray(cine.data, dtype=np.float32)
        times = [float(t) for t in cine.times_ms] if np.all(np.isfinite(cine.times_ms)) else None
        return cls(image=image, label=np.zeros(image.shape, dtype=np.uint8), spacing=cine.spacing,
                   subject_id=name, ed_frame=0, es_frame=0, slice_positions_mm=list(cine.slice_positions_mm),
                   source=str(paths[0].parent), separate_slices=separate_slices, frame_times_ms=times,
                   slice_names=slice_names)

    # ---------- saving ----------

    def save_contours(self, path: str | Path) -> Path:
        """Save the contours and every choice made on them (ED/ES per slice, slice order, slices left
        out) to one .npz file. The images are not copied: they stay in the original scan files."""
        import json

        meta = {
            "format": "tracecmr-contours-1", "subject_id": self.subject_id, "source": self.source,
            "shape": list(self.image.shape), "spacing": list(self.spacing),
            "ed_frame": self.ed_frame, "es_frame": self.es_frame,
            "ed_by_slice": {str(k): v for k, v in self.ed_by_slice.items()},
            "es_by_slice": {str(k): v for k, v in self.es_by_slice.items()},
            "separate_slices": self.separate_slices, "order": list(self.order),
            "included": {ph: sorted(v) for ph, v in self.included.items()},
            "reasons": {ph: {str(k): v for k, v in r.items()} for ph, r in self.reasons.items()},
            "papillary": self.papillary, "contoured_by": self.contoured_by,
            "septum_manual": {str(k): float(v) for k, v in self.septum_manual.items()},
        }
        curve_extra = {}
        if self.saved_curves:
            keys = sorted(self.saved_curves)
            meta["curves"] = [[int(t), int(z), name, int(len(self.saved_curves[(t, z, name)])),
                               bool((t, z, name) in self.saved_hidden)] for t, z, name in keys]
            meta["curve_slice_hash"] = {f"{t},{z}": self.slice_hash(t, z) for t, z in {k[:2] for k in keys}}
            curve_extra["curve_points"] = np.concatenate([self.saved_curves[k] for k in keys]).astype(np.float32)
        extra = {}
        if self.model_label is not None and self.model_ran is not None and self.model_ran.any():
            zt = np.argwhere(self.model_ran)  # images of those slices, so the pairs stand on their own
            extra = {"model_label": self.model_label.astype(np.uint8), "model_ran": self.model_ran,
                     "model_zt": zt.astype(np.int16),
                     "model_images": np.stack([self.image[:, :, z, t] for z, t in zt]).astype(np.float32)}
            meta["model_settings"] = _model_settings()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez_compressed(f, label=self.label.astype(np.uint8), meta=json.dumps(meta), **extra, **curve_extra)
        return path

    def load_contours(self, path: str | Path) -> None:
        """Restore contours saved with save_contours onto this (same) scan."""
        import json

        with np.load(path, allow_pickle=False) as d:
            label = d["label"].copy()
            meta = json.loads(str(d["meta"]))
            model_label = d["model_label"].copy() if "model_label" in d.files else None
            model_ran = d["model_ran"].copy() if "model_ran" in d.files else None
            curve_points = d["curve_points"].astype(float) if "curve_points" in d.files else None
        if tuple(meta["shape"]) != self.image.shape:
            raise ValueError(f"saved contours are for a scan of shape {tuple(meta['shape'])}, "
                             f"this scan is {self.image.shape}")
        self.label = label
        self.label_detailed = label.copy()
        self.ed_frame, self.es_frame = int(meta["ed_frame"]), int(meta["es_frame"])
        self.ed_by_slice = {int(k): int(v) for k, v in meta["ed_by_slice"].items()}
        self.es_by_slice = {int(k): int(v) for k, v in meta["es_by_slice"].items()}
        self.separate_slices = bool(meta["separate_slices"])
        self.order = [int(z) for z in meta["order"]]
        self.included = {ph: set(int(z) for z in v) for ph, v in meta["included"].items()}
        self.reasons = {ph: {int(k): v for k, v in r.items()} for ph, r in meta["reasons"].items()}
        self.papillary = meta.get("papillary")
        self.contoured_by = meta.get("contoured_by", "")
        self.septum_manual = {int(k): float(v) for k, v in meta.get("septum_manual", {}).items()}
        self.saved_curves, self.saved_hidden, self.saved_curve_hash = {}, set(), {}
        if curve_points is not None and meta.get("curves"):
            i = 0
            for t, z, name, n, hid in meta["curves"]:
                self.saved_curves[(int(t), int(z), name)] = curve_points[i:i + int(n)].copy()
                if hid:
                    self.saved_hidden.add((int(t), int(z), name))
                i += int(n)
            self.saved_curve_hash = {tuple(int(v) for v in k.split(",")): h
                                     for k, h in meta.get("curve_slice_hash", {}).items()}
        self.model_label, self.model_ran = model_label, model_ran

    def slice_hash(self, t: int, z: int) -> str:
        import hashlib

        return hashlib.sha1(np.ascontiguousarray(self.label[:, :, z, t]).tobytes()).hexdigest()

    def curves_still_valid(self) -> dict:
        """Saved interpolation points for slices whose labels haven't changed since they were saved."""
        ok = {k[:2] for k in self.saved_curves
              if self.saved_curve_hash.get(k[:2]) == self.slice_hash(*k[:2])}
        return {k: v for k, v in self.saved_curves.items() if k[:2] in ok}

    # ---------- ED / ES frames ----------

    def t_for(self, z: int, phase: str) -> int:
        """The frame used as `phase` ('ED' or 'ES') on slice z."""
        if phase == "ED":
            return self.ed_by_slice.get(z, self.ed_frame)
        return self.es_by_slice.get(z, self.es_frame)

    def set_phase_frame(self, phase: str, t: int, z: int | None = None) -> None:
        """Set ED or ES to frame t on slice z, or on every slice when z is None."""
        overrides = self.ed_by_slice if phase == "ED" else self.es_by_slice
        if z is None:
            if phase == "ED":
                self.ed_frame = int(t)
            else:
                self.es_frame = int(t)
            overrides.clear()
            return
        base = self.ed_frame if phase == "ED" else self.es_frame
        if int(t) == base:
            overrides.pop(z, None)
        else:
            overrides[z] = int(t)

    def frames_in_use(self) -> list[int]:
        """Every frame that is ED or ES on some slice."""
        nz = self.image.shape[2]
        return sorted({self.t_for(z, ph) for z in range(nz) for ph in ("ED", "ES")})

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
        for ph in ("ED", "ES"):
            lab = self._phase_stack(self.label, ph)
            img = self._phase_stack(self.image, ph)
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
            for (ft, z), why in self.flags.items():  # auto-contour checks override "wall covers"
                if ft == self.t_for(z, ph) and z in self.included[ph]:
                    self.reasons[ph][z] = f"check: {why}"

    # ---------- auto-contour ----------

    def run_autocontour(self, model: str = "ucl", all_frames: bool = True, compare_with: str | None = None,
                        progress=None) -> str:
        """Contour with a published model; flags slices to review; suggests ED/ES. Returns a status line."""
        from .. import autocontour as ac

        frames = list(range(self.image.shape[3])) if all_frames else self.frames_in_use()
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
        n = len({z for (t, z) in flags if t in (self.t_for(z, "ED"), self.t_for(z, "ES"))})
        msg += f" {n} slice(s) flagged for review." if n else " No slices flagged."
        return msg

    def autocontour_pairs(self, pairs: list[tuple[int, int]], model: str = "ucl",
                          compare_with: str | None = None) -> str:
        """Contour only the given (slice, frame) images; contours elsewhere are kept. Returns a status line."""
        from .. import autocontour as ac

        lab = ac.run_pairs(self.image, self.spacing, pairs, model)
        new = self.label.copy()
        for k, (z, t) in enumerate(pairs):
            old_rv = new[:, :, z, t] == L.RV_BLOODPOOL
            new[:, :, z, t] = lab[:, :, k]
            # the LV model draws no RV: keep a hand-drawn RV wherever the new LV doesn't cover it
            new[:, :, z, t][old_rv & (lab[:, :, k] == 0)] = L.RV_BLOODPOOL
        # checks on the contoured images, laid out as one volume in slice order
        flags = {}
        vol = np.zeros(self.image.shape[:3] + (1,), np.uint8)
        for k, (z, t) in enumerate(pairs):
            vol[:, :, z, 0] = lab[:, :, k]
        when = dict(pairs)
        for (_, z), why in ac.sanity_flags(vol, [0]).items():
            if z in when:
                flags[(when[z], z)] = why
        for k, (z, t) in enumerate(pairs):  # apex at end-systole: a cavity of a few pixels is unreliable
            n_blood = int((lab[:, :, k] == L.LV_BLOODPOOL).sum())
            if 0 < n_blood < 25 and (t, z) not in flags:
                flags[(t, z)] = "tiny cavity: check the endo (redraw it, or Clear it if the cavity has closed)"
        done = set(pairs)
        for t in sorted({t for _, t in pairs}):  # the end slices are the least reliable, for people and models
            zs = [z for z in self.order if np.isin(new[:, :, z, t], (L.LV_BLOODPOOL, L.LV_MYOCARDIUM)).any()]
            if len(zs) < 2:
                continue
            for z, why in ((zs[0], "base slice: double-check it's ventricle (not atrium or outflow) and fix the "
                                   "outline"),
                           (zs[-1], "apex slice: double-check the outline (or leave it out if there's no cavity)")):
                if (z, t) in done and (t, z) not in flags:
                    flags[(t, z)] = why
        if compare_with:
            other = ac.run_pairs(self.image, self.spacing, pairs, compare_with)
            ovol = np.zeros_like(vol)
            for k, (z, t) in enumerate(pairs):
                ovol[:, :, z, 0] = other[:, :, k]
            for (_, z), d in ac.disagreement(vol, ovol, [0]).items():
                if z in when:
                    flags.setdefault((when[z], z), f"models disagree (overlap {d:.0%})")
        self.label = new
        self.label_detailed = new.copy()
        if self.model_label is None or self.model_label.shape != self.label.shape:
            self.model_label = np.zeros(self.label.shape, np.uint8)
            self.model_ran = np.zeros(self.label.shape[2:], bool)
        for k, (z, t) in enumerate(pairs):
            self.model_label[:, :, z, t] = lab[:, :, k]
            self.model_ran[z, t] = True
        self.papillary = ac.MODELS[model].papillary
        self.contoured_by = ac.MODELS[model].label
        self.flags = {**{k: v for k, v in self.flags.items() if k not in {(t, z) for z, t in pairs}}, **flags}
        self.apply_suggestions()
        empty = [z for k, (z, t) in enumerate(pairs) if not lab[:, :, k].any()]
        frames = sorted({t for _, t in pairs})
        what = f"frame {frames[0]}" if len(frames) == 1 else "each slice's own frame"
        msg = f"Contoured {len(pairs)} slice(s) at {what} with the {ac.MODELS[model].label}."
        msg += f" {len(flags)} slice(s) flagged for review." if flags else " No slices flagged."
        if empty:
            msg += (f" ⚠ The model found no LV on slice(s) {', '.join(map(str, sorted(empty)))} (too small or too "
                    "little contrast, common at the apex): draw it by hand (click points around it, right-click "
                    "to close), or leave it out if there's no cavity.")
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
        ed = self._phase_stack(self.label, "ED")
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

    def _phase_stack(self, arr4d: np.ndarray, phase: str) -> np.ndarray:
        """(x, y, slice) taking every slice from its own ED or ES frame."""
        nz = arr4d.shape[2]
        return np.stack([arr4d[:, :, z, self.t_for(z, phase)] for z in range(nz)], axis=2)

    def frame(self, phase: str, reorder: bool = True) -> CardiacImage:
        """One phase as a CardiacImage; by default in the user's order, excluded slices removed."""
        zs = [z for z in self.order if z in self.included[phase]] if reorder else list(range(self.image.shape[2]))
        img, lab = self._phase_stack(self.image, phase), self._phase_stack(self.label, phase)
        return CardiacImage(
            image=img[:, :, zs],
            spacing=self.spacing,
            label=lab[:, :, zs],
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
        self.ed_by_slice.clear()
        self.es_by_slice.clear()
        if not self.separate_slices:
            return f"End-diastole = frame {self.ed_frame}, end-systole = frame {self.es_frame}."
        # separately acquired slices: each slice's own largest / smallest blood pool
        area = blood.sum(axis=0)  # (slice, frame)
        moved = 0
        for z in range(area.shape[0]):
            tz = np.flatnonzero(drawn[:, z, :].any(axis=0))
            if len(tz) < 2 or not area[z, tz].any():
                continue
            self.set_phase_frame("ED", int(tz[np.argmax(area[z, tz])]), z)
            self.set_phase_frame("ES", int(tz[np.argmin(area[z, tz])]), z)
            moved += z in self.ed_by_slice or z in self.es_by_slice
        return (f"ED/ES chosen per slice (separate cines); most slices: ED = frame {self.ed_frame}, "
                f"ES = frame {self.es_frame}; {moved} slice(s) differ.")

    def per_slice(self) -> list[dict]:
        """Segment-style per-slice results, base to apex, for slices with a blood pool at both ED and ES.

        Each row: slice, ED/ES blood-pool volume (uL), slice EF (%), ED myocardial mass (mg), and whether the
        slice counts in the final result (it must be selected for both phases). The final volumes are the
        sums over the selected slices.
        """
        from ..volumes import MYOCARDIAL_DENSITY_G_PER_ML

        dx, dy, dz = self.spacing
        vox = dx * dy * dz
        rows = []
        for z in self.order:
            ed = self.label[:, :, z, self.t_for(z, "ED")]
            es = self.label[:, :, z, self.t_for(z, "ES")]
            edv = float((ed == L.LV_BLOODPOOL).sum()) * vox
            esv = float((es == L.LV_BLOODPOOL).sum()) * vox
            if edv <= 0 or esv <= 0:
                continue
            myo = float(np.isin(ed, (L.LV_MYOCARDIUM, L.SCAR)).sum()) * vox
            rows.append({"slice": z, "edv_ul": edv, "esv_ul": esv, "ef_pct": 100 * (edv - esv) / edv,
                         "mass_ed_mg": myo * MYOCARDIAL_DENSITY_G_PER_ML,
                         "included": z in self.included["ED"] and z in self.included["ES"],
                         "es_not_set": self.t_for(z, "ES") == self.t_for(z, "ED")})
        return rows

    def rv_volumes(self) -> dict | None:
        """RV end-diastolic/end-systolic volume (uL), stroke volume and EF from the RV contours, summed over
        every slice that has one (slices are taken at their own ED/ES frames). None without RV contours."""
        dx, dy, dz = self.spacing
        vol = {}
        for ph in ("ED", "ES"):
            rv = self._phase_stack(self.label, ph) == L.RV_BLOODPOOL
            vol[ph] = float(rv.sum()) * dx * dy * dz
        if vol["ED"] <= 0 or vol["ES"] <= 0:
            return None
        sv = vol["ED"] - vol["ES"]
        return {"rvedv_ul": vol["ED"], "rvesv_ul": vol["ES"], "rvsv_ul": sv, "rvef_pct": 100 * sv / vol["ED"]}

    def es_not_set(self) -> list[int]:
        """Selected slices whose ES is still the same frame as their ED (so they look like they don't contract)."""
        both = self.included["ED"] & self.included["ES"]
        return [z for z in self.order if z in both and self.t_for(z, "ES") == self.t_for(z, "ED")
                and np.any(self.label[:, :, z, self.t_for(z, "ED")] == L.LV_BLOODPOOL)]

    def compute(self) -> tuple[FunctionResult, WallResult | None, list[str]]:
        """Volumes and wall thickness using the user's order (base first). Returns notes too."""
        notes = self.order_warnings()
        missing = self.es_not_set()
        if missing:
            notes.insert(0, f"ES isn't set on slice(s) {', '.join(map(str, missing))}: their ES is the same frame "
                            f"as ED, so they count as not contracting and EF reads too low. Set ES on each slice "
                            f"(or untick Per-slice ED/ES to use one ES for all).")
        ed, es = self.frame("ED"), self.frame("ES")
        if not np.any(ed.label == L.LV_BLOODPOOL) or not np.any(es.label == L.LV_BLOODPOOL):
            raise ValueError("Both end-diastole and end-systole need a drawn blood pool.")
        fn = compute_function(ed, es)
        try:
            man = {}
            for ph in ("ED", "ES"):  # slice index -> position in that phase's stack (the user's order)
                zs = [z for z in self.order if z in self.included[ph]]
                man[ph] = {i: self.septum_manual[z] for i, z in enumerate(zs) if z in self.septum_manual}
            wall = compute_wall(ed, es, base_end="first",  # the user's order is base-first by definition
                                mirror=self.mirror_segments, manual_septum=man)
        except Exception as e:  # wall needs myocardium; volumes are still useful without it
            wall = None
            notes.append(f"Wall thickness not computed: {e}")
        return fn, wall, notes
