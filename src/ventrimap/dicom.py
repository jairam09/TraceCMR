"""Reading cine short-axis stacks from DICOM.

Scanners export cine either as many single-frame files (one per slice and time
point) or as "enhanced" multi-frame files that pack frames into one file; Bruker
ParaVision can produce both. This module reads either, sorts frames into
(x, y, slice, time) using the geometry in the headers, and returns standard
CardiacImage objects with spacing taken from the files.

Sorting rules:
  - Slice order: position of each frame along the slice normal
    (ImagePositionPatient projected onto the cross product of the
    ImageOrientationPatient row and column directions).
  - Time order: TriggerTime (single-frame) or NominalCardiacTriggerDelayTime
    (enhanced); falls back to InstanceNumber / frame order if missing.
  - Slice spacing: median distance between neighbouring slice positions,
    which includes any gap (SliceThickness alone would not).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import CardiacImage


@dataclass
class _Frame:
    pixels: np.ndarray  # (rows, cols) after rescale
    position: np.ndarray  # ImagePositionPatient, mm
    orientation: np.ndarray  # 6 direction cosines
    pixel_spacing: tuple[float, float]  # (row spacing, column spacing) in mm
    time: float
    order: int  # tie-breaker: instance number or frame index


def _pydicom():
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading DICOM needs pydicom: pip install pydicom") from e
    return pydicom


def _rescale(ds, arr):
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr.astype(np.float32) * slope + intercept


def _frames_from_classic(ds) -> list[_Frame]:
    return [
        _Frame(
            pixels=_rescale(ds, ds.pixel_array),
            position=np.array(ds.ImagePositionPatient, dtype=float),
            orientation=np.array(ds.ImageOrientationPatient, dtype=float),
            pixel_spacing=tuple(float(v) for v in ds.PixelSpacing),
            # Explicit None check: a trigger time of 0 ms is valid, not missing.
            time=float(ds.TriggerTime) if getattr(ds, "TriggerTime", None) not in (None, "") else np.nan,
            order=int(getattr(ds, "InstanceNumber", 0) or 0),
        )
    ]


def _frames_from_enhanced(ds) -> list[_Frame]:
    shared = ds.SharedFunctionalGroupsSequence[0] if "SharedFunctionalGroupsSequence" in ds else None
    pixels = ds.pixel_array
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis]
    frames = []
    for i, pf in enumerate(ds.PerFrameFunctionalGroupsSequence):

        def get(seq_name, attr):
            for group in (pf, shared):
                if group is not None and seq_name in group:
                    item = getattr(group, seq_name)[0]
                    if attr in item:
                        return getattr(item, attr)
            return None

        spacing = get("PixelMeasuresSequence", "PixelSpacing")
        orient = get("PlaneOrientationSequence", "ImageOrientationPatient")
        pos = get("PlanePositionSequence", "ImagePositionPatient")
        t = get("CardiacSynchronizationSequence", "NominalCardiacTriggerDelayTime")
        if spacing is None or orient is None or pos is None:
            raise ValueError(f"enhanced DICOM frame {i} is missing geometry (spacing/orientation/position)")
        frames.append(
            _Frame(
                pixels=_rescale(ds, pixels[i]),
                position=np.array(pos, dtype=float),
                orientation=np.array(orient, dtype=float),
                pixel_spacing=(float(spacing[0]), float(spacing[1])),
                time=float(t) if t is not None else np.nan,
                order=i,
            )
        )
    return frames


def read_dicom_frames(paths: list[Path]) -> list[_Frame]:
    """Read every image frame from a list of DICOM files (classic or enhanced)."""
    pydicom = _pydicom()
    frames: list[_Frame] = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p))
        except Exception:  # not a DICOM file (e.g. DICOMDIR text, thumbnails)
            continue
        if "PixelData" not in ds:
            continue
        if "PerFrameFunctionalGroupsSequence" in ds:
            frames.extend(_frames_from_enhanced(ds))
        else:
            frames.extend(_frames_from_classic(ds))
    if not frames:
        raise ValueError("no DICOM image frames found")
    return frames


@dataclass
class DicomCine:
    """A sorted cine stack: data is (x, y, slice, time)."""

    data: np.ndarray
    spacing: tuple[float, float, float]
    times_ms: np.ndarray
    slice_positions_mm: np.ndarray
    subject_id: str

    def frame(self, t: int, papillary: str | None = None) -> CardiacImage:
        """One time frame as a CardiacImage (no label: contours come from elsewhere)."""
        if not 0 <= t < self.data.shape[3]:
            raise IndexError(f"frame {t} out of range (0..{self.data.shape[3] - 1})")
        return CardiacImage(
            image=self.data[:, :, :, t].copy(),
            spacing=self.spacing,
            subject_id=self.subject_id,
            phase=f"frame{t}",
            affine=np.diag([*self.spacing, 1.0]),
            papillary=papillary,
        )


def load_dicom_cine(folder: str | Path, subject_id: str | None = None) -> DicomCine:
    """Load a short-axis cine series from a folder of DICOM files.

    All files must belong to one series with one orientation. Raises if the
    frames do not form a complete slice x time grid, rather than guessing.
    """
    folder = Path(folder)
    paths = sorted(p for p in folder.rglob("*") if p.is_file())
    frames = read_dicom_frames(paths)

    o = frames[0].orientation
    for f in frames:
        if not np.allclose(f.orientation, o, atol=1e-3):
            raise ValueError("frames have different orientations: more than one series in the folder?")
        if not np.allclose(f.pixel_spacing, frames[0].pixel_spacing, rtol=1e-4):
            raise ValueError("frames have different pixel spacing")
        if f.pixels.shape != frames[0].pixels.shape:
            raise ValueError("frames have different matrix sizes")

    normal = np.cross(o[:3], o[3:])
    dist = np.array([float(np.dot(f.position, normal)) for f in frames])
    slice_pos = np.unique(np.round(dist, 3))
    n_slices = len(slice_pos)
    if len(frames) % n_slices:
        raise ValueError(f"{len(frames)} frames do not divide evenly into {n_slices} slices")
    n_times = len(frames) // n_slices

    grid = [[] for _ in range(n_slices)]
    for f, d in zip(frames, dist):
        grid[int(np.argmin(np.abs(slice_pos - round(d, 3))))].append(f)
    if any(len(g) != n_times for g in grid):
        counts = [len(g) for g in grid]
        raise ValueError(f"slices have different numbers of time frames: {counts}")

    rows, cols = frames[0].pixels.shape
    data = np.zeros((cols, rows, n_slices, n_times), dtype=np.float32)
    times = np.zeros((n_slices, n_times))
    for z, g in enumerate(grid):
        use_time = all(np.isfinite(f.time) for f in g)
        g.sort(key=(lambda f: (f.time, f.order)) if use_time else (lambda f: f.order))
        for t, f in enumerate(g):
            data[:, :, z, t] = f.pixels.T  # DICOM pixels are (rows=y, cols=x)
            times[z, t] = f.time

    row_spacing, col_spacing = frames[0].pixel_spacing  # DICOM order: (between rows, between columns)
    dz = float(np.median(np.diff(slice_pos))) if n_slices > 1 else 1.0
    return DicomCine(
        data=data,
        spacing=(col_spacing, row_spacing, abs(dz)),
        times_ms=np.nanmean(times, axis=0),
        slice_positions_mm=slice_pos,
        subject_id=subject_id or folder.name,
    )
