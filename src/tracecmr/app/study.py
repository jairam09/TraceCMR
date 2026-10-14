"""A study: the image series of one examination, shown as thumbnails like Segment's image stacks.

A Bruker or clinical export folder usually holds several series (short-axis cine,
long-axis cines, scout, late enhancement). `scan_dicom_folder` groups the files
by DICOM SeriesInstanceUID without reading pixel data, guesses what each series
is from its description, and loads a series only when the user opens it. A Bruker
ParaVision study folder (numbered scan folders with pdata/1/2dseq) is read directly
by `scan_bruker_study`, with no DICOM export needed. Some protocols save every
short-axis slice as its own single-slice cine; `combine` stacks such series into one
short-axis cine (slice order then comes from the slice positions, and the user can
still reorder slices in the panel).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .session import Session

KINDS = ("Short axis", "Long axis", "LGE", "Other")


def guess_kind(description: str) -> str:
    """Guess the series type from its DICOM description / protocol name."""
    d = " " + description.lower().replace("_", " ").replace("-", " ") + " "
    if any(k in d for k in ("lge", "late", "scar", " psir", " de ", "delayed")):
        return "LGE"
    if any(k in d for k in ("2ch", "3ch", "4ch", " lax", "long axis", "longaxis", " la ", "2 ch", "4 ch")):
        return "Long axis"
    if any(k in d for k in (" sax", " sa ", "short axis", "shortaxis", "cine sa", " sa")):
        return "Short axis"
    return "Other"


@dataclass
class Series:
    name: str
    kind: str
    source: str
    n_slices: int = 0
    n_frames: int = 0
    paths: list[Path] = field(default_factory=list)  # DICOM files, empty for a UCL file
    ucl_raw: bool = False
    position_mm: float = 0.0  # slice position along the normal (single-slice series)
    session: Session | None = None
    thumb: np.ndarray | None = None  # small 2D preview image
    geometry: tuple = ()  # (orientation rounded, rows, cols): series with equal geometry can be stacked
    parts: list["Series"] = field(default_factory=list)  # for a combined stack: the series it was made from
    stack: "Series | None" = None  # for a part: the stack it belongs to
    slice_index: int = 0  # for a part: its slice in the stack
    pv: object | None = None  # a Bruker ParaVision scan (tracecmr.bruker.PvScan) read from 2dseq

    def load(self) -> Session:
        """Open the series (cached: edits made in the app stay with it while the study is open)."""
        if self.session is None:
            if self.parts:
                self.session = _load_stack(self)
            elif self.pv is not None:
                self.session = _session_from_cine(_cine(self), self.name, self.source)
            elif self.paths:
                self.session = Session.from_dicom_files(self.paths, name=self.name)
            else:
                self.session = Session.from_ucl(self.source, with_labels=not self.ucl_raw)
            t = 0
            img = self.session.image
            self.thumb = img[:, :, img.shape[2] // 2, t]
        return self.session

    @property
    def contoured(self) -> bool:
        return self.session is not None and bool(np.any(self.session.label))

    def summary(self) -> str:
        s = f"{self.kind} · {self.n_slices} slice{'s' if self.n_slices != 1 else ''} × {self.n_frames} frames"
        return s + (" · contoured" if self.contoured else "")


def ucl_series(path: str | Path, raw: bool = False) -> Series:
    import h5py

    with h5py.File(str(path), "r") as f:
        shape = f["Images"].shape  # (x, y, slice, time)
        mid = np.asarray(f["Images"][:, :, shape[2] // 2, 0], dtype=np.float32)
    return Series(name=Path(path).stem + (" (raw)" if raw else ""), kind="Short axis", source=str(path),
                  n_slices=int(shape[2]), n_frames=int(shape[3]), ucl_raw=raw, thumb=mid)


def scan_dicom_folder(folder: str | Path) -> list[Series]:
    """Group the DICOM files under a folder into series (headers only; pixels are read on opening)."""
    import pydicom

    groups: dict[str, list] = defaultdict(list)
    for p in sorted(x for x in Path(folder).rglob("*") if x.is_file()):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        if not hasattr(ds, "ImagePositionPatient") and "PerFrameFunctionalGroupsSequence" not in ds:
            continue  # not an image (DICOMDIR, reports, ...)
        key = str(getattr(ds, "SeriesInstanceUID", "") or p.parent)  # no UID: one series per folder
        groups[key].append((p, ds))
    out = []
    for uid, items in groups.items():
        ds0 = items[0][1]
        desc = str(getattr(ds0, "SeriesDescription", "") or getattr(ds0, "ProtocolName", "") or "")
        num = getattr(ds0, "SeriesNumber", None)
        name = f"{num} {desc}".strip() if num is not None else (desc or uid[-8:])
        geom = ()
        if "PerFrameFunctionalGroupsSequence" in ds0:
            from ..dicom import enhanced_geometry

            try:  # one or more multi-frame files: slices from the frame positions (Bruker: one file per slice)
                o, positions = None, []
                for _, d in items:
                    oo, pp = enhanced_geometry(d)
                    o = oo if o is None else o
                    positions += pp
                normal = np.cross(o[:3], o[3:])
                dists = np.round([float(np.dot(q, normal)) for q in positions], 3)
                n_slices = len(np.unique(dists))
                n_frames = len(positions) // max(n_slices, 1)
                pos = float(np.mean(dists))
                geom = (tuple(np.round(o, 2)), int(ds0.Rows), int(ds0.Columns))
            except (ValueError, AttributeError, KeyError):
                n_slices, n_frames = 0, int(getattr(ds0, "NumberOfFrames", 1))
                pos = 0.0
        else:
            geom = (tuple(np.round(np.array(ds0.ImageOrientationPatient, float), 2)), int(ds0.Rows),
                    int(ds0.Columns))
            o = np.array(ds0.ImageOrientationPatient, float)
            normal = np.cross(o[:3], o[3:])
            dists = np.round([float(np.dot(np.array(d.ImagePositionPatient, float), normal)) for _, d in items], 3)
            n_slices = len(np.unique(dists))
            n_frames = len(items) // max(n_slices, 1)
            pos = float(np.mean(dists))
        series = Series(name=name, kind=guess_kind(desc), source=str(folder), n_slices=n_slices,
                        n_frames=n_frames, paths=[p for p, _ in items], position_mm=pos, geometry=geom)
        try:  # a preview from one file in the middle of the series
            mid = pydicom.dcmread(str(items[len(items) // 2][0]))
            arr = mid.pixel_array
            series.thumb = np.asarray(arr[0] if arr.ndim == 3 else arr, dtype=np.float32).T
        except Exception:
            pass
        out.append(series)
    out.sort(key=lambda s: (KINDS.index(s.kind), s.name))
    return out


def scan_bruker_study(folder: str | Path) -> list[Series]:
    """List the scans of a ParaVision study folder as series (parameters only; pixels are read on opening,
    except a small preview of each)."""
    from .. import bruker

    out = []
    for sc in bruker.scan_study(folder):
        if sc.n_frames < 1 or sc.dim != 2:
            continue
        name = f"E{sc.number} {sc.name}"
        s = Series(name=name, kind=guess_kind(f"{sc.name} {sc.protocol}"), source=str(folder),
                   n_slices=sc.n_slices, n_frames=sc.n_frames, position_mm=float(np.mean(sc.slice_distances())),
                   geometry=sc.geometry, pv=sc)
        try:
            data = bruker.load_scan(sc)
            s.thumb = data[:, :, data.shape[2] // 2, 0]
        except Exception:
            pass
        out.append(s)
    out.sort(key=lambda s: (KINDS.index(s.kind), s.pv.number))
    return out


def combine(series: list[Series], name: str | None = None) -> Series:
    """Stack several series (e.g. one cine per short-axis slice) into one short-axis cine."""
    if len(series) < 2:
        raise ValueError("pick at least two series to combine")
    if any(not s.paths and s.pv is None for s in series):
        raise ValueError("only scanner series (DICOM or ParaVision) can be combined")
    if len({s.pv is None for s in series}) != 1:
        raise ValueError("pick series from the same source (all DICOM or all ParaVision)")
    if any(s.n_slices != 1 for s in series):
        raise ValueError("only single-slice cines can be combined (pick the per-slice series)")
    if len({s.geometry for s in series}) != 1:
        raise ValueError("the series have different orientations or matrix sizes")
    ordered = sorted(series, key=lambda s: s.position_mm)
    pos = [round(s.position_mm, 2) for s in ordered]
    if len(set(pos)) != len(pos):
        raise ValueError("two series share a slice position (a repeated slice?): pick one")
    paths = [p for s in ordered for p in s.paths]
    stack = Series(name=name or f"Short-axis stack ({len(series)} slices)", kind="Short axis",
                   source=ordered[0].source, n_slices=len(ordered), n_frames=min(s.n_frames for s in ordered),
                   paths=paths,
                   thumb=ordered[len(ordered) // 2].thumb, geometry=ordered[0].geometry, parts=ordered)
    for i, part in enumerate(ordered):  # load_dicom_cine sorts slices by position too
        part.stack, part.slice_index = stack, i
    return stack


def auto_stacks(series: list[Series]) -> list[Series]:
    """Stack single-slice short-axis cines that share orientation and matrix (one cine per slice, as
    many small-animal protocols and Segment workflows use), however many slices there are.

    If a slice position was scanned more than once, the latest series (highest series number) goes in
    the stack and the others stay listed on their own (marked "earlier scan"), so the user can swap them in
    with Combine. Slices may have different numbers of frames; the stack keeps the frames all share.
    """
    groups: dict[tuple, list[Series]] = defaultdict(list)
    for s in series:
        if s.kind == "Short axis" and s.n_slices == 1 and s.stack is None and s.geometry:
            groups[s.geometry].append(s)
    stacks = []
    for g in groups.values():
        by_pos: dict[float, list[Series]] = defaultdict(list)
        for s in g:
            by_pos[round(s.position_mm, 2)].append(s)
        keep = []
        for same in by_pos.values():
            same.sort(key=_series_number)
            keep.append(same[-1])
            for older in same[:-1]:
                older.name += " (earlier scan)"
        if len(keep) >= 2:
            stacks.append(combine(keep))
    return stacks


def _series_number(s: Series) -> float:
    if s.pv is not None:
        return float(s.pv.number)
    head = s.name.split(" ", 1)[0]
    try:
        return float(head)
    except ValueError:
        return -1.0


def _load_stack(stack: Series) -> Session:
    """Load each slice's cine and stack them (base/apex order comes later, from the user or auto-order)."""
    cines = [_cine(p) for p in stack.parts]
    shapes = {c.data.shape[:2] for c in cines}
    if len(shapes) != 1:
        raise ValueError(f"slices have different matrix sizes: {sorted(shapes)}")
    n_t = min(c.data.shape[3] for c in cines)
    image = np.concatenate([c.data[:, :, :, :n_t] for c in cines], axis=2).astype(np.float32)
    positions = [float(c.slice_positions_mm[0]) for c in cines]
    dz = float(np.median(np.abs(np.diff(positions)))) if len(positions) > 1 else 1.0
    times = np.nanmean(np.stack([c.times_ms[:n_t] for c in cines]), axis=0)
    counts = sorted({c.data.shape[3] for c in cines})
    s = Session(image=image, label=np.zeros(image.shape, np.uint8), spacing=(*cines[0].spacing[:2], dz),
                subject_id=stack.name, slice_positions_mm=positions, source=stack.source, separate_slices=True,
                frame_times_ms=[float(t) for t in times] if np.all(np.isfinite(times)) else None,
                slice_names=[p.name for p in stack.parts])
    if len(counts) > 1:
        s.subject_id += f" (frames {counts[0]}–{counts[-1]} per slice; first {n_t} used)"
    return s


def _cine(s: Series):
    """One series' cine (DICOM or ParaVision): .data (x, y, slice, time), .spacing, .times_ms,
    .slice_positions_mm."""
    if s.pv is not None:
        from ..bruker import load_cine

        return load_cine(s.pv)
    from ..dicom import load_dicom_cine

    return load_dicom_cine(Path(s.paths[0]).parent, subject_id=s.name, paths=s.paths)


def _session_from_cine(cine, name: str, source: str) -> Session:
    image = np.asarray(cine.data, dtype=np.float32)
    times = cine.times_ms
    return Session(image=image, label=np.zeros(image.shape, np.uint8), spacing=tuple(cine.spacing), subject_id=name,
                   ed_frame=0, es_frame=0, slice_positions_mm=[float(x) for x in cine.slice_positions_mm],
                   source=source,
                   frame_times_ms=[float(x) for x in times] if len(times) and np.all(np.isfinite(times)) else None)
