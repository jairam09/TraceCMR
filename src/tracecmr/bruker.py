"""Reading Bruker ParaVision scans straight from the scanner's study folder (no DICOM export needed).

A ParaVision study folder holds one numbered folder per scan (1, 2, 3, ...). Each has the reconstructed images in
pdata/<n>/2dseq (raw integers) and their description in pdata/<n>/visu_pars:

    VisuCoreSize            pixels per frame (x fastest, then y)
    VisuCoreExtent          field of view in mm, so pixel size = extent / size
    VisuCoreFrameCount      frames in 2dseq, one after the other
    VisuFGOrderDesc         how frames are arranged: e.g. (16, <FG_CARDIAC_MOVIE>) then (8, <FG_SLICE>);
                            the first group listed varies fastest
    VisuCoreOrientation     direction cosines (read, phase, slice) per slice
    VisuCorePosition        position of each slice in mm
    VisuCoreDataSlope/Offs  stored value * slope + offset = image value, per frame
    VisuCoreWordType/ByteOrder

Many small-animal cine protocols scan each short-axis slice as its own scan (one slice, N cardiac frames), so a
short-axis stack is several scans with the same orientation at different positions: `scan_study` lists every
scan and `load_cine` reads one. Slice order comes from the positions along the slice normal, never from the
scan names (which are typed by hand and can be stale).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIME_GROUPS = ("FG_CARDIAC_MOVIE", "FG_MOVIE", "FG_CYCLE")  # cine frames (newer, older ParaVision versions)
SLICE_GROUPS = ("FG_SLICE",)
WORD_TYPES = {"_16BIT_SGN_INT": "i2", "_32BIT_SGN_INT": "i4", "_32BIT_FLOAT": "f4", "_8BIT_UNSGN_INT": "u1",
              "_16BIT_UNSGN_INT": "u2", "_32BIT_UNSGN_INT": "u4"}


# ---------------------------------------------------------------- parameter files (JCAMP-DX)

def _expand(body: str) -> list[str]:
    """Tokens of an array body: <strings>, (tuples), @n*(value) runs, plain numbers."""
    out: list[str] = []
    for m in re.finditer(r"@(\d+)\*\(([^)]*)\)|<[^>]*>|\([^()]*\)|[^\s()<>]+", body):
        if m.group(1):
            out.extend([m.group(2).strip()] * int(m.group(1)))
        else:
            out.append(m.group(0))
    return out


def _value(tok: str):
    tok = tok.strip()
    if tok.startswith("<") and tok.endswith(">"):
        return tok[1:-1]
    if tok.startswith("(") and tok.endswith(")"):
        return tuple(_value(t) for t in re.findall(r"<[^>]*>|[^,]+", tok[1:-1]))
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def read_params(path: str | Path) -> dict:
    """{name: value} from a ParaVision parameter file (visu_pars, method, acqp, reco).

    A scalar comes back as int, float or str. An array declared `( n )` or `( n, m )` comes back as a flat list,
    with @n*(v) runs expanded, <strings> unwrapped and (a, b) tuples as Python tuples; except a string array
    `( 65 ) <text>`, which is a single string."""
    out: dict = {}
    text = Path(path).read_text(errors="replace")
    for m in re.finditer(r"^##\$?([^=\n]+)=(.*?)(?=^##|\Z)", text, re.S | re.M):
        key = m.group(1).strip()
        raw = "\n".join(line for line in m.group(2).splitlines() if not line.startswith("$$")).strip()
        dims = re.match(r"^\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)\s*(.*)$", raw, re.S)
        if dims:
            body = dims.group(2).strip()
            if body.startswith("<") and body.endswith(">") and body.count("<") == 1:
                out[key] = body[1:-1]  # a char array: one string
            else:
                out[key] = [_value(t) for t in _expand(body)]
        else:
            out[key] = _value(raw) if raw else ""
    return out


# ---------------------------------------------------------------- one scan

@dataclass
class PvScan:
    """One reconstructed scan: what it is and how its 2dseq is laid out (pixels are read on demand)."""

    folder: Path  # the scan's folder (e.g. study/8)
    proc: int  # processed data number (pdata/<proc>)
    number: int
    name: str
    protocol: str
    size: tuple[int, int]  # (nx, ny)
    pixel_mm: tuple[float, float]  # (x, y)
    thickness_mm: float
    groups: list[tuple[int, str]]  # frame groups, fastest first: [(16, "FG_CARDIAC_MOVIE"), ...]
    n_slices: int
    n_frames: int
    orientation: np.ndarray  # (n_slices, 3, 3) rows: read, phase, slice directions
    positions: np.ndarray  # (n_slices, 3) mm
    frame_times_ms: list[float] | None = None
    heart_rate: float | None = None
    dim: int = 2
    note: str = ""
    params: dict = field(default_factory=dict, repr=False)

    @property
    def seq(self) -> Path:
        return self.folder / "pdata" / str(self.proc) / "2dseq"

    @property
    def normal(self) -> np.ndarray:
        return self.orientation[0][2]

    def slice_distances(self) -> np.ndarray:
        """Position of each slice along the slice normal (mm)."""
        return self.positions @ self.normal

    @property
    def geometry(self) -> tuple:
        """Scans with equal geometry can be stacked into one short-axis cine."""
        return (tuple(np.round(self.orientation[0].ravel(), 2)), self.size[1], self.size[0])

    def summary(self) -> str:
        return f"E{self.number} {self.name} · {self.n_slices}×{self.n_frames} · {self.size[0]}×{self.size[1]}"


def _list(v) -> list:
    return v if isinstance(v, list) else [v]


def read_scan(folder: str | Path, proc: int = 1) -> PvScan:
    """Describe one scan from its parameter files (no pixels read)."""
    folder = Path(folder)
    vp = folder / "pdata" / str(proc) / "visu_pars"
    if not vp.is_file():
        raise ValueError(f"no reconstructed images in {folder.name} (pdata/{proc}/visu_pars missing)")
    v = read_params(vp)
    acqp = read_params(folder / "acqp") if (folder / "acqp").is_file() else {}
    dim = int(v.get("VisuCoreDim", 2))
    size = [int(x) for x in _list(v["VisuCoreSize"])]
    extent = [float(x) for x in _list(v["VisuCoreExtent"])]
    n_total = int(v.get("VisuCoreFrameCount", 1))
    groups = []
    for g in _list(v.get("VisuFGOrderDesc", [])):
        if isinstance(g, tuple) and len(g) >= 2:
            groups.append((int(g[0]), str(g[1])))
    known = 1
    for n, _ in groups:
        known *= n
    if known != n_total:  # frames not described by the groups (older versions): one extra group
        groups.append((n_total // max(known, 1), "FG_UNKNOWN"))
    n_slices = int(np.prod([n for n, g in groups if g in SLICE_GROUPS])) if groups else 1
    n_frames = int(np.prod([n for n, g in groups if g in TIME_GROUPS])) if groups else 1
    note = ""
    other = [(n, g) for n, g in groups if g not in SLICE_GROUPS + TIME_GROUPS and n > 1]
    if other:
        note = "uses the first of " + ", ".join(f"{n} {g[3:].lower()}" for n, g in other)
    if dim == 3 and len(size) == 3:  # a 3D volume: its third axis is the slices
        n_slices = size[2]
    orient = np.array([float(x) for x in _list(v["VisuCoreOrientation"])]).reshape(-1, 3, 3)
    pos = np.array([float(x) for x in _list(v["VisuCorePosition"])]).reshape(-1, 3)
    if dim == 2:
        if len(orient) == 1 and n_slices > 1:
            orient = np.repeat(orient, n_slices, axis=0)
        if len(pos) == 1 and n_slices > 1:  # one position given: the others step along the normal
            step = float(_list(v.get("VisuCoreSlicePacksSliceDist", [v.get("VisuCoreFrameThickness", 1)]))[0])
            pos = pos[0] + np.outer(np.arange(n_slices) * step, orient[0][2])
    name = str(v.get("VisuSeriesExperimentComment") or acqp.get("ACQ_scan_name") or folder.name)
    name = re.sub(r"\s*\(E\d+\)\s*$", "", name).strip()
    times = None
    ft = v.get("VisuCardiacMovieFrameTime")
    if isinstance(ft, list) and ft and n_frames > 1:
        vals = [float(x[0]) if isinstance(x, tuple) else float(x) for x in ft]
        if len(vals) >= n_frames:
            times = vals[:n_frames]
    hr = v.get("VisuCardiacHeartRate")
    return PvScan(
        folder=folder, proc=proc, number=int(folder.name) if folder.name.isdigit() else 0, name=name,
        protocol=str(v.get("VisuAcquisitionProtocol") or acqp.get("ACQ_protocol_name") or ""),
        size=(size[0], size[1]), pixel_mm=(extent[0] / size[0], extent[1] / size[1]),
        thickness_mm=float(_list(v.get("VisuCoreFrameThickness", 1))[0]) if dim == 2 else extent[2] / size[2],
        groups=groups, n_slices=n_slices, n_frames=n_frames, orientation=orient, positions=pos,
        frame_times_ms=times, heart_rate=float(hr) if isinstance(hr, (int, float)) else None, dim=dim, note=note,
        params=v)


def load_scan(scan: PvScan) -> np.ndarray:
    """The scan's images as float32 (x, y, slice, time), with each frame's slope and offset applied, slices in
    the order stored (sort them with slice_distances)."""
    v = scan.params
    dtype = np.dtype(WORD_TYPES.get(str(v.get("VisuCoreWordType", "_16BIT_SGN_INT")), "i2"))
    dtype = dtype.newbyteorder("<" if str(v.get("VisuCoreByteOrder", "littleEndian")).lower().startswith("little")
                               else ">")
    nx, ny = scan.size
    raw = np.fromfile(scan.seq, dtype=dtype)
    per_frame = nx * ny * (scan.n_slices if scan.dim == 3 else 1)
    n_total = raw.size // per_frame
    if n_total * per_frame != raw.size or n_total < 1:
        raise ValueError(f"2dseq of E{scan.number} has {raw.size} values, not a whole number of "
                         f"{nx}×{ny} frames")
    frames = raw[: n_total * per_frame].reshape(n_total, per_frame).astype(np.float32)
    slope = np.array([float(x) for x in _list(v.get("VisuCoreDataSlope", 1.0))], dtype=np.float32)
    offs = np.array([float(x) for x in _list(v.get("VisuCoreDataOffs", 0.0))], dtype=np.float32)
    slope = np.resize(slope, n_total)
    offs = np.resize(offs, n_total)
    frames = frames * slope[:, None] + offs[:, None]
    if scan.dim == 3:  # (frames, z, y, x) -> (x, y, z, frames)
        vol = frames.reshape(n_total, scan.n_slices, ny, nx)
        return np.ascontiguousarray(vol.transpose(3, 2, 1, 0)[..., : max(scan.n_frames, 1)])
    # frame index -> one index per group, the first group varying fastest
    shape = [n for n, _ in scan.groups] if scan.groups else [n_total]
    idx = np.arange(n_total).reshape(shape[::-1])  # C order: last axis fastest = first group
    names = [g for _, g in scan.groups] if scan.groups else ["FG_UNKNOWN"]
    sl = [names.index(g) for g in names if g in SLICE_GROUPS]
    tm = [names.index(g) for g in names if g in TIME_GROUPS]
    take = [0] * len(names)
    out = np.zeros((nx, ny, scan.n_slices, scan.n_frames), np.float32)
    for z in range(scan.n_slices):
        for t in range(scan.n_frames):
            take = [0] * len(names)
            if sl:
                take[sl[0]] = z
            if tm:
                take[tm[0]] = t
            f = idx[tuple(take[::-1])]
            out[:, :, z, t] = frames[f].reshape(ny, nx).T  # stored x fastest: (y, x) -> (x, y)
    if v.get("VisuCoreDiskSliceOrder") == "disk_reverse_slice_order" or \
            v.get("VisuCoreDiskSliceOrder") == ["disk_reverse_slice_order"]:
        out = out[:, :, ::-1]
    return out


# ---------------------------------------------------------------- a study

def is_study(folder: str | Path) -> bool:
    """True for a ParaVision study folder (numbered scan folders with reconstructed images)."""
    folder = Path(folder)
    try:
        return any(d.is_dir() and d.name.isdigit() and any(d.glob("pdata/*/2dseq")) for d in folder.iterdir())
    except OSError:
        return False


def scan_study(folder: str | Path) -> list[PvScan]:
    """Every scan in a study folder that has images, in scan order (scans that can't be read are skipped)."""
    out = []
    for d in sorted((d for d in Path(folder).iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name)):
        procs = sorted(int(p.parent.name) for p in d.glob("pdata/*/2dseq") if p.parent.name.isdigit())
        if not procs:
            continue
        try:
            out.append(read_scan(d, procs[0]))
        except (ValueError, KeyError, IndexError):
            continue
    return out


@dataclass
class PvCine:
    """A cine ready for the app: data is (x, y, slice, time), slices sorted along the normal."""

    data: np.ndarray
    spacing: tuple[float, float, float]
    times_ms: np.ndarray
    slice_positions_mm: np.ndarray
    subject_id: str


def load_cine(scan: PvScan) -> PvCine:
    data = load_scan(scan)
    dist = scan.slice_distances()
    order = np.argsort(dist, kind="stable")
    data = data[:, :, order]
    dist = dist[order]
    dz = float(np.median(np.abs(np.diff(dist)))) if len(dist) > 1 else scan.thickness_mm
    times = np.array(scan.frame_times_ms if scan.frame_times_ms else [np.nan] * scan.n_frames, float)
    return PvCine(data=data, spacing=(scan.pixel_mm[0], scan.pixel_mm[1], dz), times_ms=times,
                  slice_positions_mm=dist, subject_id=scan.name)
