"""ED/ES timing across short-axis slices: how much does it matter to pick end-diastole and end-systole per
slice instead of once for the whole stack?

Mouse short-axis cines are usually acquired one slice after another over several minutes. If the heart rate
drifts during that time, frame k of the basal slice and frame k of the apical slice are not the same moment in
the cardiac cycle, and a single global ED/ES pair takes some slices off their true extremes. This module
measures that on a contoured stack:

* each slice's own ED and ES (largest / smallest blood pool in that slice's cine), against the global pair;
* EDV, ESV and EF computed both ways on the same contours;
* whether the per-slice ES frames *trend* with acquisition order (what drift predicts) or merely scatter.

Read the result with one caveat in mind. Taking each slice's own maximum and minimum can only raise EDV and
lower ESV, so EF(per slice) >= EF(global) even with no drift at all: noisy contours alone push the extremes
outward. That is why the slice curves are smoothed before picking extremes, and why the trend statistic is
reported: real drift moves ES steadily from the first-acquired slice to the last, noise scatters it at random.
A large EF difference without a trend is selection noise, not drift.

It needs contours on many frames of each slice ("Contour every frame"), not just at ED and ES.

The ParaVision readers at the bottom pull the timing parameters a Bruker scan recorded (trigger, ECG,
respiration, repetition time), so measured heart rates can be set against the frame shifts. Which parameters
a given ParaVision version writes for cine acquisitions has not been checked against real data yet:
`timing_params` returns whatever looks timing-related rather than assuming field names.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import labels as L


@dataclass
class SliceTiming:
    slice: int
    own_ed: int
    own_es: int
    ed_shift: int  # own frame minus the global frame, wrapped to the shortest way round the cycle
    es_shift: int
    n_frames: int  # contoured frames on this slice


@dataclass
class TimingResult:
    global_ed: int
    global_es: int
    n_frames: int  # frames in the cine
    slices: list[SliceTiming]  # slices with enough contoured frames to find their own ED/ES
    skipped: list[int] = field(default_factory=list)  # in the totals at the global frames only
    edv_global: float = 0.0
    esv_global: float = 0.0
    edv_per_slice: float = 0.0
    esv_per_slice: float = 0.0
    frame_ms: float | None = None
    es_trend: float | None = None  # Spearman rho of ES shift against acquisition order
    ed_trend: float | None = None

    @staticmethod
    def _ef(edv: float, esv: float) -> float:
        return 100.0 * (edv - esv) / edv if edv > 0 else float("nan")

    @property
    def ef_global(self) -> float:
        return self._ef(self.edv_global, self.esv_global)

    @property
    def ef_per_slice(self) -> float:
        return self._ef(self.edv_per_slice, self.esv_per_slice)

    @property
    def delta_ef(self) -> float:
        """EF points gained by picking ED/ES per slice."""
        return self.ef_per_slice - self.ef_global

    @property
    def es_spread_frames(self) -> int:
        return (max(s.es_shift for s in self.slices) - min(s.es_shift for s in self.slices)) if self.slices else 0

    @property
    def ed_spread_frames(self) -> int:
        return (max(s.ed_shift for s in self.slices) - min(s.ed_shift for s in self.slices)) if self.slices else 0

    @property
    def es_spread_ms(self) -> float | None:
        return None if self.frame_ms is None else self.es_spread_frames * self.frame_ms

    def summary(self) -> str:
        """One line for the status bar."""
        if len(self.slices) < 2:
            return "Not enough contoured frames per slice to compare ED/ES timing (contour every frame)."
        ms = f" ({self.es_spread_ms:.0f} ms)" if self.es_spread_ms is not None else ""
        trend = ""
        if self.es_trend is not None:
            trend = (", trending with slice order" if abs(self.es_trend) >= 0.6
                     else ", no clear trend with slice order")
        return (f"ES varies by {self.es_spread_frames} frame(s){ms} across slices{trend}. "
                f"EF {self.ef_global:.1f}% with one ES for all, {self.ef_per_slice:.1f}% per slice.")

    def as_row(self) -> dict:
        return {"global_ed": self.global_ed, "global_es": self.global_es, "n_frames": self.n_frames,
                "slices_measured": len(self.slices), "slices_skipped": len(self.skipped),
                "edv_global_ul": self.edv_global, "esv_global_ul": self.esv_global, "ef_global_pct": self.ef_global,
                "edv_per_slice_ul": self.edv_per_slice, "esv_per_slice_ul": self.esv_per_slice,
                "ef_per_slice_pct": self.ef_per_slice, "delta_ef_pts": self.delta_ef,
                "es_spread_frames": self.es_spread_frames, "ed_spread_frames": self.ed_spread_frames,
                "es_spread_ms": self.es_spread_ms, "frame_ms": self.frame_ms,
                "es_trend_rho": self.es_trend, "ed_trend_rho": self.ed_trend}


def _smooth_cyclic(y: np.ndarray, width: int) -> np.ndarray:
    """Moving average that wraps around: a cine is one cardiac cycle, so the last frame neighbours the first."""
    if width <= 1 or len(y) < width:
        return y.astype(float)
    k = np.ones(width) / width
    pad = width // 2
    return np.convolve(np.concatenate([y[-pad:], y, y[:pad]]), k, mode="valid")[: len(y)]


def _wrap(d: int, n: int) -> int:
    """Shortest signed distance round a cycle of n frames."""
    d = (d + n // 2) % n - n // 2
    return int(d)


def _spearman(x, y) -> float | None:
    if len(x) < 3 or len(set(y)) < 2 or len(set(x)) < 2:
        return None
    from scipy.stats import spearmanr

    rho = spearmanr(x, y).statistic
    return None if not np.isfinite(rho) else float(rho)


def phase_timing(session, min_frames: int = 4, smooth: int = 3, slices: list[int] | None = None,
                 tie: float = 0.01) -> TimingResult:
    """Compare global and per-slice ED/ES on a contoured stack.

    `slices`: which slices count (default: those selected for both ED and ES). Acquisition order is taken to be
    slice index order, which is how a serially acquired stack is numbered; pass `slices` to override the set,
    not the order. The volumes use only slices contoured at every frame they need, so both totals are made of
    the same slices. `tie`: frames within this fraction of a slice's blood-pool range of its extreme count as
    tied with it (flat tops and bottoms of the curve)."""
    lab = session.label
    nt = lab.shape[3]
    dx, dy, dz = session.spacing
    vox = dx * dy * dz
    use = sorted(slices if slices is not None else (session.included["ED"] & session.included["ES"]))
    blood = (lab == L.LV_BLOODPOOL)
    area = blood.sum(axis=(0, 1)).astype(float)  # (slice, frame)
    drawn = (lab > 0).any(axis=(0, 1))  # (slice, frame)
    use = [z for z in use if drawn[z].any() and area[z].any()]

    # global ED/ES: largest / smallest total blood pool among frames contoured on (nearly) every slice used
    if use:
        n_drawn = drawn[use].sum(axis=0)
        cand = np.flatnonzero(n_drawn >= max(1, n_drawn.max() - 1))
    else:
        cand = np.array([], dtype=int)
    if len(cand) < 2:
        return TimingResult(global_ed=session.ed_frame, global_es=session.es_frame, n_frames=nt, slices=[],
                            skipped=list(use))
    total = area[use].sum(axis=0)
    g_ed, g_es = int(cand[np.argmax(total[cand])]), int(cand[np.argmin(total[cand])])

    def extreme(curve: np.ndarray, frames: np.ndarray, g: int, largest: bool) -> int:
        """The frame where this slice's curve peaks (or bottoms out). Near the extreme a cine is flat, and
        frames within `tie` of it can't be told apart; among those, the one nearest the global frame wins, so a
        slice is only called shifted when its own data says so."""
        vals = curve[frames]
        best = vals.max() if largest else vals.min()
        span = float(vals.max() - vals.min())
        near = frames[np.abs(vals - best) <= tie * span + 1e-9]
        return int(min(near, key=lambda t: abs(_wrap(int(t) - g, nt))))

    measured, skipped = [], []
    for z in use:
        tz = np.flatnonzero(drawn[z])
        if len(tz) < min_frames:
            skipped.append(z)
            continue
        # a full cine on this slice: smooth it as a cycle; with gaps, use the contoured frames as they are
        curve = _smooth_cyclic(area[z], smooth) if len(tz) == nt else area[z]
        own_ed, own_es = extreme(curve, tz, g_ed, True), extreme(curve, tz, g_es, False)
        measured.append(SliceTiming(slice=z, own_ed=own_ed, own_es=own_es, ed_shift=_wrap(own_ed - g_ed, nt),
                                    es_shift=_wrap(own_es - g_es, nt), n_frames=int(len(tz))))

    # the same slices in both totals: every slice used must be contoured at the frames it is counted at
    own = {s.slice: (s.own_ed, s.own_es) for s in measured}
    edv_g = esv_g = edv_p = esv_p = 0.0
    for z in use:
        oe, os_ = own.get(z, (g_ed, g_es))
        if not (drawn[z, g_ed] and drawn[z, g_es] and drawn[z, oe] and drawn[z, os_]):
            continue
        edv_g += area[z, g_ed] * vox
        esv_g += area[z, g_es] * vox
        edv_p += area[z, oe] * vox
        esv_p += area[z, os_] * vox

    frame_ms = None
    times = getattr(session, "frame_times_ms", None)
    if times is not None and len(times) >= 2:
        steps = np.diff(np.asarray(times, dtype=float))
        steps = steps[steps > 0]
        frame_ms = float(np.median(steps)) if len(steps) else None

    order = [s.slice for s in measured]
    return TimingResult(global_ed=g_ed, global_es=g_es, n_frames=nt, slices=measured, skipped=skipped,
                        edv_global=edv_g, esv_global=esv_g, edv_per_slice=edv_p, esv_per_slice=esv_p,
                        frame_ms=frame_ms,
                        es_trend=_spearman(order, [s.es_shift for s in measured]),
                        ed_trend=_spearman(order, [s.ed_shift for s in measured]))


def write_csv(result: TimingResult, path: str | Path, subject: str = "") -> Path:
    """Per-slice rows (own ED/ES and shifts) followed by the summary, in one CSV."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", subject])
        w.writerow([])
        w.writerow(["slice", "own_ed", "own_es", "ed_shift_frames", "es_shift_frames", "contoured_frames"])
        for s in result.slices:
            w.writerow([s.slice, s.own_ed, s.own_es, s.ed_shift, s.es_shift, s.n_frames])
        for z in result.skipped:
            w.writerow([z, "", "", "", "", "too few frames"])
        w.writerow([])
        for k, v in result.as_row().items():
            w.writerow([k, "" if v is None else (f"{v:.4g}" if isinstance(v, float) else v)])
    return path


# ---------------------------------------------------------------- Bruker ParaVision parameter files

_TIMING = re.compile(r"trig|ecg|resp|heart|rr_|_rr|cardiac|cine|frame|movie|repetit|igate|intragate|navig|"
                     r"selfgat|gating|scantime|duration", re.IGNORECASE)


def read_jcamp(path: str | Path) -> dict:
    """Parse a ParaVision JCAMP-DX parameter file (method, acqp, visu_pars, reco) into {name: value}.

    Scalars come back as float when they parse, else str; arrays declared as `( n )` come back as a list of
    the tokens on the following lines; strings in < > lose their brackets."""
    out: dict = {}
    key, buf = None, []

    def flush():
        if key is None:
            return
        raw = " ".join(buf).strip()
        m = re.match(r"^\(\s*[\d,\s]+\)\s*(.*)$", raw, re.S)
        if m and buf and buf[0].strip().startswith("("):  # sized array: values follow the size
            body = m.group(1).strip()
            out[key] = [t.strip("<>") for t in re.findall(r"<[^>]*>|\S+", body)] if body else []
            return
        raw = raw.strip("<>")
        try:
            out[key] = float(raw)
        except ValueError:
            out[key] = raw

    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("$$"):
            continue
        if line.startswith("##"):
            flush()
            k, _, v = line[2:].partition("=")
            key, buf = k.lstrip("$").strip(), [v]
        elif key is not None:
            buf.append(line)
    flush()
    return out


def timing_params(scan_dir: str | Path) -> dict:
    """Every timing-related parameter a Bruker scan folder recorded, as {"file:name": value}.

    Looks in method and acqp in the scan folder and visu_pars / reco under pdata/*. Deliberately broad: the
    exact names for cine heart rate and trigger timing differ between ParaVision versions and have not been
    checked against real data here."""
    scan_dir = Path(scan_dir)
    files = [scan_dir / "method", scan_dir / "acqp", *sorted(scan_dir.glob("pdata/*/visu_pars")),
             *sorted(scan_dir.glob("pdata/*/reco"))]
    found = {}
    for f in files:
        if not f.is_file():
            continue
        tag = f.name if f.parent == scan_dir else f"{f.parent.name}/{f.name}"
        for k, v in read_jcamp(f).items():
            if _TIMING.search(k):
                found[f"{tag}:{k}"] = v
    return found
