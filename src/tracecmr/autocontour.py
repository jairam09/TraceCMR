"""Automatic contouring with published mouse cardiac MRI models.

Two backends, both run from the user's own download of each model (nothing is
bundled, so each model keeps its own license):

- "ucl":    Shah et al., JCMR 2026 (UNet3+, 2D, LV blood pool + myocardium).
            GPL-3.0: point TRACECMR_UCL_DIR at a clone of
            github.com/mrphys/Open-Source_Pre-Clinical_Segmentation with the
            real weights (Git LFS) in ModelWeights/.
- "calico": Calico Life Sciences, Sci Rep 2025 (3D UNet, LV + myocardium + RV).
            MIT: point TRACECMR_CALICO_DIR at a clone of
            github.com/calico/mouse_cardiac_mri_segmentation with its weights.

Both return standard labels (labels.py) on the scan's own grid, shaped
(x, y, slice, time); frames that were not run stay 0.

Preliminary comparison on UCL data (docs/lab_notebook.md, 2026-10-07): the UCL
model is more accurate on UCL's own test mice; Calico generalises reasonably to
an unseen scanner but failed on some severely dilated hearts. Neither is trusted
blindly: the app flags slices for review (see `disagreement`).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy import ndimage

from . import labels as L

Progress = Callable[[int, int], None]  # (done, total)


def _dir(env: str, default: str) -> Path:
    if env in os.environ:
        return Path(os.environ[env]).expanduser()
    home = Path.home() / ".tracecmr"
    if not home.exists() and (Path.home() / ".ventrimap").exists():  # before the rename
        home = Path.home() / ".ventrimap"
    return home / "models" / default


@dataclass
class ModelInfo:
    name: str
    label: str
    outputs_rv: bool
    papillary: str | None  # convention of the model's training labels
    license: str


MODELS = {
    "ucl": ModelInfo("ucl", "UCL model (LV, 2D)", False, "myocardium", "GPL-3.0"),
    "calico": ModelInfo("calico", "Calico model (LV + RV, 3D)", True, None, "MIT"),
}


def available() -> dict[str, str]:
    """Which models are installed, with a one-line status each."""
    out = {}
    u = _dir("TRACECMR_UCL_DIR", "ucl")
    w = _ucl_weights(u)
    out["ucl"] = "ready" if w and (u / "unet3plus.py").exists() else f"not installed (set TRACECMR_UCL_DIR; looked in {u})"
    c = _dir("TRACECMR_CALICO_DIR", "calico")
    ck = c / "model_checkpoints" / "short_axis.ckpt"
    ok = ck.exists() and ck.stat().st_size > 1_000_000  # an LFS pointer is ~130 bytes
    out["calico"] = "ready" if ok else f"not installed (set TRACECMR_CALICO_DIR; looked in {c})"
    return out


def _ucl_weights(d: Path) -> Path | None:
    for p in (d / "ModelWeights" / "SEG459.h5", d / "ModelWeights" / "real" / "SEG459.h5"):
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    return None


# ---------------------------------------------------------------- UCL model

_ucl_model = None



_load_lock = threading.Lock()  # the background preload and an Auto LV click must not build the model twice


def _load_ucl():
    global _ucl_model
    with _load_lock:
        return _load_ucl_locked()


def _load_ucl_locked():
    global _ucl_model
    if _ucl_model is None:
        d = _dir("TRACECMR_UCL_DIR", "ucl")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf

        sys.path.insert(0, str(d))
        from unet3plus import unet3plus  # the authors' architecture, from their repository

        inp = tf.keras.Input(shape=[256, 256, 1])
        net = unet3plus(inp, rank=2, add_dropout=1, dropout_rate=0.3, filters=[32, 64, 128, 256, 512],
                        kernel_size=3, encoder_block_depth=2, decoder_block_depth=1, out_channels=3,
                        out_activation="softmax", deep_supervision=True, CGM=False)
        model = tf.keras.Model(inputs=inp, outputs=net.outputs())
        model.load_weights(str(_ucl_weights(d)))
        # Only the main output is used. The extra deep-supervision outputs (4 more full-size heads) made
        # inference ~8x slower for nothing; the main output is identical either way.
        final = tf.keras.Model(inputs=model.inputs, outputs=model.outputs[0])
        # Keras predict (not a hand-made tf.function: that crashed the app on Windows when called from
        # the background thread)
        _ucl_model = lambda x: final.predict(x, verbose=0)  # noqa: E731
    return _ucl_model


# Blood-pool cutoff. The model calls a pixel blood at 50% (argmax); it tends to round off the cavity's
# corners between the papillary muscles, which moves blood into the myocardium (LV mass reads high).
# A lower cutoff lets the blood pool grow into those corners, but only inside the LV and only touching the
# existing blood pool (refine_blood). Set from validation; TRACECMR_BLOOD_CUTOFF overrides (0.5 = off).
BLOOD_CUTOFF = 0.5


# Outer-edge (epicardium) cutoff: a pixel counts as LV only if P(myo) + P(blood) >= this. At plain argmax
# (0.5) the model draws the epicardium ~half a pixel too far out, so LV mass read ~+8 mg high. Chosen on UCL's
# 15 validation mice (mass bias +8.1 -> +0.6 mg; EDV, ESV, EF unchanged), checked once on the 25 test mice
# (+6.3 -> -0.1 mg). TRACECMR_EPI_CUTOFF overrides (0.5 = plain argmax).
EPI_CUTOFF = 0.9


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def blood_cutoff() -> float:
    return _env_float("TRACECMR_BLOOD_CUTOFF", BLOOD_CUTOFF)


def epi_cutoff() -> float:
    return _env_float("TRACECMR_EPI_CUTOFF", EPI_CUTOFF)


def refine_blood(lab: np.ndarray, p_blood: np.ndarray, cutoff: float) -> np.ndarray:
    """Grow the blood pool (UCL code 2) into myocardium pixels (code 1) whose blood probability is at least
    `cutoff`, slice by slice, only inside the LV outline and only where connected to the existing pool."""
    if cutoff >= 0.5:
        return lab
    out = lab.copy()
    for z in range(lab.shape[2]):
        sl, blood = lab[:, :, z], lab[:, :, z] == 2
        if not blood.any():
            continue  # no cavity (apex cap): don't invent one
        inside = ndimage.binary_fill_holes(sl > 0)
        cand = (p_blood[:, :, z] >= cutoff) & inside & (sl == 1)
        comps, n = ndimage.label(cand | blood)
        if n == 0:
            continue
        touching = np.unique(comps[blood])
        grow = cand & np.isin(comps, touching[touching > 0])
        out[:, :, z][grow] = 2
    return out


def _ucl_probs(vol_xyz: np.ndarray, spacing) -> np.ndarray:
    """One frame (x, y, slice) -> class probabilities (x, y, slice, 3: background, myo, blood) on the
    input grid. The model expects 256x256 at 0.1 mm."""
    model = _load_ucl()
    dx, dy, _ = spacing
    f = (dx / 0.1, dy / 0.1)
    work = vol_xyz if f == (1.0, 1.0) else ndimage.zoom(vol_xyz, (f[0], f[1], 1), order=1)
    pad = _fit(work, 256)
    x = np.stack([_minmax(s) for s in np.moveaxis(pad.arr, 2, 0)])[..., None]
    prob = np.moveaxis(np.asarray(model(x), dtype=np.float32), 0, 2)  # (256, 256, slice, 3)
    prob = pad.undo(prob)
    prob[pad.undo(np.ones(pad.arr.shape[:3], bool)) == 0] = (1.0, 0.0, 0.0)  # outside the model's view
    if f != (1.0, 1.0):
        prob = ndimage.zoom(prob, [vol_xyz.shape[0] / prob.shape[0], vol_xyz.shape[1] / prob.shape[1], 1, 1],
                            order=1)
        fixed = np.zeros(vol_xyz.shape + (3,), np.float32)
        fixed[..., 0] = 1.0
        sx, sy = min(prob.shape[0], fixed.shape[0]), min(prob.shape[1], fixed.shape[1])
        fixed[:sx, :sy] = prob[:sx, :sy]
        prob = fixed
    return prob


def labels_from_probs(prob: np.ndarray, cutoff: float = 0.5, epi_cutoff: float = 0.5) -> np.ndarray:
    """UCL codes (1 myo, 2 blood) from class probabilities: argmax, keep the LV, then refine_blood.
    epi_cutoff > 0.5 makes the outer edge stricter: a pixel is LV only if P(myo) + P(blood) >= epi_cutoff."""
    lab = prob.argmax(-1).astype(np.uint8)
    if epi_cutoff > 0.5:
        trimmed = lab.copy()
        trimmed[(prob[..., 1] + prob[..., 2]) < epi_cutoff] = 0
        for z in range(lab.shape[2]):  # the cutoff tidies edges; it must not erase a faint slice (apex)
            before, after = int((lab[:, :, z] > 0).sum()), int((trimmed[:, :, z] > 0).sum())
            if before and after >= 0.5 * before:
                lab[:, :, z] = trimmed[:, :, z]
    lab = _keep_main_component(lab)
    return refine_blood(lab, prob[..., 2], cutoff)


def _ucl_volume(vol_xyz: np.ndarray, spacing, cutoff: float | None = None) -> np.ndarray:
    """One frame (x, y, slice) -> UCL codes (1 myo, 2 blood)."""
    return labels_from_probs(_ucl_probs(vol_xyz, spacing), blood_cutoff() if cutoff is None else cutoff,
                             epi_cutoff())


def _ucl_volume_unsure(vol_xyz: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Like _ucl_volume, plus each slice's "model unsure" score (quality.unsure_score)."""
    from .quality import unsure_score

    prob = _ucl_probs(vol_xyz, spacing)
    lab = labels_from_probs(prob, blood_cutoff(), epi_cutoff())
    std = _to_standard(lab, "ucl")
    return lab, np.array([unsure_score(prob[:, :, z], std[:, :, z]) for z in range(lab.shape[2])], np.float32)


_last_unsure: np.ndarray | None = None  # per-slice scores from the last UCL run (None for other models)


# ------------------------------------------------------------- Calico model

_calico = None


def _load_calico():
    global _calico
    if _calico is None:
        d = _dir("TRACECMR_CALICO_DIR", "calico")
        sys.path.insert(0, str(d))
        import torch
        from munch import Munch

        import inference_segmentation as cal  # the authors' inference code (MIT)

        torch.set_num_threads(max(1, (os.cpu_count() or 2)))
        cfg = Munch(cal.parse_config_file(str(d / "configs" / "short_axis.yml")))
        model = cal.load_model_from_checkpoint(checkpoint_path=str(d / "model_checkpoints" / "short_axis.ckpt"),
                                               model_architecture="unet3d.UNet3D",
                                               model_parameters=cfg.model_parameters, device=cal.device)
        _calico = (cal, cfg, model)
    return _calico


def _calico_volume(vol_xyz: np.ndarray, spacing) -> np.ndarray:
    """One frame -> Calico codes (1 LV blood, 2 myo, 3 RV). Trained at ~0.13 x 0.13 x 0.8 mm, 192x192."""
    cal, cfg, model = _load_calico()
    dx, dy, dz = spacing
    rs = ndimage.zoom(vol_xyz.astype(np.float32), (dx / 0.1302, dy / 0.1302, dz / 0.8), order=1)
    pad = _fit(rs, 192)
    pred = cal.infer_single_image_3d(pad.arr[..., None], model, cfg)[..., 0]
    return _resize_nearest(pad.undo(pred), vol_xyz.shape)


class _Worker:
    """One background process per model (see _model_worker.py): started on first use, kept running."""

    def __init__(self, model: str):
        self.model = model
        self.proc = None
        self.lock = threading.Lock()
        self.log_path = os.path.join(tempfile.gettempdir(), f"tracecmr_{model}_worker.log")

    def _start(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window popping up on Windows
        self._log = open(self.log_path, "a")
        self.proc = subprocess.Popen([sys.executable, "-m", "tracecmr._model_worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1,
                                     creationflags=flags, env={**os.environ})

    def ask(self, line: str) -> None:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            try:
                self.proc.stdin.write(line + "\n")
                self.proc.stdin.flush()
                while True:
                    reply = self.proc.stdout.readline()
                    if not reply:  # the worker died: report it; a fresh one starts next time
                        self.proc = None
                        raise RuntimeError(f"the {self.model} model stopped unexpectedly. "
                                           f"Details: {self._tail()}")
                    if reply.startswith("@@ok"):
                        return
                    if reply.startswith("@@err"):
                        raise RuntimeError(reply[6:].strip())
            except (BrokenPipeError, OSError) as e:
                self.proc = None
                raise RuntimeError(f"the {self.model} model stopped unexpectedly ({e}). Details: {self._tail()}")

    def _tail(self) -> str:
        try:
            with open(self.log_path) as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            return " | ".join(lines[-3:]) or self.log_path
        except OSError:
            return self.log_path

    def run(self, vol_xyz: np.ndarray, spacing) -> np.ndarray:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inp, out = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
            with open(inp, "wb") as f:
                np.savez(f, vol=vol_xyz.astype(np.float32), spacing=np.asarray(spacing, float))
            self.ask(f"run {self.model} {inp} {out}")
            with np.load(out) as d:  # close before the folder is removed (Windows file locks)
                self.last_unsure = d["unc"].copy() if "unc" in d.files else None
                return d["lab"].copy()


_workers: dict[str, _Worker] = {}


def _model_volume(model: str, vol_xyz: np.ndarray, spacing) -> np.ndarray:
    """Run a model on one volume: inside a worker process from the app, directly inside the worker."""
    global _last_unsure
    if os.environ.get("TRACECMR_INPROCESS") == "1":
        if model == "ucl":
            lab, _last_unsure = _ucl_volume_unsure(vol_xyz, spacing)
            return lab
        _last_unsure = None
        return _calico_volume(vol_xyz, spacing)
    w = _workers.setdefault(model, _Worker(model))
    lab = w.run(vol_xyz, spacing)
    _last_unsure = getattr(w, "last_unsure", None)
    return lab


# ------------------------------------------------------------------ public

def run(image_xyzt: np.ndarray, spacing, model: str = "ucl", frames: Iterable[int] | None = None,
        progress: Progress | None = None) -> np.ndarray:
    """Contour the given frames (default: all). Returns standard labels (x, y, slice, time)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    status = available()[model]
    if status != "ready":
        raise RuntimeError(f"{MODELS[model].label}: {status}")
    frames = list(range(image_xyzt.shape[3])) if frames is None else list(frames)
    out = np.zeros(image_xyzt.shape, dtype=np.uint8)
    for i, t in enumerate(frames):
        vol = image_xyzt[..., t].astype(np.float32)
        if model == "ucl":
            raw = _model_volume("ucl", vol, spacing)
            std = np.zeros_like(raw)
            std[raw == 1] = L.LV_MYOCARDIUM
            std[raw == 2] = L.LV_BLOODPOOL
        else:
            raw = _model_volume("calico", vol, spacing)
            std = np.zeros_like(raw)
            std[raw == 1] = L.LV_BLOODPOOL
            std[raw == 2] = L.LV_MYOCARDIUM
            std[raw == 3] = L.RV_BLOODPOOL
        out[..., t] = std
        if progress:
            progress(i + 1, len(frames))
    return out


def _to_standard(raw: np.ndarray, model: str) -> np.ndarray:
    std = np.zeros_like(raw)
    if model == "ucl":
        std[raw == 1] = L.LV_MYOCARDIUM
        std[raw == 2] = L.LV_BLOODPOOL
    else:
        std[raw == 1] = L.LV_BLOODPOOL
        std[raw == 2] = L.LV_MYOCARDIUM
        std[raw == 3] = L.RV_BLOODPOOL
    return std


def run_pairs(image_xyzt: np.ndarray, spacing, pairs: list[tuple[int, int]], model: str = "ucl",
              unsure_out: list | None = None) -> np.ndarray:
    """Contour only the given (slice, frame) images, in one pass through the model.

    The images are stacked in slice order as one volume (each slice from its own frame, which is how
    per-slice ED/ES works), so this costs about one frame instead of every frame of the cine.
    Returns standard labels (x, y, len(pairs)) in the order of `pairs`.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    status = available()[model]
    if status != "ready":
        raise RuntimeError(f"{MODELS[model].label}: {status}")
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    vol = np.stack([image_xyzt[:, :, pairs[i][0], pairs[i][1]] for i in order], axis=2).astype(np.float32)
    raw = _model_volume(model, vol, spacing)
    std = _to_standard(raw, model)
    out = np.zeros_like(std)
    for k, i in enumerate(order):
        out[:, :, i] = std[:, :, k]
    if unsure_out is not None:  # "model unsure" score per pair (quality.py), when the model gives one
        unc = _last_unsure
        scores = [None] * len(pairs)
        if unc is not None and len(unc) == len(pairs):
            for k, i in enumerate(order):
                scores[i] = float(unc[k])
        unsure_out[:] = scores
    return out


def preload(model: str = "ucl") -> None:
    """Load a model ahead of time (e.g. in the background when a scan opens) so the first click is quick."""
    try:
        if available().get(model) == "ready":
            if os.environ.get("TRACECMR_INPROCESS") == "1":
                _load_ucl() if model == "ucl" else _load_calico()
            else:
                _workers.setdefault(model, _Worker(model)).ask(f"load {model}")
    except Exception:  # a failed preload is not an error: Auto-contour loads (and reports) on its own
        pass


def disagreement(a: np.ndarray, b: np.ndarray, frames: Iterable[int], threshold: float = 0.8) -> dict[tuple[int, int], float]:
    """Slices where two contour sets disagree: {(frame, slice): LV overlap (Dice)} for Dice < threshold.

    Compares the whole LV (blood pool + myocardium) so the papillary convention
    does not matter. A slice contoured by only one of the two counts as Dice 0.
    """
    flags = {}
    for t in frames:
        for z in range(a.shape[2]):
            pa = np.isin(a[:, :, z, t], (L.LV_BLOODPOOL, L.LV_MYOCARDIUM))
            pb = np.isin(b[:, :, z, t], (L.LV_BLOODPOOL, L.LV_MYOCARDIUM))
            s = pa.sum() + pb.sum()
            if s == 0:
                continue
            d = 2 * (pa & pb).sum() / s
            if d < threshold:
                flags[(t, z)] = float(d)
    return flags


def sanity_flags(lab: np.ndarray, frames: Iterable[int]) -> dict[tuple[int, int], str]:
    """Single-model checks: {(frame, slice): reason} for slices that look implausible.

    - blood pool with no wall around it, or wall broken into pieces
    - LV area jumps by more than 2x versus both neighbouring slices (likely a miss or a leak)
    """
    flags: dict[tuple[int, int], str] = {}
    for t in frames:
        areas = []
        for z in range(lab.shape[2]):
            sl = lab[:, :, z, t]
            lv = np.isin(sl, (L.LV_BLOODPOOL, L.LV_MYOCARDIUM))
            areas.append(int(lv.sum()))
            blood, wall = sl == L.LV_BLOODPOOL, sl == L.LV_MYOCARDIUM
            if blood.any() and not wall.any():
                flags[(t, z)] = "blood pool but no wall"
            elif wall.any() and ndimage.label(wall)[1] > 1 and blood.any():
                flags[(t, z)] = "wall in several pieces"
        for z in range(1, lab.shape[2] - 1):
            a, lo, hi = areas[z], areas[z - 1], areas[z + 1]
            if a and lo and hi and (a > 2 * max(lo, hi) or 2 * a < min(lo, hi)):
                flags.setdefault((t, z), "area jumps versus neighbouring slices")
    return flags


# ---------------------------------------------------------------- helpers

def _minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


@dataclass
class _Fit:
    arr: np.ndarray
    ox: int
    oy: int
    src_shape: tuple

    def undo(self, lab: np.ndarray) -> np.ndarray:
        sx, sy = self.src_shape[:2]
        n = self.arr.shape[0]
        out = np.zeros((sx, sy) + lab.shape[2:], lab.dtype)
        x0, y0 = max(self.ox, 0), max(self.oy, 0)
        px, py = max(-self.ox, 0), max(-self.oy, 0)
        wx, wy = min(n, sx), min(n, sy)
        out[x0:x0 + wx, y0:y0 + wy] = lab[px:px + wx, py:py + wy]
        return out


def _fit(a: np.ndarray, n: int) -> _Fit:
    """Centre-crop or zero-pad the first two axes to n x n."""
    sx, sy = a.shape[:2]
    ox, oy = (sx - n) // 2, (sy - n) // 2
    out = np.zeros((n, n) + a.shape[2:], a.dtype)
    x0, y0 = max(ox, 0), max(oy, 0)
    px, py = max(-ox, 0), max(-oy, 0)
    wx, wy = min(n, sx), min(n, sy)
    out[px:px + wx, py:py + wy] = a[x0:x0 + wx, y0:y0 + wy]
    return _Fit(out, ox, oy, a.shape)


def _resize_nearest(lab: np.ndarray, shape) -> np.ndarray:
    f = [shape[i] / lab.shape[i] for i in range(len(shape))]
    out = ndimage.zoom(lab, f, order=0)
    fixed = np.zeros(shape, lab.dtype)
    s = tuple(slice(0, min(out.shape[i], shape[i])) for i in range(len(shape)))
    fixed[s] = out[s]
    return fixed


def _keep_main_component(lab: np.ndarray) -> np.ndarray:
    """Keep the LV (largest 3D connected blob of blood + myocardium); drop far-off specks."""
    lv = lab > 0
    comps, n = ndimage.label(lv)
    if n <= 1:
        return lab
    sizes = ndimage.sum(lv, comps, range(1, n + 1))
    keep = comps == (1 + int(np.argmax(sizes)))
    return np.where(keep, lab, 0).astype(lab.dtype)
