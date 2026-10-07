"""Automatic contouring with published mouse cardiac MRI models.

Two backends, both run from the user's own download of each model (nothing is
bundled, so each model keeps its own license):

- "ucl":    Shah et al., JCMR 2026 (UNet3+, 2D, LV blood pool + myocardium).
            GPL-3.0: point VENTRIMAP_UCL_DIR at a clone of
            github.com/mrphys/Open-Source_Pre-Clinical_Segmentation with the
            real weights (Git LFS) in ModelWeights/.
- "calico": Calico Life Sciences, Sci Rep 2025 (3D UNet, LV + myocardium + RV).
            MIT: point VENTRIMAP_CALICO_DIR at a clone of
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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy import ndimage

from . import labels as L

Progress = Callable[[int, int], None]  # (done, total)


def _dir(env: str, default: str) -> Path:
    return Path(os.environ.get(env, Path.home() / ".ventrimap" / "models" / default)).expanduser()


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
    u = _dir("VENTRIMAP_UCL_DIR", "ucl")
    w = _ucl_weights(u)
    out["ucl"] = "ready" if w and (u / "unet3plus.py").exists() else f"not installed (set VENTRIMAP_UCL_DIR; looked in {u})"
    c = _dir("VENTRIMAP_CALICO_DIR", "calico")
    ck = c / "model_checkpoints" / "short_axis.ckpt"
    ok = ck.exists() and ck.stat().st_size > 1_000_000  # an LFS pointer is ~130 bytes
    out["calico"] = "ready" if ok else f"not installed (set VENTRIMAP_CALICO_DIR; looked in {c})"
    return out


def _ucl_weights(d: Path) -> Path | None:
    for p in (d / "ModelWeights" / "SEG459.h5", d / "ModelWeights" / "real" / "SEG459.h5"):
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    return None


# ---------------------------------------------------------------- UCL model

_ucl_model = None


def _load_ucl():
    global _ucl_model
    if _ucl_model is None:
        d = _dir("VENTRIMAP_UCL_DIR", "ucl")
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
        _ucl_model = model
    return _ucl_model


def _ucl_volume(vol_xyz: np.ndarray, spacing) -> np.ndarray:
    """One frame (x, y, slice) -> UCL codes (1 myo, 2 blood). The model expects 256x256 at 0.1 mm."""
    model = _load_ucl()
    dx, dy, _ = spacing
    f = (dx / 0.1, dy / 0.1)
    work = vol_xyz if f == (1.0, 1.0) else ndimage.zoom(vol_xyz, (f[0], f[1], 1), order=1)
    pad = _fit(work, 256)
    x = np.stack([_minmax(s) for s in np.moveaxis(pad.arr, 2, 0)])[..., None]
    out = model.predict(x, verbose=0)
    out = out[0] if isinstance(out, (list, tuple)) else out
    lab = np.moveaxis(out.argmax(-1), 0, 2).astype(np.uint8)
    lab = _keep_main_component(pad.undo(lab))
    if f != (1.0, 1.0):
        lab = _resize_nearest(lab, vol_xyz.shape)
    return lab


# ------------------------------------------------------------- Calico model

_calico = None


def _load_calico():
    global _calico
    if _calico is None:
        d = _dir("VENTRIMAP_CALICO_DIR", "calico")
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
            raw = _ucl_volume(vol, spacing)
            std = np.zeros_like(raw)
            std[raw == 1] = L.LV_MYOCARDIUM
            std[raw == 2] = L.LV_BLOODPOOL
        else:
            raw = _calico_volume(vol, spacing)
            std = np.zeros_like(raw)
            std[raw == 1] = L.LV_BLOODPOOL
            std[raw == 2] = L.LV_MYOCARDIUM
            std[raw == 3] = L.RV_BLOODPOOL
        out[..., t] = std
        if progress:
            progress(i + 1, len(frames))
    return out


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
