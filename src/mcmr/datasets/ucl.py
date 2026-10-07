"""Loader for the UCL open mouse cardiac MRI dataset.

Shah W, Stuckey DJ, et al. Open-source pre-clinical image segmentation: mouse
cardiac magnetic resonance imaging datasets with a deep learning segmentation
framework. J Cardiovasc Magn Reson, 2026.
Data: https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation

File format (one .h5 per mouse, from the dataset README):
    Images           (256, 256, slices, frames)  uint16 cine short-axis stack
    Masks            same shape, uint8: 0 background, 1 LV myocardium, 2 LV blood pool;
                     frames without expert labels are all zeros
    Frames_with_ROI  0-based indices of the labeled frames (ED and ES, plus
                     extra artifact frames for a few mice)

The files store no voxel spacing. Per the paper's methods: 9.4 T Agilent scanner,
interpolated pixel size 0.1 x 0.1 mm, slice thickness 1.0 mm, contiguous slices.
Papillary muscles and trabeculae are included in the myocardium.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import labels as L
from ..io import CardiacImage, remap_labels

# From the paper's imaging protocol; the .h5 files do not record it.
UCL_SPACING = (0.1, 0.1, 1.0)


def _open(path: str | Path):
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading UCL files needs h5py: pip install h5py") from e
    return h5py.File(str(path), "r")


def labeled_frames(path: str | Path) -> list[int]:
    """0-based indices of frames that have expert labels."""
    with _open(path) as f:
        listed = sorted(int(t) for t in np.asarray(f["Frames_with_ROI"]).ravel())
        masks = f["Masks"]
        # Trust the listed frames only if they really contain labels.
        return [t for t in listed if np.any(masks[..., t])]


def find_ed_es(path: str | Path) -> dict[str, int]:
    """Identify ED (largest blood pool) and ES (smallest) among the labeled frames.

    For the few mice with extra labeled artifact frames, ED and ES are still the
    frames with the largest and smallest labeled blood pool.
    """
    frames = labeled_frames(path)
    if len(frames) < 2:
        raise ValueError(f"{Path(path).name}: need at least 2 labeled frames, found {frames}")
    with _open(path) as f:
        masks = f["Masks"]
        blood = {t: int(np.count_nonzero(masks[..., t] == 2)) for t in frames}
    return {"ED": max(blood, key=blood.get), "ES": min(blood, key=blood.get)}


def load_ucl(
    path: str | Path,
    frame: int | str = "ED",
    spacing: tuple[float, float, float] = UCL_SPACING,
) -> CardiacImage:
    """Load one cardiac frame of one UCL mouse as a CardiacImage.

    Args:
        path: a MouseN.h5 file.
        frame: "ED", "ES", or a 0-based frame index. Unlabeled frames load with label=None.
        spacing: override only if you know better than the paper's protocol.
    """
    path = Path(path)
    phase: str | None = None
    if isinstance(frame, str):
        phase = frame.upper()
        if phase not in ("ED", "ES"):
            raise ValueError(f"frame must be 'ED', 'ES' or an int, got {frame!r}")
        frame = find_ed_es(path)[phase]

    with _open(path) as f:
        images = f["Images"]
        masks = f["Masks"]
        if images.ndim != 4:
            raise ValueError(f"expected 4D Images (x, y, slice, frame), got {images.shape}")
        n_frames = images.shape[3]
        if not 0 <= frame < n_frames:
            raise IndexError(f"frame {frame} out of range (0..{n_frames - 1})")
        image = np.asarray(images[..., frame], dtype=np.float32)
        raw_mask = np.asarray(masks[..., frame])

    label = remap_labels(raw_mask.astype(np.int32), L.UCL) if raw_mask.any() else None
    return CardiacImage(
        image=image,
        spacing=spacing,
        label=label,
        subject_id=path.stem,
        phase=phase if phase else f"frame{frame}",
        affine=np.diag([*spacing, 1.0]),
        papillary="myocardium",  # per the paper: trabeculae and papillary muscles in the myocardium
    )
