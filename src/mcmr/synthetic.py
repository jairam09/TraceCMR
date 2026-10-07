"""Synthetic 'phantom' hearts with known geometry.

Real data never comes with a perfectly known answer. A phantom does: a stack
of concentric rings whose volumes and wall thickness we can compute exactly.
Every analysis function gets tested on phantoms before it touches real data.
"""

from __future__ import annotations

import numpy as np

from . import labels as L
from .io import CardiacImage


def ring_phantom(
    shape: tuple[int, int, int] = (128, 128, 10),
    spacing: tuple[float, float, float] = (0.1, 0.1, 1.0),
    blood_radius_mm: float = 2.0,
    wall_mm: float = 1.0,
    center_mm: tuple[float, float] | None = None,
    noise_sd: float = 0.0,
    seed: int = 0,
    subject_id: str = "phantom",
    phase: str | None = "ED",
) -> CardiacImage:
    """A cylinder-shaped LV: a disc of blood surrounded by a ring of myocardium.

    Defaults are mouse-sized: 4 mm blood pool diameter, 1 mm wall, 0.1 mm pixels.
    Image intensities: background 0, myocardium 0.4, blood 1.0, plus optional noise.
    """
    nx, ny, nz = shape
    dx, dy, _ = spacing
    if center_mm is None:
        center_mm = (nx * dx / 2, ny * dy / 2)

    # Physical coordinates of each pixel center, in mm.
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.hypot(xx - center_mm[0], yy - center_mm[1])

    label2d = np.zeros((nx, ny), dtype=np.uint8)
    label2d[r <= blood_radius_mm + wall_mm] = L.LV_MYOCARDIUM
    label2d[r <= blood_radius_mm] = L.LV_BLOODPOOL
    label = np.repeat(label2d[:, :, None], nz, axis=2)

    image = np.zeros(shape, dtype=np.float32)
    image[label == L.LV_MYOCARDIUM] = 0.4
    image[label == L.LV_BLOODPOOL] = 1.0
    if noise_sd > 0:
        rng = np.random.default_rng(seed)
        image += rng.normal(0.0, noise_sd, size=shape).astype(np.float32)

    return CardiacImage(
        image=image,
        spacing=spacing,
        label=label,
        subject_id=subject_id,
        phase=phase,
        affine=np.diag([*spacing, 1.0]),
    )


def tapered_phantom(
    n_slices: int = 8,
    base_blood_mm: float = 2.5,
    apex_blood_mm: float = 0.8,
    wall_mm: float = 1.0,
    apical_cap: bool = True,
    base_first: bool = True,
    spacing: tuple[float, float, float] = (0.1, 0.1, 1.0),
    size: int = 128,
) -> CardiacImage:
    """A bullet-shaped LV: blood pool radius shrinks linearly from base to apex.

    With apical_cap=True an extra myocardium-only slice closes the apex.
    base_first=False stores the slices apex-first, as some datasets do.
    """
    radii = list(np.linspace(base_blood_mm, apex_blood_mm, n_slices))
    slices = [ring_phantom((size, size, 1), spacing, r, wall_mm, phase="ED") for r in radii]
    if apical_cap:
        cap = ring_phantom((size, size, 1), spacing, apex_blood_mm, wall_mm, phase="ED")
        cap.label[cap.label == L.LV_BLOODPOOL] = L.LV_MYOCARDIUM
        cap.image[cap.label == L.LV_MYOCARDIUM] = 0.4
        slices.append(cap)
    image = np.concatenate([s.image for s in slices], axis=2)
    label = np.concatenate([s.label for s in slices], axis=2)
    if not base_first:
        image, label = image[:, :, ::-1].copy(), label[:, :, ::-1].copy()
    return CardiacImage(
        image=image, spacing=spacing, label=label, subject_id="tapered", phase="ED",
        affine=np.diag([*spacing, 1.0]),
    )


def ring_phantom_truth(
    shape: tuple[int, int, int] = (128, 128, 10),
    spacing: tuple[float, float, float] = (0.1, 0.1, 1.0),
    blood_radius_mm: float = 2.0,
    wall_mm: float = 1.0,
) -> dict[str, float]:
    """Exact (continuous) volumes for ring_phantom with the same arguments, in microliters."""
    nz = shape[2]
    height = nz * spacing[2]
    outer = blood_radius_mm + wall_mm
    blood = np.pi * blood_radius_mm**2 * height
    myo = np.pi * (outer**2 - blood_radius_mm**2) * height
    return {"blood_ul": float(blood), "myocardium_ul": float(myo), "wall_mm": wall_mm}
