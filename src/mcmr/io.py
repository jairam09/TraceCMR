"""Loading images and labels into one consistent object.

Every later step (volumes, wall thickness, scar) starts from a CardiacImage,
so all format-specific handling lives here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from . import labels as L

# "myocardium": papillary muscles/trabeculae labeled as myocardium (excluded from blood pool),
#               e.g. the UCL mouse dataset.
# "blood_pool": endocardial contour drawn around them, so they count as blood pool,
#               e.g. a Segment SOP that contours the compact wall only.
PAPILLARY_CONVENTIONS = ("myocardium", "blood_pool", None)


@dataclass
class CardiacImage:
    """One 3D short-axis stack (one cardiac phase) plus its labels.

    Arrays are indexed (x, y, slice). Spacing is in mm: (dx, dy, dz), where
    dz is the center-to-center distance between slices.
    """

    image: np.ndarray
    spacing: tuple[float, float, float]
    label: np.ndarray | None = None
    subject_id: str = "unknown"
    phase: str | None = None  # e.g. "ED" or "ES"
    affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    # How the labels treat papillary muscles and trabeculae. This cannot be reliably
    # converted after contouring, so it is carried with the data and checked whenever
    # results are combined or compared. None = unknown.
    papillary: str | None = None

    def __post_init__(self) -> None:
        if self.papillary not in PAPILLARY_CONVENTIONS:
            raise ValueError(f"papillary must be one of {PAPILLARY_CONVENTIONS}, got {self.papillary!r}")
        if self.image.ndim != 3:
            raise ValueError(f"image must be 3D (x, y, slice), got shape {self.image.shape}")
        if self.label is not None and self.label.shape != self.image.shape:
            raise ValueError(
                f"label shape {self.label.shape} does not match image shape {self.image.shape}"
            )
        if any(s <= 0 for s in self.spacing):
            raise ValueError(f"spacing must be positive, got {self.spacing}")

    @property
    def voxel_volume_mm3(self) -> float:
        """Volume of one voxel in mm^3 (= microliters)."""
        dx, dy, dz = self.spacing
        return dx * dy * dz

    def labels_present(self) -> dict[int, int]:
        """Map of label number -> voxel count, for labels that appear."""
        if self.label is None:
            return {}
        values, counts = np.unique(self.label, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def slices_with(self, label_value: int) -> list[int]:
        """Indices of slices that contain at least one voxel of label_value."""
        if self.label is None:
            return []
        has = np.any(self.label == label_value, axis=(0, 1))
        return [int(i) for i in np.flatnonzero(has)]


def remap_labels(label: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Translate a dataset's label numbers into the standard scheme.

    Raises if the array contains a value the mapping does not cover, so an
    unexpected label never slips through silently.
    """
    present = set(np.unique(label).astype(int).tolist())
    unknown = present - set(mapping)
    if unknown:
        raise ValueError(f"label values {sorted(unknown)} not in mapping {mapping}")
    out = np.zeros_like(label, dtype=np.uint8)
    for src, dst in mapping.items():
        out[label == src] = dst
    return out


def load_nifti(
    image_path: str | Path,
    label_path: str | Path | None = None,
    *,
    label_map: str | dict[int, int] = "standard",
    frame: int | None = None,
    subject_id: str | None = None,
    phase: str | None = None,
    papillary: str | None = None,
) -> CardiacImage:
    """Load a NIfTI image (and optional label file) into a CardiacImage.

    Args:
        image_path: .nii or .nii.gz image.
        label_path: matching label file, or None.
        label_map: "acdc", "standard", or a dict {dataset value: standard value}.
        frame: for 4D cine files (x, y, slice, time), which time frame to take.
        subject_id: defaults to the image file name.
        phase: optional tag such as "ED" or "ES".
        papillary: "myocardium", "blood_pool", or None if unknown. Set it whenever you know.
    """
    image_path = Path(image_path)
    img = nib.load(str(image_path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    zooms = img.header.get_zooms()

    if data.ndim == 4:
        if frame is None:
            raise ValueError(
                f"{image_path.name} is 4D with {data.shape[3]} frames; pass frame=<index>"
            )
        data = data[..., frame]
    elif data.ndim == 2:
        data = data[..., np.newaxis]
    elif data.ndim != 3:
        raise ValueError(f"unsupported image shape {data.shape}")

    spacing = tuple(float(z) for z in zooms[:3]) if len(zooms) >= 3 else (*map(float, zooms[:2]), 1.0)

    label = None
    if label_path is not None:
        lab_img = nib.load(str(label_path))
        lab = np.asarray(lab_img.get_fdata())
        if lab.ndim == 4 and frame is not None:
            lab = lab[..., frame]
        lab = np.rint(lab).astype(np.int32)
        mapping = L.LABEL_MAPS[label_map] if isinstance(label_map, str) else label_map
        label = remap_labels(lab, mapping)
        lab_zooms = tuple(float(z) for z in lab_img.header.get_zooms()[:3])
        if not np.allclose(lab_zooms, spacing, rtol=1e-3):
            raise ValueError(f"label spacing {lab_zooms} differs from image spacing {spacing}")

    if subject_id is None:
        subject_id = image_path.name.replace(".nii.gz", "").replace(".nii", "")

    return CardiacImage(
        image=data,
        spacing=spacing,  # type: ignore[arg-type]
        label=label,
        subject_id=subject_id,
        phase=phase,
        affine=np.asarray(img.affine),
        papillary=papillary,
    )


def save_nifti(ci: CardiacImage, image_path: str | Path, label_path: str | Path | None = None) -> None:
    """Write a CardiacImage back to NIfTI (used by tests and format conversion)."""
    affine = ci.affine.copy()
    # Make sure the affine encodes the spacing, so it round-trips through the header.
    for i, s in enumerate(ci.spacing):
        col = affine[:3, i]
        norm = np.linalg.norm(col)
        affine[:3, i] = col / norm * s if norm > 0 else np.eye(3)[i] * s
    img = nib.Nifti1Image(ci.image.astype(np.float32), affine)
    img.header.set_zooms(ci.spacing)
    nib.save(img, str(image_path))
    if label_path is not None and ci.label is not None:
        lab = nib.Nifti1Image(ci.label.astype(np.uint8), affine)
        lab.header.set_zooms(ci.spacing)
        nib.save(lab, str(label_path))


def sanity_check(ci: CardiacImage, species: str = "mouse") -> list[str]:
    """Return human-readable warnings about things that often indicate a bug."""
    warnings: list[str] = []
    dx, dy, dz = ci.spacing
    if species == "mouse":
        if not (0.02 <= dx <= 0.5 and 0.02 <= dy <= 0.5):
            warnings.append(
                f"in-plane spacing {dx:.3f} x {dy:.3f} mm is unusual for mouse (expected ~0.05-0.3 mm)"
            )
    elif species == "human":
        if not (0.5 <= dx <= 3.5 and 0.5 <= dy <= 3.5):
            warnings.append(
                f"in-plane spacing {dx:.2f} x {dy:.2f} mm is unusual for human (expected ~1-2 mm)"
            )
    if not np.isclose(dx, dy, rtol=0.05):
        warnings.append(f"pixels are not square ({dx:.3f} vs {dy:.3f} mm); check the header")
    if ci.label is not None:
        present = ci.labels_present()
        for needed in (L.LV_BLOODPOOL, L.LV_MYOCARDIUM):
            if needed not in present:
                warnings.append(f"no {L.NAMES[needed]} voxels in the label")
        if L.LV_BLOODPOOL in present:
            sl = ci.slices_with(L.LV_BLOODPOOL)
            if sl and (max(sl) - min(sl) + 1) != len(sl):
                warnings.append(f"LV blood pool slices are not contiguous: {sl}")
    if float(np.ptp(ci.image)) == 0.0:
        warnings.append("image is constant (all pixels identical)")
    return warnings
