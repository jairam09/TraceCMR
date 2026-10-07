"""Load one image (+ label) and report what's in it. Run this on every new dataset.

Examples:
    # A synthetic phantom, to check your install works:
    python scripts/check_dataset.py --phantom --out data/checks

    # One ACDC subject (end-diastolic frame file + its ground truth):
    python scripts/check_dataset.py \
        --image data/acdc/patient001/patient001_frame01.nii.gz \
        --label data/acdc/patient001/patient001_frame01_gt.nii.gz \
        --label-map acdc --species human --out data/checks
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mcmr import labels as L
from mcmr.io import load_nifti, sanity_check
from mcmr.synthetic import ring_phantom
from mcmr.viz import save_overlay


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, help="NIfTI image file")
    p.add_argument("--label", type=Path, help="NIfTI label file (optional)")
    p.add_argument("--label-map", default="standard", choices=sorted(L.LABEL_MAPS), help="dataset label convention")
    p.add_argument("--frame", type=int, default=None, help="time frame for 4D cine files")
    p.add_argument("--species", default="mouse", choices=["mouse", "human"])
    p.add_argument("--phantom", action="store_true", help="use a synthetic phantom instead of a file")
    p.add_argument("--out", type=Path, default=Path("data/checks"), help="folder for the overlay PNG")
    args = p.parse_args()

    if args.phantom:
        ci = ring_phantom(noise_sd=0.05)
    elif args.image is None:
        p.error("give --image (and usually --label), or --phantom")
    else:
        ci = load_nifti(args.image, args.label, label_map=args.label_map, frame=args.frame)

    dx, dy, dz = ci.spacing
    print(f"subject      {ci.subject_id}")
    print(f"shape        {ci.image.shape}  (x, y, slices)")
    print(f"spacing      {dx:.4g} x {dy:.4g} x {dz:.4g} mm")
    print(f"voxel volume {ci.voxel_volume_mm3:.4g} mm^3 (= uL)")
    print(f"intensity    min {ci.image.min():.4g}, max {ci.image.max():.4g}")
    if ci.label is not None:
        print("labels present:")
        for value, count in ci.labels_present().items():
            vol = count * ci.voxel_volume_mm3
            print(f"  {value} {L.NAMES.get(value, '?'):<15} {count:>8} voxels  {vol:10.2f} uL")
        print(f"LV blood pool on slices {ci.slices_with(L.LV_BLOODPOOL)}")

    warnings = sanity_check(ci, species=args.species)
    print("warnings:" if warnings else "warnings: none")
    for w in warnings:
        print(f"  - {w}")

    args.out.mkdir(parents=True, exist_ok=True)
    out = save_overlay(ci, args.out / f"{ci.subject_id}_overlay.png")
    print(f"overlay saved to {out}  <- open it and look at every slice")


if __name__ == "__main__":
    main()
