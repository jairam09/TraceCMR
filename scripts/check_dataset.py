"""Load one image (+ label) and report what's in it. Run this on every new dataset.

Examples:
    # A synthetic phantom, to check your install works:
    python scripts/check_dataset.py --phantom --out data/checks

    # One mouse from the UCL open dataset, at end-diastole and end-systole:
    python scripts/check_dataset.py --ucl data/ucl/Data/Mouse1.h5 --frame ED --out data/checks
    python scripts/check_dataset.py --ucl data/ucl/Data/Mouse1.h5 --frame ES --out data/checks

    # A folder of DICOM files for one cine series (frame 0 by default):
    python scripts/check_dataset.py --dicom path/to/series_folder --frame 0 --out data/checks

    # One ACDC subject (end-diastolic frame file + its ground truth):
    python scripts/check_dataset.py \
        --image data/acdc/patient001/patient001_frame01.nii.gz \
        --label data/acdc/patient001/patient001_frame01_gt.nii.gz \
        --label-map acdc --species human --out data/checks
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ventrimap import labels as L
from ventrimap.datasets import load_ucl
from ventrimap.dicom import load_dicom_cine
from ventrimap.io import load_nifti, sanity_check
from ventrimap.synthetic import ring_phantom
from ventrimap.viz import save_overlay


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, help="NIfTI image file")
    p.add_argument("--label", type=Path, help="NIfTI label file (optional)")
    p.add_argument("--label-map", default="standard", choices=sorted(L.LABEL_MAPS), help="dataset label convention")
    p.add_argument("--ucl", type=Path, help="a MouseN.h5 file from the UCL open mouse dataset")
    p.add_argument("--dicom", type=Path, help="a folder of DICOM files for one cine series (images only)")
    p.add_argument("--frame", default=None, help="time frame: an index for 4D NIfTI, or ED / ES / index for --ucl")
    p.add_argument("--species", default="mouse", choices=["mouse", "human"])
    p.add_argument("--phantom", action="store_true", help="use a synthetic phantom instead of a file")
    p.add_argument("--out", type=Path, default=Path("data/checks"), help="folder for the overlay PNG")
    args = p.parse_args()

    def as_frame(value):
        return int(value) if value is not None and str(value).lstrip("-").isdigit() else value

    if args.phantom:
        ci = ring_phantom(noise_sd=0.05)
    elif args.ucl is not None:
        ci = load_ucl(args.ucl, as_frame(args.frame) or "ED")
    elif args.dicom is not None:
        cine = load_dicom_cine(args.dicom)
        print(f"DICOM cine: {cine.data.shape[2]} slices x {cine.data.shape[3]} frames, "
              f"trigger times {cine.times_ms.min():.1f}-{cine.times_ms.max():.1f} ms")
        ci = cine.frame(int(args.frame) if args.frame is not None else 0)
    elif args.image is None:
        p.error("give --image (and usually --label), --ucl, or --phantom")
    else:
        ci = load_nifti(args.image, args.label, label_map=args.label_map, frame=as_frame(args.frame))

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
    tag = f"_{ci.phase}" if ci.phase else ""
    out = save_overlay(ci, args.out / f"{ci.subject_id}{tag}_overlay.png")
    print(f"overlay saved to {out}  <- open it and look at every slice")


if __name__ == "__main__":
    main()
