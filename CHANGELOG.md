# Changelog

## [Unreleased]

### Added
- Auto-contour (`autocontour.py`, app button): UCL or Calico model on every frame, sanity flags and optional two-model disagreement flags, ED/ES suggested from the volume curve, volume-over-time plot in the app. Models are user-installed (`docs/models.md`).
- App: play cine, Set ED/ES at the current frame; open a UCL file as a raw cine.
- `papillary.py`: experimental conversion of detailed contours to papillary-in-blood-pool (off by default).
- napari app (v0 prototype): `ventrimap` command or Plugins > VentriMap. Open a UCL file or DICOM folder, see contours over the images, set slice order by hand (flip, drag, leave slices out; warnings if the order looks reversed or contradicts scanner positions), edit contours with napari's brush, compute volumes/EF/mass and wall-thickness bullseyes, export CSV. Install with `pip install -e ".[app]"`.
- `contours.py`: mask <-> smooth curve conversion (adaptive control points, centripetal Catmull-Rom), exact curve areas; `scripts/curve_conversion_test.py`.
- App: slices chosen separately for ED and ES, with suggested choices and the reason for each (no contour, apical cap, basal slices flagged for checking when the drawn ring does not look like muscle).
- App: Snap to edge button (per slice): grows the endocardium into bright blood the model rounded off; tested on UCL test mice (helps EDV/mass, over-grows at ES, so manual, not automatic).
- App: Smooth button, Segment-style: smooths both contours on the slice being viewed (gentle, area-preserving) and hides that slice's points; Edit points brings them back; Undo.
- App: Segment-style curve editing (drag control points; add/delete points; pixel brush as fallback).
- `compute_wall(..., base_end=)` to set orientation by hand.
- `scripts/run_cohort.py`: one-row-per-mouse summary for a whole dataset.
- `scripts/predict_groups.py`: pre-registered blind group-prediction rules for UCL validation.

### Fixed
- UCL ED/ES selection could pick a partially contoured artifact frame as ES (3 of 130 mice; Mouse91 EF 87% -> 67%). Results-changing for those mice.

## [0.3.0] - 2026-10-07

### Changed
- Project renamed to **VentriMap**. Python package renamed `mcmr` -> `ventrimap` (imports change: `from ventrimap import ...`). Real-data test variable renamed `MCMR_UCL_DATA` -> `VENTRIMAP_UCL_DATA`. Entries below use the new names.
- Bullseye plots now run clockwise from 3 o'clock, matching how segments appear in the image (previously mirrored).

### Added
- VentriMap logo (`docs/assets/`): slice-stack V with the endocardial contour, light and dark, static and animated, plus an app icon.

## [0.2.0] - 2026-10-07

### Added
- `wall.py`: regional wall thickness (radial rays, sub-pixel borders), AHA-layout levels and segments, systolic thickening per segment, flags for negative thickening and papillary convention.
- `viz.save_bullseye`: bullseye plots of segment values.
- `scripts/run_wall.py`: batch wall thickness for UCL mice, CSV + bullseye plots.
- `dicom.py`: cine loader for classic and enhanced multi-frame DICOM, spacing from headers; `check_dataset.py --dicom`.
- Tests for wall thickness (phantoms, papillary bump, thickening, conventions), bullseye output and DICOM sorting/spacing.

### Fixed
- DICOM trigger time of 0 ms no longer treated as missing (caught by tests before release).

## [0.1.1] - 2026-10-06

### Added
- `orientation.py`: base/apex detection (refuses rather than guesses), per-heart detection from ED with ES cross-check, reordering to base-first, and basal/mid/apical/apical-cap level assignment.
- `synthetic.tapered_phantom`: bullet-shaped LV with optional apical cap, stored either way round.
- Visual ground truth for 16 UCL mice (`tests/data/ucl_orientation_truth.csv`) and tests against it.

## [0.1.0] - 2026-10-06

### Added
- `volumes.py`: EDV, ESV, SV, EF, LV mass (ED and ES), per-slice blood pool areas, QC flags.
- Papillary muscle convention tracked on every `CardiacImage` (`papillary`); mixing or comparing across conventions is refused unless explicitly allowed.
- `scripts/run_volumes.py`: batch volumes for a folder of UCL mice, written to CSV with a summary.
- Tests for volume geometry, spacing scaling, scar handling, QC flags and convention checks.

### Checked
- 16 UCL mice: results identical to an independent raw-file recomputation; EF 29.6-81.3%, no QC flags.

## [0.0.2] - 2026-10-06

### Added
- Loader for the UCL open mouse cardiac MRI dataset (`ventrimap.datasets.load_ucl`), with ED/ES detection and the dataset's label scheme.
- `check_dataset.py --ucl` option.
- Tests for the UCL loader on synthetic UCL-format files, plus an optional real-data test (`VENTRIMAP_UCL_DATA`).
- `h5py` dependency.

## [0.0.1] - 2026-10-06

### Added
- `CardiacImage` data structure; NIfTI loading (3D and 4D cine with frame selection) and saving.
- Standard internal label scheme and ACDC label mapping.
- Sanity checks for spacing, non-square pixels, missing labels and non-contiguous LV slices.
- Synthetic ring phantom with exact reference volumes.
- Overlay figures (LV endo red, LV epi green, RV endo magenta).
- `scripts/check_dataset.py` command-line check for any dataset.
- Tests and GitHub Actions CI.
