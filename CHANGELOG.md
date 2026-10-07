# Changelog

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
- Loader for the UCL open mouse cardiac MRI dataset (`mcmr.datasets.load_ucl`), with ED/ES detection and the dataset's label scheme.
- `check_dataset.py --ucl` option.
- Tests for the UCL loader on synthetic UCL-format files, plus an optional real-data test (`MCMR_UCL_DATA`).
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
