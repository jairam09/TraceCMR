# Changelog

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
