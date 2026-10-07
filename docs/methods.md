# Methods and conventions

Every choice that changes a result is written down here, with the reason. Papers and reviewers will ask about each one.

## Data representation (decided)

- Arrays are indexed `(x, y, slice)`, one cardiac phase per `CardiacImage`.
- Spacing is `(dx, dy, dz)` in mm, read from the file header. `dz` is the center-to-center slice distance (slice thickness + gap).
- Volumes are in µL (1 mm³ = 1 µL). Human results may also be shown in mL.
- Labels use one internal scheme (`src/mcmr/labels.py`): 0 background, 1 LV blood pool, 2 LV myocardium, 3 RV blood pool, 4 scar, 5 MVO. Loaders translate each dataset's numbers and refuse unknown values rather than guessing.

## Dataset-specific facts (decided by the source, not by us)

- UCL mouse data: spacing (0.1, 0.1, 1.0) mm taken from the paper because the files do not store it; trabeculae and papillary muscles are labeled as myocardium; ED/ES chosen as the labeled frames with largest/smallest blood pool.

## Papillary muscles and trabeculae (decided: track, never convert)

- Two conventions exist: labeled as **myocardium** (UCL dataset) or enclosed in the endocardial contour as **blood pool** (the author's Segment SOP).
- Every `CardiacImage` carries `papillary` = `"myocardium"`, `"blood_pool"` or `None` (unknown). `load_ucl` sets `"myocardium"`. `compute_function` refuses to mix conventions between ED and ES, and `check_comparable` refuses to compare results across conventions unless explicitly allowed.
- Why not convert: tested on UCL Mouse1 (2026-10-06). Morphological closing of the blood pool barely moved ESV, because at end-systole papillary muscles press against the wall and are indistinguishable from wall in the mask. Filling to the convex hull (an upper bound) changed EDV 74.7 -> 84.3 uL (+13%), ESV 22.7 -> 26.7 uL (+18%), mass 114 -> 104 mg (-9%), EF 69.6 -> 68.3%. The effect is too large to ignore and cannot be recovered reliably from labels.
- Consequence for wall thickness (Phase 3): with `"myocardium"` labels, attached papillary muscles inflate wall thickness where they sit. A papillary-aware endocardial border is needed.

## Volumes (Phase 2)

- [x] Simpson's method: per-slice area x slice spacing, summed (`volumes.py`).
- [x] Myocardial density for mass: 1.05 g/mL. Scar and MVO count toward mass, not blood pool.
- [x] QC flags: ESV >= EDV (phases swapped?), LV mass differing >15% between ED and ES, unknown papillary convention.
- [ ] Basal slice inclusion rule (UCL: follow their labels as given).
- [ ] Area from pixel counts vs from contour polygons (compare both against Segment when contour data is available).

## Slice order and levels (decided, `orientation.py`)

- UCL stacks are not consistently ordered: of 16 mice, 13 are stored base-first and 3 apex-first. Ground truth was set by eye from anatomy (base end: atria, great vessels, large RV; apex end: small LV, no RV) and is kept in `tests/data/ucl_orientation_truth.csv`.
- Candidate rules scored against that truth (ED): area-profile thirds 16/16; end-slice areas 16/16; apical cap (myocardium-only end slice) correct 7/7 when present, silent otherwise; bright tissue outside the epicardium (image-based RV/atria cue) 15/16, so it was not used.
- Rule adopted: blood-pool area of the basal third must be at least 1.25x the apical third (lowest observed at ED: 1.54); apical cap and end slices must not contradict it. Otherwise the detector raises `OrientationUncertain` and the orientation must be given by hand. It never guesses.
- Orientation is decided per heart from ED, cross-checked with ES. Single-phase results: ED 16 correct; ES 14 correct, 2 refused (Mouse60 dilated, Mouse130), 0 wrong.
- Levels: slices with blood pool are split into thirds base -> apex (`np.array_split`, extra slices to basal then mid); myocardium-only slices beyond the apex are the apical cap.
- Limitation: 16 mice with only 3 apex-first stacks. Re-validate on each new dataset.

## Wall thickness (Phase 3, version 1 implemented in `wall.py`)

- [x] **Radial rays.** Centre = blood-pool centroid; 120 rays (every 3 degrees). Along each ray, the endocardium is where the interpolated blood-pool mask first drops below 0.5 and the epicardium is where the interpolated LV mask (blood + myocardium + scar) drops below 0.5 beyond that. Sampling step 0.25 px with bilinear interpolation gives sub-pixel borders, important when a mouse wall is ~10 pixels thick. Rays with no valid wall are NaN.
- [x] **Levels and segments.** Slices are ordered base-first using the per-heart orientation (`orientation.py`), split into basal / mid / apical thirds, and each level is divided into equal angular sectors in the AHA layout (6 basal, 6 mid, 4 apical). Each segment value = mean over rays in the sector, averaged over that level's slices.
- [x] **Thickening** = (ESWT - EDWT) / EDWT x 100%, per level and segment. ED and ES are compared level by level, not slice by slice, because the base moves toward the apex in systole.
- [x] **Checks.** Phantom rings: thickness within 0.04 mm of truth at every angle, independent of centre position, scaling correctly with spacing. Real data (16 UCL mice, 2026-10-07): ED wall 0.76-1.17 mm; mean thickening correlates with EF (r = 0.85); dilated Mouse60 has the thinnest mid/apical walls (0.45-0.66 mm) and low thickening.
- **Known limitations (version 1):**
  - Rays overestimate thickness where the wall is oblique to the ray. Version 2: centreline chords.
  - Labels that count papillary muscles as myocardium (UCL) inflate thickness where they attach; visible as spikes in ray overlays (e.g. Mouse80 ED). Tested with a synthetic papillary bump.
  - Basal thickening is least reliable: 10 of 11 negative-thickening segments across 16 mice were basal, where the most basal ES slice is a partial ring at the outflow tract. These segments are flagged, not hidden.
  - Segments are numbered from the image x-axis, counter-clockwise. They are not AHA-named until a septal reference (RV insertion point) is available; UCL data has no RV labels.
- [ ] Version 2: centreline chord method; compare with rays on ellipses and real hearts.
- [ ] Septal reference from RV insertion (needs RV labels, e.g. ACDC or lab data).
- [ ] Papillary-aware endocardial border for data labeled with papillary muscles as myocardium.

## DICOM input (`dicom.py`)

- Reads classic single-frame and enhanced multi-frame DICOM. Slice order from ImagePositionPatient projected on the slice normal; time order from TriggerTime / NominalCardiacTriggerDelayTime (fallback: instance number).
- Pixel spacing from PixelSpacing (row spacing -> y, column spacing -> x); slice spacing = median distance between slice positions, so gaps are included.
- Refuses incomplete grids (missing frames, mixed orientations or matrix sizes) instead of guessing.
- Bug caught by tests during development: a trigger time of 0 ms was treated as missing, which scrambled time order. Fixed with an explicit missing-value check.
- Tested only on synthetic series so far; needs one real Bruker DICOM export to confirm.

## Scar (Phase 4, to decide)

- [ ] Default threshold method (n-SD with n = ?, or FWHM).
- [ ] Remote region selection rule.
- [ ] Minimum scar cluster size (mm²).
- [ ] Endocardial erosion (pixels) to avoid blood-pool contamination.
- [ ] MVO handling (reported with and without).
