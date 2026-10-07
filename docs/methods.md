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

## Slice order (open issue for Phase 3)

- UCL stacks are not consistently ordered: of 16 mice checked, 13 start at the base and 3 at the apex (judged by which end has the larger blood pool). Mouse130's ends are nearly equal, so that simple rule is unreliable. Volumes are unaffected; basal/mid/apical assignment needs a robust base/apex detector before Phase 3.

## Wall thickness (Phase 3, to decide)

- [ ] Radial rays (v1) vs centerline chords (v2).
- [ ] Segment scheme: AHA 17-segment default; reference = anterior RV insertion point.
- [ ] Assignment of slices to basal / mid / apical.

## Scar (Phase 4, to decide)

- [ ] Default threshold method (n-SD with n = ?, or FWHM).
- [ ] Remote region selection rule.
- [ ] Minimum scar cluster size (mm²).
- [ ] Endocardial erosion (pixels) to avoid blood-pool contamination.
- [ ] MVO handling (reported with and without).
