# Methods and conventions

Every choice that changes a result is written down here, with the reason. Papers and reviewers will ask about each one.

## Data representation (decided)

- Arrays are indexed `(x, y, slice)`, one cardiac phase per `CardiacImage`.
- Spacing is `(dx, dy, dz)` in mm, read from the file header. `dz` is the center-to-center slice distance (slice thickness + gap).
- Volumes are in µL (1 mm³ = 1 µL). Human results may also be shown in mL.
- Labels use one internal scheme (`src/mcmr/labels.py`): 0 background, 1 LV blood pool, 2 LV myocardium, 3 RV blood pool, 4 scar, 5 MVO. Loaders translate each dataset's numbers and refuse unknown values rather than guessing.

## Volumes (Phase 2, to decide)

- [ ] Papillary muscles: blood pool or myocardium? (Default: follow the source dataset; make it a setting.)
- [ ] Basal slice inclusion rule.
- [ ] Area from pixel counts vs from contour polygons (compare both against Segment).
- [ ] Myocardial density for mass: 1.05 g/mL (standard).

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
