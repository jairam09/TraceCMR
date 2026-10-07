# Lab notebook

One entry per work session: date, what you tried, what happened, what you decided. Newest at the top.

## 2026-10-07 (name, logo, demo)

- Renamed the project VentriMap (package `ventrimap`). Name check: no existing tool called VentriMap; free on PyPI. Similar names in human cardiology (Vektor vMap, Ventripoint) are different products.
- Logo: first try (16-segment bullseye) rejected as too busy. Final: the LV seen from the side as a V, built from horizontal bars like a stack of short-axis slices, with the endocardial contour in oxblood. Animated version thickens the wall in systole (what VentriMap measures); static version for README, paper and favicon. Wordmark outlined from Inter Display SemiBold (OFL).
- Built a web demo of the pipeline on Mouse40, 1, 90 and 60 (load/check, base-apex, contours, volumes, wall rays, bullseyes).
- Fixed: repo bullseye plots were mirrored relative to the image; now clockwise from 3 o'clock like the demo.

## 2026-10-07 (wall thickness + DICOM)

- Direction decided overnight: one app (napari plugin) combining auto-contour, manual correction and analysis; flagship feature = tracking the same mouse across scans. Build order: Segment AI check -> wall thickness -> longitudinal -> scar -> auto-contour -> app.
- Built `wall.py` (radial rays, sub-pixel borders, 120 rays). Phantoms: within 0.04 mm everywhere.
- 16 UCL mice: ED wall 0.76-1.17 mm; mean thickening vs EF r = 0.85. Mouse60 (DCM): thin mid/apical walls, low thickening. Mouse40 (healthy): thickening 33-96%.
- Negative thickening in 11 segments, 10 of them basal: ray overlay of Mouse80 shows the top ES slice is a partial ring at the outflow tract. Flagged, not hidden.
- Papillary spikes visible in ray overlays exactly where predicted (UCL labels papillary as myocardium).
- Built `dicom.py`. Tests caught a bug: trigger time 0 ms treated as missing -> scrambled time order. Fixed.
- Next: Segment Pre-clinical AI check; DICOM + Segment contour reader on one real lab file (ask first); wall thickness v2 (chords).

## 2026-10-06 (base/apex)

- Labeled base/apex by eye for 16 UCL mice from anatomy (montages of all labeled slices). 3 apex-first: Mouse1, 3, 90.
- Scored 4 rules. Area-profile thirds and end slices 16/16; apical cap never wrong but present in only 7; RV/atria brightness band 15/16 (missed Mouse50), not used.
- Built `orientation.py`: thirds ratio >= 1.25 plus no contradictions, else refuse. ED 16/16 correct (min ratio 1.54). ES 14 correct, 2 refused, 0 wrong. Orientation decided per heart from ED with ES cross-check.
- Visual check of levels on Mouse1 (apex-first) and Mouse2 (base-first): reordering and basal/mid/apical/cap labels look right.
- Next: wall thickness v1 (radial rays) on ring and tapered phantoms; think about the papillary problem for thickness.

## 2026-10-06 (Phase 2 start)

- Papillary convention: UCL labels papillary muscles as myocardium; my Segment SOP puts them inside the endo contour (blood pool). Tried converting UCL labels: closing barely changes ESV (papillary muscles look like wall at ES); convex hull upper bound shifts EDV +13%, ESV +18%, mass -9%. Decision: track the convention on every dataset, never convert, refuse cross-convention comparisons.
- Built `volumes.py` and `run_volumes.py`. 16 UCL mice (1, 2, 3, 10, 20 ... 130): EDV median 67.9 uL (44.8-157.1), ESV 28.1 (9.9-110.5), EF 61.4% (29.6-81.3), mass 103.8 mg (75.4-129.8). No QC flags.
- Independent check with raw h5py: identical EDV/ESV for all 16.
- Mouse60 (EDV 157, EF 30%): overlay shows a dilated thin-walled LV with good contours, consistent with the DCM group, not a bug.
- Found: slice order varies (13 base-first, 3 apex-first). Must solve before Phase 3.
- Next: robust base/apex detection; then wall thickness v1 (radial rays) on phantoms.

## 2026-10-06 (later)

- Found that the UCL mouse dataset is public (JCMR, Feb 2026): 130 mice, 9.4 T cine, LV blood pool + myocardium at ED/ES. Repo: mrphys/Open-Source_Pre-Clinical_Segmentation.
- Added `load_ucl`. Files store no spacing; used 0.1 x 0.1 x 1.0 mm from the paper's methods.
- Mouse1 check: EDV 74.7 uL, ESV 22.7 uL, EF 70%, LV mass ~114 mg. Overlay: contours sit on the LV in all slices; slice 0 = apical cap.
- License question: paper says CC BY 4.0, repo says GPL-3.0. Use data with citation; don't copy their code.
- Next: download 10-20 mice and run `check_dataset.py` on each, then start Phase 2 (volumes) and compare EF across disease groups.

## 2026-10-06

- Started the project. Repo scaffold created: NIfTI loader with label remapping (ACDC supported), sanity checks, synthetic ring phantom, overlay figures, 14 tests passing.
- Phantom check: blood pool 126.4 µL by voxel counting vs 125.7 µL exact (0.6% pixelation error at 0.1 mm pixels).
- Next: install on laptop (SETUP.md), push to GitHub, email UCL authors about the mouse dataset, register for ACDC and run `check_dataset.py` on a few subjects.
