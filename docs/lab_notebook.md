# Lab notebook

One entry per work session: date, what you tried, what happened, what you decided. Newest at the top.

## 2026-10-07 (auto-contour in the app)

- Built `autocontour.py` (UCL and Calico backends loaded from the user's own downloads; sanity flags; two-model disagreement flags) and the app's Auto-contour button, raw-cine opening, and a volume-over-time plot.
- End-to-end on UCL Mouse40 opened as a raw cine: 21 frames contoured in ~4 min on a 2-core CPU; ED/ES found at frames 0 and 9 (expert's frames); EF 79.1% vs expert 79.1%; EDV 55.7 vs 56.5 uL. Mouse40 may be in the model's training set, so this checks the workflow, not accuracy. The model also contoured the top slice at ED (not contoured by the expert): basal check should flag this more clearly (to do).
- Papillary smoothing (`papillary.py`) tried as the default per the author's decision: fills papillary slits at ED but eats compact wall where thickness varies, mostly at ES (+19% ESV on 30 mice). Kept experimental and off by default until tuned against hand-drawn smooth contours. Author's UCL mice contoured in Segment style would provide that ground truth.
- Running TensorFlow and PyTorch models in one process exceeds 8 GB; the double-check option needs more memory.
- Author review of auto-contours: better than Segment's AI in his opinion; a few errors, e.g. an endocardial tip cut off. Checked: the cut is in the model's mask itself, not the curve fit. Idea: "snap to edge" refinement.
- Snap to edge built and tested on UCL's 25 test mice: better EDV and mass, worse ESV at ES, EF unchanged within noise. Made a per-slice button rather than automatic. Smooth button changed to Segment-style (current slice, hides points) on author feedback; epicardium smoothed more (author spotted a model-mask indent).
- Control points v3 (author: points bunched at pixel steps): smooth outline (area-preserving), 18 even points, extras only at sharp tips with a 4 px minimum gap. 130-mouse test: within ~1% of pixel results, EF within 0.5 pts.

## 2026-10-07 (auto-contour model comparison, preliminary)

- Calico model (MIT, 3D, LV+myo+RV) set up: weights from the public repo (checksum matches). Runs on UCL after resampling to 0.13 mm / 0.8 mm; orientation does not matter. CPU only here: ~14 s per frame, so ~5 min per 20-frame scan without a GPU.
- UCL model (GPL, 2D UNet3+) run on its own 25 held-out test mice (split reproduced from their notebook, seed 42).
- On those 25 mice: UCL Dice blood 0.91/0.87 (ED/ES), myo 0.84/0.87, EF error mean -1.0 pts (worst 5.7). Calico (21 of the 25 done) Dice blood 0.87/0.85, myo 0.78/0.82, EF error mean -5.0 (worst 72). Calico on 45 mice so far: 4 with |EF error| > 10 pts (one missed ES, three badly dilated hearts).
- Caveat: UCL's model is on home data (same scanner and lab); the fair test is data from another lab and scanner. Remaining 85 Calico mice to run.
- Provisional plan: UCL model as default (optional separate install because of GPL), Calico as option (RV for the septal reference); flag slices where the two models disagree.
- Author decision: default analysis convention = papillary muscles in the blood pool (smooth), detailed mask kept for one-click switching.

## 2026-10-07 (ED/ES from raw cines)

- Author: Bruker gives raw cines, so ED/ES must be identified before analysis. Tried an image-only estimate (bright-blood pixel count inside the ED epicardium, summed over slices, per frame) on 130 UCL mice: ED within 1 frame of the expert's frame in 94%, ES in only 46%. Not good enough to choose ES automatically. Plan: user scrubs the cine and marks ED/ES (as in Segment); automatic suggestion from the volume curve once auto-contour has contoured every frame.
- Redesigned the app concept as a plain desktop tool (menu bar, toolbar, cine scrubber with ED/ES marking) after author feedback that the first version looked too "AI/techy".

## 2026-10-07 (napari v0 + curve conversion test)

- Built the napari app v0 (open, ED/ES, manual slice order with warnings, pixel brush editing, compute, bullseyes, CSV). Numbers identical to the pipeline (Mouse40 EF 79.1%).
- Decided (author): Segment-style curve contours with draggable points as the main editing mode; pixel brush as fallback.
- Conversion test, all 130 mice: at 24 even control points every measure stays within about 1% of the pixel method. Author then spotted the curve cutting across a sharp blood-pool tip (Mouse40 slice 4, 4 o'clock): switched to adaptive point placement (points added where the curve misses the outline by >1 px) and a centripetal Catmull-Rom curve (no overshoot with clustered points). Same ~1% agreement, shape now follows tips. Built the drag-to-edit curve editor in napari; dragging points updates the curve and results.

## 2026-10-07 (130-mouse cohort run)

- Downloaded all 130 UCL mice. The dataset gives group totals (39 control, 11 MI, 51 DCM, 29 TAC) but no per-mouse group labels; emailed the corresponding author (J. Steeden) to ask for them and to confirm voxel spacing.
- Pre-registered blind group-prediction rules (`scripts/predict_groups.py`, committed before looking at results).
- Ran `scripts/run_cohort.py` on all 130. Volumes/EF/mass for 130/130; wall thickness for 124/130.
- Bug found: in mice with extra artifact frames contoured on only some slices, the smallest blood pool was a partial frame, so ES was wrong (Mouse91: 2 of 7 slices, EF 87%). Fixed in `find_ed_es` (candidates must cover >= max-1 slices) with a test; affected Mouse91, 125, 48. Mouse91 EF now 67%.
- Orientation refused for 6 mice, all dilated (EF 15-37%): flatter area profiles. 0 known wrong, but the 114 new mice have no visual orientation truth yet; spot-check needed. Fallback for the app: one manual click.
- 8 mice flagged for ED/ES mass mismatch >15% (contour inconsistency in the source labels); to review visually.
- Cohort medians: EDV 67.9 uL, ESV 28.4, EF 57.9%, mass ED 107.4 mg / ES 115.2 mg. In line with the paper's reported test-set values (EDV 65.6, ESV 27.3, EF 59.3, mass 105.2 / 111.2; 25 mice), including ES mass > ED mass.
- Blind prediction counts: Control 91, DCM 14, TAC 13, MI 12 (expected 39 / 51 / 29 / 11). The rules catch severe phenotypes but call most of the DCM and TAC mice normal, so many DCM/TAC mice are mild by these measures. Scoring waits for the labels.

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
- Next: Segment Pre-clinical AI check; DICOM + Segment contour reader on one real lab file; wall thickness v2 (chords).

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
