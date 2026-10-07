# Lab notebook

One entry per work session: date, what you tried, what happened, what you decided. Newest at the top.

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
