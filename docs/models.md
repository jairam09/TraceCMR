# Auto-contour models

| Model | What it outlines | How you get it | Speed (2-core CPU, 10-slice frame) | Licence | Source |
| --- | --- | --- | --- | --- | --- |
| UCL (UNet3+, 2D) | LV blood pool, LV myocardium | Built into the installers from 0.3.5; older copies and source installs: one-click download (about 40 MB) | ~0.4 s one pass, ~3.5 s with the 8-orientation average (default) | GPL-3.0 | Shah et al., JCMR 2026; github.com/mrphys/Open-Source_Pre-Clinical_Segmentation |
| Calico (3D UNet) | LV blood pool, LV myocardium, RV blood pool | Manual install (below) | ~15-30 s | MIT | Calico Life Sciences, Sci Rep 2025; github.com/calico/mouse_cardiac_mri_segmentation |

A recent laptop is similar or faster. Contouring every frame of a 20-frame cine with the UCL model took 74 s on
the 2-core machine.

## The UCL model in TraceCMR

**What ships.** The authors publish a Keras/TensorFlow model (`ModelWeights/SEG459.h5`). TraceCMR converts it
unchanged to ONNX (`scripts/convert_ucl_onnx.py`, run by the `model` workflow) so it runs with ONNX Runtime, a
~15-20 MB library, instead of TensorFlow (~500 MB). Only the network's main output is kept (the deep-supervision
heads are training aids). The converted file is published once as the pre-release `ucl-model-1`
(`ucl_seg459.onnx`, 18.9 MB, with the authors' licence and a notice), and its size and SHA-256 are pinned in
`src/tracecmr/ucl_model.json`. The installers fetch it from there and check it (`scripts/fetch_ucl_model.py`);
the one-click setup does the same.

**Same answers as the original.** The conversion is checked before publishing: on three UCL mice (every slice at
ED and ES, the app's preprocessing) the largest difference in any class probability was 1.2e-5 and no pixel
changed label. Then through the whole app pipeline on UCL's 40 validation and test mice: EDV, ESV, SV, EF and
mass were identical to the TensorFlow model's on every mouse (`scripts/validate_backends.py`).

**Where TraceCMR looks for it**, in order: `TRACECMR_UCL_ONNX` (a file); `<TraceCMR>/models/ucl_seg459.onnx`
(where the installer puts it); `~/.tracecmr/models/ucl/ucl_seg459.onnx` (where the one-click setup puts it,
or `TRACECMR_UCL_DIR`). If there's no ONNX model but the authors' Keras files and TensorFlow are there (copies
set up with 0.3.1-0.3.4), it runs those instead. `TRACECMR_UCL_BACKEND=onnx|tensorflow` forces one.

**The 8-orientation average.** The model was trained on images in UCL's orientation. On a Bruker scan from
another facility, feeding the same scan turned or mirrored gave very different contours: EF from 30% to 73%
across the 8 orientations (one look each, every frame contoured, ED/ES from the volume curve). So Auto LV now runs
the model on each slice in all 8 orientations (4 quarter-turns, with and without a mirror), turns each answer
back and averages the probabilities. The result is then the same whatever the orientation: on that scan, EF
59.2% from every orientation, with identical contours. It is 8 times the work. Settings → "Steadier Auto LV"
turns it off (`TRACECMR_TTA=0`).

Averaging softens the outer edge, so the epicardium cutoff (P(myo) + P(blood) needed to count a pixel as LV) was
re-chosen for it, the same way as before: on UCL's 15 validation mice, the cutoff giving the smallest mass bias
(0.6; it is 0.9 for one pass), then reported once on the 25 test mice.

Steadier is not the same as right. On that outside scan (a cine acquired after a contrast injection) the averaged
contours still had clear errors on several slices: myocardium in pieces and no cavity on half the slices at
end-systole. The model has simply never seen scans like this: it was trained only on UCL's mice, scanner and
protocol, and this was a different kind of mouse, a different scanner and protocol, and acquired after contrast.
The fix is training data from such scans (TraceCMR keeps the model's draft next to every corrected contour, so
corrected scans become training pairs), not more tricks at run time.

## Accuracy on UCL's data

UCL's 25 held-out test mice (split reproduced from the authors' notebook: sorted IDs, `np.random.seed(42)`,
90/15/25), model at the expert's ED and ES frames, then the app's own slice order and inclusion; TraceCMR minus
expert, mean (SD):

| | EDV µL | ESV µL | EF pts | Mass mg | Worst EF error | Dice blood ED / ES | Dice myo ED / ES |
| --- | --- | --- | --- | --- | --- | --- | --- |
| One pass (0.3.4) | -2.4 (7.0) | -0.1 (2.2) | -1.1 (2.9) | -1.0 (13.1) | 7.8 | 0.907 / 0.869 | 0.845 / 0.875 |
| 8-orientation average (0.3.5) | -2.4 (6.3) | -0.0 (2.1) | -1.1 (3.0) | +1.7 (13.6) | 7.3 | 0.912 / 0.869 | 0.848 / 0.876 |

On the 15 validation mice the average was a wash too (EF -1.4 (7.0) vs -1.7 (5.5); one mouse with a hard base got
worse). On its own lab's data the model is already steady, so averaging changes little; it is there for other
scanners. Both are best cases: the model's own lab, scanner and protocol.

## Fully automatic: Auto LV on every frame, ED and ES found by the app

The same 25 test mice opened as raw cines, every frame contoured, ED/ES picked by Detect from the volume curve,
nothing taken from the expert (`scripts/validate_test25.py`, 8-orientation average):

| | Bias | SD | 95% limits | Worst |
| --- | --- | --- | --- | --- |
| EF (pts) | +0.3 | 3.2 | -6.0 to +6.6 | -6.8 |
| EDV (µL) | -2.6 | 5.9 | -14.2 to +9.1 | -19.2 |
| ESV (µL) | -1.0 | 2.2 | -5.3 to +3.3 | -6.5 |
| LV mass at ED (mg) | +1.6 | 13.5 | -24.8 to +27.9 | +41.6 |

ED was within one frame of the expert's in 88% of mice, ES in 64%: the bottom of the volume curve is flat, so a
frame or two either way changes ESV little. Detect counts a slice as contoured wherever Auto LV looked, even if it
found no cavity there (0.3.5): before, a frame where the apical slices had emptied at end-systole could be passed
over for an earlier one (UCL Mouse1, every frame contoured in the app: ES frame 6 instead of 9, EF 58.5% instead
of 71.3%; expert 69.6%).

## Installing by hand

Only needed without the installers. In the app, the first click on Auto LV (or Settings → Set up
auto-contouring…) downloads the model into `~/.tracecmr/models/ucl` and installs ONNX Runtime into TraceCMR's own
Python, about 40 MB. The manual equivalent:

```bash
pip install "tracecmr[auto]"     # or: pip install "onnxruntime>=1.20,<2"
mkdir -p ~/.tracecmr/models/ucl
curl -L -o ~/.tracecmr/models/ucl/ucl_seg459.onnx \
  https://github.com/jairam09/TraceCMR/releases/download/ucl-model-1/ucl_seg459.onnx
# check: sha256 7afe39b3870c5ae66cd94103b7b189cdc8153e86c88b1481cb074414bb4f908a

# Calico model (Git LFS)
git clone https://github.com/calico/mouse_cardiac_mri_segmentation ~/.tracecmr/models/calico
cd ~/.tracecmr/models/calico && git lfs pull       # ~67 MB
pip install torch torchvision torchio munch
```

`TRACECMR_CALICO_DIR` points elsewhere. The app lists a model as "(not installed)" until its weights are found.
Running both models in one session (the "double-check" option) needs about 8 GB of free memory.

## How they compare (preliminary, 2026-10-07)

On UCL's 25 held-out test mice, at the expert's ED/ES frames (one pass):

| | UCL model | Calico model |
| --- | --- | --- |
| Blood-pool Dice (ED / ES) | 0.91 / 0.87 | 0.87 / 0.85 |
| Myocardium Dice (ED / ES) | 0.84 / 0.87 | 0.78 / 0.82 |
| EF error, mean (worst) | -1.0 (5.7) pts | -5.0 (72) pts |

The UCL model is tested on its own lab's data here, so it has a home advantage. Calico failed on some severely
dilated hearts. A fair comparison needs data neither model has seen.

## Checks on model output

- Sanity flags on each slice: blood pool with no wall, wall in several pieces, LV area jumping by more than 2x
  versus both neighbours.
- Optional second model: slices where the two models' LV outlines overlap less than 80% are flagged.
- Flagged slices appear in orange in the slice list ("check: ...").

## Papillary convention

Both models label papillary muscles as myocardium (UCL certainly; Calico's training convention is not
documented). An experimental converter (`papillary.py`) moves them into the blood pool; it is off by default
until it has been tuned against hand-drawn contours (see `methods.md`).
