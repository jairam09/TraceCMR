# Auto-contour models

VentriMap does not bundle any model. Each is downloaded from its authors and keeps its own license.

| Model | What it outlines | Speed (CPU, 9-slice scan) | License | Source |
| --- | --- | --- | --- | --- |
| UCL (UNet3+, 2D) | LV blood pool, LV myocardium | ~10 s per frame | GPL-3.0 | Shah et al., JCMR 2026; github.com/mrphys/Open-Source_Pre-Clinical_Segmentation |
| Calico (3D UNet) | LV blood pool, LV myocardium, RV blood pool | ~15-30 s per frame | MIT | Calico Life Sciences, Sci Rep 2025; github.com/calico/mouse_cardiac_mri_segmentation |

Times are from a 2-core cloud machine with no GPU; a recent laptop is similar or faster.

## Installing

Both repositories keep their weights in Git LFS, so a plain download gives only small pointer files.

```bash
# UCL model
git clone https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation ~/.ventrimap/models/ucl
cd ~/.ventrimap/models/ucl && git lfs pull          # needs Git LFS; ~156 MB
pip install tensorflow

# Calico model
git clone https://github.com/calico/mouse_cardiac_mri_segmentation ~/.ventrimap/models/calico
cd ~/.ventrimap/models/calico && git lfs pull       # ~67 MB
pip install torch torchvision torchio munch
```

Other locations work if you set `VENTRIMAP_UCL_DIR` / `VENTRIMAP_CALICO_DIR`. The app lists a model as "(not installed)" until its weights are found.

Running both models in one session (the "double-check" option) needs about 8 GB of free memory.

## How they compare (preliminary, 2026-10-07)

On UCL's 25 held-out test mice, at the expert's ED/ES frames:

| | UCL model | Calico model |
| --- | --- | --- |
| Blood-pool Dice (ED / ES) | 0.91 / 0.87 | 0.87 / 0.85 |
| Myocardium Dice (ED / ES) | 0.84 / 0.87 | 0.78 / 0.82 |
| EF error, mean (worst) | -1.0 (5.7) pts | -5.0 (72) pts |

The UCL model is tested on its own lab's data here, so it has a home advantage. Calico failed on some severely dilated hearts. A fair comparison needs data neither model has seen.

End-to-end check (UCL Mouse40 opened as a raw cine, all 21 frames contoured by the UCL model): ED and ES were found at frames 0 and 9, the same frames the expert chose; EF 79.1% vs 79.1% expert; EDV 55.7 vs 56.5 uL. Mouse40 is not one of UCL's test mice, so the model may have seen it in training.

## Checks on model output

- Sanity flags on each slice: blood pool with no wall, wall in several pieces, LV area jumping by more than 2x versus both neighbours.
- Optional second model: slices where the two models' LV outlines overlap less than 80% are flagged.
- Flagged slices appear in orange in the slice list ("check: ...").

## Papillary convention

Both models label papillary muscles as myocardium (UCL certainly; Calico's training convention is not documented). An experimental converter (`papillary.py`) moves them into the blood pool; it is off by default until it has been tuned against hand-drawn contours (see `methods.md`).
