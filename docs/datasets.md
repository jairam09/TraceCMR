# Datasets

No data is committed to this repository. Download each dataset yourself into `data/<name>/` (the `data/` folder is ignored by git). Record the license and citation here the day you download it.

| Dataset | Species / sequence | Labels | Access | License | Loader |
| --- | --- | --- | --- | --- | --- |
| UCL open mouse cardiac MRI | Mouse, 9.4 T cine SAX, 130 mice (39 control, 11 MI + biomaterial patch, 51 DCM, 29 TAC) | LV blood pool + LV myocardium at ED and ES (no RV, no LGE) | https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation | Paper states CC BY 4.0; repository is labeled GPL-3.0 (see note) | `tracecmr.datasets.load_ucl` |
| ACDC | Human, short-axis cine | RV, LV myocardium, LV cavity at ED and ES | Register at https://acdc.creatis.insa-lyon.fr/ | Check terms on site before redistributing | `load_nifti(..., label_map="acdc")` |
| EMIDEC | Human, short-axis LGE | Myocardium, infarct, MVO | http://emidec.com/ | CC-BY-NC-SA | To add |
| CMR-MULTI | Human cine + LGE, incl. scar | Chambers, myocardium, scar | https://huggingface.co/datasets/cxhQAQ/CMR-MULTI | Non-commercial research only | Optional |
| Calico mouse model | Mouse cine (pretrained weights only, no training data) | n/a | https://github.com/calico/mouse_cardiac_mri_segmentation | MIT | Model to benchmark (Phase 5) |

## UCL dataset

**Downloading.** The full repository is about 5 GB (130 files of roughly 30-60 MB). To fetch only some mice, use a partial clone:

```bash
cd data
git clone --filter=blob:none --no-checkout https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation.git ucl
cd ucl
git checkout HEAD -- README.md LICENSE Paper/PreClinical_Segmentation_Paper.pdf
git checkout HEAD -- Data/Mouse1.h5 Data/Mouse2.h5 Data/Mouse3.h5   # add more as needed
```

For all 130 mice: `git checkout HEAD -- Data`. Their model weights (`ModelWeights/SEG459.h5`, ~156 MB) are stored with Git LFS: install Git LFS and run `git lfs pull` to get them.

**Format** (from the dataset README, confirmed on Mouse1.h5): `Images` and `Masks` are `(256, 256, slices, frames)`; masks use 0 background, 1 LV myocardium, 2 LV blood pool; `Frames_with_ROI` lists 0-based labeled frames; unlabeled frames are all zeros. A few mice have extra labeled frames with inflow artifacts. `find_ed_es()` picks ED and ES as the labeled frames with the largest and smallest blood pool.

**Spacing.** The files store no voxel spacing. Per the paper: interpolated pixel size 0.1 x 0.1 mm, slice thickness 1.0 mm, contiguous slices. The loader uses `(0.1, 0.1, 1.0)`. Check on Mouse1: EDV 74.7 uL, ESV 22.7 uL, EF 70%, LV mass ~114 mg, all physiologically plausible.

**Conventions.** Trabeculae and papillary muscles are included in the myocardium. In Mouse1, slice 0 is the apical cap (myocardium only) and the top slices are unlabeled.

**License note.** The JCMR article itself is CC BY 4.0, but it names no separate licence for the data, code or weights; they live in the authors' GitHub repository, which carries GPL-3.0. Either way they are shared openly for reuse with credit. Practical rule: cite the paper wherever the data or model is used; redistribute data files (e.g. the test mice for testers) with credit and the repository's LICENSE; do **not** copy their code into this MIT-licensed project. The app downloads their model from their repository on the user's request (`model_setup.py`), so TraceCMR never ships it. Corresponding author: Jennifer Steeden (UCL).

## Label conventions

| Dataset | Native labels | `label_map` |
| --- | --- | --- |
| UCL | 0 background, 1 LV myocardium, 2 LV blood pool | `"ucl"` |
| ACDC | 0 background, 1 RV cavity, 2 myocardium, 3 LV cavity | `"acdc"` |
| EMIDEC | To be filled in when downloaded | To add |

## Citations

- UCL: Shah W, Stuckey DJ, Yao T, Wrobel M, Deniszczyc E, Feng Z, Anderson S, Campbell R, Muthurangu V, Steeden J. Open-source pre-clinical image segmentation: mouse cardiac magnetic resonance imaging datasets with a deep learning segmentation framework. J Cardiovasc Magn Reson. 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13237537/
- ACDC: Bernard O, et al. Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved? IEEE Trans Med Imaging. 2018;37(11):2514-2525. doi:10.1109/TMI.2018.2837502
- EMIDEC: Lalande A, et al. Emidec: a database usable for the automatic evaluation of myocardial infarction from delayed-enhancement cardiac MRI. Data. 2020;5(4):89. doi:10.3390/data5040089
- Calico: Yoganathan, et al. Automated cardiac MRI analysis for robust profiling of heart failure models in mice. Sci Rep. 2025. https://www.nature.com/articles/s41598-025-30810-y
