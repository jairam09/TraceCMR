<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/tracecmr-logo-dark.svg">
    <img src="docs/assets/tracecmr-logo.svg" alt="TraceCMR" width="320">
  </picture>
</p>

# TraceCMR

Open-source analysis for **mouse cardiac MRI**: LV volumes and function, regional wall thickness and thickening maps, and (planned) tracking the same heart across scans and LGE scar quantification.

> **Status: early development (v0.3.0).** Data loading (UCL, NIfTI, DICOM), sanity checks, overlays, global LV function (volumes, EF, mass) and regional wall thickness work. Scar and longitudinal tracking are in progress. Not yet validated for research use.

## What it will do

| Output | Status |
| --- | --- |
| Load NIfTI images and labels, check spacing and labels, draw overlays | Working |
| Load the UCL open mouse cardiac MRI dataset (130 mice), find ED/ES | Working |
| EDV, ESV, stroke volume, EF, LV mass, with QC flags | Working (v0.1) |
| Detect base vs apex, reorder slices, assign basal/mid/apical levels | Working (v0.1.1) |
| Regional end-diastolic wall thickness and systolic thickening, bullseye plots | Working (v0.2, radial-ray method) |
| Read cine series from DICOM (classic and enhanced multi-frame) | Working (v0.2, tested on synthetic data) |
| Desktop app (napari): cine playback, ED/ES marking, slice order and per-phase slice choice, drag-point contour editing, compute | Working (prototype) |
| Automatic contouring with published models (UCL, Calico), review flags, ED/ES from the volume curve | Working (prototype; models installed separately, see docs/models.md) |
| Track the same mouse across scans (per-segment change maps) | Planned |
| LGE scar size, % of myocardium, transmurality (n-SD and FWHM) | Planned (v0.3) |
| Bruker ParaVision and Segment (.mat) input | Planned |

## Install

```bash
git clone https://github.com/jairam09/tracecmr.git
cd tracecmr
pip install -e ".[dev]"
pytest            # all tests should pass
```

To use the app: `pip install -e ".[app]"`, then run `tracecmr` (or open it from napari's Plugins menu). Auto-contour models are optional downloads: see [docs/models.md](docs/models.md).

## Quick start

Check that everything works on a synthetic phantom heart:

```bash
python scripts/check_dataset.py --phantom --out data/checks
```

Check a real mouse from the [UCL open dataset](docs/datasets.md#ucl-dataset):

```bash
python scripts/check_dataset.py --ucl data/ucl/Data/Mouse1.h5 --frame ED --out data/checks
```

Compute EDV, ESV, SV, EF and LV mass for every downloaded UCL mouse:

```bash
python scripts/run_volumes.py --ucl-dir data/ucl/Data --out results/ucl_volumes.csv
```

Results record the papillary muscle convention of the labels (UCL: papillary muscles counted as myocardium). The pipeline refuses to compare results across conventions; see [docs/methods.md](docs/methods.md).

Regional wall thickness and thickening, with bullseye plots for each mouse:

```bash
python scripts/run_wall.py --ucl-dir data/ucl/Data --out results/wall
```

Check a DICOM cine series (a folder of .dcm files):

```bash
python scripts/check_dataset.py --dicom path/to/series --out data/checks
```

Check a human dataset (example: one ACDC subject, which uses its own label numbers):

```bash
python scripts/check_dataset.py \
  --image data/acdc/patient001/patient001_frame01.nii.gz \
  --label data/acdc/patient001/patient001_frame01_gt.nii.gz \
  --label-map acdc --species human --out data/checks
```

This prints shape, voxel spacing, label volumes and warnings, and saves an overlay PNG (red = LV endocardium, green = LV epicardium, magenta = RV endocardium). **Always open the overlay and look at every slice before trusting any number.**

In Python:

```python
from tracecmr import load_nifti, sanity_check
ci = load_nifti("image.nii.gz", "label.nii.gz", label_map="acdc")
print(ci.spacing, ci.labels_present())
print(sanity_check(ci, species="human"))
```

## Conventions

All data is translated into one label scheme on load (`src/tracecmr/labels.py`): 0 background, 1 LV blood pool, 2 LV myocardium, 3 RV blood pool, 4 scar, 5 MVO. Arrays are indexed `(x, y, slice)`; spacing is in mm, and volumes are in µL (1 mm³ = 1 µL). Analysis conventions are documented in [docs/methods.md](docs/methods.md).

## Data

No data is stored in this repository. See [docs/datasets.md](docs/datasets.md) for the public datasets used in development, their licenses and how to cite them.

## Contributing and support

Bug reports and questions are welcome as GitHub issues. See [CONTRIBUTING.md](CONTRIBUTING.md).

## AI assistance

Parts of this project were developed with AI assistance. See [docs/ai_usage.md](docs/ai_usage.md).

## License

MIT. See [LICENSE](LICENSE).
