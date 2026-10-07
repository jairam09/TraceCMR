# mouse-cmr-pipeline

Open-source Python pipeline for analyzing **mouse cardiac MRI**: LV volumes and function, regional wall thickness and thickening, and LGE scar quantification.

> **Status: early development (v0.0.1).** Data loading, sanity checks and overlay figures work. Volumes, wall thickness and scar are in progress. Not yet validated for research use.

## What it will do

| Output | Status |
| --- | --- |
| Load NIfTI images and labels, check spacing and labels, draw overlays | Working |
| EDV, ESV, stroke volume, EF, LV mass | Planned (v0.1) |
| Regional end-diastolic wall thickness and systolic thickening, bullseye plots | Planned (v0.2) |
| LGE scar size, % of myocardium, transmurality (n-SD and FWHM) | Planned (v0.3) |
| Automatic segmentation (existing models, then nnU-Net) | Planned |
| Bruker ParaVision and Segment (.mat) input | Planned |

## Install

```bash
git clone https://github.com/<your-username>/mouse-cmr-pipeline.git
cd mouse-cmr-pipeline
pip install -e ".[dev]"
pytest            # all tests should pass
```

## Quick start

Check that everything works on a synthetic phantom heart:

```bash
python scripts/check_dataset.py --phantom --out data/checks
```

Check a real dataset (example: one ACDC subject, which uses its own label numbers):

```bash
python scripts/check_dataset.py \
  --image data/acdc/patient001/patient001_frame01.nii.gz \
  --label data/acdc/patient001/patient001_frame01_gt.nii.gz \
  --label-map acdc --species human --out data/checks
```

This prints shape, voxel spacing, label volumes and warnings, and saves an overlay PNG (red = LV endocardium, green = LV epicardium, magenta = RV endocardium). **Always open the overlay and look at every slice before trusting any number.**

In Python:

```python
from mcmr import load_nifti, sanity_check
ci = load_nifti("image.nii.gz", "label.nii.gz", label_map="acdc")
print(ci.spacing, ci.labels_present())
print(sanity_check(ci, species="human"))
```

## Conventions

All data is translated into one label scheme on load (`src/mcmr/labels.py`): 0 background, 1 LV blood pool, 2 LV myocardium, 3 RV blood pool, 4 scar, 5 MVO. Arrays are indexed `(x, y, slice)`; spacing is in mm, and volumes are in µL (1 mm³ = 1 µL). Analysis conventions are documented in [docs/methods.md](docs/methods.md).

## Data

No data is stored in this repository. See [docs/datasets.md](docs/datasets.md) for the public datasets used in development, their licenses and how to cite them.

## Contributing and support

Bug reports and questions are welcome as GitHub issues. See [CONTRIBUTING.md](CONTRIBUTING.md).

## AI assistance

Parts of this project were developed with AI assistance. See [docs/ai_usage.md](docs/ai_usage.md).

## License

MIT. See [LICENSE](LICENSE).
