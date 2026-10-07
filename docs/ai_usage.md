# AI usage disclosure

JOSS (and increasingly other journals) require a statement of which AI tools were used, for what, and confirmation that the human author reviewed and validated all AI-assisted output and made the core design decisions. Keep this log current; add a line whenever AI meaningfully helps.

## Tools

- Claude (Anthropic), Claude Opus 5.5, via claude.ai.

## Log

| Date | What | Nature of assistance | Human review |
| --- | --- | --- | --- |
| 2026-10-06 | Project plan and build guide | Drafting the phased plan; researching datasets, nnU-Net and JOSS requirements | Plan reviewed and adopted by author |
| 2026-10-06 | Initial repo scaffold (`io.py`, `labels.py`, `synthetic.py`, `viz.py`, `check_dataset.py`, tests, docs) | Code generation and test scaffolding | To do: read every file, run tests, confirm overlays |
| 2026-10-06 | UCL dataset loader (`datasets/ucl.py`), tests, dataset docs | Code generation; inspected a real UCL file to confirm format and ED/ES frames; spacing taken from the UCL paper | To do: read the loader, rerun the Mouse1 check yourself |
| 2026-10-06 | `volumes.py`, papillary convention tracking, `run_volumes.py`, tests | Code generation; ran on 16 UCL mice with an independent recomputation; tested papillary conversion and documented why it was rejected | Author raised the papillary convention issue and chose to track rather than convert; to do: read `volumes.py` |
| 2026-10-06 | `orientation.py`, tapered phantom, orientation tests | Code generation; drew anatomy montages of 16 mice and proposed visual base/apex labels; scored candidate rules against them | To do: re-check the 16 visual labels in `tests/data/ucl_orientation_truth.csv` yourself (they are the ground truth) and read `orientation.py` |
| 2026-10-07 | `wall.py`, bullseye plots, `run_wall.py`, `dicom.py`, tests | Code generation; ran wall thickness on 16 UCL mice with ray overlays and bullseyes; investigated negative basal thickening; tests caught a DICOM trigger-time bug | To do: read `wall.py` and `dicom.py`; check the Mouse80 ray overlay and bullseyes |

## Design decisions made by the author

(Record here the decisions you make yourself: conventions, methods chosen, what to validate against, scope.)

- 2026-10-06: Identified that UCL labels papillary muscles as myocardium while my Segment SOP includes them in the blood pool; decided conventions must be tracked and never silently mixed.
- 2026-10-07: Chose a single app built as a napari plugin (auto-contour + manual correction + analysis) rather than a standalone app; chose tracking the same mouse across scans as the flagship feature; decided to check Segment Pre-clinical's mouse AI before building auto-contouring.
