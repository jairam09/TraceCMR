# Changelog

## [Unreleased]

### Changed
- Septum markers on the Wall overlay are anchored to the wall: the septum line runs from the endocardium to just past the epicardium, the RV insertion points are short ticks across the epicardium (they used to start at the centre and could stick out past the heart).
- Menu arrows on the Wall and Save buttons sit in their own space (no blue patch over the icon).
- Menu arrows on the Wall and Save buttons sit in their own space (no blue patch over the icon).
- The ED wall bullseye now uses the same blue (thin) → red (thick) colours as the Wall lines, so a colour means the same thickness in both.
- Bullseyes resize with the panel: square, side by side, titles on top and colour bars underneath, numbers sized to fit (hidden when there's no room), thinner segment borders. They no longer overlap or squash when the window is small.
- Toolbar: scrolls sideways when the window is narrower than it (buttons on the right, including Wall, were cut off on laptop screens); slightly more compact; the zoom button is now Center (it was a second "Fit").
- Wall thickness now uses the centerline method (the standard used by Segment and most papers): thickness is measured perpendicular to the wall at points midway between endo and epi. The old radial rays read up to 43% too thick where the wall is slanted (oval hearts); on 10 real UCL hearts the average end-diastolic wall is 8% thinner than before (0.91 vs 0.99 mm) and % thickening barely changes. Live updates stay fast (only the edited slice is recomputed).
- The editable points on contours are called interpolation points throughout (Segment's term); the button that shows them again after Smooth is now Points.
- Left-drag on the image never pans again (it clashed with editing dots, especially the epicardium). New Pan toggle button (hand icon, or Space): while on, left-drag moves the image and nothing is edited. Right-drag still pans anytime.
- Renamed VentriMap to **TraceCMR** (package `tracecmr`, command `tracecmr`, new logo wordmark). Everything made with VentriMap still works: settings are copied over on first start, old `VENTRIMAP_*` environment variables are still read, and saved `.ventrimap.npz` contours and the old "VentriMap contours" folder still load. New saves use `.tracecmr.npz` in "Documents/TraceCMR results".

### Added
- Set the septum by hand (Wall button arrow → Set the septum on this slice, then click the middle of the septum; Esc cancels), like Segment's RV insertion point. A slice set by hand overrides the automatic septum for its level, works with no RV traced at all, is saved with the mouse, and shows as a magenta line; Reset (this slice or all) goes back to automatic.
- Anatomical wall segments: where the RV meets the LV gives the septum and the two RV insertion points, and the 16 wall segments are numbered in the AHA order from there (segment 1 anterior, septum = 2-3, apical 2), the same reference at ED and ES. Levels without an RV borrow the nearest level's septum; with no RV traced at all it falls back to the image axis and says so. The bullseyes are drawn in the standard AHA view (anterior at the top, septum on the left), and the Wall overlay marks the septum (magenta line) and insertion points. Settings → Wall thickness → Mirror segment numbering, for scans seen from the base.
- Wall button arrow: choose Thickness or Thickening ED → ES. Thickening colours each line by how much that part of the wall thickens from ED to ES on this slice (green = healthy, 40%+; yellow = weak, about 20% or less; red = thins), shown on the ED or ES outline, whichever frame is being viewed. The thickening bullseye uses the same colours (-40% to 80%).
- Wall button (key W): draws the wall-thickness lines on the slice being viewed, from endo to epi through the centerline, coloured blue (thin) to red (thick), with the slice's mean, thinnest and thickest wall in the status line. Follows the slice and frame, and updates after edits.
- Fit button: redraws this slice's endocardium around the bright blood in the image. Unlike Snap (which only adds) it also gives dark muscle the outline took in back to the wall, so it fixes the rounded, misplaced outlines the model often draws at end-systole. It refuses when the slice has no clear bright blood. Undo reverts.
- Grow and Shrink buttons (keys ] and [): push the selected contour (Endo / Epi / RV) out or pull it in by half a pixel all the way round on the slice being viewed, keeping its shape and your dots. Made for an endocardium the model drew too small where the blood is less bright. The endo never crosses the epi; smoothed contours stay smoothed; Undo reverts each press.
- After Auto LV, the base and apex slices it contoured get an orange ⚠ "double-check" note in the slice list (base: make sure it's ventricle, not atrium or outflow; apex: check the outline or leave it out). The note clears as soon as you edit that slice.
- Segment-style saving: the first Save (or Ctrl+S) opens a save window that shows where the file goes and lets you rename it or pick another folder; after that Ctrl+S saves to the same file. Save as… (Save button menu, or Ctrl+Shift+S) picks a new name or place. A confirmation lists what was written (contours, results table, images) with the folder and an Open folder button (can be turned off). Reopening a mouse loads from wherever you saved it.
- Brightness and contrast (Segment style): Contrast button on the toolbar with sliders, Auto (stretch to the slice being viewed) and Reset; or hold Ctrl and drag the image (left/right = contrast, up/down = brightness).
- Results are live: volumes, EF, mass and the wall maps update by themselves after each edit once ED and ES both have contours, and appear straight away when you open a mouse with saved contours (no Compute press needed).
- Per-slice table: rows read "Slice N", clicking a row goes to that slice, and the row of the slice being viewed is highlighted.
- Auto-contour keeps the model's own contours next to your corrections in the saved file (with the images), and scripts/collect_corrections.py gathers them into a training set for fine-tuning.
- scripts/validate_test25.py: whole-pipeline validation on UCL's 25 held-out test mice (expert frames and fully automatic), one folder per mouse plus summary and Bland-Altman plots.
- Raw scans and results are kept apart: Settings → Folders sets a read-only raw scans folder and a results folder (refused if one is inside the other). Each mouse gets its own results sub-folder with its contours, results table and images. Contours saved by older versions still load.
- Export images (and Save): raw | contoured PNG of every contoured slice at ED and ES, plus a base-to-apex montage per phase.
- Raw button (key H) hides all contours to look at the raw image; nothing is deleted.
- Auto-contour (`autocontour.py`, app button): UCL or Calico model on every frame, sanity flags and optional two-model disagreement flags, ED/ES suggested from the volume curve, volume-over-time plot in the app. Models are user-installed (`docs/models.md`).
- App: play cine, Set ED/ES at the current frame; open a UCL file as a raw cine.
- App: zoom buttons (−, +, Fit heart) and keys (-, =, F) for trackpad users.
- App: "Model folder…" button to point at a downloaded model; the choice is remembered.
- App: Series column on the left (like Segment's thumbnails): open a whole DICOM study, click a series to work on it, combine one-cine-per-slice series into a short-axis stack.
- Per-slice ED/ES (Segment style) for stacks built from one cine per slice; a study folder auto-stacks such slices, listed under the stack. Time bar under the image with draggable ED/ES markers; slice/frame/time overlay in the image corner.
- Auto-contour does only the frame being viewed (every slice; at ED or ES, each slice's own ED or ES) in one model pass: about 5 s instead of 90 s. The model loads in the background when a scan opens. "Contour every frame" is still available. New scans start with ED = ES = frame 0.
- Deeper red/green contours, thicker lines, dots filled with their contour colour.
- Cleaner layout: icon toolbar on top (open, auto-contour, endo/epi, smooth, snap, dots, clear, undo, brush, zoom, settings), series and slices on the left, time bar with ED/ES at the bottom, results on the right; model and other options moved to Settings. Opening a single slice's folder opens the whole mouse.
- Right ventricle: yellow RV contour (draw, edit, auto-contour with the Calico model alongside the UCL LV model), RV volumes and RVEF in Results and the CSV. Epicardium is now light blue. Calico is loaded before TensorFlow (the reverse order crashed).
- UCL model ~8x faster (only its main output is computed; the unused deep-supervision heads were most of the cost): under 1 s per frame on a 2-core machine, identical contours. RV is off by default.
- Separate Auto LV and Auto RV buttons; the RV model always runs in its own process, so the app never loads PyTorch next to TensorFlow.
- Endo / Epi / RV buttons pick the contour being edited: only its dots show and can be grabbed; Smooth and Clear act on it alone.
- Per-slice results table (EDV, ESV, EF, mass per slice, Segment style) with the total over the selected slices; updates as you edit.
- Both models now run in a background worker process (kept running between clicks): the app never loads TensorFlow or PyTorch itself, so a model failure is reported instead of closing the app, and a new worker starts on the next click.
- Calico removed from the app (Auto RV, the RV option and the two-model double-check). The RV is drawn by hand with the yellow RV tool; Calico support stays in autocontour.py for the model-comparison scripts only.
- Slice/time scrollbars removed (slice list or Up/Down; time bar or Left/Right). Save (Ctrl+S) stores contours and all choices to a contours folder; they reload automatically when the mouse is reopened, and the previous mouse is saved when switching. "Mice" list on the left shows every mouse in a data folder (✓ = saved contours).
- App: Segment-style contour editing with no modes: drag a dot, click the line to add one, right-click a dot to delete, drag the background to pan, click points to draw a missing contour (right-click closes); trash can clears a slice; Ctrl+Z undoes every edit.
- `papillary.py`: experimental conversion of detailed contours to papillary-in-blood-pool (off by default).
- napari app (v0 prototype): `ventrimap` command or Plugins > VentriMap. Open a UCL file or DICOM folder, see contours over the images, set slice order by hand (flip, drag, leave slices out; warnings if the order looks reversed or contradicts scanner positions), edit contours with napari's brush, compute volumes/EF/mass and wall-thickness bullseyes, export CSV. Install with `pip install -e ".[app]"`.
- `contours.py`: mask <-> smooth curve conversion (adaptive control points, centripetal Catmull-Rom), exact curve areas; `scripts/curve_conversion_test.py`.
- App: slices chosen separately for ED and ES, with suggested choices and the reason for each (no contour, apical cap, basal slices flagged for checking when the drawn ring does not look like muscle).
- App: Snap to edge button (per slice): grows the endocardium into bright blood the model rounded off; tested on UCL test mice (helps EDV/mass, over-grows at ES, so manual, not automatic).
- App: Smooth button, Segment-style: smooths both contours on the slice being viewed (gentle, area-preserving) and hides that slice's points; Edit points brings them back; Undo.
- App: Segment-style curve editing (drag control points; add/delete points; pixel brush as fallback).
- `compute_wall(..., base_end=)` to set orientation by hand.
- `scripts/run_cohort.py`: one-row-per-mouse summary for a whole dataset.
- `scripts/predict_groups.py`: pre-registered blind group-prediction rules for UCL validation.

### Fixed
- Contours no longer creep each time a mouse is reopened: the exact interpolation points (and which contours are smoothed) are saved with the contours and restored on reopen. Before, the points were rebuilt from the saved pixels with a little smoothing, so sharp tips (especially the RV's) rounded off slightly with every open-edit-save. Slices changed some other way since saving (e.g. Auto LV) are rebuilt from their pixels as before.
- Undo now also puts the labels back: after an edit had reached the results (they update live), Undo restored the outline on screen but volumes, EF and the wall lines kept the undone edit (e.g. Fit, then Undo).
- Auto LV could return nothing on a faint apex slice: the outer-edge cutoff (P >= 0.9) erased slices where the model was only moderately sure. The cutoff now only tidies edges and never removes more than half a slice. When the model genuinely finds no LV on a slice, the status line now says so and suggests drawing it by hand.
- Contrast Reset button did nothing.
- Tiny contours (apex at end-systole) no longer get 18 dots piled into a clump: the number of dots scales with the contour's length (4 minimum, about 4 px apart). Auto LV also flags a cavity of under 25 pixels as "tiny cavity: check" (orange) so you redraw or clear it.
- Drawing a new contour: a misplaced point can be dragged before right-clicking to close.
- Left-drag on empty image pans again (it was removed to stop near-miss pans). It now pans only when the drag starts well away from every dot and line; a drag that starts near a dot grabs the nearest dot.
- Adjusting dots could slide the image around: a left-drag that just missed a dot panned the view (common on a trackpad). Left-drag now never pans (pan with right- or middle-drag); dots are easier to grab (larger click radius, and a drag starting near a dot grabs the nearest one).
- Auto LV overwrote slices you had already contoured (and erased a hand-drawn RV). It now only contours slices with no LV contour at that frame; when every slice has one, it asks whether to redo just the slice being viewed. A hand-drawn RV is always kept, and Ctrl+Z undoes a whole Auto LV run.
- EF could read far too low with no warning when per-slice ES was set on only some slices (the rest kept ES = ED). Compute now warns in orange under the button, the per-slice table shows "ES not set" on those slices, and the note lists them.
- Left panel no longer squashes on short windows: mice and series lists size to their contents, the slice list gets the room (no sideways scrolling; notes shown as a tooltip and an orange ⚠ name), and the whole panel scrolls if needed. Combine button only appears when there are single-slice series to combine.
- LV mass read high (~+6-8 mg): the model drew the epicardium about half a pixel too far out. Stricter outer-edge cutoff (P(LV) >= 0.9), chosen on the 15 validation mice (+8.1 -> +0.6 mg), checked once on the 25 test mice (+6.3 -> -0.1 mg); EDV, ESV and EF unchanged. A lower blood cutoff for the endocardial corners was tested and did not help (the model is too confident there); kept off. Results-changing (mass).
- Smoothed contours no longer get their dots back when ED/ES is set (or detected) on another slice; only slices whose contours actually changed are rebuilt.
- UCL ED/ES selection could pick a partially contoured artifact frame as ES (3 of 130 mice; Mouse91 EF 87% -> 67%). Results-changing for those mice.

## [0.3.0] - 2026-10-07

### Changed
- Project renamed to **VentriMap**. Python package renamed `mcmr` -> `ventrimap` (imports change: `from ventrimap import ...`). Real-data test variable renamed `MCMR_UCL_DATA` -> `VENTRIMAP_UCL_DATA`. Entries below use the new names.
- Bullseye plots now run clockwise from 3 o'clock, matching how segments appear in the image (previously mirrored).

### Added
- VentriMap logo (`docs/assets/`): slice-stack V with the endocardial contour, light and dark, static and animated, plus an app icon.

## [0.2.0] - 2026-10-07

### Added
- `wall.py`: regional wall thickness (radial rays, sub-pixel borders), AHA-layout levels and segments, systolic thickening per segment, flags for negative thickening and papillary convention.
- `viz.save_bullseye`: bullseye plots of segment values.
- `scripts/run_wall.py`: batch wall thickness for UCL mice, CSV + bullseye plots.
- `dicom.py`: cine loader for classic and enhanced multi-frame DICOM, spacing from headers; `check_dataset.py --dicom`.
- Tests for wall thickness (phantoms, papillary bump, thickening, conventions), bullseye output and DICOM sorting/spacing.

### Fixed
- DICOM trigger time of 0 ms no longer treated as missing (caught by tests before release).

## [0.1.1] - 2026-10-06

### Added
- `orientation.py`: base/apex detection (refuses rather than guesses), per-heart detection from ED with ES cross-check, reordering to base-first, and basal/mid/apical/apical-cap level assignment.
- `synthetic.tapered_phantom`: bullet-shaped LV with optional apical cap, stored either way round.
- Visual ground truth for 16 UCL mice (`tests/data/ucl_orientation_truth.csv`) and tests against it.

## [0.1.0] - 2026-10-06

### Added
- `volumes.py`: EDV, ESV, SV, EF, LV mass (ED and ES), per-slice blood pool areas, QC flags.
- Papillary muscle convention tracked on every `CardiacImage` (`papillary`); mixing or comparing across conventions is refused unless explicitly allowed.
- `scripts/run_volumes.py`: batch volumes for a folder of UCL mice, written to CSV with a summary.
- Tests for volume geometry, spacing scaling, scar handling, QC flags and convention checks.

### Checked
- 16 UCL mice: results identical to an independent raw-file recomputation; EF 29.6-81.3%, no QC flags.

## [0.0.2] - 2026-10-06

### Added
- Loader for the UCL open mouse cardiac MRI dataset (`ventrimap.datasets.load_ucl`), with ED/ES detection and the dataset's label scheme.
- `check_dataset.py --ucl` option.
- Tests for the UCL loader on synthetic UCL-format files, plus an optional real-data test (`VENTRIMAP_UCL_DATA`).
- `h5py` dependency.

## [0.0.1] - 2026-10-06

### Added
- `CardiacImage` data structure; NIfTI loading (3D and 4D cine with frame selection) and saving.
- Standard internal label scheme and ACDC label mapping.
- Sanity checks for spacing, non-square pixels, missing labels and non-contiguous LV slices.
- Synthetic ring phantom with exact reference volumes.
- Overlay figures (LV endo red, LV epi green, RV endo magenta).
- `scripts/check_dataset.py` command-line check for any dataset.
- Tests and GitHub Actions CI.
