# Changelog

## [Unreleased]

### Added
- ED/ES timing across slices (`tracecmr.drift`): each slice's own end-diastole and end-systole against the global pair, EDV/ESV/EF computed both ways on the same contours, and whether the per-slice ES trends with acquisition order (drifting heart rate) or just scatters (contour noise). Ties at the flat top and bottom of a slice's curve go to the global frame, so a slice is only called shifted when its data says so. With Per-slice ED/ES on, Detect reports the spread and the EF difference; Export all writes `<mouse>_ED_ES_timing.csv` when there are enough contoured frames. Reads timing parameters from Bruker ParaVision method/acqp/visu_pars files (field names not yet checked against real cine data).
- Installers for Windows (.exe) and macOS (.pkg, Apple silicon), built with conda constructor: each carries its own Python and installs TraceCMR offline from bundled wheels. Windows gets Desktop and Start-menu shortcuts and an uninstaller; macOS gets TraceCMR.app in Applications. No model weights included (each keeps its own licence); TensorFlow for auto-contour is an optional one-click step. A GitHub Actions workflow (`installers`) builds both on real Windows and Mac machines, installs each, checks the app starts, and attaches the installers to the run.

### Fixed
- Windows installer: the Desktop and Start-menu shortcuts were never created. The install step runs with a bare PATH where `powershell` is not found; it is now called by its full path (also in the uninstaller). The installer build now fails if either shortcut is missing. The shortcut opens the app normally (it was set to start minimised, a leftover from the old .bat launcher).
- "No model installed. See the README" (testers have no README) now reads "Auto-contouring is not set up on this computer yet. Contour by hand for now."
- The export summary shows the images folder with the right separator on macOS.

### Changed
- Installers: the last install step is about 4x faster. pip no longer pre-compiles every package (about 12,000 files, most never used, each one scanned by antivirus); a short warm-up import compiles just the modules TraceCMR loads, so the first launch is not slowed. On macOS the TraceCMR folder is handed to the logged-in user so Python can cache compiled modules. Version 0.3.1; 0.3.2 adds the fixes above.
- Switching to the Wall tab is instant. The bullseyes were rebuilt from scratch every time the tab was picked (about 0.2 s); they are now rebuilt only when the numbers or the panel size change. The wall-lines layer is also created while the mouse is opening instead of on the first click on Wall.
- ED | ES bar: the mouse and slice name sits in the view's corner (it was cut off on a narrow window), the controls are shorter, and the crop menu's list no longer draws a stray outline around the chosen item.
- ED | ES: Lock crop. Set the window you want on one slice (crop, zoom and pan, e.g. on the slice with the widest ventricle), press Lock crop, and every slice is shown in exactly that window, in the view and in the saved figures, until you unlock it.
- ED | ES crop has three settings: this slice (tight around the heart on this slice), same for all slices (one crop around the heart on every slice, set by the widest one, so every slice is shown at the same scale and place, in the view and in the All slices figure) and whole image.
- The ED | ES view zooms and pans: mouse wheel or + / − zoom both images together (about the cursor), drag moves them, and double-click, F or Ctrl+0 fit them again. The scale bar follows the zoom.
- Fixed: Restore after a crash brought the labels back but the contours on screen stayed as before (and overwrote the restored labels at the next edit). The recovery file now carries the points as they were when it was written. The restore prompt is dark from its first frame and appears once the window has settled, instead of flashing.
- Grabbing a dot: no pause between pressing and the dot following the mouse (a full redraw ran on every press), and zoomed in closely a click anywhere on the drawn dot now takes it (the grab radius was 12 screen pixels even when the dot itself was drawn bigger than that, so clicks on its rim missed).
- Dragging a contour dot is much smoother, especially zoomed in: while dragging, only that dot and its contour are redrawn (not every dot and line on the slice), contours are drawn as closed lines with no fill to rebuild, and the graphics card no longer recompiles the contour shader on every mouse move (a vispy quirk, worked around in app/_vispy_fixes.py).
- The ED | ES panel is drawn directly (no chart library in the live view), so it redraws in a few milliseconds: popping it out, resizing it, scrubbing slices and dragging contrast all stay smooth, and it never stretches, just grows, shrinks or re-stacks between side by side and ED above ES. Saved figures are unchanged. Other charts show an evenly scaled placeholder while resizing instead of a stretched one.
- ED | ES is a view over the image instead of a popup or panel: Ctrl+E swaps the image for the pair, filling the whole image area, with one slim bar for the switches, the figure buttons and ×. Ctrl+E, Esc or × bring the image back. Nothing new to dock, pop out or resize, so the interface stays as it was.
- The ED | ES window opens already drawn, with no white flash; panels being resized show their dark background instead of white.
- ROI colours: lime instead of white (a white outline disappears on bright blood).
- Every status line, hover card, tooltip, tour step, warning and flag rewritten to be short and plain (about half the length on average); keys shown as (S), (Ctrl+S) etc. The export error box no longer asks you to forward the error.
- The Compute button is gone: results already recompute by themselves a moment after every edit. While ED or ES contours are still missing, the Wall notes say what is missing instead.
- The Results tab bar has a thin outline, and the selected tab a raised, outlined look, so the three tabs read as one control.
- Results panel organised into tabs, one per analysis: Function (volumes, EF, mass, volume curve, per-slice table), Wall (bullseyes, wall lines on the image, thickness or thickening, septum) and Regions (ROIs, CNR/SNR). The scan name, status and Compute stay above the tabs and the export buttons below them; each tab scrolls on its own. Ctrl+1/2/3 switch tabs, the Wall tool (W) opens the Wall tab and the ROI tool (R) the Regions tab, and picking Wall turns the wall lines on. The toolbar stays the same in every tab.
- Drawing a region (or turning ROI on) opens the Results panel if it is hidden and scrolls down to the Regions of interest box, so the numbers are in view.
- Keyboard shortcuts work wherever the focus is: S, P, Delete and the zoom keys go through napari's key handling (fed every key no other widget used), so they work after clicking a button or a list, not only with the image focused. The slice, mouse and series lists no longer take the keyboard when clicked, so letter keys stay shortcuts instead of jumping to a row starting with that letter.
- ROI button has its own icon (a rectangle with handles), no longer like Points.
- Hide contours (H) is instant both ways: bringing the contours back no longer rebuilds them from the masks (about 0.8 s on a 20-frame scan, now about 0.04 s).
- Points and Smooth reorganised. Points is now an on/off switch (key P): on, the selected contour always shows its interpolation points; off, a clean view of the contours with no dots (left-drag pans). Picking Endo, Epi or RV (or Q) turns points back on. Smooth (key S) smooths the selected contour and keeps its points, so it can be adjusted straight away; contours no longer get stuck without points after smoothing and switching contours.
- Hover hints are small dark cards with a title, one line on what the button does and its shortcut as keycaps, instead of plain tooltips; moving along the toolbar shows the next card straight away.
- Brightness and contrast is a floating card over the image (drag it by its title, Esc or × closes) with value readouts and the Ctrl + drag tip as a keycap.
- Toolbar menus redrawn in TraceCMR's style: rounded dark cards with an icon per entry, a check mark for on/off entries, and each shortcut shown as a keycap (e.g. [Ctrl] [S]). Menus and tooltips now appear instantly (Windows' fade-in animation is off).
- Universal keys: + (or =) zooms in and − zooms out, with or without Ctrl and on the keypad; Ctrl+0 centers on the heart; Delete or Backspace clears the contour on the slice (Undo brings it back).
- The app exits cleanly the moment its window closes (no chance of a crash report while Python tears down).
- Toolbar reorganised into groups (file · contour · edit · view · wall · panels), one button per job: Open scan has a menu for UCL files, Snap/Fit/Grow/Shrink are under Refine, and Hide contours, Brightness and contrast and Zoom are under View. Their keys work as before. Thin dividers separate the groups.
- Panels: the scan list and results panels have a slim header with their name and a × to hide them; View → Panels has a tick for each panel (Scan list, Toolbar, Results, Time bar) to hide or bring it back, plus Show all panels (Ctrl+Shift+R). Hiding a panel says in the status bar where to get it back. Panels no longer tear off into floating windows, and napari's close button (which destroyed the panel) is gone.
- Start screen: the logo's red trace draws itself in once when the app opens (about 1.6 s, gentle start and finish, after a short pause).
- Smooth panel resizing: charts (volume curve, bullseyes) show a scaled snapshot while a panel edge is dragged and render once the drag pauses, instead of rebuilding on every pixel (a drag step went from ~220 ms to ~60 ms in testing; most of the rest is screen drawing, faster on a real graphics card).
- Per-slice table: short one-line headers (EDV, ESV, EF, Mass) with the units in the box title, so all five columns fit even a narrow side panel.
- Splash progress line moves continuously while the app loads (driven by the program loading its parts) instead of jumping in steps.
- Wording: "Open mouse" is now "Open scan" (toolbar button and hints), and the start screen says "Open a scan to begin".
- Typography: the whole interface (TraceCMR panel, napari menus, start screen) and the charts and exported images now use Inter, bundled with the app, so TraceCMR looks the same on every computer instead of mixing Segoe UI with the logo's font. Inter is the interface font of Linear, Figma, Notion and GitLab. The TraceCMR wordmark is redrawn in Inter Display Medium: lighter so it balances the thin slice bars, slightly tighter, and closer to the mark (scripts/make_wordmark.py regenerates it). The start screen hint is shorter and sits within the logo's width.
- Septum markers on the Wall overlay are anchored to the wall: the septum line runs from the endocardium to just past the epicardium, the RV insertion points are short ticks across the epicardium (they used to start at the centre and could stick out past the heart).
- Menu arrows on the Wall and Save buttons sit in their own space (no blue patch over the icon).
- The ED wall bullseye now uses the same blue (thin) → red (thick) colours as the Wall lines, so a colour means the same thickness in both.
- Bullseyes resize with the panel: square, side by side, titles on top and colour bars underneath, numbers sized to fit (hidden when there's no room), thinner segment borders. They no longer overlap or squash when the window is small.
- Toolbar: scrolls sideways when the window is narrower than it (buttons on the right, including Wall, were cut off on laptop screens); slightly more compact; the zoom button is now Center (it was a second "Fit").
- Wall thickness now uses the centerline method (the standard used by Segment and most papers): thickness is measured perpendicular to the wall at points midway between endo and epi. The old radial rays read up to 43% too thick where the wall is slanted (oval hearts); on 10 real UCL hearts the average end-diastolic wall is 8% thinner than before (0.91 vs 0.99 mm) and % thickening barely changes. Live updates stay fast (only the edited slice is recomputed).
- The editable points on contours are called interpolation points throughout (Segment's term); the button that shows them again after Smooth is now Points.
- Left-drag on the image never pans again (it clashed with editing dots, especially the epicardium). New Pan toggle button (hand icon, or Space): while on, left-drag moves the image and nothing is edited. Right-drag still pans anytime.
- Renamed VentriMap to **TraceCMR** (package `tracecmr`, command `tracecmr`, new logo wordmark). Everything made with VentriMap still works: settings are copied over on first start, old `VENTRIMAP_*` environment variables are still read, and saved `.ventrimap.npz` contours and the old "VentriMap contours" folder still load. New saves use `.tracecmr.npz` in "Documents/TraceCMR results".

### Added
- ED | ES side by side (View ▾, or Ctrl+E): a resizable window with the slice at end-diastole and end-systole next to each other, contours drawn, at the same crop and scale. It follows the slice (↑/↓ work in it too), every edit and the brightness/contrast. Switches for contours, scale bar and crop to the heart. Save figure… writes that slice as PNG (300 dpi), SVG or PDF; All slices… writes every slice in the analysis, base to apex, ED | ES per row.
- Redo (toolbar button, Ctrl+Y or Ctrl+Shift+Z) for contour edits; a new edit ends the redo history.
- Auto-save and crash recovery: every minute, unsaved changes are written to a recovery file (results folder, .recovery). A normal save removes it; if TraceCMR closed before saving, reopening that mouse offers to restore the changes.
- Ruler: ROI ▾ → Ruler draws a line and shows its length in mm on the image and in the Regions table (with the mean signal along it); rulers are saved with the mouse and kept out of CNR comparisons.
- Help → Report a problem: saves a report (versions, system, model status, the end of the log), copies it to the clipboard, opens its folder, and opens an email to the support address once one is set.
- ROI tool (toolbar ROI button, key R): draw a circle (drag from the centre) or a freehand outline on the image and get its area, mean signal and SD (hover for slice, frame, pixel count and range). The Regions of interest box in Results lists every region (double-click to rename, e.g. scar, remote, air; click to jump to its slice) and compares any two: contrast-to-noise (CNR = |mean A − mean B| ÷ noise SD) and signal-to-noise, with the noise taken from a region you pick (ideally air). Regions are saved with the mouse's contours, exported as a CSV with Export all (or Export CSV), and drawn in their own colours with their names. Esc or R leaves ROI mode; Delete removes the selected region.
- First-run tour: the first time TraceCMR opens, a short spotlight tour (9 stops, about a minute) dims the window, lights up one part at a time and explains it. Next/Back/Skip or the arrow keys, Enter and Esc. Shown once; replay it from Help → Take the tour. TRACECMR_NO_TOUR=1 turns it off.
- Keyboard shortcuts card: press ? (or F1, or Help → Keyboard shortcuts) to see every shortcut on one card.
- New keys: Q switches the contour being edited (Endo → Epi → RV), K plays or pauses the cine, G switches between a smooth and a pixel image.
- Smooth image (View → Smooth image, or G): draws the scan smoothly (bicubic) like Segment, or as the scanner's real pixels. Remembered between runs; contours are unaffected.
- Toolbar buttons Scans, Results and Focus (top right): Scans and Results show or hide the side panels and light up while their panel shows; Focus (Ctrl+Shift+F) hides both side panels and the time bar so the scan fills the window, and brings back what was open when clicked again.
- The image scales with the viewing area: hiding a panel, Focus or maximising the window makes the scan bigger, keeping the framing (napari kept the zoom fixed, so extra room was just black).
- Desktop app launcher: `tracecmr-app` starts TraceCMR with no console window. A splash screen with the logo and a thin red progress line appears straight away while the app loads; messages and crash details go to Documents/TraceCMR results/logs/tracecmr_log.txt. The main window is built hidden and appears in one step once drawn (no white flash). The Desktop and Start-menu shortcuts open it directly. A model folder given once through TRACECMR_UCL_DIR is remembered.
- Start screen text drawn smoothly (no pixel-grid snapping) and a little larger and brighter, so it no longer looks broken up on black.
- Windows shortcuts on the Desktop and in the Start menu carry TraceCMR's app ID, so the taskbar button shows the TraceCMR logo and "Pin to taskbar" pins TraceCMR (scripts/windows/make_shortcut.ps1, run by the update script).
- TraceCMR start screen instead of napari's: before a mouse is open the image area shows just the TraceCMR logo, one line on how to start and the version (no napari logo, shortcuts or tips). A mouse folder or UCL .h5 file dragged onto it opens in TraceCMR. The app also registers under TraceCMR's name and taskbar ID, so napari no longer puts its own logo back on the window when the theme changes.
- The TraceCMR logo is the app's icon: on the window, on the Windows taskbar (TraceCMR registers as its own app instead of showing Python's icon), and on a TraceCMR shortcut that the update script puts on the Desktop.
- Double-check flags: four independent checks mark the slices to look at first (⚠ in the slice list, reasons on hover, and a list under the results): the model was unsure along much of the outline; the outline doesn't match the image (Fit would change it a lot, which catches confidently wrong outlines); the anatomy doesn't add up (size jump between neighbours, ES cavity bigger than ED, endo touching epi, implausible wall); base/apex. They update after every edit, and the model-unsure flag clears once the slice has been edited. On UCL's 25 held-out test mice they flagged 40% of slices and caught 91% of the real errors (thresholds chosen on the 15 validation mice; scripts/validate_checks.py, docs/methods.md).
- Export all results (button under the results, or the Save menu): saves, then writes into the mouse's results folder an Excel workbook (Summary with the settings used, Per slice, Wall segments with AHA numbers and names, Notes), the ED-wall and thickening bullseyes and the slice images, and adds or updates the mouse's row in study_summary.xlsx in the results folder: one row per mouse, one column per measure (volumes, EF, mass, RV, mean and per-segment wall thickness and thickening), ready for Prism or R. Without openpyxl the same tables are written as CSV. Needs `openpyxl` (added to the requirements; run TraceCMR update).
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
- Double-check flags never second-guess your work: they (and Auto LV's own base/apex notes) appear only while a slice is still exactly the outline Auto LV drew; any edit (points, brush, Fit, Grow, Snap) makes it yours and clears them, and hand-drawn slices are never flagged. (Including the papillary muscle used to be flagged as "outline doesn't match the image".)
- Opening another mouse in the same window could hang in an endless redraw loop (when the last view was a late frame on a near-empty slice), and every previously opened mouse's editor kept reacting to clicks and slice changes in the background. Old editors are now detached, overlays are removed before the image, and the editor ignores position changes caused by its own redraw.
- Export all crashed on Windows while writing the notes (characters like → can't be written in Windows' default encoding); every exported file is now written as UTF-8, and a failed export shows a message instead of only the status line.
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
