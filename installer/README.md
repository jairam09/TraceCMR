# TraceCMR installers

**Windows:** one `.exe` someone double-clicks: no Python, conda or command line on their side. It installs
TraceCMR for the current user (no administrator rights), adds Desktop and Start-menu shortcuts, and appears in
Windows' "Installed apps" with an uninstaller.

**macOS (Apple silicon):** one `.pkg`. It installs TraceCMR in the user's home folder and puts TraceCMR.app in
their Applications folder (Launchpad, Spotlight, Dock). Intel Macs are not built yet.

Nothing of the user's (scans, contours, results) lives in the install folder.

## Releases (normal route)

Push a version tag (`git tag v0.3.5 && git push origin v0.3.5`). The **release** workflow builds both
installers and the patch wheel, tests them on clean Windows and Mac machines (install, the first session, Auto
LV setup, and patch updates including a rollback), and only then publishes a GitHub release. The installers keep
the same names every release, so `https://github.com/jairam09/TraceCMR/releases/latest/download/TraceCMR-Windows.exe`
(and `TraceCMR-macOS.pkg`) always give the newest. Installed copies update themselves from the release's
`latest.json` and wheel (see docs/updates.md).

To build installers without releasing: GitHub → Actions → **installers** → **Run workflow** (the installers are
attached to the run under Artifacts).

## Build on Windows by hand (fallback)

1. Install [Miniforge](https://conda-forge.org/download/) once.
2. Open the **Miniforge Prompt** and run `installer\build_installer.bat` from the repository.
3. The installer appears in `installer\dist\`. Details of each step are in `installer\build_log.txt`.

The build needs internet; the finished installer does not.

## How it works

`constructor` makes an installer that lays down its own Python 3.12. `build_installer.bat` first turns
TraceCMR and all its packages (napari, PyQt6, numpy, ...) into wheels with that same Python version and zips
them; the installer carries the zip, and `post_install.bat` installs from it offline with pip. The result is the
same package set used during development, rather than conda-forge's Qt builds.

## Models

No model weights are included, on purpose: each published model keeps its own licence (UCL: GPL-3.0; Calico:
MIT). The first click on **Auto LV** offers a one-time setup that downloads UCL's model from the authors'
repository (checked against their published SHA-256) and installs TensorFlow from PyPI, about 500 MB in all,
with progress and a Cancel button (`src/tracecmr/model_setup.py`). Settings → "Set up auto-contouring…" does the
same. Everything else works without it.

The wheels are built with `h5py<3.15`, the range TensorFlow 2.21 accepts, so setting up auto-contour adds
TensorFlow without replacing packages the app already uses.

## Unsigned installers: what testers will see

Neither installer is code-signed yet, so the first open shows a warning:

- **Windows** (SmartScreen, "Windows protected your PC"): **More info** → **Run anyway**.
- **macOS** ("cannot be opened because it is from an unidentified developer"): right-click the .pkg →
  **Open** → **Open**, or System Settings → Privacy & Security → **Open Anyway**.

Signing removes the warnings but costs money (Apple Developer: $99/year; Windows certificates more). Worth it
once people outside the first testers install it.
