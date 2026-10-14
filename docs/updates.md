# Updates

TraceCMR installed with the Windows or Mac installer updates itself. A tester installs once; after that, new
versions arrive as small patches, the way games and browsers update.

## What the user sees

1. A few seconds after TraceCMR opens, if a newer version is out, a card in the corner says
   **TraceCMR 0.3.5 is available**, with what's new. **Update** or **Later**. (Help → Check for updates does the
   same at any time; Settings → Updates turns the check at start off.)
2. **Update** downloads it (well under 1 MB) and checks it against the published SHA-256.
3. **Restart now** saves the open mouse if it has changes, closes TraceCMR, installs the update and opens
   TraceCMR again, about half a minute in all. **When I close TraceCMR** installs it at the next close instead.
4. The reopened TraceCMR says **Updated** with the new version, or, if anything went wrong, that it is still
   on the old version and where the log is.

## How it works

Every GitHub release has `latest.json`, the TraceCMR wheel (only TraceCMR's own code, not Python, napari or
Qt) and the two installers. The app reads `releases/latest/download/latest.json`; nothing about the user or their
scans is sent.

Installing happens after TraceCMR has closed, by a helper the app copies out of its package
(`src/tracecmr/_apply_update.py`, standard library only), because it replaces the package:

1. wait for TraceCMR to end (on Windows also the `tracecmr-app.exe` launcher, whose file is replaced)
2. copy every file of the installed version aside (from its RECORD)
3. `pip install --no-deps --no-index --force-reinstall <wheel>`: offline, TraceCMR only, no other package moves
4. import the new version in a fresh Python (with the window module) and check its version number
5. if 3 or 4 failed, remove what was installed and put the copied files back, then check again
6. write the result for the next start, and reopen TraceCMR the way its shortcut does

Everything is logged to `Documents/TraceCMR results/logs/update_log.txt` (Help → Report a problem includes the
app log).

## When a patch isn't enough

A patch only changes TraceCMR. If a release needs another package or a newer one (its `requires` in
`latest.json` aren't met by the installed copy), or a newer Python, the card offers **Download** for the new
installer instead. The installers don't install over an existing copy (checked on both systems), so the old
copy is removed first: on Windows, Settings → Apps → TraceCMR → Uninstall; on a Mac, move the TraceCMR folder in
the home folder and TraceCMR in the home folder's Applications to the Trash. Results and settings live outside
the install folder and are kept.

Source installs (`git clone`, `pip install -e`) never patch themselves: Help → Check for updates says to
`git pull` instead, and the check at start is off for them.

## Testing

- `tests/test_updater.py`: version order, the manifest, requirement checks, checked downloads, and the helper's
  install and rollback on a stand-in install.
- `scripts/make_test_patches.py` makes pretend releases from a real wheel (a good one, a broken one, one that
  needs new packages); `scripts/check_update.py` runs them through a real install headless, and
  `scripts/check_update_gui.py` runs one through the window (card, Update, Restart now, reopen).
- The release workflow runs both on clean Windows and Mac machines before anything is published.

Environment: `TRACECMR_UPDATE_URL` points the app at another `latest.json` (testing);
`TRACECMR_NO_UPDATE_CHECK=1` turns the check at start off.
