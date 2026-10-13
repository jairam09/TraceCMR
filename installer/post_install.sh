#!/bin/bash
# macOS: runs once, at the end of installation, with $PREFIX set to the new TraceCMR folder.
# Installs TraceCMR and its packages from the bundled wheels (no internet needed), then puts a TraceCMR app in
# the user's Applications folder so it opens from Launchpad, Spotlight and the Dock like any other app.
LOG="$PREFIX/install_log.txt"
echo "TraceCMR install $(date)" > "$LOG"

"$PREFIX/bin/python" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$PREFIX/wheels.zip" "$PREFIX/wheels" >> "$LOG" 2>&1 || { echo "UNZIP FAILED" >> "$LOG"; exit 1; }
# --no-compile: pre-compiling every package (about 12,000 files) was most of the install time. The warm-up import
# below compiles just the modules TraceCMR actually loads, so the first launch stays quick.
"$PREFIX/bin/python" -m pip install --no-index --no-compile --find-links "$PREFIX/wheels" "tracecmr[app]" >> "$LOG" 2>&1 \
    || { echo "PIP FAILED" >> "$LOG"; exit 1; }
echo "PACKAGES OK $(date +%T)" >> "$LOG"
QT_QPA_PLATFORM=offscreen "$PREFIX/bin/python" -c "import napari, tracecmr.app._widget, tracecmr.app.compare" >> "$LOG" 2>&1
echo "WARMUP DONE $(date +%T)" >> "$LOG"
rm -rf "$PREFIX/wheels" "$PREFIX/wheels.zip"

# The installer may run as root; the app belongs to whoever is logged in at the screen.
OWNER="$(stat -f %Su /dev/console 2>/dev/null || id -un)"
OWNER_HOME="$(dscl . -read "/Users/$OWNER" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
[ -n "$OWNER_HOME" ] || OWNER_HOME="$HOME"
APP="$OWNER_HOME/Applications/TraceCMR.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/MacOS/TraceCMR" <<LAUNCH
#!/bin/bash
cd "\$HOME/Documents" 2>/dev/null || cd "\$HOME"
exec "$PREFIX/bin/tracecmr-app" "\$@"
LAUNCH
chmod +x "$APP/Contents/MacOS/TraceCMR"
cp "$PREFIX/tracecmr.icns" "$APP/Contents/Resources/tracecmr.icns" 2>/dev/null
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>TraceCMR</string>
  <key>CFBundleDisplayName</key><string>TraceCMR</string>
  <key>CFBundleIdentifier</key><string>org.tracecmr.TraceCMR</string>
  <key>CFBundleExecutable</key><string>TraceCMR</string>
  <key>CFBundleIconFile</key><string>tracecmr</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.3.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
chown -R "$OWNER" "$APP" "$OWNER_HOME/Applications" 2>/dev/null
# The installer may have run as root: hand the TraceCMR folder to the user so Python can cache compiled modules there.
chown -R "$OWNER" "$PREFIX" 2>/dev/null
echo "APP OK: $APP" >> "$LOG"
echo "INSTALL OK" >> "$LOG"
exit 0
