#!/bin/bash
# macOS: double-click to add auto-contour. The segmentation models need TensorFlow (about 400 MB, downloaded
# now, so this needs internet). The model weights themselves are a separate download; see the README.
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "Installing TensorFlow for TraceCMR's auto-contour. This needs internet and takes a few minutes."
if "$HERE/bin/python" -m pip install "tensorflow>=2.16" >> "$HERE/install_log.txt" 2>&1; then
  echo "Done. Restart TraceCMR to use auto-contour."
else
  echo "Failed. The details are in install_log.txt in $HERE."
fi
read -n 1 -s -r -p "Press any key to close."
