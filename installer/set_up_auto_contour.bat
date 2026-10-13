@echo off
title TraceCMR - set up auto-contour
rem Optional: the segmentation models need TensorFlow (about 400 MB, downloaded now, so this needs internet).
rem The model weights themselves are a separate download; see the README.
set "HERE=%~dp0"
echo Installing TensorFlow for TraceCMR's auto-contour. This needs internet and takes a few minutes.
echo Leave this window open.
"%HERE%python.exe" -m pip install "tensorflow>=2.16" >> "%HERE%install_log.txt" 2>&1
if errorlevel 1 (echo Failed. The details are in install_log.txt in this folder. & pause & exit /b 1)
echo Done. Restart TraceCMR to use auto-contour.
pause
