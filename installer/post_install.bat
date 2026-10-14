@echo off
rem Runs once, at the end of installation, inside the new TraceCMR folder (%PREFIX%).
rem The installer runs this with a minimal PATH, so PowerShell is called by its full path.
rem Installs TraceCMR and its packages from the bundled wheels (no internet needed), then makes shortcuts.
set "LOG=%PREFIX%\install_log.txt"
echo TraceCMR install %date% %time% > "%LOG%"

"%PREFIX%\python.exe" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "%PREFIX%\wheels.zip" "%PREFIX%\wheels" >> "%LOG%" 2>&1
if errorlevel 1 (echo UNZIP FAILED >> "%LOG%" & exit /b 1)

rem --no-compile: pre-compiling every package (about 12,000 files) was most of the install time. Instead, the
rem warm-up import below compiles just the modules TraceCMR actually loads, so the first launch stays quick.
"%PREFIX%\python.exe" -m pip install --no-index --no-compile --find-links "%PREFIX%\wheels" "tracecmr[app,auto]" >> "%LOG%" 2>&1
if errorlevel 1 (echo PIP FAILED >> "%LOG%" & exit /b 1)
echo PACKAGES OK %time% >> "%LOG%"
set "QT_QPA_PLATFORM=offscreen"
"%PREFIX%\python.exe" -c "import napari, tracecmr.app._widget, tracecmr.app.compare, onnxruntime" >> "%LOG%" 2>&1
set "QT_QPA_PLATFORM="
echo WARMUP DONE %time% >> "%LOG%"

rem Auto LV: UCL's model (with the authors' licence) goes in models\, where TraceCMR looks first; then run it once
if not exist "%PREFIX%\models" mkdir "%PREFIX%\models"
move /y "%PREFIX%\ucl_seg459.onnx" "%PREFIX%\models\ucl_seg459.onnx" >> "%LOG%" 2>&1
move /y "%PREFIX%\UCL-LICENSE.txt" "%PREFIX%\models\UCL-LICENSE.txt" >> "%LOG%" 2>&1
"%PREFIX%\python.exe" -c "import sys, numpy as np, onnxruntime as ort; s = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']); x = np.zeros((1, 256, 256, 1), np.float32); print('AUTO LV OK', ort.__version__, s.run(None, {s.get_inputs()[0].name: x})[0].shape)" "%PREFIX%\models\ucl_seg459.onnx" >> "%LOG%" 2>&1
if errorlevel 1 echo AUTO LV CHECK FAILED: the app will offer the one-click setup >> "%LOG%"

rem the wheels are installed now; drop the copies (about half the install size)
rmdir /s /q "%PREFIX%\wheels" >> "%LOG%" 2>&1
del /q "%PREFIX%\wheels.zip" >> "%LOG%" 2>&1

set "ICON=%PREFIX%\Lib\site-packages\tracecmr\app\resources\tracecmr.ico"
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PREFIX%\make_shortcut.ps1" -Target "%PREFIX%\Scripts\tracecmr-app.exe" -WorkDir "%USERPROFILE%\Documents" -Icon "%ICON%" >> "%LOG%" 2>&1
if errorlevel 1 (echo shortcut fallback used >> "%LOG%") else (echo SHORTCUTS OK >> "%LOG%")
echo INSTALL OK >> "%LOG%"
exit /b 0
