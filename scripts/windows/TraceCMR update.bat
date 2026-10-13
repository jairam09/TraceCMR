@echo off
title TraceCMR update
set "LOG=%USERPROFILE%\Documents\Projects\tracecmr_setup_log.txt"
set "REPO=%USERPROFILE%\Documents\Projects\mouse-cmr-pipeline"
set "ENV=%USERPROFILE%\ventrimap-env"
echo Updating TraceCMR (about a minute). Leave this window open.
echo update %date% %time% >> "%LOG%"
"%ENV%\Scripts\python.exe" -m pip uninstall -y ventrimap >> "%LOG%" 2>&1
"%ENV%\Scripts\python.exe" -m pip install -e "%REPO%[app]" >> "%LOG%" 2>&1
if errorlevel 1 (echo UPDATE FAILED >> "%LOG%" & echo Update failed. Tell Claude. & pause & exit /b 1)
echo UPDATE OK >> "%LOG%"
rem Shortcuts (Desktop and Start menu) open the desktop app directly: no console window, logo on the taskbar.
powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\windows\make_shortcut.ps1" -Target "%ENV%\Scripts\tracecmr-app.exe" -WorkDir "%USERPROFILE%\Documents" -Icon "%REPO%\src\tracecmr\app\resources\tracecmr.ico" >> "%LOG%" 2>&1
if errorlevel 1 (echo shortcut not created properly >> "%LOG%") else (echo shortcut OK >> "%LOG%")
echo Done. Starting TraceCMR...
rem First start passes the model folder once; the app remembers it from then on.
set "TRACECMR_UCL_DIR=%USERPROFILE%\Documents\Projects\ventrimap-models\ucl"
start "" "%ENV%\Scripts\tracecmr-app.exe"
exit
