@echo off
rem Runs once, at the end of installation, inside the new TraceCMR folder (%PREFIX%).
rem Installs TraceCMR and its packages from the bundled wheels (no internet needed), then makes shortcuts.
set "LOG=%PREFIX%\install_log.txt"
echo TraceCMR install %date% %time% > "%LOG%"

"%PREFIX%\python.exe" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "%PREFIX%\wheels.zip" "%PREFIX%\wheels" >> "%LOG%" 2>&1
if errorlevel 1 (echo UNZIP FAILED >> "%LOG%" & exit /b 1)

"%PREFIX%\python.exe" -m pip install --no-index --find-links "%PREFIX%\wheels" "tracecmr[app]" >> "%LOG%" 2>&1
if errorlevel 1 (echo PIP FAILED >> "%LOG%" & exit /b 1)
echo PACKAGES OK >> "%LOG%"

rem the wheels are installed now; drop the copies (about half the install size)
rmdir /s /q "%PREFIX%\wheels" >> "%LOG%" 2>&1
del /q "%PREFIX%\wheels.zip" >> "%LOG%" 2>&1

set "ICON=%PREFIX%\Lib\site-packages\tracecmr\app\resources\tracecmr.ico"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PREFIX%\make_shortcut.ps1" -Target "%PREFIX%\Scripts\tracecmr-app.exe" -WorkDir "%USERPROFILE%\Documents" -Icon "%ICON%" >> "%LOG%" 2>&1
if errorlevel 1 (echo shortcut fallback used >> "%LOG%") else (echo SHORTCUTS OK >> "%LOG%")
echo INSTALL OK >> "%LOG%"
exit /b 0
