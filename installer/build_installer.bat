@echo off
title Build the TraceCMR installer
rem Builds installer\TraceCMR-<version>-Windows-x86_64.exe on this Windows machine.
rem Needs: conda (Miniforge: https://conda-forge.org/download/) and internet, for the build only.
rem The finished installer works offline.
setlocal
set "HERE=%~dp0"
set "REPO=%HERE%.."
set "BUILD=%HERE%build"
set "LOG=%HERE%build_log.txt"
echo Building the TraceCMR installer. This takes several minutes; the details go to build_log.txt.
echo build %date% %time% > "%LOG%"

where conda >nul 2>&1
if errorlevel 1 (
  echo conda was not found. Install Miniforge from https://conda-forge.org/download/ , then run this again
  echo from the "Miniforge Prompt".
  pause & exit /b 1
)

echo [1/4] build tools (Python 3.12 + constructor)...
if not exist "%BUILD%\py\python.exe" call conda create -y -p "%BUILD%\py" -c conda-forge python=3.12 pip constructor >> "%LOG%" 2>&1
if errorlevel 1 goto failed
set "PY=%BUILD%\py\python.exe"

echo [2/4] TraceCMR and every package it needs, as wheels...
if exist "%BUILD%\wheels" rmdir /s /q "%BUILD%\wheels"
"%PY%" -m pip wheel --wheel-dir "%BUILD%\wheels" "%REPO%[app]" >> "%LOG%" 2>&1
if errorlevel 1 goto failed

echo [3/4] packing the wheels...
if exist "%HERE%wheels.zip" del /q "%HERE%wheels.zip"
"%PY%" -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])" "%HERE%wheels" "%BUILD%\wheels" >> "%LOG%" 2>&1
if errorlevel 1 goto failed

echo [4/4] building the installer...
pushd "%HERE%"
call "%BUILD%\py\Scripts\constructor.exe" . --output-dir "%HERE%dist" >> "%LOG%" 2>&1
set "ERR=%errorlevel%"
popd
if not "%ERR%"=="0" goto failed

echo BUILD OK >> "%LOG%"
echo.
echo Done: the installer is in installer\dist\
dir /b "%HERE%dist\*.exe"
pause
exit /b 0

:failed
echo BUILD FAILED >> "%LOG%"
echo Build failed. Send Claude installer\build_log.txt.
pause
exit /b 1
