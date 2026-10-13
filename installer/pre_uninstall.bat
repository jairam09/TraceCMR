@echo off
rem Runs before TraceCMR is uninstalled: remove the shortcuts post_install.bat made.
rem Contours and results live with the user's data, not here, so nothing of theirs is touched.
for /f "usebackq delims=" %%d in (`%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do del /q "%%d\TraceCMR.lnk" 2>nul
for /f "usebackq delims=" %%d in (`%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "[Environment]::GetFolderPath('Programs')"`) do del /q "%%d\TraceCMR.lnk" 2>nul
exit /b 0
