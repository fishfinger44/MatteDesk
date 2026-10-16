@echo off
rem MatteDesk - start the local server and open the browser (no Docker needed).
rem First start: finds Python 3.11+ (installs Python 3.11 via winget when missing), creates .venv, installs requirements.
cd /d "%~dp0"
set /p MD_VER=<VERSION
title MatteDesk v%MD_VER%
if exist .venv\Scripts\python.exe goto :run

call :findpy
if not defined MD_PY (
  echo [MatteDesk] Python 3.11+ not found. Installing Python 3.11 with winget ^(about 30 MB, current user only^) ...
  winget install --id Python.Python.3.11 -e --scope user --silent --accept-package-agreements --accept-source-agreements
  call :findpy
)
if not defined MD_PY (
  echo [MatteDesk] Python could not be installed automatically.
  echo Install Python 3.11 manually: https://www.python.org/downloads/release/python-3119/
  echo ^(tick "Add python.exe to PATH"^), then run this file again.
  goto :err
)
echo [MatteDesk] Python: %MD_PY%
echo [MatteDesk] Creating the .venv environment ...
"%MD_PY%" -m venv --system-site-packages .venv || goto :err
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :err

:run
rem Port 8765 already taken by an older MatteDesk (other window, app preview)? Ask, then stop it - otherwise the new
rem server cannot start and the browser keeps showing the old one.
call :freeport
if errorlevel 2 goto :eof
if errorlevel 1 goto :portbusy
start "" http://127.0.0.1:8765
echo [MatteDesk v%MD_VER%] Server: http://127.0.0.1:8765  (close this window to stop it)
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8765
goto :eof

:findpy
rem MD_PY = full path of a Python >= 3.11 (py launcher, PATH, default per-user install folder); empty when none.
set "MD_PY="
for /f "delims=" %%P in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "MD_PY=%%P"
if defined MD_PY exit /b 0
for /f "delims=" %%P in ('python -c "import sys;print(sys.executable if sys.version_info[:2]>=(3,11) else '')" 2^>nul') do set "MD_PY=%%P"
if defined MD_PY exit /b 0
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "MD_PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
exit /b 0

:freeport
rem exit 0 = port free (or freed), 1 = cannot start, 2 = the user keeps the running server
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\server_port.ps1" -Port 8765 -Check
if errorlevel 2 exit /b 1
if not errorlevel 1 exit /b 0
echo [MatteDesk] A MatteDesk server is already running on port 8765 (an older version or started from another window).
echo [MatteDesk] Stopping it aborts any computation in progress.
choice /c YN /t 15 /d Y /m "Stop it and start v%MD_VER%? Y = yes, N = no (yes in 15 s)"
if errorlevel 2 goto :keepold
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\server_port.ps1" -Port 8765 -Stop
if errorlevel 1 exit /b 1
echo [MatteDesk] Old server stopped.
exit /b 0

:keepold
echo [MatteDesk] Keeping the running server and opening it in the browser.
start "" http://127.0.0.1:8765
exit /b 2

:portbusy
echo [MatteDesk] Port 8765 is busy and could not be freed - close the program using it and run this file again.
pause
goto :eof

:err
echo [MatteDesk] Installation failed.
pause
