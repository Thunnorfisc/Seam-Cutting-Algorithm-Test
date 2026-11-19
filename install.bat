@echo off
rem Double-click to install the Python packages this project needs.
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then run this again.
  pause
  exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo This project needs Python 3.10 or newer. This one is:
  %PY% --version
  pause
  exit /b 1
)

echo Installing the packages in requirements.txt ...
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo pip could not install everything. See the messages above.
  pause
  exit /b 1
)

echo.
%PY% tools\check_setup.py
pause
