@echo off
rem Double-click to start the seam carving page. Close this window to stop it.
cd /d "%~dp0"
where py >nul 2>nul && (py frontend\serve.py %*) || (python frontend\serve.py %*)
if errorlevel 1 pause
