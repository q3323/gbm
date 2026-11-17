@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\package_backend_release.py"
pause
