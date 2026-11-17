@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\publish_predictions.py"
if errorlevel 1 (
  echo.
  echo Publish failed.
) else (
  echo.
  echo Publish completed.
)
pause
