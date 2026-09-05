@echo off
setlocal
cd /d "%~dp0"
python "%~dp0scripts\run_germany_ablation_4rates.py" %*
if errorlevel 1 (
  echo.
  echo Experiment failed. Check the output and log files.
)
pause
