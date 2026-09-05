@echo off
setlocal
cd /d "%~dp0"
"%USERPROFILE%\.conda\envs\sfc_ppo\python.exe" "%~dp0scripts\run_us_ablation_4rates.py"
if errorlevel 1 (
  echo.
  echo Experiment failed. Check the output and log files.
)
pause
