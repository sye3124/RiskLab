@echo off
setlocal
cd /d "%~dp0"
echo RiskLab numerical environment setup. Python 3.12 is recommended.
python --version
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Run python run.py using the activated virtual environment, or START_ANALYSIS.bat.
echo To open the app, run python run.py --app or START_APP.bat.
pause
exit /b 0
:failed
echo Setup did not complete. Check Python installation and the error above.
pause
exit /b 1
