@echo off
setlocal
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py -3
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install -q --disable-pip-version-check -r requirements.txt -r requirements-train.txt || (pause & exit /b 1)
python train.py || (pause & exit /b 1)
python test.py --quick || (pause & exit /b 1)
python export_runtime.py || (pause & exit /b 1)
echo.
echo Retrained. Re-deploy to Vercel to publish the new models ^(commit runtime\*.json^).
pause
