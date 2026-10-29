@echo off
setlocal
cd /d "%~dp0"
title Green Grid Allocator

set PY=python
where python >nul 2>nul || set PY=py -3
%PY% --version >nul 2>nul || (echo Python 3.9+ is required. Install it from python.org and tick "Add to PATH". & pause & exit /b 1)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  %PY% -m venv .venv || (echo Could not create venv. & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"

if not exist ".venv\.deps_ok" (
  echo [2/3] Installing dependencies...
  python -m pip install -q --disable-pip-version-check -r requirements.txt || (echo pip install failed. & pause & exit /b 1)
  echo ok> ".venv\.deps_ok"
)

if not exist "runtime\model.json" (
  echo [3/3] First run: training models ^(takes a few minutes^)...
  python -m pip install -q --disable-pip-version-check -r requirements-train.txt || (echo pip install failed. & pause & exit /b 1)
  python train.py || (echo Training failed. & pause & exit /b 1)
  python export_runtime.py || (echo Export failed. & pause & exit /b 1)
)

python app.py
pause
