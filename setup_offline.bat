@echo off
rem One-time setup (needs Internet): install packages and download all models into .\models
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Dang tao moi truong Python...
    python -m venv .venv || goto :error
)
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
where nvidia-smi >nul 2>&1 && (
    .venv\Scripts\python.exe -m pip install -r requirements-cuda.txt || goto :error
)
.venv\Scripts\python.exe main.py download %* || goto :error
echo.
echo Hoan tat. Tu gio app chay offline - co the rut mang.
pause
exit /b 0

:error
echo Cai dat that bai. Kiem tra Internet va Python 3.10 - 3.13.
pause
exit /b 1
