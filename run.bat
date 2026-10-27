@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Dang cai dat lan dau...
    python -m venv .venv || goto :error
    .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
)
start "" .venv\Scripts\pythonw.exe main.py %*
exit /b 0

:error
echo Cai dat that bai. Kiem tra Python 3.10+ da duoc cai.
pause
exit /b 1
