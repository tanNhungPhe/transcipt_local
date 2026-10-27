@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe goto :setup
if not exist models\nllb-1.3B\model.bin goto :setup
goto :run

:setup
echo Lan dau chay: cai dat va tai model (can Internet mot lan)...
call setup_offline.bat || exit /b 1

:run
start "" .venv\Scripts\pythonw.exe main.py %*
exit /b 0
