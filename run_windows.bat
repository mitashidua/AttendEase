@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo First run setup_windows.bat
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python app.py
pause
