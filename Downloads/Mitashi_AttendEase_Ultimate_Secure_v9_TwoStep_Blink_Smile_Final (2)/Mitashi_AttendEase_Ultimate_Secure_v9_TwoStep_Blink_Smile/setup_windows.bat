@echo off
setlocal
cd /d "%~dp0"
echo ===============================================
echo   AttendEase Secure Kiosk - Windows Setup
echo ===============================================
where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py
) else (
  set PY=python
)
%PY% --version || goto :python_error
if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless >nul 2>nul
pip install -r requirements.txt || goto :fail
python self_check.py || goto :fail
echo.
echo Setup complete. Starting AttendEase...
python app.py
goto :eof
:python_error
echo Python was not found. Install Python 3.10-3.12 64-bit and tick Add Python to PATH.
pause
exit /b 1
:fail
echo.
echo Setup failed. Copy the error above and send it for debugging.
pause
exit /b 1
