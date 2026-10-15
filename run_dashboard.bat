@echo off
REM ============================================================
REM  Industrial Predictive Maintenance Dashboard - Launcher
REM ============================================================
echo Starting Industrial Predictive Maintenance Dashboard...
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found on PATH. Install Python 3.10/3.11 and try again.
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing/verifying dependencies from requirements.txt ...
pip install -r requirements.txt

echo.
echo Launching Streamlit app...
streamlit run app.py

pause
