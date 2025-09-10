@echo off
echo Starting EEG Analysis Dashboard...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import streamlit, numpy, pandas, matplotlib, seaborn, scipy, pyedflib, plotly" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install requirements. Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM Run the dashboard
echo Starting Streamlit dashboard...
echo The dashboard will open in your web browser at http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.
streamlit run eeg_dashboard.py

pause



