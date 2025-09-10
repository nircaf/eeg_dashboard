# EEG Analysis Dashboard Launcher
Write-Host "Starting EEG Analysis Dashboard..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import streamlit, numpy, pandas, matplotlib, seaborn, scipy, pyedflib, plotly" 2>$null
    Write-Host "All dependencies are installed!" -ForegroundColor Green
} catch {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install requirements. Please run: pip install -r requirements.txt" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Run the dashboard
Write-Host "Starting Streamlit dashboard..." -ForegroundColor Green
Write-Host "The dashboard will open in your web browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Cyan
Write-Host ""
streamlit run eeg_dashboard.py



