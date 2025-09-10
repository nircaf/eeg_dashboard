# EEG Analysis Dashboard - Setup Guide

## Quick Start

### Option 1: Using the provided scripts (Recommended for Windows)

1. **Double-click `run_dashboard.bat`** or **Right-click `run_dashboard.ps1` → "Run with PowerShell"**

The script will automatically:
- Check if Python is installed
- Install required packages
- Start the dashboard

### Option 2: Manual setup

1. **Install Python 3.8+**
   - Download from https://python.org
   - **IMPORTANT**: Check "Add Python to PATH" during installation

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run eeg_dashboard.py
   ```

## Troubleshooting

### Python not found
- **Problem**: "Python was not found" error
- **Solution**: 
  1. Install Python from https://python.org
  2. During installation, check "Add Python to PATH"
  3. Restart your command prompt/PowerShell

### Package installation fails
- **Problem**: pip install fails
- **Solution**:
  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

### Permission errors on Windows
- **Problem**: Permission denied when installing packages
- **Solution**: Run PowerShell as Administrator, then:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  .\run_dashboard.ps1
  ```

### Port already in use
- **Problem**: Port 8501 is already in use
- **Solution**: The dashboard will automatically use the next available port (8502, 8503, etc.)

## Using the Dashboard

1. **Open your web browser** and go to the URL shown in the terminal (usually http://localhost:8501)

2. **Upload an EDF file**:
   - Use the file uploader in the sidebar
   - Or click "Load Example File" to use the provided example

3. **View the analysis**:
   - File validation results
   - Signal quality metrics
   - Interactive visualizations
   - Artifact detection results

## Example File

The dashboard includes an example EDF file at:
`Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf`

You can use this to test the dashboard functionality.

## System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM (8GB+ recommended for large EEG files)
- **Browser**: Any modern web browser (Chrome, Firefox, Safari, Edge)

## Features Overview

- ✅ **EDF File Validation**: Checks file format and integrity
- ✅ **Signal Quality Assessment**: SNR, artifacts, flat lines
- ✅ **Interactive Visualizations**: Raw signals, spectrograms, power spectra
- ✅ **Artifact Detection**: Muscle, eye movement, electrode pops
- ✅ **Frequency Analysis**: Delta, Theta, Alpha, Beta, Gamma bands
- ✅ **Quality Recommendations**: Automated suggestions for data improvement

## Support

If you encounter any issues:
1. Check this setup guide
2. Ensure all requirements are installed
3. Try running the test script: `python test_dashboard.py`
4. Check the terminal output for error messages



