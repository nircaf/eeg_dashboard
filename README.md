# EEG Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing EEG (Electroencephalography) files in EDF format. This tool provides file validation, signal quality assessment, and state-of-the-art EEG analysis features.

## Features

### File Validation
- Validates EDF file format and integrity
- Extracts metadata (patient ID, recording info, duration, etc.)
- Checks file structure and channel information

### Signal Quality Assessment
- Signal-to-noise ratio calculation
- Dynamic range analysis
- Artifact detection and quantification
- Flat line detection
- Signal statistics (kurtosis, skewness)

### Advanced Analysis
- **Raw Data Table**: Interactive table showing raw EEG values with time stamps
- **Channel Information**: Detailed channel types, units, and positions using MNE
- **MNE Integration**: Full MNE-Python support for professional EEG analysis
- Frequency band analysis (Delta, Theta, Alpha, Beta, Gamma)
- Power spectral density visualization
- Spectrogram analysis
- Raw EEG signal plotting with interactive controls
- Data export functionality (CSV download)

### State-of-the-art Tests
- Muscle artifact detection
- Eye movement artifact detection
- Electrode pop detection
- Comprehensive quality metrics

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run eeg_dashboard.py
```

## Usage

1. Open the dashboard in your web browser (usually http://localhost:8501)
2. The example EDF file will be automatically loaded on first visit
3. Upload your own EDF file using the sidebar or use the example file
4. View the analysis results including:
   - File validation status
   - Signal quality metrics
   - **Raw data table** with interactive controls
   - Interactive visualizations
   - Artifact detection results
   - Quality recommendations
   - Data export functionality

## Example File

The dashboard includes an example EDF file at `Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf` that you can use to test the functionality.

## Requirements

- Python 3.8+
- EDF files in valid European Data Format
- Sufficient memory for large EEG files (the dashboard limits to 20 channels for performance)

## Technical Details

The dashboard uses:
- **MNE-Python** for EDF file reading, validation, and advanced EEG analysis
- **scipy** for signal processing and statistical analysis
- **plotly** for interactive visualizations
- **streamlit** for the web interface

## Quality Assessment

The dashboard provides a comprehensive quality score based on:
- Signal-to-noise ratio
- Artifact levels
- Flat line detection
- Channel integrity

Results are color-coded:
- 🟢 Green: Excellent quality
- 🟡 Yellow: Good quality with minor issues
- 🔴 Red: Poor quality requiring attention

## Troubleshooting

### MNE filtering errors
- **Problem**: "RuntimeError: inst.filter requires raw data to be loaded"
- **Solution**: The dashboard automatically handles data loading. If you encounter this error, ensure you're using the latest version of the dashboard.

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
