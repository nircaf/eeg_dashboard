#!/usr/bin/env python3
"""
Demo script to show the EEG Dashboard features
"""

import os
import sys

def print_banner():
    """Print a nice banner for the demo"""
    print("=" * 60)
    print("🧠 EEG Analysis Dashboard - Feature Demo")
    print("=" * 60)
    print()

def check_files():
    """Check if all required files exist"""
    required_files = [
        "eeg_dashboard.py",
        "requirements.txt",
        "Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf"
    ]
    
    print("📁 Checking required files...")
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_exist = False
    
    print()
    return all_exist

def show_features():
    """Show the main features of the dashboard"""
    print("🚀 Dashboard Features:")
    print()
    
    features = [
        ("📊 Raw Data Table", "Interactive table showing raw EEG values with time stamps"),
        ("🔬 Channel Information", "Detailed channel types, units, and positions using MNE"),
        ("📈 Signal Visualization", "Raw EEG plots, spectrograms, and power spectra"),
        ("🔍 Quality Assessment", "Signal-to-noise ratio, artifact detection, flat line detection"),
        ("🧠 Frequency Analysis", "Delta, Theta, Alpha, Beta, Gamma band analysis"),
        ("⚠️ Artifact Detection", "Muscle artifacts, eye movements, electrode pops"),
        ("💾 Data Export", "Download raw data as CSV files"),
        ("📋 Auto-Loading", "Example file automatically loads on first visit"),
        ("🎛️ Interactive Controls", "Time range selection, channel selection, duration controls"),
        ("🔧 MNE Integration", "Full MNE-Python support for professional EEG analysis")
    ]
    
    for feature, description in features:
        print(f"   {feature}: {description}")
    
    print()

def show_usage():
    """Show how to use the dashboard"""
    print("📖 How to Use:")
    print()
    print("1. 🚀 Start the dashboard:")
    print("   streamlit run eeg_dashboard.py")
    print()
    print("2. 🌐 Open your browser:")
    print("   http://localhost:8501")
    print()
    print("3. 📁 The example file loads automatically!")
    print("   - View the raw data table in the first tab")
    print("   - Explore different visualizations")
    print("   - Check signal quality metrics")
    print("   - Export data if needed")
    print()
    print("4. 🔄 Upload your own EDF file:")
    print("   - Use the file uploader in the sidebar")
    print("   - Or click 'Load Example' to reload the example")
    print()

def show_tabs():
    """Show what each tab contains"""
    print("📑 Dashboard Tabs:")
    print()
    
    tabs = [
        ("Raw Data Table", [
            "Interactive data table with time stamps",
            "Adjustable time range and duration",
            "Configurable number of rows to display",
            "Data statistics (sampling rate, channels, etc.)",
            "CSV export functionality"
        ]),
        ("Channel Info", [
            "Detailed channel information using MNE",
            "Channel types, units, and positions",
            "Channel type summary and counts",
            "MNE file information and metadata",
            "Professional EEG analysis features"
        ]),
        ("Raw EEG", [
            "Multi-channel EEG signal plots",
            "Interactive time range selection",
            "Zoom and pan capabilities",
            "Up to 8 channels displayed simultaneously"
        ]),
        ("Spectrogram", [
            "Frequency-time analysis",
            "Channel selection dropdown",
            "Color-coded power representation",
            "Interactive spectrogram visualization"
        ]),
        ("Power Spectrum", [
            "Power spectral density analysis",
            "Frequency band annotations",
            "Average PSD across all channels",
            "Delta, Theta, Alpha, Beta, Gamma bands"
        ]),
        ("Artifact Analysis", [
            "Muscle artifact detection results",
            "Eye movement artifact detection",
            "Electrode pop detection",
            "Channel-wise artifact reporting"
        ])
    ]
    
    for tab_name, features in tabs:
        print(f"   📊 {tab_name}:")
        for feature in features:
            print(f"      • {feature}")
        print()

def main():
    """Main demo function"""
    print_banner()
    
    # Check files
    if not check_files():
        print("❌ Some required files are missing. Please ensure all files are present.")
        return 1
    
    # Show features
    show_features()
    
    # Show usage
    show_usage()
    
    # Show tabs
    show_tabs()
    
    print("🎉 Ready to run! Start the dashboard with:")
    print("   streamlit run eeg_dashboard.py")
    print()
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



