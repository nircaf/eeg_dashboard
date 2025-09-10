#!/usr/bin/env python3
"""
Test script to verify MNE integration in the EEG dashboard
"""

import os
import sys

def test_mne_import():
    """Test if MNE can be imported"""
    try:
        import mne
        print(f"✅ MNE imported successfully - Version: {mne.__version__}")
        return True
    except ImportError as e:
        print(f"❌ MNE import failed: {e}")
        return False

def test_edf_reading():
    """Test if MNE can read EDF files"""
    try:
        import mne
        example_path = "Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf"
        
        if not os.path.exists(example_path):
            print(f"❌ Example EDF file not found: {example_path}")
            return False
        
        # Test reading the EDF file
        raw = mne.io.read_raw_edf(example_path, preload=False, verbose=False)
        print(f"✅ EDF file read successfully")
        print(f"   Channels: {len(raw.ch_names)}")
        print(f"   Sampling rate: {raw.info['sfreq']} Hz")
        print(f"   Duration: {raw.times[-1]:.2f} seconds")
        print(f"   Channel names: {raw.ch_names[:5]}...")  # Show first 5 channels
        
        return True
    except Exception as e:
        print(f"❌ EDF reading failed: {e}")
        return False

def test_dashboard_imports():
    """Test if the dashboard can import all required modules"""
    try:
        # Test individual imports
        import streamlit as st
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import signal
        from scipy.stats import kurtosis, skew
        import mne
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import os
        
        print("✅ All dashboard imports successful")
        return True
    except ImportError as e:
        print(f"❌ Dashboard import failed: {e}")
        return False

def test_dashboard_functions():
    """Test if dashboard functions can be imported"""
    try:
        # Import the dashboard module
        sys.path.append('.')
        from eeg_dashboard import validate_edf_file, load_edf_data, get_channel_info
        
        print("✅ Dashboard functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Dashboard function import failed: {e}")
        return False

def main():
    """Run all MNE integration tests"""
    print("🧠 MNE Integration Test Suite")
    print("=" * 50)
    print()
    
    tests = [
        ("MNE Import", test_mne_import),
        ("EDF Reading", test_edf_reading),
        ("Dashboard Imports", test_dashboard_imports),
        ("Dashboard Functions", test_dashboard_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🔍 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All MNE integration tests passed!")
        print("The dashboard is ready to use with MNE-Python.")
    else:
        print("⚠️  Some tests failed. Please install MNE-Python:")
        print("   pip install mne")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())




