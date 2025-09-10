#!/usr/bin/env python3
"""
Test script to verify the EEG dashboard functionality
"""

import os
import sys
import subprocess

def test_edf_file():
    """Test if the example EDF file exists and is accessible"""
    example_path = "Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf"
    
    if os.path.exists(example_path):
        print(f"✅ Example EDF file found: {example_path}")
        file_size = os.path.getsize(example_path)
        print(f"   File size: {file_size:,} bytes")
        return True
    else:
        print(f"❌ Example EDF file not found: {example_path}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'streamlit',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'pyedflib',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def test_dashboard_syntax():
    """Test if the dashboard script has valid syntax"""
    try:
        with open('eeg_dashboard.py', 'r') as f:
            code = f.read()
        
        compile(code, 'eeg_dashboard.py', 'exec')
        print("✅ Dashboard script syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in dashboard script: {e}")
        return False
    except FileNotFoundError:
        print("❌ Dashboard script not found")
        return False

def main():
    """Run all tests"""
    print("🧠 EEG Dashboard Test Suite")
    print("=" * 40)
    
    tests = [
        ("EDF File Check", test_edf_file),
        ("Package Imports", test_imports),
        ("Dashboard Syntax", test_dashboard_syntax)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The dashboard is ready to use.")
        print("\nTo run the dashboard:")
        print("   streamlit run eeg_dashboard.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the dashboard.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
