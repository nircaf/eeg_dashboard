#!/usr/bin/env python3
"""
Test script to verify the MNE preload fix
"""

import os
import sys

def test_mne_preload():
    """Test MNE preload functionality"""
    try:
        import mne
        print(f"✅ MNE imported successfully - Version: {mne.__version__}")
        
        # Test with the example EDF file
        example_path = "Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf"
        
        if not os.path.exists(example_path):
            print(f"❌ Example EDF file not found: {example_path}")
            return False
        
        # Test 1: Read without preload (should work for validation)
        print("🔍 Testing EDF reading without preload...")
        raw_no_preload = mne.io.read_raw_edf(example_path, preload=False, verbose=False)
        print(f"   ✅ EDF read without preload: {raw_no_preload}")
        print(f"   Preload status: {raw_no_preload.preload}")
        
        # Test 2: Load data into memory
        print("🔍 Testing data loading...")
        raw_no_preload.load_data()
        print(f"   ✅ Data loaded successfully")
        print(f"   Preload status after load_data(): {raw_no_preload.preload}")
        
        # Test 3: Apply filtering (should work now)
        print("🔍 Testing filtering...")
        raw_no_preload.filter(l_freq=0.1, h_freq=100.0, fir_design='firwin', verbose=False)
        print(f"   ✅ Filtering applied successfully")
        
        # Test 4: Apply notch filter
        print("🔍 Testing notch filtering...")
        raw_no_preload.notch_filter(freqs=50.0, verbose=False)
        print(f"   ✅ Notch filtering applied successfully")
        
        print("🎉 All MNE preload tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MNE preload test failed: {e}")
        return False

def test_dashboard_functions():
    """Test dashboard functions with preload fix"""
    try:
        # Import the dashboard module
        sys.path.append('.')
        from eeg_dashboard import validate_edf_file, load_edf_data, preprocess_eeg_data
        
        print("✅ Dashboard functions imported successfully")
        
        # Test validation
        example_path = "Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf"
        if os.path.exists(example_path):
            print("🔍 Testing file validation...")
            validation_result = validate_edf_file(example_path)
            if validation_result['valid']:
                print("   ✅ File validation successful")
            else:
                print(f"   ❌ File validation failed: {validation_result.get('error', 'Unknown error')}")
                return False
        
        # Test data loading
        print("🔍 Testing data loading...")
        eeg_data = load_edf_data(example_path)
        if eeg_data is not None:
            print("   ✅ Data loading successful")
            print(f"   Channels: {eeg_data['n_channels']}")
            print(f"   Samples: {eeg_data['n_samples']}")
            print(f"   Raw preload status: {eeg_data['raw'].preload}")
        else:
            print("   ❌ Data loading failed")
            return False
        
        print("🎉 All dashboard function tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard function test failed: {e}")
        return False

def main():
    """Run all preload fix tests"""
    print("🧠 MNE Preload Fix Test Suite")
    print("=" * 50)
    print()
    
    tests = [
        ("MNE Preload Functionality", test_mne_preload),
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
        print("🎉 All preload fix tests passed!")
        print("The dashboard should now work correctly with MNE filtering.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
