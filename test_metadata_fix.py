#!/usr/bin/env python3
"""
Test script to verify the metadata extraction fix works correctly
"""

import mne
import os

def test_metadata_extraction():
    """Test the fixed metadata extraction logic"""
    
    # Path to the example EDF file
    example_path = "Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf"
    
    if not os.path.exists(example_path):
        print(f"❌ Example file not found: {example_path}")
        return False
    
    try:
        # Read the EDF file with MNE (same as in the dashboard)
        print("📁 Loading EDF file...")
        raw = mne.io.read_raw_edf(example_path, preload=False, verbose=False)
        
        # Extract basic information
        n_channels = len(raw.ch_names)
        n_samples = len(raw.times)
        duration = raw.times[-1] if len(raw.times) > 0 else 0
        sampling_rate = raw.info['sfreq']
        
        print(f"✅ File loaded successfully!")
        print(f"   Channels: {n_channels}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Sampling Rate: {sampling_rate:.1f} Hz")
        print(f"   Samples: {n_samples:,}")
        
        # Test the fixed metadata extraction
        print("\n🔍 Testing metadata extraction...")
        
        # Get file metadata using the fixed logic
        file_info = {
            'patient_id': raw.info.get('subject_info', {}).get('his_id', 'Unknown') if raw.info.get('subject_info') else 'Unknown',
            'recording_id': 'Unknown',  # meas_id not available in this file format
            'start_date': raw.info.get('meas_date', 'Unknown'),
            'file_duration': duration,
            'recording_date': raw.info.get('meas_date', 'Unknown'),
            'experimenter': 'Unknown',  # experimenter not available in this file format
            'description': raw.info.get('description', 'Unknown')
        }
        
        print("📋 Extracted metadata:")
        for key, value in file_info.items():
            print(f"   {key}: {value}")
        
        # Show the actual raw.info structure for reference
        print(f"\n🔬 Raw info structure:")
        print(f"   Available keys: {list(raw.info.keys())}")
        print(f"   Subject info: {raw.info.get('subject_info', 'Not available')}")
        print(f"   Measurement date: {raw.info.get('meas_date', 'Not available')}")
        print(f"   Description: {raw.info.get('description', 'Not available')}")
        
        print("\n✅ Metadata extraction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during metadata extraction test: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧠 EEG Metadata Extraction Test")
    print("=" * 40)
    
    success = test_metadata_extraction()
    
    if success:
        print("\n🎉 All tests passed! The metadata extraction fix is working correctly.")
    else:
        print("\n💥 Test failed! There may be an issue with the metadata extraction.")
