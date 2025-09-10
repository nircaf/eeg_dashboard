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
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set MNE logging to reduce verbosity
mne.set_log_level('WARNING')

# Page configuration
st.set_page_config(
    page_title="EEG Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def validate_edf_file(file_path):
    """Validate if the uploaded file is a valid EDF file or a pickled MNE Raw object"""
    try:
        # If the file is a pickled MNE Raw, load it via pickle
        if isinstance(file_path, str) and file_path.lower().endswith(('.pkl', '.pickle')):
            with open(file_path, 'rb') as f:
                raw = pickle.load(f)
        else:
            # Assume EDF by default
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        
        # Extract basic information
        n_channels = len(raw.ch_names)
        n_samples = len(raw.times)
        duration = raw.times[-1] if len(raw.times) > 0 else 0
        sampling_rate = raw.info.get('sfreq', np.nan)
        
        # Get file metadata (safe access)
        file_info = {
            'patient_id': raw.info.get('subject_info', {}).get('his_id', 'Unknown') if raw.info.get('subject_info') else 'Unknown',
            'recording_id': raw.info.get('meas_id', {}).get('version', 'Unknown') if raw.info.get('meas_id') else 'Unknown',
            'start_date': raw.info.get('meas_date', 'Unknown'),
            'file_duration': duration,
            'recording_date': raw.info.get('meas_date', 'Unknown'),
            'experimenter': raw.info.get('experimenter', 'Unknown'),
            'description': raw.info.get('description', 'Unknown') if raw.info.get('description') is not None else 'Unknown'
        }
        
        return {
            'valid': True,
            'n_channels': n_channels,
            'n_samples': n_samples,
            'duration': duration,
            'sampling_rate': sampling_rate,
            'channel_names': raw.ch_names,
            'file_info': file_info,
            'raw_info': raw.info,
            'raw_obj': raw
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def load_edf_data(file_path, max_channels=20):
    """Load EDF data with error handling using MNE"""
    try:
        # Read the EDF file with MNE
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Get basic information
        n_channels = len(raw.ch_names)
        n_samples = len(raw.times)
        sampling_rate = raw.info['sfreq']
        
        # Limit channels for performance
        channels_to_load = min(n_channels, max_channels)
        
        # Get data (channels x samples)
        data = raw.get_data()[:channels_to_load, :]
        channel_names = raw.ch_names[:channels_to_load]
        
        # All channels have the same sampling rate in MNE
        sampling_rates = [sampling_rate] * channels_to_load
        
        return {
            'data': data,
            'channel_names': channel_names,
            'sampling_rates': sampling_rates,
            'n_channels': channels_to_load,
            'n_samples': n_samples,
            'raw': raw  # Keep raw object for additional MNE functionality
        }
    except Exception as e:
        st.error(f"Error loading EDF data: {str(e)}")
        return None

def calculate_eeg_quality_metrics(data, sampling_rate):
    """Calculate comprehensive EEG quality metrics"""
    metrics = {}
    
    # Basic signal statistics
    metrics['mean_amplitude'] = np.mean(np.abs(data))
    metrics['std_amplitude'] = np.std(data)
    metrics['dynamic_range'] = np.max(data) - np.min(data)
    
    # Signal quality indicators
    metrics['kurtosis'] = np.mean(kurtosis(data, axis=1))
    metrics['skewness'] = np.mean(skew(data, axis=1))
    
    # Power spectral density
    freqs, psd = signal.welch(data, fs=sampling_rate, nperseg=min(1024, data.shape[1]//4))
    
    # Frequency band analysis
    delta_power = np.mean(psd[:, (freqs >= 0.5) & (freqs <= 4)], axis=1)
    theta_power = np.mean(psd[:, (freqs >= 4) & (freqs <= 8)], axis=1)
    alpha_power = np.mean(psd[:, (freqs >= 8) & (freqs <= 13)], axis=1)
    beta_power = np.mean(psd[:, (freqs >= 13) & (freqs <= 30)], axis=1)
    gamma_power = np.mean(psd[:, (freqs >= 30) & (freqs <= 100)], axis=1)
    
    metrics['delta_power'] = np.mean(delta_power)
    metrics['theta_power'] = np.mean(theta_power)
    metrics['alpha_power'] = np.mean(alpha_power)
    metrics['beta_power'] = np.mean(beta_power)
    metrics['gamma_power'] = np.mean(gamma_power)
    
    # Artifact detection
    # High amplitude artifacts
    threshold = 3 * np.std(data)
    high_amp_artifacts = np.sum(np.abs(data) > threshold, axis=1)
    metrics['high_amp_artifact_ratio'] = np.mean(high_amp_artifacts) / data.shape[1]
    
    # Flat line detection
    flat_threshold = 0.1 * np.std(data)
    flat_lines = np.sum(np.abs(np.diff(data, axis=1)) < flat_threshold, axis=1)
    metrics['flat_line_ratio'] = np.mean(flat_lines) / (data.shape[1] - 1)
    
    # Signal-to-noise ratio estimation
    signal_power = np.mean(np.var(data, axis=1))
    noise_power = np.mean(np.var(np.diff(data, axis=1), axis=1))
    metrics['snr_estimate'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    return metrics, freqs, psd

def detect_artifacts(data, sampling_rate):
    """Detect various types of artifacts in EEG data"""
    artifacts = {}
    
    # Muscle artifacts (high frequency, high amplitude)
    freqs, psd = signal.welch(data, fs=sampling_rate, nperseg=min(1024, data.shape[1]//4))
    muscle_power = np.mean(psd[:, (freqs >= 20) & (freqs <= 100)], axis=1)
    muscle_threshold = np.mean(muscle_power) + 2 * np.std(muscle_power)
    artifacts['muscle_artifacts'] = np.sum(muscle_power > muscle_threshold)
    
    # Eye movement artifacts (low frequency, high amplitude)
    eye_power = np.mean(psd[:, (freqs >= 0.5) & (freqs <= 4)], axis=1)
    eye_threshold = np.mean(eye_power) + 2 * np.std(eye_power)
    artifacts['eye_movement_artifacts'] = np.sum(eye_power > eye_threshold)
    
    # Electrode pop artifacts (sudden spikes)
    diff_data = np.diff(data, axis=1)
    spike_threshold = 3 * np.std(diff_data)
    artifacts['electrode_pops'] = np.sum(np.abs(diff_data) > spike_threshold, axis=1)
    
    return artifacts

def plot_raw_eeg(data, channel_names, sampling_rate, start_time=0, duration=10):
    """Plot raw EEG signals with a consistent y-axis across subplots"""
    n_channels = min(len(channel_names), 8)  # Limit to 8 channels for readability
    end_sample = min(int((start_time + duration) * sampling_rate), data.shape[1])
    start_sample = int(start_time * sampling_rate)
    
    # Extract the segment to plot
    seg = data[:n_channels, start_sample:end_sample]
    time_axis = np.arange(start_sample, end_sample) / sampling_rate

    # Compute global y limits from the plotted segment and add small padding
    if seg.size > 0:
        global_min = float(np.nanmin(seg))
        global_max = float(np.nanmax(seg))
        if global_max == global_min:
            # Avoid zero-range axis
            padding = 1.0 if global_max == 0 else abs(global_max) * 0.1
        else:
            padding = 0.05 * (global_max - global_min)
        y_range = [global_min - padding, global_max + padding]
    else:
        y_range = None

    fig = make_subplots(
        rows=n_channels, cols=1,
        subplot_titles=channel_names[:n_channels],
        vertical_spacing=0.02
    )
    
    for i in range(n_channels):
        y_data = data[i, start_sample:end_sample]
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=y_data,
                mode='lines',
                name=channel_names[i],
                line=dict(width=0.8)
            ),
            row=i+1, col=1
        )
        # Apply same y range to each subplot if available
        if y_range is not None:
            fig.update_yaxes(range=y_range, row=i+1, col=1)
    
    fig.update_layout(
        height=200 * n_channels,
        showlegend=False,
        title=f"Raw EEG Signals ({start_time:.1f}s - {start_time + duration:.1f}s)"
    )
    
    return fig

def plot_spectrogram(data, channel_names, sampling_rate, channel_idx=0):
    """Plot spectrogram for a selected channel (frequency range 0-45 Hz)"""
    if channel_idx >= len(channel_names):
        channel_idx = 0
    
    freqs, times, Sxx = signal.spectrogram(
        data[channel_idx],
        fs=sampling_rate,
        nperseg=min(1024, data.shape[1]//8),
        noverlap=min(512, data.shape[1]//16)
    )
    
    # Limit frequencies to 0-45 Hz for display
    freq_mask = (freqs >= 0) & (freqs <= 45)
    if np.any(freq_mask):
        freqs_disp = freqs[freq_mask]
        Sxx_disp = Sxx[freq_mask, :]
    else:
        # fallback to full range if mask empty
        freqs_disp = freqs
        Sxx_disp = Sxx

    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx_disp + 1e-10),
        x=times,
        y=freqs_disp,
        colorscale='Viridis',
        colorbar=dict(title="Power (dB)")
    ))
    
    fig.update_layout(
        title=f"Spectrogram - {channel_names[channel_idx]}",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=500
    )
    # Ensure y-axis shows 0-45 Hz (or the available display range)
    try:
        fig.update_yaxes(range=[0, 45])
    except Exception:
        pass
    
    return fig

def plot_power_spectrum(psd, freqs, channel_names):
    """Plot power spectral density"""
    fig = go.Figure()
    
    # Plot average PSD across all channels
    avg_psd = np.mean(psd, axis=0)
    fig.add_trace(go.Scatter(
        x=freqs,
        y=10 * np.log10(avg_psd + 1e-10),
        mode='lines',
        name='Average PSD',
        line=dict(width=2)
    ))
    
    # Add frequency band annotations
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4, 'blue'),
        'Theta (4-8 Hz)': (4, 8, 'green'),
        'Alpha (8-13 Hz)': (8, 13, 'orange'),
        'Beta (13-30 Hz)': (13, 30, 'red'),
        'Gamma (30-100 Hz)': (30, 100, 'purple')
    }
    
    for band_name, (low, high, color) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        if np.any(band_mask):
            fig.add_vrect(
                x0=low, x1=high,
                fillcolor=color, opacity=0.2,
                annotation_text=band_name,
                annotation_position="top"
            )
    
    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)",
        height=400
    )
    
    return fig

def get_channel_info(raw):
    """Get detailed channel information using MNE"""
    channel_info = []
    # Prefer the Raw object's API for channel types (more robust)
    try:
        ch_types = raw.get_channel_types()
    except Exception:
        # Fallback if method not available or fails
        ch_types = ['unknown'] * len(raw.ch_names)

    for i, ch_name in enumerate(raw.ch_names):
        ch_type = ch_types[i] if i < len(ch_types) else 'unknown'

        # Safely extract unit and location information from raw.info['chs']
        unit = 'Unknown'
        pos = "Unknown"
        chs = raw.info.get('chs', None)
        if isinstance(chs, (list, tuple)) and i < len(chs):
            ch_entry = chs[i]
            # unit may be stored as an int code; attempt to display raw value if present
            unit = ch_entry.get('unit', 'Unknown') if isinstance(ch_entry, dict) else ch_entry
            loc = ch_entry.get('loc', None) if isinstance(ch_entry, dict) else None
            if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 3 and not np.isnan(loc[0]):
                pos = f"({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})"

        ch_info = {
            'Channel': ch_name,
            'Type': ch_type,
            'Unit': unit,
            'Position': pos
        }
        channel_info.append(ch_info)
    return pd.DataFrame(channel_info)

def create_raw_data_table(data, channel_names, sampling_rate, start_time=0, duration=5):
    """Create a raw data table showing EEG values"""
    # Calculate sample range
    start_sample = int(start_time * sampling_rate)
    end_sample = int((start_time + duration) * sampling_rate)
    end_sample = min(end_sample, data.shape[1])
    
    # Create time axis
    time_axis = np.arange(start_sample, end_sample) / sampling_rate
    
    # Create DataFrame with time and channel data
    df_data = {'Time (s)': time_axis}
    
    # Add each channel as a column
    for i, channel_name in enumerate(channel_names):
        if i < data.shape[0]:  # Make sure we don't exceed available channels
            df_data[channel_name] = data[i, start_sample:end_sample]
    
    df = pd.DataFrame(df_data)
    
    return df

# Main Streamlit app
def main():
    st.markdown('<h1 class="main-header">🧠 EEG Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    st.sidebar.header("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file (EDF or pickled MNE Raw)",
        type=['edf', 'pkl', 'pickle'],
        help="Upload a valid EDF (.edf) or a pickled MNE Raw object (.pkl / .pickle)"
    )
    
    # Example file option
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Example File:**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Load Example"):
            example_path = "Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf"
            try:
                uploaded_file = example_path
                st.session_state['example_loaded'] = True
                st.sidebar.success("Example loaded!")
            except:
                st.sidebar.error("Example not found!")
    with col2:
        if st.button("Clear"):
            uploaded_file = None
            st.session_state['example_loaded'] = False
            st.sidebar.info("Cleared!")
    
    # Auto-load example file on first visit
    if uploaded_file is None and not st.session_state.get('example_loaded', False):
        example_path = "Tests/edf_files/FIRST EXAMPLE PATIENTSHORT.edf"
        if os.path.exists(example_path):
            uploaded_file = example_path
            st.session_state['example_loaded'] = True
            st.sidebar.info("📁 Example file auto-loaded")
    
    if uploaded_file is not None:
        # Handle both uploaded files and example file path
        if isinstance(uploaded_file, str):
            file_path = uploaded_file
        else:
            # Save uploaded file temporarily
            with open("temp_uploaded_file.edf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_path = "temp_uploaded_file.edf"
        
        # Validate EDF file
        st.header("📋 File Validation")
        with st.spinner("Validating EDF file..."):
            validation_result = validate_edf_file(file_path)
        
        if validation_result['valid']:
            st.success("✅ Valid EDF file detected!")
            
            # Display file information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Channels", validation_result['n_channels'])
            with col2:
                st.metric("Duration", f"{validation_result['duration']:.1f}s")
            with col3:
                st.metric("Sampling Rate", f"{validation_result['sampling_rate']:.1f} Hz")
            with col4:
                st.metric("Samples", f"{validation_result['n_samples']:,}")
            
            # Display additional MNE file info
            if 'raw_info' in validation_result:
                st.subheader("📋 File Details")
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.write(f"**Patient ID:** {validation_result['file_info']['patient_id']}")
                    st.write(f"**Recording ID:** {validation_result['file_info']['recording_id']}")
                
                with info_col2:
                    st.write(f"**Recording Date:** {validation_result['file_info']['start_date']}")
                    st.write(f"**Experimenter:** {validation_result['file_info']['experimenter']}")
                
                with info_col3:
                    st.write(f"**Data Type:** {validation_result['raw_info'].get('data_type', 'Unknown')}")
                    st.write(f"**Description:** {validation_result['file_info']['description']}")
            
            # Load data
            st.header("📊 Data Analysis")
            with st.spinner("Loading EEG data..."):
                # If validation returned a raw object (e.g., pickled MNE Raw), use it directly
                if validation_result.get('raw_obj') is not None:
                    raw = validation_result['raw_obj']
                    try:
                        n_channels = len(raw.ch_names)
                        n_samples = len(raw.times)
                        sampling_rate = raw.info.get('sfreq', np.nan)
                        channels_to_load = min(n_channels, 20)
                        data = raw.get_data()[:channels_to_load, :]
                        channel_names = raw.ch_names[:channels_to_load]
                        sampling_rates = [sampling_rate] * channels_to_load
                        
                        eeg_data = {
                            'data': data,
                            'channel_names': channel_names,
                            'sampling_rates': sampling_rates,
                            'n_channels': channels_to_load,
                            'n_samples': n_samples,
                            'raw': raw
                        }
                    except Exception as e:
                        st.error(f"Error using provided Raw object: {e}")
                        eeg_data = None
                else:
                    eeg_data = load_edf_data(file_path)
            
            if eeg_data is not None:
                # Calculate quality metrics
                with st.spinner("Calculating quality metrics..."):
                    metrics, freqs, psd = calculate_eeg_quality_metrics(
                        eeg_data['data'], 
                        eeg_data['sampling_rates'][0]
                    )
                    artifacts = detect_artifacts(
                        eeg_data['data'], 
                        eeg_data['sampling_rates'][0]
                    )
                
                # Display quality metrics
                st.subheader("🔍 Signal Quality Assessment")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
                    st.metric("Signal-to-Noise Ratio", f"{metrics['snr_estimate']:.1f} dB")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
                    st.metric("Dynamic Range", f"{metrics['dynamic_range']:.1f} μV")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    artifact_ratio = metrics['high_amp_artifact_ratio']
                    if artifact_ratio < 0.01:
                        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
                        st.metric("Artifact Level", "Low", delta=f"{artifact_ratio*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif artifact_ratio < 0.05:
                        st.markdown('<div class="metric-card warning-metric">', unsafe_allow_html=True)
                        st.metric("Artifact Level", "Medium", delta=f"{artifact_ratio*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card error-metric">', unsafe_allow_html=True)
                        st.metric("Artifact Level", "High", delta=f"{artifact_ratio*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    flat_ratio = metrics['flat_line_ratio']
                    if flat_ratio < 0.01:
                        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
                        st.metric("Flat Line Ratio", "Good", delta=f"{flat_ratio*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card warning-metric">', unsafe_allow_html=True)
                        st.metric("Flat Line Ratio", "Poor", delta=f"{flat_ratio*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
                    st.metric("Kurtosis", f"{metrics['kurtosis']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
                    st.metric("Skewness", f"{metrics['skewness']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Frequency band analysis
                st.subheader("🌊 Frequency Band Analysis")
                band_col1, band_col2, band_col3, band_col4, band_col5 = st.columns(5)
                
                with band_col1:
                    st.metric("Delta (0.5-4 Hz)", f"{metrics['delta_power']:.2f}")
                with band_col2:
                    st.metric("Theta (4-8 Hz)", f"{metrics['theta_power']:.2f}")
                with band_col3:
                    st.metric("Alpha (8-13 Hz)", f"{metrics['alpha_power']:.2f}")
                with band_col4:
                    st.metric("Beta (13-30 Hz)", f"{metrics['beta_power']:.2f}")
                with band_col5:
                    st.metric("Gamma (30-100 Hz)", f"{metrics['gamma_power']:.2f}")
                
                # Visualization tabs
                st.subheader("📈 Visualizations")
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Raw Data Table", "Channel Info", "Raw EEG", "Spectrogram", "Power Spectrum", "Artifact Analysis"])
                
                with tab1:
                    st.subheader("Raw EEG Data Table")
                    st.markdown("**Interactive data table showing raw EEG values**")
                    
                    # Controls for data table
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        start_time = st.slider("Start Time (s)", 0.0, max(1.0, eeg_data['n_samples']/eeg_data['sampling_rates'][0]-5), 0.0, key="table_start")
                    with col2:
                        duration = st.slider("Duration (s)", 1.0, 10.0, 5.0, key="table_duration")
                    with col3:
                        max_rows = st.selectbox("Max Rows", [100, 500, 1000, 5000], index=0)
                    
                    # Create and display raw data table
                    raw_df = create_raw_data_table(
                        eeg_data['data'], 
                        eeg_data['channel_names'], 
                        eeg_data['sampling_rates'][0],
                        start_time, 
                        duration
                    )
                    
                    # Limit rows for performance
                    display_df = raw_df.head(max_rows)
                    
                    st.dataframe(
                        display_df, 
                        use_container_width=True,
                        height=400
                    )
                    
                    # Show data info and statistics
                    st.info(f"Showing {len(display_df)} rows of {len(raw_df)} total rows. Data range: {start_time:.1f}s - {start_time + duration:.1f}s")
                    
                    # Data statistics
                    st.subheader("📊 Data Statistics")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Sampling Rate", f"{eeg_data['sampling_rates'][0]:.1f} Hz")
                        st.metric("Total Channels", len(eeg_data['channel_names']))
                    
                    with stats_col2:
                        st.metric("Data Points", f"{len(raw_df):,}")
                        st.metric("Time Resolution", f"{1/eeg_data['sampling_rates'][0]*1000:.1f} ms")
                    
                    with stats_col3:
                        # Calculate some basic stats for the displayed data
                        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:  # More than just time column
                            mean_val = display_df[numeric_cols[1:]].mean().mean()
                            std_val = display_df[numeric_cols[1:]].std().mean()
                            st.metric("Mean Amplitude", f"{mean_val:.2f} μV")
                            st.metric("Std Deviation", f"{std_val:.2f} μV")
                    
                    # Export functionality
                    st.subheader("💾 Export Data")
                    csv_data = display_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"eeg_data_{start_time:.1f}s_to_{start_time + duration:.1f}s.csv",
                        mime="text/csv"
                    )
                
                with tab2:
                    st.subheader("Channel Information")
                    st.markdown("**Detailed information about EEG channels using MNE**")
                    
                    # Get channel information
                    if 'raw' in eeg_data:
                        channel_df = get_channel_info(eeg_data['raw'])
                        
                        # Display channel information table
                        st.dataframe(channel_df, use_container_width=True)
                        
                        # Channel type summary
                        st.subheader("📊 Channel Type Summary")
                        type_counts = channel_df['Type'].value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            for ch_type, count in type_counts.items():
                                st.metric(f"{ch_type} Channels", count)
                        
                        # Additional MNE info
                        st.subheader("🔬 MNE File Information")
                        info_col1, info_col2 = st.columns(2)
                        
                        # Show total duration explicitly and then list raw.info items dynamically
                        with info_col1:
                            try:
                                st.write(f"**Total Duration:** {eeg_data['raw'].times[-1]:.2f} seconds")
                            except Exception:
                                st.write("**Total Duration:** Unknown")
                            
                            info_items = list(eeg_data['raw'].info.items())
                            half = (len(info_items) + 1) // 2
                            for k, v in info_items[:half]:
                                st.write(f"**{k}:** {v}")
                        
                        with info_col2:
                            info_items = list(eeg_data['raw'].info.items())
                            half = (len(info_items) + 1) // 2
                            for k, v in info_items[half:]:
                                st.write(f"**{k}:** {v}")
                    else:
                        st.warning("Raw MNE object not available for detailed channel information")
                
                with tab3:
                    st.subheader("Raw EEG Signals")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        start_time = st.slider("Start Time (s)", 0.0, max(1.0, eeg_data['n_samples']/eeg_data['sampling_rates'][0]-10), 0.0, key="plot_start")
                    with col2:
                        duration = st.slider("Duration (s)", 1.0, 30.0, 10.0, key="plot_duration")
                    
                    fig_raw = plot_raw_eeg(
                        eeg_data['data'], 
                        eeg_data['channel_names'], 
                        eeg_data['sampling_rates'][0],
                        start_time, 
                        duration
                    )
                    st.plotly_chart(fig_raw, use_container_width=True)
                
                with tab4:
                    st.subheader("Spectrogram Analysis")
                    channel_idx = st.selectbox(
                        "Select Channel for Spectrogram",
                        range(len(eeg_data['channel_names'])),
                        format_func=lambda x: eeg_data['channel_names'][x]
                    )
                    
                    fig_spec = plot_spectrogram(
                        eeg_data['data'], 
                        eeg_data['channel_names'], 
                        eeg_data['sampling_rates'][0],
                        channel_idx
                    )
                    st.plotly_chart(fig_spec, use_container_width=True)
                
                with tab5:
                    st.subheader("Power Spectral Density")
                    fig_psd = plot_power_spectrum(psd, freqs, eeg_data['channel_names'])
                    st.plotly_chart(fig_psd, use_container_width=True)
                
                with tab6:
                    st.subheader("Artifact Detection Results")
                    
                    artifact_col1, artifact_col2, artifact_col3 = st.columns(3)
                    
                    with artifact_col1:
                        st.metric("Muscle Artifacts", f"{artifacts['muscle_artifacts']} channels")
                    with artifact_col2:
                        st.metric("Eye Movement Artifacts", f"{artifacts['eye_movement_artifacts']} channels")
                    with artifact_col3:
                        st.metric("Electrode Pops", f"{np.sum(artifacts['electrode_pops'])} total")
                    
                    # Artifact visualization
                    if len(eeg_data['channel_names']) > 0:
                        artifact_data = {
                            'Channel': eeg_data['channel_names'],
                            'Muscle Artifacts': [1 if i < artifacts['muscle_artifacts'] else 0 for i in range(len(eeg_data['channel_names']))],
                            'Eye Movement': [1 if i < artifacts['eye_movement_artifacts'] else 0 for i in range(len(eeg_data['channel_names']))],
                            'Electrode Pops': artifacts['electrode_pops']
                        }
                        
                        df_artifacts = pd.DataFrame(artifact_data)
                        st.dataframe(df_artifacts, use_container_width=True)
                
                # Summary and recommendations
                st.subheader("📋 Analysis Summary")
                
                # Compute numeric indicators
                n_channels = len(eeg_data['channel_names'])
                snr = metrics.get('snr_estimate', float('nan'))
                high_amp_ratio = metrics.get('high_amp_artifact_ratio', float('nan'))
                flat_ratio = metrics.get('flat_line_ratio', float('nan'))
                kurt = metrics.get('kurtosis', float('nan'))
                skewness = metrics.get('skewness', float('nan'))
                delta = metrics.get('delta_power', float('nan'))
                theta = metrics.get('theta_power', float('nan'))
                alpha = metrics.get('alpha_power', float('nan'))
                beta = metrics.get('beta_power', float('nan'))
                gamma = metrics.get('gamma_power', float('nan'))
                alpha_beta_ratio = (alpha / beta) if (beta and beta > 0) else float('nan')
                
                muscle_count = artifacts.get('muscle_artifacts', 0)
                eye_count = artifacts.get('eye_movement_artifacts', 0)
                epops = artifacts.get('electrode_pops', 0)
                try:
                    electrode_pops_total = int(np.sum(epops))
                except Exception:
                    electrode_pops_total = int(epops) if isinstance(epops, (int, float)) else 0
                
                muscle_ratio = muscle_count / n_channels if n_channels > 0 else float('nan')
                eye_ratio = eye_count / n_channels if n_channels > 0 else float('nan')
                
                # Numeric summary display
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                with sum_col1:
                    st.metric("SNR (dB)", f"{snr:.1f}" if np.isfinite(snr) else "N/A")
                    st.metric("Channels", n_channels)
                    st.metric("High-amp artifact %", f"{high_amp_ratio*100:.2f}%")
                with sum_col2:
                    st.metric("Flat-line %", f"{flat_ratio*100:.2f}%")
                    st.metric("Muscle artifact channels", f"{muscle_count}")
                    st.metric("Eye artifact channels", f"{eye_count}")
                with sum_col3:
                    st.metric("Alpha/Beta ratio", f"{alpha_beta_ratio:.2f}" if not np.isnan(alpha_beta_ratio) else "N/A")
                    st.metric("Kurtosis", f"{kurt:.2f}")
                    st.metric("Skewness", f"{skewness:.2f}")
                
                # Compute a weighted quality score (0-100)
                score = 0.0
                # SNR contribution (0..40) — 20 dB maps to full 40
                if np.isfinite(snr):
                    score += min(max((snr / 20.0) * 40.0, 0), 40)
                # Artifact contribution (0..30) — lower is better
                if np.isfinite(high_amp_ratio):
                    score += max(0, 30 - min(high_amp_ratio * 100 * 10, 30))
                else:
                    score += 15
                # Flat-line contribution (0..15)
                if np.isfinite(flat_ratio):
                    score += max(0, 15 - min(flat_ratio * 100 * 150, 15))
                else:
                    score += 7.5
                # Muscle+Eye contribution (0..15)
                score += max(0, 15 - min((muscle_ratio + eye_ratio) * 100 * 5, 15))
                
                quality_percent = float(np.clip(score, 0, 100))
                
                # Show overall quality and numeric percent
                if quality_percent >= 75:
                    st.success(f"✅ **Overall Quality: EXCELLENT** — {quality_percent:.0f}%")
                elif quality_percent >= 50:
                    st.warning(f"⚠️ **Overall Quality: GOOD** — {quality_percent:.0f}%")
                else:
                    st.error(f"❌ **Overall Quality: POOR** — {quality_percent:.0f}%")
                
                # Detailed explanation in an expander
                with st.expander("Show analysis details & scoring logic"):
                    st.markdown("### Scoring breakdown (higher is better)")
                    st.write("The dashboard combines multiple indicators into a single 0–100 score. Below are the computed numeric values used in scoring:")
                    details = {
                        "SNR (dB)": snr,
                        "High_amp_artifact_ratio": high_amp_ratio,
                        "Flat_line_ratio": flat_ratio,
                        "Muscle_artifact_ratio": muscle_ratio,
                        "Eye_artifact_ratio": eye_ratio,
                        "Alpha_power": alpha,
                        "Beta_power": beta,
                        "Alpha/Beta": alpha_beta_ratio,
                        "Kurtosis": kurt,
                        "Skewness": skewness,
                        "Electrode_pops_total": electrode_pops_total
                    }
                    # Pretty table
                    details_df = pd.DataFrame.from_dict(details, orient='index', columns=['Value'])
                    st.dataframe(details_df, use_container_width=True, height=240)
                    
                    st.markdown("#### Score formula (summary)")
                    st.markdown("- SNR: up to 40 points (20 dB → 40 points)")
                    st.markdown("- High-amplitude artifacts: up to 30 points (lower is better)")
                    st.markdown("- Flat-lines: up to 15 points (lower is better)")
                    st.markdown("- Muscle + Eye artifacts: up to 15 points (lower is better)")
                    st.markdown(f"**Final quality score:** {quality_percent:.1f}%")
                    st.markdown("**Thresholds used for recommendations:**")
                    st.write("- High-amplitude artifact ratio > 5% considered high")
                    st.write("- Flat-line ratio > 1% considered problematic")
                    st.write("- Muscle/Eye artifacts affecting >30% of channels considered high")
                
                # Recommendations (more informative/numeric)
                st.subheader("💡 Recommendations")
                recommendations = []
                
                if high_amp_ratio > 0.05:
                    recommendations.append(f"Consider artifact removal techniques for high-amplitude artifacts (detected {high_amp_ratio*100:.2f}%).")
                
                if flat_ratio > 0.01:
                    recommendations.append(f"Check electrode connectivity - flat-line ratio is {flat_ratio*100:.2f}%.")
                
                if muscle_ratio > 0.3:
                    recommendations.append(f"Muscle artifacts on {muscle_count}/{n_channels} channels — consider filtering or artifact rejection.")
                
                if eye_ratio > 0.3:
                    recommendations.append(f"Eye-movement artifacts on {eye_count}/{n_channels} channels — consider EOG correction.")
                
                if not np.isnan(alpha_beta_ratio) and alpha_beta_ratio < 0.5:
                    recommendations.append(f"Low alpha/beta ratio ({alpha_beta_ratio:.2f}) — may indicate reduced alpha activity.")
                
                if electrode_pops_total > 0:
                    recommendations.append(f"Detected {electrode_pops_total} electrode pop events — inspect raw traces around spikes.")
                
                if not recommendations:
                    recommendations.append("No specific recommendations - data quality is good.")
                
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
        
        else:
            st.error(f"❌ Invalid EDF file: {validation_result['error']}")
    
    else:
        st.info("👆 Please upload an EDF file to begin analysis")
        
        # Show example of what the dashboard can do
        st.markdown("---")
        st.subheader("🔬 What This Dashboard Does")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **File Validation:**
            - Validates EDF file format
            - Checks file integrity
            - Extracts metadata
            
            **Signal Quality Assessment:**
            - Signal-to-noise ratio
            - Dynamic range analysis
            - Artifact detection
            - Flat line detection
            """)
        
        with col2:
            st.markdown("""
            **Advanced Analysis:**
            - Frequency band analysis
            - Power spectral density
            - Spectrogram visualization
            - Raw signal plotting
            
            **State-of-the-art Tests:**
            - Muscle artifact detection
            - Eye movement detection
            - Electrode pop detection
            - Signal quality metrics
            """)

if __name__ == "__main__":
    main()
