import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson
from io import StringIO, BytesIO
import re
from datetime import datetime
import base64

# Set page config with custom theme
st.set_page_config(
    page_title="SpectrAnalys",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern scientific design
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --background-color: #f8f9fa;
        --card-background: #ffffff;
        --text-color: #2c3e50;
        --border-color: #e0e0e0;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .custom-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Card styling */
    .scientific-card {
        background: var(--card-background);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .scientific-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--secondary-color);
        display: inline-block;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--background-color);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--background-color);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--background-color);
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Status messages */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        border-top: 1px solid var(--border-color);
        color: #666;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Apply scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#2c3e50',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'xtick.color': '#2c3e50',
    'ytick.color': '#2c3e50',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.major.size': 6,
    'ytick.minor.size': 3,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#2c3e50',
    'legend.fancybox': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.facecolor': 'white',
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'errorbar.capsize': 3,
})

# Function to load spectrum data
@st.cache_data
def load_spectrum(uploaded_file):
    """Load spectrum from uploaded file"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        data = pd.read_csv(StringIO(content), sep='\t', header=None, names=['x', 'y'])
        return data.dropna().sort_values('x').reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {e}")
        return None

# Function to normalize spectrum
def normalize_spectrum(x, y, norm_method, norm_range=None):
    """Normalize spectrum using different methods"""
    if norm_method == "Maximum intensity":
        return y / y.max() if y.max() != 0 else y
    
    elif norm_method == "Area":
        area = simpson(y, x)
        if area != 0:
            return y / area
        return y
    
    elif norm_method == "Peak intensity (range)":
        if norm_range is not None:
            mask = (x >= norm_range[0]) & (x <= norm_range[1])
            if np.any(mask):
                max_in_range = y[mask].max()
                if max_in_range != 0:
                    return y / max_in_range
        return y / y.max() if y.max() != 0 else y
    
    return y

# Function to align x ranges
def align_x_ranges(spectra_dict):
    """Align all spectra to common x range"""
    if not spectra_dict:
        return spectra_dict
    
    # Find common x range
    min_x = max([spec['data']['x'].min() for spec in spectra_dict.values()])
    max_x = min([spec['data']['x'].max() for spec in spectra_dict.values()])
    
    if min_x >= max_x:
        return spectra_dict
    
    # Interpolate all spectra to common x grid
    common_x = np.linspace(min_x, max_x, 1000)
    
    aligned_spectra = {}
    for name, spec in spectra_dict.items():
        x_orig = spec['data']['x'].values
        y_orig = spec['data']['y'].values
        y_interp = np.interp(common_x, x_orig, y_orig)
        
        aligned_spectra[name] = {
            'data': pd.DataFrame({'x': common_x, 'y': y_interp}),
            'original_data': spec['data'],
            'color': spec['color']
        }
    
    return aligned_spectra

# Function to extract x ranges from string
def parse_x_ranges(range_str):
    """Parse x ranges from string like '100-200, 300-400'"""
    if not range_str or range_str == "":
        return None
    
    ranges = []
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = part.split('-')
                ranges.append((float(start), float(end)))
            except:
                continue
    
    return ranges if ranges else None

# Function to crop spectrum to ranges and create data for broken axis
def crop_to_ranges_multi(x, y, ranges):
    """Crop spectrum to multiple ranges and return list of (x_segment, y_segment)"""
    if ranges is None:
        return [(x, y)]
    
    segments = []
    for start, end in ranges:
        mask = (x >= start) & (x <= end)
        if np.any(mask):
            segments.append((x[mask], y[mask]))
    
    return segments

# Function to create combined plot with all four visualization types
# Function to create combined plot with all four visualization types
def create_combined_plot(spectra_dict, x_label, y_label, title,
                         raw_offset_step, norm_offset_step, fill_area,
                         norm_method, x_ranges=None):
    """Create scientific plot with all four visualization types in subplots"""
    
    # Prepare normalized spectra
    normalized_spectra = {}
    for name, spec in spectra_dict.items():
        data = spec['data']
        y_norm = normalize_spectrum(
            data['x'].values,
            data['y'].values,
            norm_method,
            None
        )
        normalized_spectra[name] = {
            'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
            'color': spec['color']
        }
    
    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Define the four visualization types
    viz_configs = [
        (axes[0, 0], "Raw Spectra", spectra_dict, 0, False, False, False, x_label, y_label),
        (axes[0, 1], f"Normalized Spectra ({norm_method})", normalized_spectra, 0, False, True, False, x_label, f"Normalized Intensity ({norm_method})"),
        (axes[1, 0], f"Raw Spectra + Offset (step = {raw_offset_step})", spectra_dict, raw_offset_step, False, False, True, x_label, y_label),
        (axes[1, 1], f"Normalized Spectra + Offset (step = {norm_offset_step})", normalized_spectra, norm_offset_step, fill_area, True, True, x_label, f"Normalized Intensity ({norm_method})")
    ]
    
    for ax, subplot_title, spectra, offset_step, fill, normalized, use_offset, xl, yl in viz_configs:
        # Store handles and labels for legend
        handles = []
        labels = []
        
        spectra_items = list(spectra.items())
        
        if x_ranges is None or len(x_ranges) == 0:
            # Simple plot without broken axis
            for idx, (name, spec) in enumerate(spectra_items):
                data = spec['data']
                x = data['x'].values
                y = data['y'].values
                color = spec['color']
                
                display_name = name.replace('.txt', '')
                
                # Apply cumulative offset if requested
                if use_offset:
                    offset = idx * offset_step
                else:
                    offset = 0
                
                y_plot = y + offset
                
                if fill and normalized:
                    ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
                else:
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
                
                handles.append(line_handle[0])
                labels.append(display_name)
            
            ax.set_xlabel(xl, fontsize=10, fontweight='bold')
            ax.set_ylabel(yl, fontsize=10, fontweight='bold')
            ax.set_title(subplot_title, fontsize=11, fontweight='bold')
            
        else:
            # Broken axis plot with multiple x-ranges
            n_ranges = len(x_ranges)
            
            for range_idx, (start, end) in enumerate(x_ranges):
                for idx, (name, spec) in enumerate(spectra_items):
                    data = spec['data']
                    x_full = data['x'].values
                    y_full = data['y'].values
                    color = spec['color']
                    
                    display_name = name.replace('.txt', '')
                    
                    # Crop to current range
                    mask = (x_full >= start) & (x_full <= end)
                    if not np.any(mask):
                        continue
                    
                    x = x_full[mask]
                    y = y_full[mask]
                    
                    # Apply cumulative offset if requested
                    if use_offset:
                        offset = idx * offset_step
                    else:
                        offset = 0
                    
                    y_plot = y + offset
                    
                    # Plot
                    if fill and normalized and use_offset:
                        ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    elif fill and normalized:
                        ax.fill_between(x, 0, y_plot, alpha=0.3, color=color)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    else:
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    
                    # Add to legend only for first range
                    if range_idx == 0 and idx == 0:
                        handles.append(line_handle[0])
                        labels.append(display_name)
                
                # Add vertical line for range boundaries
                ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
                ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            
            ax.set_xlabel(xl, fontsize=10, fontweight='bold')
            ax.set_ylabel(yl, fontsize=10, fontweight='bold')
            ax.set_title(subplot_title, fontsize=11, fontweight='bold')
        
        # Add legend with correct order for offset plots
        if handles:
            # For offset plots, reverse the legend order so top curve appears first
            if use_offset:
                legend = ax.legend(reversed(handles), reversed(labels), loc='best', fontsize=8,
                                  frameon=True, edgecolor='black', prop={'weight': 'bold'})
            else:
                legend = ax.legend(handles, labels, loc='best', fontsize=8,
                                  frameon=True, edgecolor='black', prop={'weight': 'bold'})
            
            # Set legend text colors to match line colors
            for text, handle in zip(legend.get_texts(), handles):
                text.set_color(handle.get_color())
        
        ax.tick_params(direction='in', length=5, width=1)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.25)
    
    return fig

# Function for peak analysis
def analyze_peaks(spectra_dict, x_ranges=None, peak_width=20):
    """Analyze peaks in spectra"""
    results = []
    
    for name, spec in spectra_dict.items():
        data = spec['data']
        x = data['x'].values
        y = data['y'].values
        
        if x_ranges is not None:
            # Crop to ranges for peak analysis
            x_cropped = []
            y_cropped = []
            for start, end in x_ranges:
                mask = (x >= start) & (x <= end)
                if np.any(mask):
                    x_cropped.extend(x[mask])
                    y_cropped.extend(y[mask])
            x = np.array(x_cropped)
            y = np.array(y_cropped)
        
        if len(x) == 0:
            continue
        
        # Find peaks
        peaks, properties = find_peaks(y, height=np.max(y)*0.1, prominence=np.max(y)*0.05)
        
        for peak_idx in peaks:
            peak_x = x[peak_idx]
            peak_y = y[peak_idx]
            
            # Calculate area around peak
            left_idx = max(0, peak_idx - peak_width)
            right_idx = min(len(x), peak_idx + peak_width)
            area = simpson(y[left_idx:right_idx+1], x[left_idx:right_idx+1])
            
            results.append({
                'Spectrum': name.replace('.txt', ''),
                'Peak position': peak_x,
                'Intensity': peak_y,
                'Area': area
            })
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# Main app
def main():
    # Custom header
    st.markdown("""
    <div class="custom-header">
        <h1>🔬 SpectrAnalys</h1>
        <p>Advanced Spectroscopic Data Analysis Platform | High-Precision Peak Detection & Correlation Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Data Import")
        uploaded_files = st.file_uploader(
            "Upload spectra files (.txt format, tab-separated)",
            type=['txt'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.success(f"✅ Loaded {len(uploaded_files)} files")
            
            # Load data
            spectra_data = {}
            for file in uploaded_files:
                data = load_spectrum(file)
                if data is not None:
                    spectra_data[file.name] = {
                        'data': data,
                        'color': None
                    }
            
            if spectra_data:
                st.markdown("---")
                st.markdown("### 📋 Spectrum Selection")
                
                # Select and order spectra
                selected_spectra = st.multiselect(
                    "Choose spectra to display",
                    options=list(spectra_data.keys()),
                    default=list(spectra_data.keys())
                )
                
                if selected_spectra:
                    # Order spectra
                    ordered_spectra = []
                    for name in selected_spectra:
                        ordered_spectra.append(name)
                    
                    # Assign colors with default distinct colors
                    colors = {}
                    st.markdown("---")
                    st.markdown("### 🎨 Color Assignment")
                    
                    # Define default color palette
                    default_colors = [
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
                    ]
                    
                    for i, name in enumerate(ordered_spectra):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{name.replace('.txt', '')}**")
                        with col2:
                            default_color = default_colors[i % len(default_colors)]
                            colors[name] = st.color_picker(
                                f"Color {i+1}",
                                value=default_color,
                                key=f"color_{name}"
                            )
                    
                    # Update spectra data with colors
                    for name in ordered_spectra:
                        spectra_data[name]['color'] = colors[name]
                    
                    st.markdown("---")
                    st.markdown("### ⚙️ Processing Options")
                    
                    # Common x range option
                    common_x_range = st.checkbox("Align all spectra to common x range", value=False)
                    
                    # X-axis ranges
                    st.markdown("#### 📊 X-axis Ranges")
                    x_range_option = st.radio(
                        "Select range mode",
                        ["Full range", "Custom ranges (multiple)"]
                    )
                    
                    x_ranges = None
                    if x_range_option == "Custom ranges (multiple)":
                        range_input = st.text_area(
                            "Enter ranges (e.g., 100-150, 350-450, 600-800)",
                            placeholder="100-150, 350-450, 600-800",
                            help="Each range will be displayed as a separate segment on the same graph"
                        )
                        if range_input:
                            x_ranges = parse_x_ranges(range_input)
                            if x_ranges:
                                st.info(f"📌 Selected {len(x_ranges)} ranges: {', '.join([f'{r[0]:.0f}-{r[1]:.0f}' for r in x_ranges])}")
                    
                    # Axis labels
                    st.markdown("#### 🏷️ Axis Labels")
                    x_label = st.text_input("X-axis label", value="Raman shift (cm⁻¹)")
                    y_label = st.text_input("Y-axis label", value="Intensity (a.u.)")
                    
                    # Normalization options
                    st.markdown("#### 📐 Normalization")
                    norm_method = st.selectbox(
                        "Normalization method",
                        ["Maximum intensity", "Area", "Peak intensity (range)"],
                        index=0
                    )
                    
                    norm_range = None
                    if norm_method == "Peak intensity (range)":
                        norm_range_input = st.text_input(
                            "Peak range for normalization (e.g., 800-1000)",
                            placeholder="800-1000"
                        )
                        if norm_range_input:
                            try:
                                start, end = norm_range_input.split('-')
                                norm_range = (float(start), float(end))
                            except:
                                st.warning("Invalid range format")
                    
                    # Offset options
                    st.markdown("#### 📈 Offset Settings")
                    col1, col2 = st.columns(2)
                    with col1:
                        raw_offset_step = st.slider(
                            "Raw spectra offset step",
                            min_value=0.0,
                            max_value=50000.0,
                            value=1000.0,
                            step=100.0,
                            key="raw_offset_step"
                        )
                    with col2:
                        norm_offset_step = st.slider(
                            "Normalized spectra offset step",
                            min_value=0.0,
                            max_value=5.0,
                            value=0.5,
                            step=0.05,
                            key="norm_offset_step"
                        )
                    
                    # Fill area option
                    fill_area = st.checkbox("Fill area under normalized spectra", value=False)
                    
                    # Peak analysis options
                    st.markdown("---")
                    st.markdown("### 🔍 Peak Analysis")
                    analyze_peaks_flag = st.checkbox("Enable advanced peak analysis", value=False)
                    
                    if analyze_peaks_flag:
                        peak_width = st.slider(
                            "Peak width for area calculation (points)",
                            min_value=5,
                            max_value=100,
                            value=20,
                            step=5
                        )
                    
                    # Parameter correlation
                    st.markdown("---")
                    st.markdown("### 📊 Parameter Correlation")
                    param_correlation = st.checkbox("Enable correlation analysis", value=False)
                    
                    if param_correlation:
                        st.info("💡 Assign numeric values to each spectrum for correlation analysis")
                        param_values = {}
                        for name in ordered_spectra:
                            param_values[name] = st.number_input(
                                f"Value for {name.replace('.txt', '')}",
                                value=float(len(param_values) + 1),
                                step=1.0,
                                key=f"param_{name}"
                            )
                        
                        param_label = st.text_input("Parameter label", value="Sample number")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar footer with info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p>🔬 SpectrAnalys v2.0<br>Scientific Spectroscopic Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if uploaded_files and 'spectra_data' in locals() and spectra_data and ordered_spectra:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(ordered_spectra)}</div>
                <div class="metric-label">Spectra Loaded</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if x_ranges:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(x_ranges)}</div>
                    <div class="metric-label">X-axis Ranges</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">Full</div>
                    <div class="metric-label">X-axis Range</div>
                </div>
                """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{norm_method[:10]}</div>
                <div class="metric-label">Normalization</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'✓' if analyze_peaks_flag else '✗'}</div>
                <div class="metric-label">Peak Analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Apply common x range if selected
        current_spectra = spectra_data
        if common_x_range:
            current_spectra = align_x_ranges(current_spectra)
        
        # Filter spectra based on selection
        filtered_spectra = {name: current_spectra[name] for name in ordered_spectra if name in current_spectra}
        
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs([
            "📊 Combined Spectra Visualization",
            "🔍 Advanced Peak Analysis", 
            "📈 Parameter Correlation"
        ])
        
        with tab1:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Comprehensive Spectra Analysis")
            st.markdown("*All visualization modes combined for comprehensive spectral comparison*")
            
            if filtered_spectra:
                fig = create_combined_plot(
                    filtered_spectra, x_label, y_label,
                    "SpectrAnalys - Multi-Mode Spectral Visualization",
                    raw_offset_step, norm_offset_step, fill_area,
                    norm_method, x_ranges
                )
                st.pyplot(fig)
                plt.close()
                
                # Download button for combined plot
                from io import BytesIO
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode()
                st.markdown(f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <a href="data:image/png;base64,{b64}" download="spectra_combined_plot.png">
                        <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       color: white; border: none; border-radius: 8px; 
                                       padding: 0.5rem 1rem; cursor: pointer;">
                            📥 Download Combined Plot (PNG)
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No spectra selected for visualization")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Peak Detection and Analysis")
            
            if analyze_peaks_flag and filtered_spectra:
                peaks_df = analyze_peaks(filtered_spectra, x_ranges, peak_width)
                if not peaks_df.empty:
                    # Display peak statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Peaks Detected", len(peaks_df))
                    with col2:
                        st.metric("Unique Spectra", peaks_df['Spectrum'].nunique())
                    with col3:
                        st.metric("Avg Peak Intensity", f"{peaks_df['Intensity'].mean():.2f}")
                    
                    st.markdown("---")
                    st.dataframe(peaks_df, use_container_width=True)
                    
                    # Download button for peak analysis
                    csv = peaks_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download peak analysis as CSV",
                        data=csv,
                        file_name=f"peak_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualize peaks
                    st.markdown("---")
                    st.subheader("Peak Visualization")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for name, spec in filtered_spectra.items():
                        data = spec['data']
                        x = data['x'].values
                        y = data['y'].values
                        
                        if x_ranges is not None:
                            # Crop to ranges for visualization
                            x_cropped = []
                            y_cropped = []
                            for start, end in x_ranges:
                                mask = (x >= start) & (x <= end)
                                if np.any(mask):
                                    x_cropped.extend(x[mask])
                                    y_cropped.extend(y[mask])
                            x = np.array(x_cropped)
                            y = np.array(y_cropped)
                        
                        if len(x) == 0:
                            continue
                        
                        ax.plot(x, y, color=spec['color'], linewidth=1.5, label=name.replace('.txt', ''), alpha=0.7)
                        
                        # Mark peaks
                        spec_peaks = peaks_df[peaks_df['Spectrum'] == name.replace('.txt', '')]
                        for _, peak in spec_peaks.iterrows():
                            ax.axvline(peak['Peak position'], color=spec['color'], 
                                      linestyle='--', alpha=0.5, linewidth=1)
                            ax.text(peak['Peak position'], peak['Intensity']*0.8, 
                                   f"{peak['Peak position']:.1f}", 
                                   fontsize=8, ha='center', fontweight='bold')
                    
                    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
                    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
                    ax.set_title("Detected Peaks Visualization", fontsize=12, fontweight='bold')
                    ax.legend(loc='best', fontsize=10, frameon=True, edgecolor='black', prop={'weight': 'bold'})
                    ax.tick_params(direction='in', length=5, width=1)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("ℹ️ No peaks detected in the selected range. Try adjusting the x-axis ranges or peak detection parameters.")
            else:
                st.info("🔍 Enable advanced peak analysis in the sidebar to detect and analyze peaks in your spectra.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Parameter Correlation Analysis")
            
            if param_correlation and filtered_spectra and param_values:
                # Prepare data for correlation
                param_list = []
                intensity_list = []
                area_list = []
                position_list = []
                
                # Get peak analysis results if available
                if analyze_peaks_flag and 'peaks_df' in locals() and not peaks_df.empty:
                    for name in ordered_spectra:
                        if name in param_values:
                            spec_peaks = peaks_df[peaks_df['Spectrum'] == name.replace('.txt', '')]
                            if not spec_peaks.empty:
                                # Take the most intense peak
                                main_peak = spec_peaks.loc[spec_peaks['Intensity'].idxmax()]
                                param_list.append(param_values[name])
                                intensity_list.append(main_peak['Intensity'])
                                area_list.append(main_peak['Area'])
                                position_list.append(main_peak['Peak position'])
                
                if param_list:
                    # Calculate correlation coefficients
                    from scipy.stats import pearsonr
                    
                    corr_intensity = pearsonr(param_list, intensity_list)[0] if len(param_list) > 2 else 0
                    corr_area = pearsonr(param_list, area_list)[0] if len(param_list) > 2 else 0
                    corr_position = pearsonr(param_list, position_list)[0] if len(param_list) > 2 else 0
                    
                    # Display correlation metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Intensity Correlation", f"{corr_intensity:.3f}", 
                                 delta="strong" if abs(corr_intensity) > 0.7 else "weak")
                    with col2:
                        st.metric("Area Correlation", f"{corr_area:.3f}",
                                 delta="strong" if abs(corr_area) > 0.7 else "weak")
                    with col3:
                        st.metric("Position Correlation", f"{corr_position:.3f}",
                                 delta="strong" if abs(corr_position) > 0.7 else "weak")
                    
                    st.markdown("---")
                    
                    # Create correlation plots
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].scatter(param_list, intensity_list, c='#1f77b4', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                    axes[0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                    axes[0].set_ylabel("Peak Intensity (a.u.)", fontsize=11, fontweight='bold')
                    axes[0].set_title(f"Intensity vs {param_label}\n(r = {corr_intensity:.3f})", fontsize=12, fontweight='bold')
                    axes[0].grid(True, alpha=0.3, linestyle='--')
                    
                    axes[1].scatter(param_list, area_list, c='#2ca02c', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                    axes[1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                    axes[1].set_ylabel("Peak Area", fontsize=11, fontweight='bold')
                    axes[1].set_title(f"Area vs {param_label}\n(r = {corr_area:.3f})", fontsize=12, fontweight='bold')
                    axes[1].grid(True, alpha=0.3, linestyle='--')
                    
                    axes[2].scatter(param_list, position_list, c='#d62728', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                    axes[2].set_xlabel(param_label, fontsize=11, fontweight='bold')
                    axes[2].set_ylabel("Peak Position (cm⁻¹)", fontsize=11, fontweight='bold')
                    axes[2].set_title(f"Position vs {param_label}\n(r = {corr_position:.3f})", fontsize=12, fontweight='bold')
                    axes[2].grid(True, alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show correlation table
                    st.markdown("---")
                    st.subheader("Correlation Data Table")
                    corr_data = pd.DataFrame({
                        'Spectrum': [name.replace('.txt', '') for name in ordered_spectra if name in param_values],
                        param_label: param_list,
                        'Intensity': intensity_list,
                        'Area': area_list,
                        'Position': position_list
                    })
                    st.dataframe(corr_data, use_container_width=True)
                    
                    # Download button
                    csv = corr_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download correlation data as CSV",
                        data=csv,
                        file_name=f"correlation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("ℹ️ Enable peak analysis and ensure peaks are detected to perform correlation analysis")
            else:
                st.info("📊 Enable parameter correlation in the sidebar and assign numeric values to spectra for correlation analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export options section
        st.markdown("---")
        st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
        st.subheader("📤 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export raw data
            if filtered_spectra:
                export_data = pd.DataFrame()
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    export_data[f"{name.replace('.txt', '')}_x"] = data['x']
                    export_data[f"{name.replace('.txt', '')}_y"] = data['y']
                
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Export Raw Data (CSV)",
                    data=csv,
                    file_name=f"raw_spectra_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Export normalized data
            if filtered_spectra:
                export_norm = pd.DataFrame()
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    y_norm = normalize_spectrum(data['x'].values, data['y'].values, norm_method, norm_range)
                    export_norm[f"{name.replace('.txt', '')}_x"] = data['x']
                    export_norm[f"{name.replace('.txt', '')}_y_norm"] = y_norm
                
                csv_norm = export_norm.to_csv(index=False)
                st.download_button(
                    label="📥 Export Normalized Data (CSV)",
                    data=csv_norm,
                    file_name=f"normalized_spectra_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Export session info
            session_info = f"""SpectrAnalys Analysis Session
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Spectra Files: {', '.join(ordered_spectra)}
Normalization Method: {norm_method}
X-axis Ranges: {x_ranges if x_ranges else 'Full range'}
Raw Offset Step: {raw_offset_step}
Normalized Offset Step: {norm_offset_step}
Fill Area: {fill_area}
Peak Analysis: {analyze_peaks_flag}
Correlation Analysis: {param_correlation}
"""
            st.download_button(
                label="📄 Export Session Info",
                data=session_info,
                file_name=f"spectranalys_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Welcome screen with instructions
        st.markdown("## 🎯 Welcome to SpectrAnalys")
        st.markdown("Advanced spectroscopic data analysis platform for researchers and scientists")
        
        st.markdown("### 📖 Quick Start Guide:")
        st.markdown("""
        1. **Upload Data** - Select one or more .txt files with two columns (x y, tab-separated)
        2. **Configure Analysis** - Choose spectra, assign colors, set normalization and offset parameters
        3. **Visualize** - Explore combined spectra visualization with multiple display modes
        4. **Analyze Peaks** - Detect and characterize spectral peaks automatically
        5. **Correlate Parameters** - Investigate relationships between spectral features and experimental parameters
        6. **Export Results** - Download processed data, plots, and analysis results
        """)
        
        st.markdown("### ✨ Key Features:")
        st.markdown("""
        - 🔬 **Multi-Mode Visualization** - Raw, normalized, and offset spectra in one comprehensive view
        - 📊 **Broken Axis Support** - Display multiple x-axis ranges with gaps between them
        - 🎨 **Customizable Colors** - Individual color assignment for each spectrum
        - 📈 **Automatic Peak Detection** - Find peaks, calculate areas, and analyze intensities
        - 🔗 **Parameter Correlation** - Correlate spectral features with experimental parameters
        - 💾 **Data Export** - Download processed data in CSV format with publication-ready plots
        - 📐 **Multiple Normalization Methods** - Maximum intensity, area, or custom peak range normalization
        - 📏 **Cumulative Offset** - Add offsets to spectra for clear visualization (1st: 0, 2nd: +step, 3rd: +2×step)
        """)
        
        st.markdown("### 📁 File Format:")
        st.markdown("Your .txt files should contain two columns separated by tabs:")
        st.code("""
100.5    1250.3
101.2    1248.7
102.0    1251.5
...
        """, language="text")
        
        st.markdown("**Ready to analyze your spectra? 👈 Upload your files using the sidebar to get started!**")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🔬 SpectrAnalys v2.0 | Scientific Spectroscopic Analysis Platform | Built with Streamlit & Python</p>
        <p style="font-size: 0.75rem;">© 2024 SpectrAnalys - Advanced Spectroscopy Data Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
