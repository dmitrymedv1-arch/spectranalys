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
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Set page config with custom theme
st.set_page_config(
    page_title="SpectrAnalys",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for independent tabs
if 'peak_analysis_triggered' not in st.session_state:
    st.session_state.peak_analysis_triggered = False
if 'peak_analysis_results' not in st.session_state:
    st.session_state.peak_analysis_results = None
if 'peak_analysis_x_range' not in st.session_state:
    st.session_state.peak_analysis_x_range = (None, None)
if 'correlation_ready' not in st.session_state:
    st.session_state.correlation_ready = False
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = None
if 'spectra_loaded' not in st.session_state:
    st.session_state.spectra_loaded = False
if 'cached_spectra_data' not in st.session_state:
    st.session_state.cached_spectra_data = None
if 'excluded_peaks' not in st.session_state:
    st.session_state.excluded_peaks = set()

# NEW: Heatmap session state variables
if 'heatmap_params' not in st.session_state:
    st.session_state.heatmap_params = {}
if 'heatmap_param_type' not in st.session_state:
    st.session_state.heatmap_param_type = 'Temperature (°C)'
if 'heatmap_interpolation' not in st.session_state:
    st.session_state.heatmap_interpolation = 'gaussian'
if 'heatmap_colormap' not in st.session_state:
    st.session_state.heatmap_colormap = 'viridis'
if 'heatmap_applied' not in st.session_state:
    st.session_state.heatmap_applied = False
if 'heatmap_spectra_matrix' not in st.session_state:
    st.session_state.heatmap_spectra_matrix = None
if 'heatmap_spectra_norm_matrix' not in st.session_state:
    st.session_state.heatmap_spectra_norm_matrix = None
if 'heatmap_x_grid' not in st.session_state:
    st.session_state.heatmap_x_grid = None
if 'heatmap_y_values' not in st.session_state:
    st.session_state.heatmap_y_values = None
if 'heatmap_ordered_names' not in st.session_state:
    st.session_state.heatmap_ordered_names = []
if 'heatmap_y_label' not in st.session_state:
    st.session_state.heatmap_y_label = 'Temperature (°C)'
if 'heatmap_x_ranges' not in st.session_state:
    st.session_state.heatmap_x_ranges = None

# NEW: Spectral Markers session state variables
if 'spectral_markers' not in st.session_state:
    st.session_state.spectral_markers = []  # List of dicts: {type, position, width, name, color}
if 'spectral_markers_selected_plot' not in st.session_state:
    st.session_state.spectral_markers_selected_plot = 0
if 'spectral_markers_preview_position' not in st.session_state:
    st.session_state.spectral_markers_preview_position = None
if 'spectral_markers_preview_width' not in st.session_state:
    st.session_state.spectral_markers_preview_width = 0
if 'spectral_markers_show_values' not in st.session_state:
    st.session_state.spectral_markers_show_values = True
if 'spectral_markers_line_color' not in st.session_state:
    st.session_state.spectral_markers_line_color = '#000000'
if 'spectral_markers_region_color' not in st.session_state:
    st.session_state.spectral_markers_region_color = '#ff0000'
if 'spectral_markers_temp_name' not in st.session_state:
    st.session_state.spectral_markers_temp_name = ""
if 'spectral_markers_temp_color' not in st.session_state:
    st.session_state.spectral_markers_temp_color = '#000000'
if 'spectral_markers_show_preview' not in st.session_state:
    st.session_state.spectral_markers_show_preview = True

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
    
    /* Separator styling */
    .separator {
        text-align: center;
        margin: 20px 0;
        font-size: 20px;
        font-weight: bold;
        color: #667eea;
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
    'lines.linewidth': 0.6,
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
        # Clean data - remove NaN and inf values
        data = data.dropna()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data = data.sort_values('x').reset_index(drop=True)
        return data
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {e}")
        return None
        
# Function to normalize spectrum
def normalize_spectrum(x, y, norm_method, norm_range=None, x_ranges_for_rest=None):
    """Normalize spectrum using different methods
    
    Parameters:
    - x: x-axis values
    - y: y-axis values
    - norm_method: normalization method string
    - norm_range: tuple (start, end) for peak intensity range normalization
    - x_ranges_for_rest: list of tuples [(start1, end1), (start2, end2)] for "Maximum rest intensity" method
    """
    # Check if array is empty or contains invalid values
    if len(y) == 0:
        return y
    
    # Clean data
    y = np.array(y)
    mask = np.isfinite(y)
    if not np.any(mask):
        return y
    
    y_clean = y[mask]
    
    if norm_method == "Maximum intensity":
        max_val = y_clean.max()
        if max_val != 0 and np.isfinite(max_val):
            result = y / max_val
            # Replace inf and nan with 0
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result
        return y
    
    elif norm_method == "Peak intensity (range)":
        if norm_range is not None:
            mask_range = (x >= norm_range[0]) & (x <= norm_range[1])
            if np.any(mask_range):
                y_in_range = y[mask_range]
                y_in_range_clean = y_in_range[np.isfinite(y_in_range)]
                if len(y_in_range_clean) > 0:
                    max_in_range = y_in_range_clean.max()
                    if max_in_range != 0 and np.isfinite(max_in_range):
                        result = y / max_in_range
                        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                        return result
        
        max_val = y_clean.max()
        if max_val != 0 and np.isfinite(max_val):
            result = y / max_val
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result
        return y
    
    elif norm_method == "Maximum rest intensity":
        if x_ranges_for_rest is not None and len(x_ranges_for_rest) > 0:
            # Collect all y-values from the specified ranges (visible portions)
            y_in_ranges = []
            for start, end in x_ranges_for_rest:
                mask_range = (x >= start) & (x <= end)
                if np.any(mask_range):
                    y_in_range = y[mask_range]
                    y_in_range_clean = y_in_range[np.isfinite(y_in_range)]
                    if len(y_in_range_clean) > 0:
                        y_in_ranges.extend(y_in_range_clean)
            
            # If we found data in the ranges, normalize by the maximum in those ranges
            if len(y_in_ranges) > 0:
                max_in_ranges = np.max(y_in_ranges)
                if max_in_ranges != 0 and np.isfinite(max_in_ranges):
                    result = y / max_in_ranges
                    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                    return result
            
            # Fallback to global maximum if no data found in ranges
            max_val = y_clean.max()
            if max_val != 0 and np.isfinite(max_val):
                result = y / max_val
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                return result
            return y
        else:
            # If no ranges provided, behave like regular Maximum intensity
            max_val = y_clean.max()
            if max_val != 0 and np.isfinite(max_val):
                result = y / max_val
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                return result
            return y
    
    return y

def align_x_ranges(spectra_dict):
    """Align all spectra to common x range"""
    if not spectra_dict:
        return spectra_dict
    
    valid_specs = {}
    for name, spec in spectra_dict.items():
        if len(spec['data']['x']) > 0 and len(spec['data']['y']) > 0:
            valid_specs[name] = spec
    
    if not valid_specs:
        return spectra_dict
    
    min_x = max([spec['data']['x'].min() for spec in valid_specs.values()])
    max_x = min([spec['data']['x'].max() for spec in valid_specs.values()])
    
    if min_x >= max_x:
        return spectra_dict
    
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

def gradient_fill(ax, x, y, color, offset=0):
    """Create gradient fill from top (alpha=0.3) to bottom (alpha=0.9)"""
    from matplotlib.collections import PolyCollection
    import matplotlib.pyplot as plt
    
    verts = []
    for i in range(len(x)):
        verts.append((x[i], y[i] + offset))
    for i in range(len(x)-1, -1, -1):
        verts.append((x[i], offset))
    
    poly = plt.Polygon(verts, closed=True, facecolor=color, alpha=0.3, edgecolor='none')
    ax.add_patch(poly)
    
    n_layers = 20
    for i in range(n_layers):
        alpha_i = 0.3 + (i / n_layers) * 0.6  # from 0.3 to 0.9
        y_level = offset + (y + offset - offset) * (i / n_layers)
        for j in range(len(x)-1):
            ax.fill_between(x[j:j+2], offset, y_level[j:j+2], 
                           color=color, alpha=0.03, linewidth=0)

def create_individual_plot(spectra_dict, x_label, y_label, title,
                           offset_step, fill_area, normalized, use_offset,
                           x_ranges, subtract_min_intensity, fill_alpha,
                           show_grid, line_width, fig_width, fig_height,
                           legend_fontsize=8, legend_position="right", legend_offset=1.02):
    """Create individual scientific plot with download button"""
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    handles = []
    labels = []
    
    spectra_items = list(spectra_dict.items())
    
    if x_ranges is None or len(x_ranges) == 0:
        # Simple plot without broken axis
        for idx, (name, spec) in enumerate(spectra_items):
            data = spec['data']
            x = data['x'].values
            y = data['y'].values
            color = spec['color']
            
            display_name = name.replace('.txt', '')
            
            # Apply subtract minimum intensity if requested
            if subtract_min_intensity and normalized:
                if len(y) > 0:
                    y = y - y.min()
                else:
                    y = y
            
            # Apply cumulative offset if requested
            if use_offset:
                offset = idx * offset_step
            else:
                offset = 0
            
            y_plot = y + offset
            
            if fill_area and normalized:
                # Check if data is valid for fill_between
                if len(x) > 0 and len(y_plot) > 0:
                    # Clean data for fill_between
                    mask = np.isfinite(x) & np.isfinite(y_plot) & np.isfinite(offset)
                    x_clean = x[mask]
                    y_clean = y_plot[mask]
                    if len(x_clean) > 1:
                        ax.fill_between(x_clean, offset, y_clean, alpha=fill_alpha, color=color)
                line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name)
            else:
                line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name)
            
            handles.append(line_handle[0])
            labels.append(display_name)
        
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
        
    else:
        # Broken axis plot with multiple x-ranges
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
                
                # Apply subtract minimum intensity if requested
                if subtract_min_intensity and normalized:
                    y = y - y.min()
                
                # Apply cumulative offset if requested
                if use_offset:
                    offset = idx * offset_step
                else:
                    offset = 0
                
                y_plot = y + offset
                
                # Plot
                if fill_area and normalized:
                    ax.fill_between(x, offset, y_plot, alpha=fill_alpha, color=color)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                else:
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                
                # Add to handles only for first range
                if range_idx == 0:
                    handles.append(line_handle[0])
                    labels.append(display_name)
            
            # Add vertical line for range boundaries
            ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    
    # Add legend with customizable settings
    if handles:
        # Adjust legend position
        if legend_position == "right":
            bbox_anchor = (legend_offset, 0.5)
            loc = 'center left'
        elif legend_position == "best":
            bbox_anchor = None
            loc = 'best'
        else:
            # For specific positions like 'upper right', 'upper left', etc.
            bbox_anchor = None
            loc = legend_position
        
        # Create legend
        if use_offset and legend_position == "right":
            # Reverse order for offset plots
            reversed_handles = list(reversed(handles))
            reversed_labels = list(reversed(labels))
            legend = ax.legend(reversed_handles, reversed_labels, 
                              loc=loc, 
                              bbox_to_anchor=bbox_anchor,
                              fontsize=legend_fontsize,
                              frameon=True, 
                              edgecolor='black', 
                              prop={'weight': 'bold'})
            for text, handle in zip(legend.get_texts(), reversed_handles):
                text.set_color(handle.get_color())
        else:
            legend = ax.legend(handles, labels, 
                              loc=loc, 
                              bbox_to_anchor=bbox_anchor,
                              fontsize=legend_fontsize,
                              frameon=True, 
                              edgecolor='black', 
                              prop={'weight': 'bold'})
            if legend_position == "right":
                for text, handle in zip(legend.get_texts(), handles):
                    text.set_color(handle.get_color())
        
        # Adjust legend box if it's too large
        if len(handles) > 10:
            legend._legend_box.align = "left"
            # Set number of columns for large legends
            if len(handles) > 15:
                legend._ncol = 2
    
    ax.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    else:
        ax.grid(False)
    
    # Dynamic right margin adjustment based on legend size and position
    if legend_position == "right" and len(handles) > 0:
        # Calculate margin based on number of spectra and font size
        estimated_legend_width = min(0.35, 0.15 + (len(handles) * legend_fontsize / 300))
        right_margin = min(0.95, legend_offset + estimated_legend_width)
        plt.tight_layout()
        plt.subplots_adjust(right=right_margin)
    else:
        plt.tight_layout()
    
    return fig

# Function to create combined plot with all four visualization types (vertical layout)
def create_combined_plot(spectra_dict, x_label, y_label, title,
                         raw_offset_step, norm_offset_step, fill_area,
                         norm_method, x_ranges=None, fill_alpha=0.3,
                         show_grid=True, line_width=1.5,
                         legend_fontsize=8, legend_position="right", legend_offset=1.02):
    """Create scientific plot with all four visualization types in vertical subplots"""
    
    # Prepare normalized spectra
    normalized_spectra = {}
    for name, spec in spectra_dict.items():
        data = spec['data']
        y_norm = normalize_spectrum(
            data['x'].values,
            data['y'].values,
            norm_method,
            None,
            x_ranges  # Pass x_ranges for "Maximum rest intensity" method
        )
        normalized_spectra[name] = {
            'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
            'color': spec['color']
        }
    
    # Create figure with 4 subplots vertically (4 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=(12, 18))
    
    # Define the four visualization types - titles removed
    viz_configs = [
        (axes[0], spectra_dict, 0, False, False, False, x_label, y_label),
        (axes[1], normalized_spectra, 0, False, True, False, x_label, f"Normalized {y_label}"),
        (axes[2], spectra_dict, raw_offset_step, False, False, True, x_label, y_label),
        (axes[3], normalized_spectra, norm_offset_step, fill_area, True, True, x_label, f"Normalized {y_label}")
    ]
    
    for ax, spectra, offset_step, fill, normalized, use_offset, xl, yl in viz_configs:
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
                    ax.fill_between(x, offset, y_plot, alpha=fill_alpha, color=color)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name)
                else:
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name)
                
                handles.append(line_handle[0])
                labels.append(display_name)
            
            ax.set_xlabel(xl, fontsize=10, fontweight='bold')
            ax.set_ylabel(yl, fontsize=10, fontweight='bold')
            
        else:
            # Broken axis plot with multiple x-ranges
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
                        # Check if data is valid
                        if len(x) > 0 and len(y_plot) > 0:
                            mask = np.isfinite(x) & np.isfinite(y_plot) & np.isfinite(offset)
                            x_clean = x[mask]
                            y_clean = y_plot[mask]
                            if len(x_clean) > 1:
                                ax.fill_between(x_clean, offset, y_clean, alpha=fill_alpha, color=color)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                    elif fill and normalized:
                        if len(x) > 0 and len(y_plot) > 0:
                            mask = np.isfinite(x) & np.isfinite(y_plot)
                            x_clean = x[mask]
                            y_clean = y_plot[mask]
                            if len(x_clean) > 1:
                                ax.fill_between(x_clean, 0, y_clean, alpha=fill_alpha, color=color)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                    else:
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                    
                    # Add to handles only for first range
                    if range_idx == 0:
                        handles.append(line_handle[0])
                        labels.append(display_name)
                
                # Add vertical line for range boundaries
                ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
                ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            
            ax.set_xlabel(xl, fontsize=10, fontweight='bold')
            ax.set_ylabel(yl, fontsize=10, fontweight='bold')
        
        # Add legend with customizable settings
        if handles:
            # Adjust legend position
            if legend_position == "right":
                bbox_anchor = (legend_offset, 0.5)
                loc = 'center left'
            elif legend_position == "best":
                bbox_anchor = None
                loc = 'best'
            else:
                bbox_anchor = None
                loc = legend_position
            
            # Create legend
            if use_offset and legend_position == "right":
                reversed_handles = list(reversed(handles))
                reversed_labels = list(reversed(labels))
                legend = ax.legend(reversed_handles, reversed_labels, 
                                  loc=loc, 
                                  bbox_to_anchor=bbox_anchor,
                                  fontsize=legend_fontsize,
                                  frameon=True, 
                                  edgecolor='black', 
                                  prop={'weight': 'bold'})
                for text, handle in zip(legend.get_texts(), reversed_handles):
                    text.set_color(handle.get_color())
            else:
                legend = ax.legend(handles, labels, 
                                  loc=loc, 
                                  bbox_to_anchor=bbox_anchor,
                                  fontsize=legend_fontsize,
                                  frameon=True, 
                                  edgecolor='black', 
                                  prop={'weight': 'bold'})
                if legend_position == "right":
                    for text, handle in zip(legend.get_texts(), handles):
                        text.set_color(handle.get_color())
            
            # Adjust legend box if it's too large
            if len(handles) > 10:
                legend._legend_box.align = "left"
                if len(handles) > 15:
                    legend._ncol = 2
        
        ax.tick_params(direction='in', length=5, width=1)
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.grid(False)
    
    # Adjust layout with dynamic right margin
    if legend_position == "right":
        right_margin = min(0.92, legend_offset + 0.05)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4, right=right_margin)
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4)
    
    return fig

# Improved FWHM calculation function
def calculate_fwhm(x, y, peak_idx):
    """Calculate Full Width at Half Maximum for a peak with improved robustness"""
    peak_y = y[peak_idx]
    half_max = peak_y / 2
    
    # Initialize variables
    x_left = x[0]
    x_right = x[-1]
    
    # Find left crossing - search backwards from peak
    left_idx = peak_idx
    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1
    
    # If we found a point below half_max, interpolate
    if left_idx > 0 and y[left_idx] < half_max:
        # Interpolate for more accurate left boundary
        x_left = np.interp(half_max, [y[left_idx], y[left_idx+1]], [x[left_idx], x[left_idx+1]])
    elif left_idx == 0:
        # Check if we need to interpolate from start
        if y[0] < half_max:
            x_left = np.interp(half_max, [y[0], y[1]], [x[0], x[1]])
        else:
            x_left = x[0]
    
    # Find right crossing - search forward from peak
    right_idx = peak_idx
    while right_idx < len(y) - 1 and y[right_idx] > half_max:
        right_idx += 1
    
    # If we found a point below half_max, interpolate
    if right_idx < len(y) - 1 and y[right_idx] < half_max:
        # Interpolate for more accurate right boundary
        x_right = np.interp(half_max, [y[right_idx], y[right_idx-1]], [x[right_idx], x[right_idx-1]])
    elif right_idx == len(y) - 1:
        # Check if we need to interpolate from end
        if y[-1] < half_max:
            x_right = np.interp(half_max, [y[-2], y[-1]], [x[-2], x[-1]])
        else:
            x_right = x[-1]
    
    # Return FWHM, handle edge cases
    fwhm = x_right - x_left
    return max(fwhm, 0)

# Function for peak analysis with manual range selection
def analyze_peaks_manual_range(spectra_dict, x_range, peak_width=20):
    """Analyze peaks in spectra within manually selected x-range"""
    results = []
    
    for name, spec in spectra_dict.items():
        data = spec['data']
        x_full = data['x'].values
        y_full = data['y'].values
        
        # Crop to manual range
        if x_range[0] is not None and x_range[1] is not None:
            mask = (x_full >= x_range[0]) & (x_full <= x_range[1])
            x = x_full[mask]
            y = y_full[mask]
        else:
            x = x_full
            y = y_full
        
        if len(x) == 0:
            continue
        
        # Find peaks with more robust parameters
        peaks, properties = find_peaks(y, height=np.max(y)*0.05, prominence=np.max(y)*0.03, distance=5)
        
        for peak_idx in peaks:
            peak_x = x[peak_idx]
            peak_y = y[peak_idx]
            
            # Calculate area around peak
            left_idx = max(0, peak_idx - peak_width)
            right_idx = min(len(x), peak_idx + peak_width)
            area = simpson(y[left_idx:right_idx+1], x[left_idx:right_idx+1])
            
            # Calculate FWHM using improved function
            fwhm = calculate_fwhm(x, y, peak_idx)
            
            results.append({
                'Spectrum': name.replace('.txt', ''),
                'Peak position': peak_x,
                'Intensity': peak_y,
                'Area': area,
                'FWHM': fwhm,
                'Include': True  # New column for checkbox
            })
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# Function to create peak visualization with manual range
def create_peak_visualization(spectra_dict, x_range, peaks_df):
    """Create peak visualization with selected range boundaries"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter to include only peaks marked as included
    included_peaks = peaks_df[peaks_df['Include'] == True]
    
    for name, spec in spectra_dict.items():
        data = spec['data']
        x = data['x'].values
        y = data['y'].values
        
        # Clean data
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) == 0:
            continue
            
        color = spec['color']
        
        ax.plot(x, y, color=color, linewidth=1.5, label=name.replace('.txt', ''), alpha=0.7)
        
        # Mark peaks for this spectrum that are included
        spec_peaks = included_peaks[included_peaks['Spectrum'] == name.replace('.txt', '')]
        for _, peak in spec_peaks.iterrows():
            ax.axvline(peak['Peak position'], color=color, 
                      linestyle='--', alpha=0.5, linewidth=1)
            ax.text(peak['Peak position'], peak['Intensity']*0.8, 
                   f"{peak['Peak position']:.1f}", 
                   fontsize=8, ha='center', fontweight='bold')
    
    # Add manual range boundaries
    if x_range[0] is not None:
        ax.axvline(x_range[0], color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Left boundary: {x_range[0]:.1f}')
    if x_range[1] is not None:
        ax.axvline(x_range[1], color='blue', linestyle='-', linewidth=2, alpha=0.7, label=f'Right boundary: {x_range[1]:.1f}')
    
    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Intensity (a.u.)", fontsize=11, fontweight='bold')
    ax.set_title("Peak Analysis with Selected Range", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, edgecolor='black', prop={'weight': 'bold'})
    ax.tick_params(direction='in', length=5, width=1)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

# NEW FUNCTION: Create comparison plot with difference analysis
def create_comparison_plot(spectrum_a_data, spectrum_b_data, name_a, name_b,
                           x_label, y_label, norm_method, norm_range,
                           offset_step, fill_area, fill_alpha, subtract_min_intensity,
                           show_grid, line_width, fig_width, fig_height,
                           legend_fontsize, legend_position, legend_offset,
                           colormap_name, smooth_difference, smooth_sigma,
                           symmetric_scale, difference_threshold):
    """Create comparison plot with two spectra and difference analysis"""
    
    # Prepare data for spectrum A
    x_a = spectrum_a_data['data']['x'].values
    y_a_raw = spectrum_a_data['data']['y'].values
    y_a_norm = normalize_spectrum(x_a, y_a_raw, norm_method, norm_range)
    
    # Prepare data for spectrum B
    x_b = spectrum_b_data['data']['x'].values
    y_b_raw = spectrum_b_data['data']['y'].values
    y_b_norm = normalize_spectrum(x_b, y_b_raw, norm_method, norm_range)
    
    # Interpolate both spectra to common x grid
    common_x_min = max(x_a.min(), x_b.min())
    common_x_max = min(x_a.max(), x_b.max())
    common_x = np.linspace(common_x_min, common_x_max, 2000)
    
    y_a_interp = np.interp(common_x, x_a, y_a_norm)
    y_b_interp = np.interp(common_x, x_b, y_b_norm)
    
    # Apply subtract minimum intensity if requested
    if subtract_min_intensity:
        y_a_interp = y_a_interp - y_a_interp.min()
        y_b_interp = y_b_interp - y_b_interp.min()
    
    # Calculate difference (Sample - Reference)
    y_diff = y_b_interp - y_a_interp
    
    # Apply smoothing if requested
    if smooth_difference:
        y_diff_smoothed = gaussian_filter1d(y_diff, sigma=smooth_sigma)
    else:
        y_diff_smoothed = y_diff
    
    # Calculate statistics
    mean_diff = np.mean(y_diff_smoothed)
    max_abs_diff = np.max(np.abs(y_diff_smoothed))
    rms_diff = np.sqrt(np.mean(y_diff_smoothed**2))
    correlation = pearsonr(y_a_interp, y_b_interp)[0]
    
    # Create figure with two subplots - INCREASED spacing between plots
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(fig_width, fig_height * 1.2),
                                             gridspec_kw={'height_ratios': [1, 0.6]})
    
    # TOP PLOT: Both spectra with offset
    # Apply offset to spectra for visualization
    offset = 0
    y_a_plot = y_a_interp + offset
    y_b_plot = y_b_interp + offset
    
    if fill_area:
        ax_top.fill_between(common_x, offset, y_a_plot, alpha=fill_alpha, 
                           color=spectrum_a_data['color'], label=f"{name_a} (Reference)")
        ax_top.fill_between(common_x, offset, y_b_plot, alpha=fill_alpha, 
                           color=spectrum_b_data['color'], label=f"{name_b} (Sample)")
    
    ax_top.plot(common_x, y_a_plot, color=spectrum_a_data['color'], 
               linewidth=line_width, label=f"{name_a} (Reference)")
    ax_top.plot(common_x, y_b_plot, color=spectrum_b_data['color'], 
               linewidth=line_width, label=f"{name_b} (Sample)")
    
    ax_top.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax_top.set_ylabel("Norm. Intensity", fontsize=10, fontweight='bold')
    # REMOVED: ax_top.set_title(...)
    
    # Legend for top plot
    handles = ax_top.get_legend_handles_labels()[0]
    labels = ax_top.get_legend_handles_labels()[1]
    if legend_position == "right":
        bbox_anchor = (legend_offset, 0.5)
        loc = 'center left'
    elif legend_position == "best":
        bbox_anchor = None
        loc = 'best'
    else:
        bbox_anchor = None
        loc = legend_position
    
    ax_top.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_anchor,
                 fontsize=legend_fontsize, frameon=True, edgecolor='black', prop={'weight': 'bold'})
    
    ax_top.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax_top.grid(True, alpha=0.3, linestyle='--')
    else:
        ax_top.grid(False)
    
    # BOTTOM PLOT: Difference with gradient fill between curve and zero line
    # Set symmetric y-limits if requested
    if symmetric_scale:
        y_limit = max_abs_diff * 1.1
        ax_bottom.set_ylim(-y_limit, y_limit)
    else:
        y_min = np.min(y_diff_smoothed)
        y_max = np.max(y_diff_smoothed)
        y_margin = (y_max - y_min) * 0.1
        ax_bottom.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Create filled polygons between curve and zero line with colormap-based colors
    # We need to create segmented fills where each segment's color corresponds to its y-value
    
    # Create a colormap object
    cmap = plt.get_cmap(colormap_name)
    
    # Normalize y-values for colormap mapping
    if symmetric_scale:
        norm = plt.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)
    else:
        norm = plt.Normalize(vmin=np.min(y_diff_smoothed), vmax=np.max(y_diff_smoothed))
    
    # Plot zero line
    ax_bottom.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)
    
    # Create gradient fill between curve and zero line
    # Iterate through segments and fill with appropriate colors
    for i in range(len(common_x) - 1):
        x_seg = [common_x[i], common_x[i+1], common_x[i+1], common_x[i]]
        
        y1 = y_diff_smoothed[i]
        y2 = y_diff_smoothed[i+1]
        
        # Determine if the segment crosses zero
        if y1 * y2 >= 0:  # Both values have same sign or one is zero
            # Fill from curve to zero line
            y_seg = [y1, y2, 0, 0]
            # Use average color for the segment
            avg_y = (y1 + y2) / 2
            color = cmap(norm(avg_y))
            ax_bottom.fill(x_seg, y_seg, color=color, alpha=0.6, edgecolor='none', zorder=2)
        else:
            # Segment crosses zero - split into two polygons
            # Find interpolation point where y=0
            t = -y1 / (y2 - y1)  # Interpolation factor
            x_zero = common_x[i] + t * (common_x[i+1] - common_x[i])
            
            # First polygon (from y1 to 0)
            x_seg1 = [common_x[i], x_zero, x_zero, common_x[i]]
            y_seg1 = [y1, 0, 0, y1]
            color1 = cmap(norm(y1/2))
            ax_bottom.fill(x_seg1, y_seg1, color=color1, alpha=0.6, edgecolor='none', zorder=2)
            
            # Second polygon (from 0 to y2)
            x_seg2 = [x_zero, common_x[i+1], common_x[i+1], x_zero]
            y_seg2 = [0, y2, y2, 0]
            color2 = cmap(norm(y2/2))
            ax_bottom.fill(x_seg2, y_seg2, color=color2, alpha=0.6, edgecolor='none', zorder=2)
    
    # Plot the difference line on top
    ax_bottom.plot(common_x, y_diff_smoothed, color='black', linewidth=1.2, alpha=0.8, zorder=3, label='Difference profile')
    
    # Highlight significant differences if threshold is set
    if difference_threshold > 0:
        # Find regions where |difference| > threshold
        above_threshold = np.abs(y_diff_smoothed) > difference_threshold
        if np.any(above_threshold):
            # Find contiguous regions
            diff_indices = np.where(above_threshold)[0]
            regions = []
            start_idx = diff_indices[0]
            for i in range(1, len(diff_indices)):
                if diff_indices[i] > diff_indices[i-1] + 1:
                    regions.append((common_x[start_idx], common_x[diff_indices[i-1]]))
                    start_idx = diff_indices[i]
            regions.append((common_x[start_idx], common_x[diff_indices[-1]]))
            
            # Highlight regions with light yellow background
            y_min, y_max = ax_bottom.get_ylim()
            for start, end in regions:
                ax_bottom.axvspan(start, end, alpha=0.2, color='yellow', zorder=0)
    
    ax_bottom.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax_bottom.set_ylabel('Intensity Diff.', fontsize=10, fontweight='bold')
    # REMOVED: ax_bottom.set_title(...)
    
    ax_bottom.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax_bottom.grid(True, alpha=0.3, linestyle='--')
    else:
        ax_bottom.grid(False)
    
    # Add horizontal line at zero
    ax_bottom.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Create colorbar with SAME WIDTH as the plots and OFFSET like legend
    # Create a scalar mappable for the colorbar
    if symmetric_scale:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff))
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(y_diff_smoothed), vmax=np.max(y_diff_smoothed)))
    sm.set_array([])
    
    # Add colorbar with adjusted position to match plot width and add offset like legend
    # Position: [left, bottom, width, height] in figure coordinates
    # Get current axes position
    pos = ax_bottom.get_position()
    # Use legend_offset to control colorbar position (same offset as legend)
    colorbar_offset = legend_offset + 0.50  # Slightly more than legend for visual balance
    # Create colorbar axes to the right with same height as bottom plot
    cbar_ax = fig.add_axes([pos.x1 + (colorbar_offset - 1.0) * 0.1, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Amplitude', fontsize=9, fontweight='bold')
    
    # Add legend for significant differences if needed
    if difference_threshold > 0 and np.any(above_threshold):
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='yellow', alpha=0.3, label=f'|Difference| > {difference_threshold:.3f}')]
        ax_bottom.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Adjust layout with increased spacing between subplots
    plt.subplots_adjust(hspace=0.4)  # INCREASED from 0.3 to 0.4
    
    # Adjust right margin for legend if needed
    if legend_position == "right":
        right_margin = min(0.92, legend_offset + 0.05)
        plt.subplots_adjust(right=right_margin)
    else:
        plt.tight_layout()
    
    return fig, (mean_diff, max_abs_diff, rms_diff, correlation)

# NEW FUNCTION: Create heatmap from spectra matrix
def create_heatmap(spectra_matrix, x_grid, y_values, x_label, y_label, 
                   colorbar_label, colormap, interpolation, title, 
                   fig_width=10, fig_height=8, log_scale=False, show_grid=True):
    """Create a heatmap from spectra matrix with specified parameters"""
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Prepare data for heatmap
    data_matrix = np.array(spectra_matrix)
    
    # Apply log scaling if requested
    if log_scale:
        # Avoid log of zero or negative values
        data_matrix = np.maximum(data_matrix, 1e-10)
        data_matrix = np.log(data_matrix)
        colorbar_label = f"log ({colorbar_label})"
    
    # Create heatmap with imshow - transpose matrix for correct orientation
    # X-axis: parameter (y_values), Y-axis: Raman shift (x_grid)
    extent = [y_values[0], y_values[-1], x_grid[0], x_grid[-1]]
    
    # Transpose the matrix so that rows = Raman shift, columns = parameter
    data_matrix_transposed = data_matrix.T
    
    # Use exact min and max values for color scale
    data_clean = data_matrix_transposed[np.isfinite(data_matrix_transposed)]
    if len(data_clean) > 0:
        # Используем точные минимум и максимум для цветовой шкалы
        vmin = np.nanmin(data_matrix_transposed)
        vmax = np.nanmax(data_matrix_transposed)
    else:
        vmin = 0
        vmax = 1
    
    # Use imshow with specified interpolation
    im = ax.imshow(data_matrix_transposed, 
                   extent=extent, 
                   aspect='auto', 
                   origin='lower',
                   cmap=colormap,
                   interpolation=interpolation,
                   interpolation_stage='data' if interpolation != 'none' else None,
                   vmin=vmin,
                   vmax=vmax)
    
    # Add colorbar with custom ticks showing min and max values
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=20, fontweight='bold')
    
    # Определяем количество делений (5-7 всего, включая min и max)
    n_ticks = 6  # 4 промежуточных + min + max = 6
    tick_positions = np.linspace(vmin, vmax, n_ticks)
    
    # Форматируем значения для отображения
    if log_scale:
        # Для логарифмической шкалы показываем значения с 2 знаками после запятой
        tick_labels = [f"{val:.2f}" for val in tick_positions]
    else:
        # Определяем формат в зависимости от диапазона значений
        if abs(vmax - vmin) < 0.01:
            # Очень маленький диапазон - показываем с 4 знаками
            tick_labels = [f"{val:.4f}" for val in tick_positions]
        elif abs(vmax - vmin) < 1:
            # Маленький диапазон - показываем с 3 знаками
            tick_labels = [f"{val:.3f}" for val in tick_positions]
        elif abs(vmax - vmin) < 10:
            # Средний диапазон - показываем с 2 знаками
            tick_labels = [f"{val:.2f}" for val in tick_positions]
        elif abs(vmax - vmin) < 1000:
            # Большой диапазон - показываем с 1 знаком
            tick_labels = [f"{val:.1f}" for val in tick_positions]
        else:
            # Очень большой диапазон - показываем целые числа
            tick_labels = [f"{int(val)}" for val in tick_positions]
    
    # Устанавливаем деления на цветовой шкале
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=16)
    
    # Добавляем рамку вокруг цветовой шкалы для лучшей читаемости
    cbar.ax.spines['top'].set_visible(True)
    cbar.ax.spines['bottom'].set_visible(True)
    cbar.ax.spines['left'].set_visible(True)
    cbar.ax.spines['right'].set_visible(True)
    cbar.ax.spines['top'].set_color('black')
    cbar.ax.spines['bottom'].set_color('black')
    cbar.ax.spines['left'].set_color('black')
    cbar.ax.spines['right'].set_color('black')
    cbar.ax.spines['top'].set_linewidth(0.5)
    cbar.ax.spines['bottom'].set_linewidth(0.5)
    cbar.ax.spines['left'].set_linewidth(0.5)
    cbar.ax.spines['right'].set_linewidth(0.5)
    
    # Set labels
    ax.set_xlabel(y_label, fontsize=20, fontweight='bold')
    ax.set_ylabel(x_label, fontsize=20, fontweight='bold')
    
    # Improve tick formatting
    ax.tick_params(direction='in', length=5, width=1, labelsize=15)
    
    # Apply grid settings based on show_grid parameter
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    else:
        ax.grid(False)
    
    plt.tight_layout()
    return fig

def prepare_heatmap_data(spectra_dict, ordered_spectra, heatmap_params, norm_method, norm_range, x_ranges):
    """Prepare data matrices for heatmap generation"""
    
    # Get all spectra data and interpolate to common x grid
    all_x = []
    for name in ordered_spectra:
        if name in spectra_dict:
            x_vals = spectra_dict[name]['data']['x'].values
            if len(x_vals) > 0:
                all_x.extend(x_vals)
    
    if not all_x:
        return None, None, None, None

    if x_ranges is not None and len(x_ranges) > 0:
        # Use the min of all range starts and max of all range ends
        # This ensures we cover all specified ranges
        range_starts = [start for start, end in x_ranges]
        range_ends = [end for start, end in x_ranges]
        x_min = min(range_starts)
        x_max = max(range_ends)
        
        # Verify that there is data in these ranges for all spectra
        # If not, fallback to data-derived range
        has_data_in_ranges = True
        for name in ordered_spectra:
            if name in spectra_dict:
                x_vals = spectra_dict[name]['data']['x'].values
                # Check if spectrum has any data points within the ranges
                in_range = False
                for start, end in x_ranges:
                    mask = (x_vals >= start) & (x_vals <= end)
                    if np.any(mask):
                        in_range = True
                        break
                if not in_range:
                    has_data_in_ranges = False
                    break
        
        if not has_data_in_ranges:
            # Fallback to data-derived range if no data in specified ranges
            x_min = max([spectra_dict[name]['data']['x'].min() for name in ordered_spectra if name in spectra_dict])
            x_max = min([spectra_dict[name]['data']['x'].max() for name in ordered_spectra if name in spectra_dict])
    else:
        # Use full common range
        x_min = max([spectra_dict[name]['data']['x'].min() for name in ordered_spectra if name in spectra_dict])
        x_max = min([spectra_dict[name]['data']['x'].max() for name in ordered_spectra if name in spectra_dict])
    
    if x_min >= x_max:
        return None, None, None, None
    
    common_x = np.linspace(x_min, x_max, 2000)
    
    # Prepare matrices
    spectra_matrix = []
    spectra_norm_matrix = []
    y_values = []
    
    for name in ordered_spectra:
        if name not in spectra_dict or name not in heatmap_params:
            continue
        
        # Get spectrum data
        data = spectra_dict[name]['data']
        x_orig = data['x'].values
        y_orig = data['y'].values
        
        # If x_ranges specified, create mask for ranges and crop data
        if x_ranges is not None and len(x_ranges) > 0:
            # Create mask for all ranges combined
            mask_total = np.zeros_like(x_orig, dtype=bool)
            for start, end in x_ranges:
                mask_range = (x_orig >= start) & (x_orig <= end)
                mask_total = mask_total | mask_range
            
            # Crop to ranges
            if np.any(mask_total):
                x_cropped = x_orig[mask_total]
                y_cropped = y_orig[mask_total]
            else:
                # Fallback to full data if no points in ranges
                st.warning(f"No data points found in specified ranges for {name}. Using full spectrum.")
                x_cropped = x_orig
                y_cropped = y_orig
        else:
            x_cropped = x_orig
            y_cropped = y_orig
        
        # Interpolate to common x grid
        # If x_cropped is empty, skip this spectrum
        if len(x_cropped) == 0:
            st.warning(f"Empty data after cropping for {name}. Skipping.")
            continue
        
        # Interpolate to common x grid
        y_interp = np.interp(common_x, x_cropped, y_cropped)
        
        # Normalize spectrum (pass x_ranges for normalization method)
        y_norm = normalize_spectrum(
            common_x, 
            y_interp, 
            norm_method, 
            norm_range,
            x_ranges
        )
        
        # Store in matrices
        spectra_matrix.append(y_interp)
        spectra_norm_matrix.append(y_norm)
        
        # Get parameter value
        param_value = heatmap_params[name]
        y_values.append(param_value)
    
    if not spectra_matrix:
        return None, None, None, None
    
    return np.array(spectra_matrix), np.array(spectra_norm_matrix), common_x, np.array(y_values)

# NEW FUNCTION: Create spectral markers plot
# NEW FUNCTION: Create spectral markers plot
def create_spectral_markers_plot(spectra_dict, x_label, y_label, offset_step, 
                                  fill_area, normalized, use_offset, x_ranges,
                                  subtract_min_intensity, fill_alpha, show_grid,
                                  line_width, fig_width, fig_height, legend_fontsize,
                                  legend_position, legend_offset, markers, 
                                  preview_position, preview_width, show_x_values,
                                  is_region_mode=False, show_preview=True):  # NEW parameter
    """Create plot with spectral markers (lines and regions)"""
    
    # Create the base plot using create_individual_plot but without legend
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    handles = []
    labels = []
    
    spectra_items = list(spectra_dict.items())
    
    if x_ranges is None or len(x_ranges) == 0:
        # Simple plot without broken axis
        for idx, (name, spec) in enumerate(spectra_items):
            data = spec['data']
            x = data['x'].values
            y = data['y'].values
            color = spec['color']
            
            display_name = name.replace('.txt', '')
            
            # Apply subtract minimum intensity if requested
            if subtract_min_intensity and normalized:
                if len(y) > 0:
                    y = y - y.min()
                else:
                    y = y
            
            # Apply cumulative offset if requested
            if use_offset:
                offset = idx * offset_step
            else:
                offset = 0
            
            y_plot = y + offset
            
            if fill_area and normalized:
                # Check if data is valid for fill_between
                if len(x) > 0 and len(y_plot) > 0:
                    # Clean data for fill_between
                    mask = np.isfinite(x) & np.isfinite(y_plot) & np.isfinite(offset)
                    x_clean = x[mask]
                    y_clean = y_plot[mask]
                    if len(x_clean) > 1:
                        ax.fill_between(x_clean, offset, y_clean, alpha=fill_alpha, color=color)
                line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name)
            else:
                line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name)
            
            handles.append(line_handle[0])
            labels.append(display_name)
        
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
        
    else:
        # Broken axis plot with multiple x-ranges
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
                
                # Apply subtract minimum intensity if requested
                if subtract_min_intensity and normalized:
                    y = y - y.min()
                
                # Apply cumulative offset if requested
                if use_offset:
                    offset = idx * offset_step
                else:
                    offset = 0
                
                y_plot = y + offset
                
                # Plot
                if fill_area and normalized:
                    ax.fill_between(x, offset, y_plot, alpha=fill_alpha, color=color)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                else:
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                
                # Add to handles only for first range
                if range_idx == 0:
                    handles.append(line_handle[0])
                    labels.append(display_name)
            
            # Add vertical line for range boundaries
            ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    
    # Add legend with customizable settings
    if handles:
        # Adjust legend position
        if legend_position == "right":
            bbox_anchor = (legend_offset, 0.5)
            loc = 'center left'
        elif legend_position == "best":
            bbox_anchor = None
            loc = 'best'
        else:
            # For specific positions like 'upper right', 'upper left', etc.
            bbox_anchor = None
            loc = legend_position
        
        # Create legend
        if use_offset and legend_position == "right":
            # Reverse order for offset plots
            reversed_handles = list(reversed(handles))
            reversed_labels = list(reversed(labels))
            legend = ax.legend(reversed_handles, reversed_labels, 
                              loc=loc, 
                              bbox_to_anchor=bbox_anchor,
                              fontsize=legend_fontsize,
                              frameon=True, 
                              edgecolor='black', 
                              prop={'weight': 'bold'})
            for text, handle in zip(legend.get_texts(), reversed_handles):
                text.set_color(handle.get_color())
        else:
            legend = ax.legend(handles, labels, 
                              loc=loc, 
                              bbox_to_anchor=bbox_anchor,
                              fontsize=legend_fontsize,
                              frameon=True, 
                              edgecolor='black', 
                              prop={'weight': 'bold'})
            if legend_position == "right":
                for text, handle in zip(legend.get_texts(), handles):
                    text.set_color(handle.get_color())
        
        # Adjust legend box if it's too large
        if len(handles) > 10:
            legend._legend_box.align = "left"
            if len(handles) > 15:
                legend._ncol = 2
    
    # Get y-limits for text placement at top
    y_min, y_max = ax.get_ylim()
    y_text_position = y_max + (y_max - y_min) * 0.02  # Position above top edge
    
    # Draw all markers
    for marker in markers:
        # Skip pending markers (they are shown as preview instead)
        if marker.get('pending', False):
            continue
            
        marker_type = marker.get('type', 'line')
        position = marker.get('position')
        width = marker.get('width', 0)
        name = marker.get('name', '')
        color = marker.get('color', '#000000')
        
        if marker_type == 'line':
            # Draw vertical line
            ax.axvline(x=position, color=color, linestyle='--', linewidth=1.2, alpha=1.0)
            
            # Add text label above top edge if show_x_values is enabled
            if show_x_values:
                label_text = f"{position:.1f}" if not name else f"{name} {position:.1f}"
                ax.text(position, y_text_position, label_text, 
                       ha='center', va='bottom', fontsize=8, 
                       color=color, fontweight='bold',
                       rotation=45, rotation_mode='anchor')
        
        elif marker_type == 'region':
            # Draw region as semi-transparent vertical band
            half_width = width / 2
            x_start = position - half_width
            x_end = position + half_width
            ax.axvspan(x_start, x_end, alpha=0.3, color=color)
            
            # Draw boundary lines
            ax.axvline(x=x_start, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axvline(x=x_end, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
            # Draw center line
            ax.axvline(x=position, color=color, linestyle='--', linewidth=1.0, alpha=0.5)
            
            # Add text label above top edge if show_x_values is enabled
            if show_x_values:
                label_text = f"{position:.1f} ± {half_width:.1f}" if not name else f"{name} {position:.1f} ± {half_width:.1f}"
                ax.text(position, y_text_position, label_text, 
                       ha='center', va='bottom', fontsize=8, 
                       color=color, fontweight='bold',
                       rotation=45, rotation_mode='anchor')
    
    # Draw preview line/region (only if show_preview is True)
    if show_preview and preview_position is not None:
        if is_region_mode and preview_width > 0:
            # Preview region
            half_width = preview_width / 2
            x_start = preview_position - half_width
            x_end = preview_position + half_width
            ax.axvspan(x_start, x_end, alpha=0.2, color='gray')
            ax.axvline(x=preview_position, color='gray', linestyle='-.', linewidth=1.0, alpha=0.5)
            ax.axvline(x=x_start, color='gray', linestyle='-.', linewidth=0.5, alpha=0.3)
            ax.axvline(x=x_end, color='gray', linestyle='-.', linewidth=0.5, alpha=0.3)
            
            # Add preview text if show_x_values is enabled
            if show_x_values:
                preview_text = f"Preview: {preview_position:.1f} ± {half_width:.1f}"
                ax.text(preview_position, y_text_position, preview_text, 
                       ha='center', va='bottom', fontsize=8, 
                       color='gray', fontweight='bold', alpha=0.7,
                       rotation=45, rotation_mode='anchor')
        else:
            # Preview line
            ax.axvline(x=preview_position, color='gray', linestyle='-.', linewidth=1.0, alpha=0.5)
            
            # Add preview text if show_x_values is enabled
            if show_x_values:
                preview_text = f"Preview: {preview_position:.1f}"
                ax.text(preview_position, y_text_position, preview_text, 
                       ha='center', va='bottom', fontsize=8, 
                       color='gray', fontweight='bold', alpha=0.7,
                       rotation=45, rotation_mode='anchor')
    
    ax.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    else:
        ax.grid(False)
    
    # Dynamic right margin adjustment based on legend size and position
    if legend_position == "right" and len(handles) > 0:
        # Calculate margin based on number of spectra and font size
        estimated_legend_width = min(0.35, 0.15 + (len(handles) * legend_fontsize / 300))
        right_margin = min(0.95, legend_offset + estimated_legend_width)
        plt.tight_layout()
        plt.subplots_adjust(right=right_margin, top=0.92)  # Make room for labels above
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for labels above
    
    return fig

# Main app
def main():
    # Custom header with logo
    import os
    from PIL import Image
    
    # Check if logo exists
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        # Display logo centered with text below
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo, width=250, use_container_width=False)
            st.markdown("""
            <div style="text-align: center;">
                <h1 style="margin: 0; color: white; font-size: 2rem; font-weight: 700;">SpectrAnalys</h1>
                <p style="margin: 0.5rem 0 0 0; color: white; opacity: 0.9;">Advanced Spectroscopic Data Analysis Platform | High-Precision Peak Detection & Correlation Analysis</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback to text header if logo not found
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
        
        # NEW: Remove all spectra button
        if uploaded_files and st.session_state.get('spectra_loaded', False):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🗑️ Remove all spectra", type="secondary", use_container_width=True):
                    # Clear all spectrum-related session state but preserve UI settings
                    st.session_state.spectra_loaded = False
                    st.session_state.cached_spectra_data = None
                    st.session_state.peak_analysis_triggered = False
                    st.session_state.peak_analysis_results = None
                    st.session_state.correlation_ready = False
                    st.session_state.excluded_peaks = set()
                    st.session_state.heatmap_applied = False
                    st.session_state.heatmap_params = {}
                    st.session_state.spectral_markers = []
                    # Clear file uploader by rerunning
                    st.rerun()
            with col2:
                st.markdown("")
        
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
                    
                    # Normalization options - Area removed
                    st.markdown("#### 📐 Normalization")
                    
                    # Build normalization options based on x_range_option
                    if x_range_option == "Custom ranges (multiple)":
                        norm_options = ["Maximum intensity", "Peak intensity (range)", "Maximum rest intensity"]
                    else:
                        norm_options = ["Maximum intensity", "Peak intensity (range)"]
                    
                    norm_method = st.selectbox(
                        "Normalization method",
                        norm_options,
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
                    
                    # NEW: Fill transparency slider (replaces fill_type)
                    fill_alpha = 0.3
                    if fill_area:
                        fill_alpha = st.slider(
                            "Fill transparency",
                            min_value=0.2,
                            max_value=0.9,
                            value=0.3,
                            step=0.1,
                            help="0.2 = more transparent, 0.9 = more opaque"
                        )
                    
                    # NEW: Subtract minimum intensity checkbox
                    subtract_min_intensity = st.checkbox("Subtract minimum intensity (start from zero)", value=False)
                    
                    # NEW: Plot settings (Grid and Linewidth)
                    st.markdown("#### 🎨 Plot Settings")
                    col1, col2 = st.columns(2)
                    with col1:
                        show_grid = st.checkbox("Show grid on plots", value=True)
                    with col2:
                        line_width = st.slider(
                            "Spectrum line thickness",
                            min_value=0.5,
                            max_value=3.0,
                            value=1.5,
                            step=0.1
                        )
                    
                    # NEW: Figure size selector for individual plots
                    st.markdown("#### 📐 Plot Size (Width × Height)")
                    fig_size_options = {
                        "3×3": (3, 3),
                        "3×4": (4, 3),
                        "3×5": (5, 3),
                        "3×6": (6, 3),
                        "3×7": (7, 3),
                        "3×9": (9, 3)
                    }
                    selected_size = st.selectbox(
                        "Select plot dimensions (width × height in inches)",
                        options=list(fig_size_options.keys()),
                        index=2
                    )
                    fig_width, fig_height = fig_size_options[selected_size]

                    # Legend settings
                    st.markdown("#### 🏷️ Legend Settings")
                    col1, col2 = st.columns(2)
                    with col1:
                        legend_fontsize = st.slider(
                            "Legend font size",
                            min_value=4,
                            max_value=16,
                            value=8,
                            step=1,
                            key="legend_fontsize"
                        )
                    with col2:
                        legend_position = st.selectbox(
                            "Legend position",
                            options=["right", "best", "upper right", "upper left", "lower left", "lower right"],
                            index=0,
                            key="legend_position"
                        )
                    
                    # Legend size (width adjustment)
                    legend_offset = st.slider(
                        "Legend offset from plot (0.5-2.0)",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.02,
                        step=0.02,
                        help="Higher value moves legend further right (for right position)"
                    )
                    
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
                    
                    # NEW: Heatmap Parameters Section
                    st.markdown("---")
                    st.markdown("### 📊 Heatmap Parameters")
                    st.markdown("*Assign numeric values (temperature, concentration, etc.) to each spectrum for heatmap visualization*")
                    
                    # Parameter type selection
                    heatmap_param_type = st.selectbox(
                        "Parameter type",
                        options=["Temperature (°C)", "Concentration (x)", "Custom"],
                        index=0,
                        key="heatmap_param_type_select"
                    )
                    
                    # Custom label if Custom is selected
                    heatmap_custom_label = ""
                    if heatmap_param_type == "Custom":
                        heatmap_custom_label = st.text_input(
                            "Custom parameter label",
                            value="Parameter",
                            key="heatmap_custom_label"
                        )
                    
                    # Determine the label for the heatmap y-axis
                    if heatmap_param_type == "Temperature (°C)":
                        heatmap_y_label = "Temperature (°C)"
                    elif heatmap_param_type == "Concentration (x)":
                        heatmap_y_label = "Concentration (x)"
                    else:
                        heatmap_y_label = heatmap_custom_label if heatmap_custom_label else "Parameter"
                    
                    # Create input fields for each spectrum
                    st.markdown("#### Assign values to spectra:")
                    
                    # Use a container with no rerun on change
                    heatmap_params_temp = {}
                    for name in ordered_spectra:
                        display_name = name.replace('.txt', '')
                        # Use a unique key for each input
                        param_key = f"heatmap_{name}"
                        heatmap_params_temp[name] = st.number_input(
                            f"{display_name}",
                            value=st.session_state.heatmap_params.get(name, len(heatmap_params_temp) + 1.0),
                            step=0.1,
                            format="%.1f",
                            key=param_key
                        )
                    
                    # Heatmap visualization settings
                    st.markdown("#### 🎨 Heatmap Settings")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Interpolation methods
                        interpolation_options = {
                            'none': 'None (discrete)',
                            'bilinear': 'Bilinear (smooth)',
                            'bicubic': 'Bicubic (smooth)',
                            'spline16': 'Spline16 (very smooth)',
                            'spline36': 'Spline36 (very smooth)',
                            'gaussian': 'Gaussian (blur)',
                            'lanczos': 'Lanczos (sharp)'
                        }
                        heatmap_interpolation = st.selectbox(
                            "Interpolation method",
                            options=list(interpolation_options.keys()),
                            format_func=lambda x: interpolation_options[x],
                            index=5,  # Default to 'gaussian'
                            key="heatmap_interpolation_select"
                        )
                    
                    with col2:
                        # Colormap selection - 10 options
                        colormap_options = {
                            'viridis': 'Viridis (perceptual)',
                            'plasma': 'Plasma (perceptual)',
                            'inferno': 'Inferno (perceptual)',
                            'magma': 'Magma (perceptual)',
                            'cividis': 'Cividis (colorblind)',
                            'Spectral_r': 'Spectral (rainbow)',
                            'coolwarm': 'Cool-Warm (diverging)',
                            'RdYlBu_r': 'Red-Yellow-Blue (diverging)',
                            'jet': 'Jet (classic)',
                            'turbo': 'Turbo (improved jet)'
                        }
                        heatmap_colormap = st.selectbox(
                            "Color palette",
                            options=list(colormap_options.keys()),
                            format_func=lambda x: colormap_options[x],
                            index=0,
                            key="heatmap_colormap_select"
                        )
                    
                    # Apply button for heatmaps
                    apply_heatmap = st.button(
                        "🔄 Apply for heatmaps",
                        use_container_width=True,
                        key="apply_heatmap_button"
                    )
                    
                    if apply_heatmap:
                        # Store heatmap parameters in session state
                        st.session_state.heatmap_params = heatmap_params_temp
                        st.session_state.heatmap_param_type = heatmap_param_type
                        st.session_state.heatmap_interpolation = heatmap_interpolation
                        st.session_state.heatmap_colormap = heatmap_colormap
                        st.session_state.heatmap_y_label = heatmap_y_label
                        st.session_state.heatmap_applied = True
                        st.session_state.heatmap_ordered_names = ordered_spectra
                        st.session_state.heatmap_x_ranges = x_ranges  # <-- СОХРАНЯЕМ x_ranges в session_state
                        
                        # Prepare heatmap data
                        spectra_matrix, spectra_norm_matrix, x_grid, y_values = prepare_heatmap_data(
                            spectra_data, ordered_spectra, heatmap_params_temp, 
                            norm_method, norm_range, x_ranges
                        )
                        
                        if spectra_matrix is not None:
                            st.session_state.heatmap_spectra_matrix = spectra_matrix
                            st.session_state.heatmap_spectra_norm_matrix = spectra_norm_matrix
                            st.session_state.heatmap_x_grid = x_grid
                            st.session_state.heatmap_y_values = y_values
                            st.success(f"✅ Heatmap data prepared! {len(y_values)} spectra with {len(x_grid)} points each.")
                        else:
                            st.error("❌ Failed to prepare heatmap data. Check that all spectra are valid.")
                            st.session_state.heatmap_applied = False
                    
                    # Color Assignment moved to the bottom
                    st.markdown("---")
                    st.markdown("### 🎨 Color Assignment")
                    
                    # Define default color palette
                    default_colors = [
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
                    ]
                    
                    colors = {}
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
                    
                    # Store in session state for independent tabs
                    st.session_state.spectra_loaded = True
                    st.session_state.cached_spectra_data = {
                        'spectra_data': spectra_data,
                        'ordered_spectra': ordered_spectra,
                        'x_label': x_label,
                        'y_label': y_label,
                        'norm_method': norm_method,
                        'norm_range': norm_range,
                        'raw_offset_step': raw_offset_step,
                        'norm_offset_step': norm_offset_step,
                        'fill_area': fill_area,
                        'fill_alpha': fill_alpha,
                        'subtract_min_intensity': subtract_min_intensity,
                        'show_grid': show_grid,
                        'line_width': line_width,
                        'x_ranges': x_ranges,
                        'common_x_range': common_x_range,
                        'fig_width': fig_width,
                        'fig_height': fig_height,
                        'analyze_peaks_flag': analyze_peaks_flag,
                        'peak_width': peak_width if analyze_peaks_flag else 20,
                        'param_correlation': param_correlation,
                        'param_values': param_values if param_correlation else None,
                        'param_label': param_label if param_correlation else "Parameter",
                        'legend_fontsize': legend_fontsize,
                        'legend_position': legend_position,
                        'legend_offset': legend_offset,
                        # NEW: Heatmap settings
                        'heatmap_params': st.session_state.heatmap_params,
                        'heatmap_param_type': st.session_state.heatmap_param_type,
                        'heatmap_interpolation': st.session_state.heatmap_interpolation,
                        'heatmap_colormap': st.session_state.heatmap_colormap,
                        'heatmap_y_label': st.session_state.heatmap_y_label,
                        'heatmap_applied': st.session_state.heatmap_applied
                    }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar footer with info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p>🔬 SpectrAnalys v2.0<br>Scientific Spectroscopic Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if uploaded_files and st.session_state.get('spectra_loaded', False) and st.session_state.cached_spectra_data:
        # Load cached data
        cached = st.session_state.cached_spectra_data
        spectra_data = cached['spectra_data']
        ordered_spectra = cached['ordered_spectra']
        x_label = cached['x_label']
        y_label = cached['y_label']
        norm_method = cached['norm_method']
        norm_range = cached['norm_range']
        raw_offset_step = cached['raw_offset_step']
        norm_offset_step = cached['norm_offset_step']
        fill_area = cached['fill_area']
        fill_alpha = cached['fill_alpha']
        subtract_min_intensity = cached['subtract_min_intensity']
        show_grid = cached['show_grid']
        line_width = cached['line_width']
        x_ranges = cached['x_ranges']
        common_x_range = cached['common_x_range']
        fig_width = cached['fig_width']
        fig_height = cached['fig_height']
        analyze_peaks_flag = cached['analyze_peaks_flag']
        peak_width = cached['peak_width']
        param_correlation = cached['param_correlation']
        param_values = cached['param_values']
        param_label = cached['param_label']
        
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
        
        # Create tabs for different analysis views - NEW: Added comparison tab
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Combined Spectra Visualization",
            "🔍 Advanced Peak Analysis", 
            "📈 Parameter Correlation",
            "🔀 Compare Two Spectra",
            "📏 Spectral Markers"
        ])
        
        with tab1:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Comprehensive Spectra Analysis")
            st.markdown("*All visualization modes combined for comprehensive spectral comparison*")
            
            # Prepare normalized spectra for individual plots
            normalized_spectra = {}
            for name, spec in filtered_spectra.items():
                data = spec['data']
                y_norm = normalize_spectrum(
                    data['x'].values,
                    data['y'].values,
                    norm_method,
                    norm_range,
                    x_ranges  # Pass x_ranges for "Maximum rest intensity" method
                )
                normalized_spectra[name] = {
                    'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
                    'color': spec['color']
                }
            
            # Apply subtract minimum intensity if requested
            if subtract_min_intensity:
                for name in normalized_spectra:
                    y_vals = normalized_spectra[name]['data']['y'].values
                    if len(y_vals) > 0:
                        y_min = y_vals.min()
                        normalized_spectra[name]['data']['y'] = y_vals - y_min
                    else:
                        normalized_spectra[name]['data']['y'] = y_vals
            
            # Define the four visualization configurations (titles removed)
            viz_configs = [
                (filtered_spectra, 0, False, False, False, y_label),
                (normalized_spectra, 0, False, True, False, f"Normalized {y_label}"),
                (filtered_spectra, raw_offset_step, False, False, True, y_label),
                (normalized_spectra, norm_offset_step, fill_area, True, True, f"Normalized {y_label}")
            ]
            
            # Create and display individual plots
            for idx, (spectra, offset_step, fill, normalized, use_offset, yl) in enumerate(viz_configs):
                fig = create_individual_plot(
                    spectra, x_label, yl, "",
                    offset_step, fill, normalized, use_offset,
                    x_ranges, subtract_min_intensity, fill_alpha,
                    show_grid, line_width, fig_width, fig_height,
                    legend_fontsize=cached['legend_fontsize'],
                    legend_position=cached['legend_position'],
                    legend_offset=cached['legend_offset']
                )
                st.pyplot(fig)
                
                # Download button for individual plot
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode()
                plot_name = f"spectra_plot_{idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                st.markdown(f"""
                <div style="text-align: center; margin-top: 0rem; margin-bottom: 1rem;">
                    <a href="data:image/png;base64,{b64}" download="{plot_name}">
                        <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       color: white; border: none; border-radius: 8px; 
                                       padding: 0.3rem 0.8rem; cursor: pointer; font-size: 0.8rem;">
                            📥 Download Plot {idx+1} (PNG, 600 dpi)
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
                plt.close()
                
                # Add separator between plots
                if idx < len(viz_configs) - 1:
                    st.markdown('<div class="separator">****</div>', unsafe_allow_html=True)
            
            # NEW: Heatmap plots section (appears only if heatmap is applied)
            if st.session_state.get('heatmap_applied', False):
                st.markdown('<div class="separator">═══════════════════════════════════════════════════</div>', unsafe_allow_html=True)
                st.subheader("🔥 Heatmap Visualization")
                st.markdown("*Spectral evolution heatmaps showing intensity distribution as function of parameter*")
                
                # Get heatmap data from session state
                spectra_matrix = st.session_state.get('heatmap_spectra_matrix')
                spectra_norm_matrix = st.session_state.get('heatmap_spectra_norm_matrix')
                x_grid = st.session_state.get('heatmap_x_grid')
                y_values = st.session_state.get('heatmap_y_values')
                heatmap_y_label = st.session_state.get('heatmap_y_label', 'Parameter')
                heatmap_interpolation = st.session_state.get('heatmap_interpolation', 'gaussian')
                heatmap_colormap = st.session_state.get('heatmap_colormap', 'viridis')
                heatmap_x_ranges = st.session_state.get('heatmap_x_ranges', None)
                
                if spectra_matrix is not None and x_grid is not None and y_values is not None:
                    # Determine if we should use log scale
                    # Use log scale if intensity values span more than 2 orders of magnitude
                    min_val = np.min(spectra_matrix[spectra_matrix > 0]) if np.any(spectra_matrix > 0) else 1
                    max_val = np.max(spectra_matrix)
                    use_log = (max_val / min_val) > 100 if min_val > 0 else False
                    
                    # Create heatmap for raw intensity
                    fig_heatmap = create_heatmap(
                        spectra_matrix, x_grid, y_values,
                        x_label, heatmap_y_label,
                        "Intensity (a.u.)",
                        heatmap_colormap, heatmap_interpolation,
                        f"Intensity Heatmap: {heatmap_y_label} vs Raman shift",
                        fig_width=12, fig_height=8,
                        log_scale=use_log
                    )
                    st.pyplot(fig_heatmap)
                    
                    # Download button for heatmap
                    buf = BytesIO()
                    fig_heatmap.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                    buf.seek(0)
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 0rem; margin-bottom: 1rem;">
                        <a href="data:image/png;base64,{b64}" download="heatmap_intensity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                            <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                           color: white; border: none; border-radius: 8px; 
                                           padding: 0.3rem 0.8rem; cursor: pointer; font-size: 0.8rem;">
                                📥 Download Heatmap (Intensity) (PNG, 600 dpi)
                            </button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                    plt.close(fig_heatmap)
                    
                    # Create heatmap for normalized intensity
                    fig_heatmap_norm = create_heatmap(
                        spectra_norm_matrix, x_grid, y_values,
                        x_label, heatmap_y_label,
                        "Normalized Intensity (a.u.)",
                        heatmap_colormap, heatmap_interpolation,
                        f"Normalized Intensity Heatmap: {heatmap_y_label} vs Raman shift",
                        fig_width=12, fig_height=8,
                        log_scale=True
                    )
                    st.pyplot(fig_heatmap_norm)
                    
                    # Download button for normalized heatmap
                    buf = BytesIO()
                    fig_heatmap_norm.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                    buf.seek(0)
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 0rem; margin-bottom: 1rem;">
                        <a href="data:image/png;base64,{b64}" download="heatmap_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                            <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                           color: white; border: none; border-radius: 8px; 
                                           padding: 0.3rem 0.8rem; cursor: pointer; font-size: 0.8rem;">
                                📥 Download Heatmap (Normalized) (PNG, 600 dpi)
                            </button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                    plt.close(fig_heatmap_norm)
                    
                    # Show parameter values used
                    st.markdown("#### 📊 Heatmap Parameters Used:")
                    param_df = pd.DataFrame({
                        'Spectrum': [name.replace('.txt', '') for name in st.session_state.heatmap_ordered_names if name in st.session_state.heatmap_params],
                        heatmap_y_label: [st.session_state.heatmap_params[name] for name in st.session_state.heatmap_ordered_names if name in st.session_state.heatmap_params]
                    })
                    st.dataframe(param_df, use_container_width=True)
                    
                    # Add interpolation and colormap info
                    st.caption(f"Interpolation: {heatmap_interpolation} | Colormap: {heatmap_colormap} | Log scale: {'Yes' if use_log else 'No'} for intensity, Yes for normalized")
                else:
                    st.warning("⚠️ Heatmap data not available. Please click 'Apply for heatmaps' in the sidebar.")
            else:
                # Show hint for heatmap
                st.info("💡 To generate heatmaps, assign numeric values to spectra in the sidebar and click 'Apply for heatmaps'.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Peak Detection and Analysis")
            st.markdown("*Select spectral range manually for precise peak analysis*")
            
            if analyze_peaks_flag and filtered_spectra:
                # Display full spectra for range selection
                st.markdown("#### 📊 Select Analysis Range")
                st.markdown("Use the sliders below to select left and right boundaries for peak analysis")
                
                # Get global x range
                all_x = []
                for spec in filtered_spectra.values():
                    all_x.extend(spec['data']['x'].values)
                global_min_x = float(np.min(all_x))
                global_max_x = float(np.max(all_x))
                
                # Create range sliders
                col1, col2 = st.columns(2)
                with col1:
                    left_boundary = st.slider(
                        "Left boundary (cm⁻¹)",
                        min_value=global_min_x,
                        max_value=global_max_x,
                        value=global_min_x,
                        step=(global_max_x - global_min_x) / 100,
                        key="left_boundary"
                    )
                with col2:
                    right_boundary = st.slider(
                        "Right boundary (cm⁻¹)",
                        min_value=global_min_x,
                        max_value=global_max_x,
                        value=global_max_x,
                        step=(global_max_x - global_min_x) / 100,
                        key="right_boundary"
                    )
                
                # Ensure left < right
                if left_boundary >= right_boundary:
                    st.warning("⚠️ Left boundary must be less than right boundary")
                    manual_range = (None, None)
                else:
                    manual_range = (left_boundary, right_boundary)
                
                # Display full spectra with range boundaries
                fig_range, ax_range = plt.subplots(figsize=(12, 5))
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    ax_range.plot(data['x'].values, data['y'].values, 
                                 color=spec['color'], linewidth=1.5, 
                                 label=name.replace('.txt', ''), alpha=0.7)
                
                # Add range boundaries
                if left_boundary < right_boundary:
                    ax_range.axvline(left_boundary, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Left: {left_boundary:.1f}')
                    ax_range.axvline(right_boundary, color='blue', linestyle='-', linewidth=2, alpha=0.7, label=f'Right: {right_boundary:.1f}')
                    # Highlight selected range
                    ax_range.axvspan(left_boundary, right_boundary, alpha=0.2, color='gray')
                
                ax_range.set_xlabel(x_label, fontsize=11, fontweight='bold')
                ax_range.set_ylabel(y_label, fontsize=11, fontweight='bold')
                ax_range.set_title("Full Spectra with Selected Analysis Range", fontsize=12, fontweight='bold')
                ax_range.legend(loc='best', fontsize=9)
                ax_range.tick_params(direction='in', length=5, width=1)
                ax_range.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                st.pyplot(fig_range)
                plt.close()
                
                # Run analysis button
                if st.button("🚀 Analyze", key="run_peak_analysis"):
                    if left_boundary < right_boundary:
                        with st.spinner("Analyzing peaks..."):
                            peaks_df = analyze_peaks_manual_range(
                                filtered_spectra, 
                                manual_range, 
                                peak_width
                            )
                            st.session_state.peak_analysis_results = peaks_df
                            st.session_state.peak_analysis_triggered = True
                            st.session_state.peak_analysis_x_range = manual_range
                            st.session_state.excluded_peaks = set()
                            st.success(f"✅ Peak analysis complete! Found {len(peaks_df)} peaks total.")
                    else:
                        st.error("Please select a valid range (left < right)")
                
                # Display results if analysis has been run
                if st.session_state.peak_analysis_triggered and st.session_state.peak_analysis_results is not None:
                    peaks_df = st.session_state.peak_analysis_results.copy()
                    
                    if not peaks_df.empty:
                        st.markdown("---")
                        st.subheader("📊 Peak Analysis Results")
                        st.markdown("*Check/Uncheck peaks to include/exclude them from visualization and correlation analysis*")
                        
                        # Display peak statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Peaks Detected", len(peaks_df))
                        with col2:
                            st.metric("Unique Spectra", peaks_df['Spectrum'].nunique())
                        with col3:
                            st.metric("Avg Peak Intensity", f"{peaks_df['Intensity'].mean():.2f}")
                        with col4:
                            st.metric("Avg FWHM", f"{peaks_df['FWHM'].mean():.2f}")
                        
                        st.markdown("---")
                        
                        # Create editable dataframe with checkboxes
                        # Generate unique IDs for each row
                        peaks_df['temp_id'] = range(len(peaks_df))
                        
                        # Display dataframe with checkboxes using st.data_editor
                        edited_df = st.data_editor(
                            peaks_df[['Spectrum', 'Peak position', 'Intensity', 'Area', 'FWHM', 'Include', 'temp_id']],
                            column_config={
                                'Include': st.column_config.CheckboxColumn(
                                    "Include?",
                                    help="Check to include this peak in analysis",
                                    default=True
                                ),
                                'temp_id': None  # Hide the temp_id column
                            },
                            disabled=['Spectrum', 'Peak position', 'Intensity', 'Area', 'FWHM'],
                            hide_index=True,
                            use_container_width=True,
                            key="peak_editor"
                        )
                        
                        # Update the Include column based on user edits
                        if edited_df is not None:
                            # Create a mapping from temp_id to new Include value
                            include_map = dict(zip(edited_df['temp_id'], edited_df['Include']))
                            # Update peaks_df
                            peaks_df['Include'] = peaks_df['temp_id'].map(include_map)
                            # Store updated dataframe in session state
                            st.session_state.peak_analysis_results = peaks_df.drop('temp_id', axis=1)
                            peaks_df = peaks_df.drop('temp_id', axis=1)
                        else:
                            peaks_df = peaks_df.drop('temp_id', axis=1)
                        
                        # Download button for peak analysis (with current include/exclude status)
                        csv = peaks_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download peak analysis as CSV",
                            data=csv,
                            file_name=f"peak_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualize peaks with selected range (only included peaks)
                        st.markdown("---")
                        st.subheader("🔍 Peak Visualization")
                        st.markdown("*Only checked peaks are shown*")
                        fig_peaks = create_peak_visualization(
                            filtered_spectra, 
                            st.session_state.peak_analysis_x_range,
                            peaks_df
                        )
                        st.pyplot(fig_peaks)
                        
                        # Download peak visualization
                        buf = BytesIO()
                        fig_peaks.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 1rem;">
                            <a href="data:image/png;base64,{b64}" download="peak_visualization.png">
                                <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                               color: white; border: none; border-radius: 8px; 
                                               padding: 0.5rem 1rem; cursor: pointer;">
                                    📥 Download Peak Visualization (PNG)
                                </button>
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                        plt.close()
                        
                        # Store for correlation tab
                        st.session_state.correlation_ready = True
                        st.session_state.correlation_peaks_df = peaks_df
                    else:
                        st.info("ℹ️ No peaks detected in the selected range. Try adjusting the range boundaries.")
            else:
                st.info("🔍 Enable advanced peak analysis in the sidebar to detect and analyze peaks in your spectra.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Parameter Correlation Analysis")
            st.markdown("*Correlate spectral features (Intensity, Area, Position, FWHM) with experimental parameters*")
            
            if param_correlation and param_values and st.session_state.get('correlation_ready', False) and st.session_state.get('correlation_peaks_df') is not None:
                peaks_df = st.session_state.correlation_peaks_df
                
                # Filter to only include peaks marked as Include=True
                peaks_df = peaks_df[peaks_df['Include'] == True]
                
                if peaks_df.empty:
                    st.warning("⚠️ No peaks are currently included. Please check at least one peak in the Peak Analysis tab.")
                else:
                    # Prepare data for correlation
                    param_list = []
                    intensity_list = []
                    area_list = []
                    position_list = []
                    fwhm_list = []
                    
                    for name in ordered_spectra:
                        if name in param_values:
                            spec_peaks = peaks_df[peaks_df['Spectrum'] == name.replace('.txt', '')]
                            if not spec_peaks.empty:
                                # Take the most intense peak from included peaks
                                main_peak = spec_peaks.loc[spec_peaks['Intensity'].idxmax()]
                                param_list.append(param_values[name])
                                intensity_list.append(main_peak['Intensity'])
                                area_list.append(main_peak['Area'])
                                position_list.append(main_peak['Peak position'])
                                fwhm_list.append(main_peak['FWHM'])
                    
                    if param_list:
                        # Calculate correlation coefficients
                        corr_intensity = pearsonr(param_list, intensity_list)[0] if len(param_list) > 2 else 0
                        corr_area = pearsonr(param_list, area_list)[0] if len(param_list) > 2 else 0
                        corr_position = pearsonr(param_list, position_list)[0] if len(param_list) > 2 else 0
                        corr_fwhm = pearsonr(param_list, fwhm_list)[0] if len(param_list) > 2 else 0
                        
                        # Display correlation metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Intensity Correlation", f"{corr_intensity:.3f}", 
                                     delta="strong" if abs(corr_intensity) > 0.7 else "weak")
                        with col2:
                            st.metric("Area Correlation", f"{corr_area:.3f}",
                                     delta="strong" if abs(corr_area) > 0.7 else "weak")
                        with col3:
                            st.metric("Position Correlation", f"{corr_position:.3f}",
                                     delta="strong" if abs(corr_position) > 0.7 else "weak")
                        with col4:
                            st.metric("FWHM Correlation", f"{corr_fwhm:.3f}",
                                     delta="strong" if abs(corr_fwhm) > 0.7 else "weak")
                        
                        st.markdown("---")
                        
                        # Create correlation plots (4 plots: Intensity, Area, Position, FWHM)
                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                        
                        # Intensity plot
                        axes[0, 0].scatter(param_list, intensity_list, c='#1f77b4', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[0, 0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[0, 0].set_ylabel("Peak Intensity (a.u.)", fontsize=11, fontweight='bold')
                        axes[0, 0].set_title(f"Intensity vs {param_label}\n(r = {corr_intensity:.3f})", fontsize=12, fontweight='bold')
                        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
                        
                        # Area plot
                        axes[0, 1].scatter(param_list, area_list, c='#2ca02c', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[0, 1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[0, 1].set_ylabel("Peak Area", fontsize=11, fontweight='bold')
                        axes[0, 1].set_title(f"Area vs {param_label}\n(r = {corr_area:.3f})", fontsize=12, fontweight='bold')
                        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
                        
                        # Position plot
                        axes[1, 0].scatter(param_list, position_list, c='#d62728', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[1, 0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[1, 0].set_ylabel("Peak Position (cm⁻¹)", fontsize=11, fontweight='bold')
                        axes[1, 0].set_title(f"Position vs {param_label}\n(r = {corr_position:.3f})", fontsize=12, fontweight='bold')
                        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
                        
                        # FWHM plot
                        axes[1, 1].scatter(param_list, fwhm_list, c='#9467bd', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[1, 1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[1, 1].set_ylabel("FWHM (cm⁻¹)", fontsize=11, fontweight='bold')
                        axes[1, 1].set_title(f"FWHM vs {param_label}\n(r = {corr_fwhm:.3f})", fontsize=12, fontweight='bold')
                        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Show correlation table
                        st.markdown("---")
                        st.subheader("Correlation Data Table")
                        corr_data = pd.DataFrame({
                            'Spectrum': [name.replace('.txt', '') for name in ordered_spectra if name in param_values and name.replace('.txt', '') in peaks_df['Spectrum'].values],
                            param_label: param_list,
                            'Intensity': intensity_list,
                            'Area': area_list,
                            'Position': position_list,
                            'FWHM': fwhm_list
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
                        st.info("ℹ️ No matching peaks found for correlation analysis. Make sure peaks are detected and included.")
            elif param_correlation and not st.session_state.get('correlation_ready', False):
                st.info("📊 Please run peak analysis in the 'Advanced Peak Analysis' tab first to obtain peak data for correlation")
            elif param_correlation and not param_values:
                st.info("📊 Please assign parameter values in the sidebar for correlation analysis")
            else:
                st.info("📊 Enable parameter correlation in the sidebar and assign numeric values to spectra for correlation analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # NEW TAB: Compare Two Spectra
        with tab4:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("🔀 Spectral Difference Analysis")
            st.markdown("*Compare two spectra to identify differences and visualize them with heatmaps*")
            
            if len(ordered_spectra) >= 2:
                # Select two spectra for comparison
                col1, col2 = st.columns(2)
                with col1:
                    spectrum_a_name = st.selectbox(
                        "Reference Spectrum",
                        options=ordered_spectra,
                        index=0,
                        key="ref_spectrum"
                    )
                with col2:
                    spectrum_b_name = st.selectbox(
                        "Sample Spectrum",
                        options=ordered_spectra,
                        index=min(1, len(ordered_spectra)-1),
                        key="sample_spectrum"
                    )
                
                # Option to swap difference direction
                swap_direction = st.checkbox("Swap difference direction (Sample - Reference)", value=True)
                
                st.markdown("---")
                
                # Difference analysis settings
                st.markdown("#### 🎨 Difference Plot Settings")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Colormap selection (10 palettes)
                    colormap_options = {
                        'RdBu_r': 'Red-Blue (diverging)',
                        'coolwarm': 'Cool-Warm (diverging)',
                        'seismic': 'Seismic (diverging)',
                        'PiYG': 'Pink-Green (diverging)',
                        'BrBG': 'Brown-Blue-Green (diverging)',
                        'RdYlBu': 'Red-Yellow-Blue (diverging)',
                        'Spectral': 'Spectral (rainbow)',
                        'viridis': 'Viridis (perceptual)',
                        'plasma': 'Plasma (perceptual)',
                        'magma': 'Magma (perceptual)'
                    }
                    selected_colormap = st.selectbox(
                        "Color palette for difference heatmap",
                        options=list(colormap_options.keys()),
                        format_func=lambda x: colormap_options[x],
                        index=0
                    )
                
                with col2:
                    # Smoothing option
                    smooth_difference = st.checkbox("Apply smoothing to difference profile", value=False)
                    smooth_sigma = 1.0
                    if smooth_difference:
                        smooth_sigma = st.slider(
                            "Smoothing sigma",
                            min_value=0.5,
                            max_value=5.0,
                            value=1.5,
                            step=0.5,
                            help="Higher values produce smoother difference profiles"
                        )
                
                col1, col2 = st.columns(2)
                with col1:
                    # Symmetric color scale
                    symmetric_scale = st.checkbox("Symmetric color scale (centered at zero)", value=True)
                
                with col2:
                    # Significance threshold
                    difference_threshold = st.number_input(
                        "Significance threshold (highlight regions with |difference| > threshold)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.01,
                        format="%.3f",
                        help="Regions with absolute difference exceeding this value will be highlighted"
                    )
                
                st.markdown("---")
                
                # Get the selected spectra data
                if spectrum_a_name and spectrum_b_name:
                    spectrum_a = filtered_spectra[spectrum_a_name]
                    spectrum_b = filtered_spectra[spectrum_b_name]
                    
                    name_a = spectrum_a_name.replace('.txt', '')
                    name_b = spectrum_b_name.replace('.txt', '')
                    
                    # Create comparison plot
                    with st.spinner("Generating comparison plot..."):
                        fig, (mean_diff, max_abs_diff, rms_diff, correlation) = create_comparison_plot(
                            spectrum_a, spectrum_b, name_a, name_b,
                            x_label, y_label, norm_method, norm_range,
                            norm_offset_step, fill_area, fill_alpha, subtract_min_intensity,
                            show_grid, line_width, fig_width, fig_height,
                            cached['legend_fontsize'], cached['legend_position'], cached['legend_offset'],
                            selected_colormap, smooth_difference, smooth_sigma,
                            symmetric_scale, difference_threshold
                        )
                        
                        # Display statistics
                        st.markdown("#### 📊 Difference Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Difference", f"{mean_diff:.4f}")
                        with col2:
                            st.metric("Max |Difference|", f"{max_abs_diff:.4f}")
                        with col3:
                            st.metric("RMS Difference", f"{rms_diff:.4f}")
                        with col4:
                            st.metric("Spectral Correlation", f"{correlation:.4f}", 
                                     delta="strong" if abs(correlation) > 0.7 else "weak")
                        
                        # Display the plot
                        st.pyplot(fig)
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            # Download plot as PNG
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                            buf.seek(0)
                            b64 = base64.b64encode(buf.getvalue()).decode()
                            st.markdown(f"""
                            <div style="text-align: center;">
                                <a href="data:image/png;base64,{b64}" download="comparison_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                   color: white; border: none; border-radius: 8px; 
                                                   padding: 0.5rem 1rem; cursor: pointer;">
                                        📥 Download Comparison Plot (PNG)
                                    </button>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Export difference data
                            # Recalculate difference data for export
                            x_a = spectrum_a['data']['x'].values
                            y_a_raw = spectrum_a['data']['y'].values
                            y_a_norm = normalize_spectrum(x_a, y_a_raw, norm_method, norm_range)
                            x_b = spectrum_b['data']['x'].values
                            y_b_raw = spectrum_b['data']['y'].values
                            y_b_norm = normalize_spectrum(x_b, y_b_raw, norm_method, norm_range)
                            
                            common_x_min = max(x_a.min(), x_b.min())
                            common_x_max = min(x_a.max(), x_b.max())
                            common_x_exp = np.linspace(common_x_min, common_x_max, 2000)
                            y_a_interp_exp = np.interp(common_x_exp, x_a, y_a_norm)
                            y_b_interp_exp = np.interp(common_x_exp, x_b, y_b_norm)
                            
                            if subtract_min_intensity:
                                y_a_interp_exp = y_a_interp_exp - y_a_interp_exp.min()
                                y_b_interp_exp = y_b_interp_exp - y_b_interp_exp.min()
                            
                            if swap_direction:
                                y_diff_exp = y_b_interp_exp - y_a_interp_exp
                            else:
                                y_diff_exp = y_a_interp_exp - y_b_interp_exp
                            
                            diff_df = pd.DataFrame({
                                'x': common_x_exp,
                                f'{name_a}_normalized': y_a_interp_exp,
                                f'{name_b}_normalized': y_b_interp_exp,
                                'difference': y_diff_exp
                            })
                            
                            csv_diff = diff_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Export Difference Data (CSV)",
                                data=csv_diff,
                                file_name=f"difference_data_{name_a}_{name_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        plt.close(fig)
                        
            else:
                st.warning("⚠️ Please load at least 2 spectra to use the comparison feature.")
                st.info("Upload multiple .txt files to compare different samples or treatments.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        

        # NEW TAB 5: Spectral Markers
        with tab5:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("📏 Spectral Markers")
            st.markdown("*Add vertical lines or regions to track spectral features across different conditions*")
            
            # Check if there are spectra to work with
            if filtered_spectra:
                # Prepare normalized spectra (for selection)
                normalized_spectra = {}
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    y_norm = normalize_spectrum(
                        data['x'].values,
                        data['y'].values,
                        norm_method,
                        norm_range,
                        x_ranges
                    )
                    normalized_spectra[name] = {
                        'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
                        'color': spec['color']
                    }
                
                # Apply subtract minimum intensity if requested
                if subtract_min_intensity:
                    for name in normalized_spectra:
                        y_vals = normalized_spectra[name]['data']['y'].values
                        if len(y_vals) > 0:
                            y_min = y_vals.min()
                            normalized_spectra[name]['data']['y'] = y_vals - y_min
                        else:
                            normalized_spectra[name]['data']['y'] = y_vals
                
                # Define the four visualization configurations
                viz_configs = [
                    (filtered_spectra, 0, False, False, False, y_label),
                    (normalized_spectra, 0, False, True, False, f"Normalized {y_label}"),
                    (filtered_spectra, raw_offset_step, False, False, True, y_label),
                    (normalized_spectra, norm_offset_step, fill_area, True, True, f"Normalized {y_label}")
                ]
                
                plot_labels = [
                    "1. Raw Spectra (No Offset)",
                    "2. Normalized Spectra (No Offset)",
                    "3. Raw Spectra (With Offset)",
                    "4. Normalized Spectra (With Offset)"
                ]
                
                # Let user select which plot to work with
                st.markdown("#### Select plot for marker placement")
                selected_plot_idx = st.radio(
                    "Choose a plot type:",
                    options=range(len(plot_labels)),
                    format_func=lambda x: plot_labels[x],
                    index=st.session_state.spectral_markers_selected_plot,
                    key="markers_plot_select"
                )
                st.session_state.spectral_markers_selected_plot = selected_plot_idx
                
                # Get the selected spectra and settings
                selected_spectra, selected_offset_step, selected_fill, selected_normalized, selected_use_offset, selected_yl = viz_configs[selected_plot_idx]
                
                # Get global x range for sliders
                all_x = []
                for spec in selected_spectra.values():
                    all_x.extend(spec['data']['x'].values)
                if all_x:
                    global_min_x = float(np.min(all_x))
                    global_max_x = float(np.max(all_x))
                else:
                    st.warning("No data available in selected spectra")
                    st.stop()
                
                # Check if there's a pending line to convert to region
                pending_line = None
                for marker in st.session_state.spectral_markers:
                    if marker.get('pending', False):
                        pending_line = marker
                        break
                
                # --- Add New Marker Section (FIRST) ---
                st.markdown("---")
                st.markdown("#### ✏️ Add New Marker")
                
                if pending_line is None:
                    # --- LINE MODE ---
                    st.info("📍 **Step 1: Add a vertical line**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Position slider for line
                        marker_position = st.slider(
                            "Line position (X value)",
                            min_value=global_min_x,
                            max_value=global_max_x,
                            value=(global_min_x + global_max_x) / 2,
                            step=(global_max_x - global_min_x) / 500,
                            key="marker_position_slider"
                        )
                        # Update preview position
                        st.session_state.spectral_markers_preview_position = marker_position
                    
                    with col2:
                        # Color picker for new marker
                        marker_color = st.color_picker(
                            "Marker color",
                            value=st.session_state.spectral_markers_temp_color,
                            key="marker_color_picker"
                        )
                        st.session_state.spectral_markers_temp_color = marker_color
                    
                    # Name input
                    marker_name = st.text_input(
                        "Marker name (optional)",
                        value=st.session_state.spectral_markers_temp_name,
                        placeholder="e.g., I, Peak 1, Marker A",
                        key="marker_name_input"
                    )
                    st.session_state.spectral_markers_temp_name = marker_name
                    
                    # Add Line button
                    if st.button("➕ Add Line", use_container_width=True):
                        # Add a new line marker with pending=True to enable region expansion
                        new_marker = {
                            'type': 'line',
                            'position': marker_position,
                            'width': 0,
                            'name': marker_name if marker_name else "",
                            'color': marker_color,
                            'pending': True  # This line is pending for region expansion
                        }
                        st.session_state.spectral_markers.append(new_marker)
                        st.session_state.spectral_markers_temp_name = ""
                        st.success(f"✅ Line added at position {marker_position:.1f}. Now you can expand it to a region!")
                        st.rerun()
                    
                    # Display info about existing markers
                    if st.session_state.spectral_markers:
                        confirmed_count = len([m for m in st.session_state.spectral_markers if not m.get('pending', False)])
                        st.info(f"ℹ️ You have {confirmed_count} confirmed markers. Add a new line above.")
                
                else:
                    # --- REGION MODE ---
                    st.info(f"📍 **Step 2: Expand line at {pending_line['position']:.1f} to a region**")
                    st.markdown("*Use the slider below to set the region width (half-width on each side)*")
                    
                    # Width slider for region
                    region_width = st.slider(
                        "Region half-width",
                        min_value=0.1,
                        max_value=(global_max_x - global_min_x) / 4,
                        value=min(5.0, (global_max_x - global_min_x) / 20),
                        step=0.1,
                        key="region_width_slider"
                    )
                    st.session_state.spectral_markers_preview_width = region_width
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Confirm Region", use_container_width=True):
                            # Convert pending line to region
                            for i, marker in enumerate(st.session_state.spectral_markers):
                                if marker.get('pending', False):
                                    st.session_state.spectral_markers[i]['type'] = 'region'
                                    st.session_state.spectral_markers[i]['width'] = region_width
                                    st.session_state.spectral_markers[i]['pending'] = False
                                    st.success(f"✅ Region added: center={marker['position']:.1f}, half-width={region_width:.1f}")
                                    st.session_state.spectral_markers_preview_width = 0
                                    st.rerun()
                                    break
                    
                    with col2:
                        if st.button("❌ Cancel Region", use_container_width=True):
                            # Remove the pending marker (keep it as a line)
                            for i, marker in enumerate(st.session_state.spectral_markers):
                                if marker.get('pending', False):
                                    st.session_state.spectral_markers[i]['pending'] = False
                                    st.session_state.spectral_markers[i]['type'] = 'line'
                                    st.info(f"⏹️ Line kept at position {marker['position']:.1f} (not expanded to region)")
                                    st.session_state.spectral_markers_preview_width = 0
                                    st.rerun()
                                    break
                
                # --- PLOT WITH MARKERS (SECOND) ---
                st.markdown("---")
                st.markdown("#### 📊 Plot with Markers")
                
                # Show global options for the plot
                col1, col2 = st.columns(2)
                with col1:
                    show_values = st.checkbox(
                        "Show X values on plot",
                        value=st.session_state.spectral_markers_show_values,
                        key="markers_show_values"
                    )
                    st.session_state.spectral_markers_show_values = show_values
                
                with col2:
                    # Show/hide preview checkbox
                    show_preview = st.checkbox(
                        "Show preview line/region (during editing)",
                        value=st.session_state.spectral_markers_show_preview,
                        key="markers_show_preview"
                    )
                    st.session_state.spectral_markers_show_preview = show_preview
                    if not show_preview:
                        st.caption("🔒 Preview hidden. All markers are confirmed.")
                
                # Create the plot
                fig_markers = create_spectral_markers_plot(
                    selected_spectra, x_label, selected_yl,
                    selected_offset_step, selected_fill, selected_normalized,
                    selected_use_offset, x_ranges, subtract_min_intensity,
                    fill_alpha, show_grid, line_width, fig_width, fig_height,
                    cached['legend_fontsize'], cached['legend_position'],
                    cached['legend_offset'],
                    st.session_state.spectral_markers,
                    st.session_state.spectral_markers_preview_position,
                    st.session_state.spectral_markers_preview_width,
                    st.session_state.spectral_markers_show_values,
                    pending_line is not None,
                    st.session_state.spectral_markers_show_preview
                )
                
                # Display the plot
                st.pyplot(fig_markers)
                
                # Download button for the plot
                buf = BytesIO()
                fig_markers.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode()
                st.markdown(f"""
                <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
                    <a href="data:image/png;base64,{b64}" download="spectral_markers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                        <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       color: white; border: none; border-radius: 8px; 
                                       padding: 0.4rem 1rem; cursor: pointer; font-size: 0.9rem;">
                            📥 Download Plot with Markers (PNG, 600 dpi)
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
                plt.close(fig_markers)
                
                # --- Display and manage existing markers (THIRD) ---
                if st.session_state.spectral_markers:
                    st.markdown("---")
                    st.markdown("#### 📋 Existing Markers")
                    
                    # Prepare data for table
                    markers_data = []
                    for i, marker in enumerate(st.session_state.spectral_markers):
                        # Skip pending markers (they are shown as preview)
                        if marker.get('pending', False):
                            continue
                        
                        marker_type = marker['type']
                        position = marker['position']
                        width = marker['width'] if marker_type == 'region' else '-'
                        name = marker['name'] if marker['name'] else ''
                        color = marker['color']
                        markers_data.append({
                            '#': i + 1,
                            'Name': name,
                            'Type': marker_type.capitalize(),
                            'Position': f"{position:.1f}",
                            'Width': f"{width:.1f}" if width != '-' else '-',
                            'Color': color,
                            'index': i
                        })
                    
                    if markers_data:
                        # Display markers in a table with delete buttons
                        df_markers = pd.DataFrame(markers_data)
                        
                        # Use a container for the table
                        with st.container():
                            # Create columns for each marker row with delete button
                            for idx, row in df_markers.iterrows():
                                col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1.5, 1, 1, 1.2, 0.8, 0.8])
                                with col1:
                                    st.write(f"**{row['#']}**")
                                with col2:
                                    st.write(row['Name'] if row['Name'] else "—")
                                with col3:
                                    st.write(row['Type'])
                                with col4:
                                    st.write(row['Position'])
                                with col5:
                                    st.write(row['Width'])
                                with col6:
                                    # Color indicator
                                    st.markdown(f"<div style='width:20px;height:20px;background-color:{row['Color']};border:1px solid #ccc;border-radius:3px;'></div>", unsafe_allow_html=True)
                                with col7:
                                    if st.button("🗑️", key=f"delete_marker_{row['index']}"):
                                        # Remove marker
                                        st.session_state.spectral_markers.pop(row['index'])
                                        st.rerun()
                            
                            # Clear all markers button
                            if st.button("🗑️ Clear All Markers", use_container_width=True):
                                st.session_state.spectral_markers = []
                                st.session_state.spectral_markers_preview_position = None
                                st.session_state.spectral_markers_preview_width = 0
                                st.rerun()
                    else:
                        st.info("No confirmed markers yet. Use the controls above to add lines or regions.")
                else:
                    st.info("No markers added yet. Use the controls above to add lines or regions.")
                
            else:
                st.warning("⚠️ No spectra loaded. Please load spectra in the sidebar first.")
            
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
                    temp_df = pd.DataFrame({
                        f"{name.replace('.txt', '')}_x": data['x'].values,
                        f"{name.replace('.txt', '')}_y": data['y'].values
                    })
                    if export_data.empty:
                        export_data = temp_df
                    else:
                        export_data = pd.concat([export_data, temp_df], axis=1)
                
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
                    y_norm = normalize_spectrum(
                        data['x'].values, 
                        data['y'].values, 
                        norm_method, 
                        norm_range,
                        x_ranges  # Pass x_ranges for "Maximum rest intensity" method
                    )
                    # Используем временный DataFrame для каждого спектра, чтобы избежать ошибки несовпадения длин
                    temp_df = pd.DataFrame({
                        f"{name.replace('.txt', '')}_x": data['x'].values,
                        f"{name.replace('.txt', '')}_y_norm": y_norm
                    })
                    if export_norm.empty:
                        export_norm = temp_df
                    else:
                        export_norm = pd.concat([export_norm, temp_df], axis=1)
                
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
Fill Transparency: {fill_alpha}
Subtract Minimum Intensity: {subtract_min_intensity}
Grid Enabled: {show_grid}
Line Width: {line_width}
Peak Analysis: {analyze_peaks_flag}
Correlation Analysis: {param_correlation}
Spectral Markers: {len(st.session_state.spectral_markers)} markers
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
        6. **Compare Spectra** - Analyze differences between two spectra with heatmap visualization
        7. **Add Markers** - Place vertical lines and regions to track spectral features
        8. **Export Results** - Download processed data, plots, and analysis results
        """)
        
        st.markdown("### ✨ Key Features:")
        st.markdown("""
        - 🔬 **Multi-Mode Visualization** - Raw, normalized, and offset spectra in one comprehensive view
        - 📊 **Broken Axis Support** - Display multiple x-axis ranges with gaps between them
        - 🎨 **Customizable Colors** - Individual color assignment for each spectrum
        - 📈 **Automatic Peak Detection** - Find peaks, calculate areas, and analyze intensities
        - 🔗 **Parameter Correlation** - Correlate spectral features with experimental parameters
        - 🔀 **Spectral Comparison** - Compare two spectra with difference analysis and heatmap visualization
        - 🔥 **Heatmap Generation** - Visualize spectral evolution as function of temperature or concentration
        - 📏 **Spectral Markers** - Add vertical lines and regions to track peak positions across conditions
        - 💾 **Data Export** - Download processed data in CSV format with publication-ready plots
        - 📐 **Multiple Normalization Methods** - Maximum intensity or custom peak range normalization
        - 📏 **Cumulative Offset** - Add offsets to spectra for clear visualization (1st: 0, 2nd: +step, 3rd: +2×step)
        - 🎛️ **Adjustable Transparency** - Control fill opacity from 0.2 to 0.9
        - ⚙️ **Grid & Line Thickness** - Customize plot appearance
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
        <p style="font-size: 0.75rem;">© 2026 SpectrAnalys - Advanced Spectroscopy Data Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
