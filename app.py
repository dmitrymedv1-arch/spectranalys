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
from scipy.signal import savgol_filter, correlate, correlation_lags
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu, shapiro, kstest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

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

# NEW: Statistical analysis session state
if 'pca_results' not in st.session_state:
    st.session_state.pca_results = None
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None
if 'statistical_test_results' not in st.session_state:
    st.session_state.statistical_test_results = None

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
    ax_top.set_ylabel(f"Normalized {y_label}", fontsize=10, fontweight='bold')
    
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
    ax_bottom.set_ylabel('Intensity Difference (a.u.)', fontsize=10, fontweight='bold')
    
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
    cbar.set_label('Difference Intensity (a.u.)', fontsize=9, fontweight='bold')
    
    # Add legend for significant differences if needed
    if difference_threshold > 0 and np.any(above_threshold):
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='yellow', alpha=0.3, label=f'|Difference| > {difference_threshold:.3f}')]
        ax_bottom.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Adjust layout with increased spacing between subplots
    plt.subplots_adjust(hspace=0.4)
    
    # Adjust right margin for legend if needed
    if legend_position == "right":
        right_margin = min(0.92, legend_offset + 0.05)
        plt.subplots_adjust(right=right_margin)
    else:
        plt.tight_layout()
    
    return fig, (mean_diff, max_abs_diff, rms_diff, correlation)

# NEW FUNCTION: Create comparison plot with difference DECOMPOSITION
def create_comparison_decomposition_plot(spectrum_a_data, spectrum_b_data, name_a, name_b,
                                         x_label, y_label, norm_method, norm_range,
                                         show_grid, line_width, fig_width, fig_height,
                                         legend_fontsize, legend_position, legend_offset):
    """
    Create comparison plot with difference decomposition analysis
    Identifies scaling, shift, broadening components of the difference
    """
    
    # Prepare data for spectrum A (Reference)
    x_a = spectrum_a_data['data']['x'].values
    y_a_raw = spectrum_a_data['data']['y'].values
    y_a_norm = normalize_spectrum(x_a, y_a_raw, norm_method, norm_range)
    
    # Prepare data for spectrum B (Sample)
    x_b = spectrum_b_data['data']['x'].values
    y_b_raw = spectrum_b_data['data']['y'].values
    y_b_norm = normalize_spectrum(x_b, y_b_raw, norm_method, norm_range)
    
    # Interpolate both spectra to common x grid
    common_x_min = max(x_a.min(), x_b.min())
    common_x_max = min(x_a.max(), x_b.max())
    common_x = np.linspace(common_x_min, common_x_max, 2000)
    
    y_ref = np.interp(common_x, x_a, y_a_norm)
    y_samp = np.interp(common_x, x_b, y_b_norm)
    
    # Clean data (remove zeros and non-positive values for ratio calculations)
    mask = (y_ref > 0) & (y_samp > 0)
    y_ref_masked = y_ref[mask]
    y_samp_masked = y_samp[mask]
    x_masked = common_x[mask]
    
    # === STEP 1: Determine scaling factor ===
    # Optimal scaling factor that minimizes MSE
    if len(y_ref_masked) > 0 and len(y_samp_masked) > 0:
        k_opt = np.sum(y_ref_masked * y_samp_masked) / np.sum(y_ref_masked**2)
        # More robust: median of ratios
        ratios = y_samp_masked / y_ref_masked
        # Remove outliers (beyond 2 sigma)
        if len(ratios) > 0:
            median_ratio = np.median(ratios)
            std_ratio = np.std(ratios)
            ratios_filtered = ratios[np.abs(ratios - median_ratio) < 2 * std_ratio]
            if len(ratios_filtered) > 0:
                k_opt_robust = np.median(ratios_filtered)
            else:
                k_opt_robust = k_opt
        else:
            k_opt_robust = k_opt
    else:
        k_opt_robust = 1.0
    
    # Apply scaling
    y_ref_scaled = y_ref * k_opt_robust
    
    # === STEP 2: Determine shift ===
    # Compute cross-correlation between scaled reference and sample
    # Use a narrow range around the main peak region for better shift detection
    # or use full range with normalization
    correlation_full = correlate(y_ref_scaled, y_samp, mode='same')
    lags = correlation_lags(len(y_ref_scaled), len(y_samp), mode='same')
    
    # Find the lag with maximum correlation
    if len(correlation_full) > 0:
        max_corr_idx = np.argmax(np.abs(correlation_full))
        shift_lag = lags[max_corr_idx]
        # Convert lag to shift in cm⁻¹
        x_step = common_x[1] - common_x[0]
        shift_cm = shift_lag * x_step
    else:
        shift_cm = 0.0
    
    # Apply shift
    if abs(shift_cm) > 1e-6:
        # Shift the scaled reference
        shift_indices = int(round(shift_lag))
        if shift_indices > 0:
            y_ref_shifted = np.concatenate([np.zeros(shift_indices), y_ref_scaled[:-shift_indices]])
        elif shift_indices < 0:
            y_ref_shifted = np.concatenate([y_ref_scaled[-shift_indices:], np.zeros(-shift_indices)])
        else:
            y_ref_shifted = y_ref_scaled.copy()
    else:
        y_ref_shifted = y_ref_scaled.copy()
        shift_cm = 0.0
    
    # === STEP 3: Determine broadening ===
    # Find peaks in both spectra and compare FWHM
    def find_peaks_for_fwhm(x, y):
        if len(x) == 0 or len(y) == 0:
            return []
        # Normalize for peak detection
        y_norm_peak = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
        peaks, _ = find_peaks(y_norm_peak, height=0.1, prominence=0.05, distance=10)
        peak_results = []
        for p in peaks:
            if p > 2 and p < len(x) - 2:
                fwhm = calculate_fwhm(x, y_norm_peak, p)
                if fwhm > 0:
                    peak_results.append((x[p], fwhm, y_norm_peak[p]))
        return peak_results
    
    # Find peaks in both spectra (use the same x range for consistency)
    peaks_ref = find_peaks_for_fwhm(common_x, y_ref_scaled)
    peaks_samp = find_peaks_for_fwhm(common_x, y_samp)
    
    # Compare FWHM for matching peaks (nearest neighbor matching)
    fwhm_ratios = []
    if len(peaks_ref) > 0 and len(peaks_samp) > 0:
        for pos_ref, fwhm_ref, height_ref in peaks_ref:
            # Find nearest peak in sample
            distances = [abs(pos_ref - pos_s[0]) for pos_s in peaks_samp]
            if len(distances) > 0:
                min_idx = np.argmin(distances)
                if distances[min_idx] < 50:  # Maximum distance in cm⁻¹ for matching
                    fwhm_samp = peaks_samp[min_idx][1]
                    if fwhm_ref > 0:
                        fwhm_ratios.append(fwhm_samp / fwhm_ref)
    
    # Calculate median broadening factor
    if len(fwhm_ratios) > 0:
        broadening_factor = np.median(fwhm_ratios)
        # Remove outliers
        std_broad = np.std(fwhm_ratios)
        broad_filtered = [f for f in fwhm_ratios if abs(f - broadening_factor) < 2 * std_broad]
        if len(broad_filtered) > 0:
            broadening_factor = np.median(broad_filtered)
    else:
        broadening_factor = 1.0
    
    # Apply broadening (conceptually - this is just an estimate)
    # We'll use this for the contribution calculation
    
    # === STEP 4: Calculate contributions ===
    # 1. Original difference
    diff_original = y_samp - y_ref
    
    # 2. Difference after scaling correction
    diff_scaling = y_samp - y_ref_scaled
    
    # 3. Difference after scaling + shift correction
    diff_shift = y_samp - y_ref_shifted
    
    # 4. Estimate residual (unexplained)
    # We'll use the remaining difference after all corrections
    diff_residual = diff_shift.copy()
    
    # Calculate variance explained by each component
    var_original = np.var(diff_original)
    var_scaling = np.var(diff_scaling)
    var_shift = np.var(diff_shift)
    var_residual = np.var(diff_residual)
    
    # Calculate contributions
    total_var = var_original + 1e-10  # Avoid division by zero
    contribution_scaling = (var_original - var_scaling) / total_var * 100
    contribution_shift = (var_scaling - var_shift) / total_var * 100
    contribution_broadening = 0.0  # We don't have a direct measurement
    contribution_other = (var_shift) / total_var * 100
    
    # Clamp contributions to sensible ranges
    contribution_scaling = max(0, min(100, contribution_scaling))
    contribution_shift = max(0, min(100, contribution_shift))
    contribution_other = max(0, min(100, contribution_other))
    
    # Normalize to sum to 100
    total_contrib = contribution_scaling + contribution_shift + contribution_other
    if total_contrib > 0:
        contribution_scaling = contribution_scaling / total_contrib * 100
        contribution_shift = contribution_shift / total_contrib * 100
        contribution_other = contribution_other / total_contrib * 100
    
    # === STEP 5: Create visualization ===
    fig = plt.figure(figsize=(fig_width, fig_height * 1.8))
    
    # Create 4 subplots in 2x2 grid with increased spacing
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3)
    
    # Subplot 1: Original spectra with difference
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(common_x, y_ref, color='blue', linewidth=line_width, label=f'{name_a} (Reference)', alpha=0.8)
    ax1.plot(common_x, y_samp, color='red', linewidth=line_width, label=f'{name_b} (Sample)', alpha=0.8)
    ax1.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax1.set_ylabel(f'Normalized {y_label}', fontsize=10, fontweight='bold')
    ax1.set_title('Original Spectra', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=legend_fontsize)
    if show_grid:
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 2: Scaling effect
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(common_x, y_ref, color='blue', linewidth=line_width, label=f'{name_a} (Reference)', alpha=0.8)
    ax2.plot(common_x, y_ref_scaled, color='green', linewidth=line_width, 
             label=f'{name_a} × {k_opt_robust:.3f}', alpha=0.8)
    ax2.plot(common_x, y_samp, color='red', linewidth=line_width, label=f'{name_b} (Sample)', alpha=0.6, linestyle='--')
    ax2.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax2.set_ylabel(f'Normalized {y_label}', fontsize=10, fontweight='bold')
    ax2.set_title(f'Scaling Effect (k={k_opt_robust:.3f})', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=legend_fontsize)
    if show_grid:
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 3: Shift effect
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(common_x, y_ref_scaled, color='blue', linewidth=line_width, 
             label=f'{name_a} (scaled)', alpha=0.8)
    ax3.plot(common_x, y_ref_shifted, color='purple', linewidth=line_width,
             label=f'{name_a} (shifted, {shift_cm:.2f} cm⁻¹)', alpha=0.8)
    ax3.plot(common_x, y_samp, color='red', linewidth=line_width, label=f'{name_b} (Sample)', alpha=0.6, linestyle='--')
    ax3.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax3.set_ylabel(f'Normalized {y_label}', fontsize=10, fontweight='bold')
    ax3.set_title(f'Shift Effect (Δ = {shift_cm:.2f} cm⁻¹)', fontsize=11, fontweight='bold')
    ax3.legend(loc='best', fontsize=legend_fontsize)
    if show_grid:
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 4: Residual difference (after all corrections)
    ax4 = fig.add_subplot(gs[1, 1])
    diff_residual_plot = y_samp - y_ref_shifted
    ax4.plot(common_x, diff_residual_plot, color='black', linewidth=line_width)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax4.fill_between(common_x, 0, diff_residual_plot, alpha=0.3, color='red' if np.any(diff_residual_plot > 0) else 'blue')
    ax4.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax4.set_ylabel('Residual Difference', fontsize=10, fontweight='bold')
    ax4.set_title(f'Residual (after scaling + shift)', fontsize=11, fontweight='bold')
    # Set symmetric y-limits if possible
    max_res = np.max(np.abs(diff_residual_plot))
    if max_res > 0:
        ax4.set_ylim(-max_res * 1.1, max_res * 1.1)
    if show_grid:
        ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Add a text box with decomposition summary
    summary_text = (
        f"Decomposition Summary:\n"
        f"─────────────────────\n"
        f"Scaling factor: {k_opt_robust:.3f}\n"
        f"Shift: {shift_cm:.2f} cm⁻¹\n"
        f"Broadening: {broadening_factor:.3f}×\n"
        f"─────────────────────\n"
        f"Contributions:\n"
        f"  Scaling:   {contribution_scaling:.1f}%\n"
        f"  Shift:     {contribution_shift:.1f}%\n"
        f"  Other:     {contribution_other:.1f}%"
    )
    
    # Add a text box at the bottom of the figure
    fig.text(0.02, 0.01, summary_text, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             transform=fig.transFigure)
    
    # Adjust layout with more spacing
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.08, right=0.95)
    
    return fig, {
        'scaling_factor': k_opt_robust,
        'shift': shift_cm,
        'broadening_factor': broadening_factor,
        'contribution_scaling': contribution_scaling,
        'contribution_shift': contribution_shift,
        'contribution_other': contribution_other,
        'residual_variance': var_residual
    }

# NEW FUNCTION: Create heatmap from spectra matrix
def create_heatmap(spectra_matrix, x_grid, y_values, x_label, y_label, 
                   colorbar_label, colormap, interpolation, title, 
                   fig_width=10, fig_height=8, log_scale=False):
    """Create a heatmap from spectra matrix with specified parameters"""
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Prepare data for heatmap
    data_matrix = np.array(spectra_matrix)
    
    # Apply log scaling if requested
    if log_scale:
        # Avoid log of zero or negative values
        data_matrix = np.maximum(data_matrix, 1e-10)
        data_matrix = np.log10(data_matrix)
        colorbar_label = f"log10({colorbar_label})"
    
    # Create heatmap with imshow - transpose matrix for correct orientation
    # X-axis: parameter (y_values), Y-axis: Raman shift (x_grid)
    extent = [y_values[0], y_values[-1], x_grid[0], x_grid[-1]]
    
    # Transpose the matrix so that rows = Raman shift, columns = parameter
    data_matrix_transposed = data_matrix.T
    
    # Use exact min and max values for color scale
    data_clean = data_matrix_transposed[np.isfinite(data_matrix_transposed)]
    if len(data_clean) > 0:
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
    cbar.set_label(colorbar_label, fontsize=14, fontweight='bold')
    
    n_ticks = 6
    tick_positions = np.linspace(vmin, vmax, n_ticks)
    
    if log_scale:
        tick_labels = [f"{val:.2f}" for val in tick_positions]
    else:
        if abs(vmax - vmin) < 0.01:
            tick_labels = [f"{val:.4f}" for val in tick_positions]
        elif abs(vmax - vmin) < 1:
            tick_labels = [f"{val:.3f}" for val in tick_positions]
        elif abs(vmax - vmin) < 10:
            tick_labels = [f"{val:.2f}" for val in tick_positions]
        elif abs(vmax - vmin) < 1000:
            tick_labels = [f"{val:.1f}" for val in tick_positions]
        else:
            tick_labels = [f"{int(val)}" for val in tick_positions]
    
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=12)
    
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
    
    ax.set_xlabel(y_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(x_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.tick_params(direction='in', length=5, width=1)
    
    plt.tight_layout()
    return fig

# NEW FUNCTION: Create PCA visualization
def create_pca_visualization(spectra_dict, ordered_spectra, n_components=2, normalize_data=True):
    """Perform PCA on spectra data and create visualizations"""
    
    # Prepare data matrix
    all_x = []
    for name in ordered_spectra:
        if name in spectra_dict:
            x_vals = spectra_dict[name]['data']['x'].values
            if len(x_vals) > 0:
                all_x.extend(x_vals)
    
    if not all_x:
        return None, None, None
    
    # Use common x range
    x_min = max([spectra_dict[name]['data']['x'].min() for name in ordered_spectra if name in spectra_dict])
    x_max = min([spectra_dict[name]['data']['x'].max() for name in ordered_spectra if name in spectra_dict])
    
    if x_min >= x_max:
        return None, None, None
    
    common_x = np.linspace(x_min, x_max, 500)
    
    # Create data matrix (spectra as rows)
    data_matrix = []
    spectrum_names = []
    
    for name in ordered_spectra:
        if name in spectra_dict:
            data = spectra_dict[name]['data']
            y_interp = np.interp(common_x, data['x'].values, data['y'].values)
            data_matrix.append(y_interp)
            spectrum_names.append(name.replace('.txt', ''))
    
    if not data_matrix:
        return None, None, None
    
    data_matrix = np.array(data_matrix)
    
    # Normalize data
    if normalize_data:
        scaler = StandardScaler()
        data_matrix_scaled = scaler.fit_transform(data_matrix.T).T
    else:
        data_matrix_scaled = data_matrix
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, len(data_matrix_scaled), len(data_matrix_scaled[0])))
    pca_result = pca.fit_transform(data_matrix_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Score plot (PC1 vs PC2)
    ax1 = axes[0]
    ax1.scatter(pca_result[:, 0], pca_result[:, 1], s=100, alpha=0.7, 
                c=range(len(spectrum_names)), cmap='viridis')
    # Add labels for each point
    for i, name in enumerate(spectrum_names):
        ax1.annotate(name, (pca_result[i, 0], pca_result[i, 1]), 
                    fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax1.set_title('PCA Score Plot', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # 2. Scree plot
    ax2 = axes[1]
    n_components_plot = min(10, len(explained_variance))
    ax2.bar(range(1, n_components_plot + 1), explained_variance[:n_components_plot] * 100, 
            alpha=0.7, color='steelblue')
    ax2.plot(range(1, n_components_plot + 1), cumulative_variance[:n_components_plot] * 100, 
             'ro-', linewidth=2, markersize=8, label='Cumulative')
    ax2.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Explained Variance (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Scree Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    ax2.set_xticks(range(1, n_components_plot + 1))
    
    # 3. Loading plot (PC1)
    ax3 = axes[2]
    loadings = pca.components_[0]
    # Show only top 30 loading values for clarity
    n_loadings = min(30, len(common_x))
    indices = np.argsort(np.abs(loadings))[-n_loadings:]
    ax3.bar(range(len(indices)), loadings[indices], alpha=0.7, color='coral')
    ax3.set_xlabel('Feature Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Loading Value', fontsize=11, fontweight='bold')
    ax3.set_title('PC1 Loading Plot (Top Features)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    return fig, {
        'pca': pca,
        'scores': pca_result,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'loadings': pca.components_,
        'spectrum_names': spectrum_names,
        'common_x': common_x
    }

# NEW FUNCTION: Create cluster analysis visualization
def create_cluster_visualization(spectra_dict, ordered_spectra, n_clusters=3, method='hierarchical'):
    """Perform cluster analysis on spectra data"""
    
    # Prepare data matrix (same as PCA)
    all_x = []
    for name in ordered_spectra:
        if name in spectra_dict:
            x_vals = spectra_dict[name]['data']['x'].values
            if len(x_vals) > 0:
                all_x.extend(x_vals)
    
    if not all_x:
        return None
    
    x_min = max([spectra_dict[name]['data']['x'].min() for name in ordered_spectra if name in spectra_dict])
    x_max = min([spectra_dict[name]['data']['x'].max() for name in ordered_spectra if name in spectra_dict])
    
    if x_min >= x_max:
        return None
    
    common_x = np.linspace(x_min, x_max, 500)
    
    data_matrix = []
    spectrum_names = []
    
    for name in ordered_spectra:
        if name in spectra_dict:
            data = spectra_dict[name]['data']
            y_interp = np.interp(common_x, data['x'].values, data['y'].values)
            data_matrix.append(y_interp)
            spectrum_names.append(name.replace('.txt', ''))
    
    if not data_matrix:
        return None
    
    data_matrix = np.array(data_matrix)
    
    # Normalize data
    scaler = StandardScaler()
    data_matrix_scaled = scaler.fit_transform(data_matrix.T).T
    
    # Perform clustering
    if method == 'hierarchical':
        # Hierarchical clustering
        linkage_matrix = linkage(data_matrix_scaled, method='ward')
        
        # Create dendrogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Dendrogram
        dendrogram(linkage_matrix, ax=ax1, labels=spectrum_names, leaf_rotation=90, leaf_font_size=10)
        ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Spectra', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Distance', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Apply clustering to get labels
        cluster = AgglomerativeClustering(n_clusters=n_clusters)
        labels = cluster.fit_predict(data_matrix_scaled)
        
        # Show clusters in 2D (using PCA for visualization)
        from sklearn.decomposition import PCA
        pca_temp = PCA(n_components=2)
        pca_result = pca_temp.fit_transform(data_matrix_scaled)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i in range(n_clusters):
            mask = labels == i
            ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=colors[i % len(colors)], label=f'Cluster {i+1}', s=100, alpha=0.7)
            # Add labels
            for j, idx in enumerate(np.where(mask)[0]):
                ax2.annotate(spectrum_names[idx], (pca_result[idx, 0], pca_result[idx, 1]), 
                            fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('PC1', fontsize=11, fontweight='bold')
        ax2.set_ylabel('PC2', fontsize=11, fontweight='bold')
        ax2.set_title(f'Clusters (n={n_clusters})', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        return fig, labels, spectrum_names
        
    else:
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_matrix_scaled)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Show clusters in 2D using PCA
        from sklearn.decomposition import PCA
        pca_temp = PCA(n_components=2)
        pca_result = pca_temp.fit_transform(data_matrix_scaled)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i in range(n_clusters):
            mask = labels == i
            ax1.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=colors[i % len(colors)], label=f'Cluster {i+1}', s=100, alpha=0.7)
            for j, idx in enumerate(np.where(mask)[0]):
                ax1.annotate(spectrum_names[idx], (pca_result[idx, 0], pca_result[idx, 1]), 
                            fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('PC1', fontsize=11, fontweight='bold')
        ax1.set_ylabel('PC2', fontsize=11, fontweight='bold')
        ax1.set_title(f'K-Means Clusters (n={n_clusters})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Elbow method plot (inertia vs clusters)
        inertias = []
        k_range = range(1, min(10, len(spectrum_names)))
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(data_matrix_scaled)
            inertias.append(km.inertia_)
        
        ax2.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Inertia', fontsize=11, fontweight='bold')
        ax2.set_title('Elbow Method for Optimal Clusters', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        return fig, labels, spectrum_names

# NEW FUNCTION: Perform statistical tests on peak parameters
def perform_statistical_tests(peaks_df, group_mapping):
    """Perform statistical tests on peak parameters between groups"""
    
    if peaks_df.empty or not group_mapping:
        return None
    
    results = {}
    
    # Parameters to test
    params = ['Intensity', 'Area', 'Peak position', 'FWHM']
    
    # Get unique groups
    groups = list(set(group_mapping.values()))
    
    for param in params:
        if param not in peaks_df.columns:
            continue
        
        # Collect data by group
        group_data = {}
        for group in groups:
            group_spectra = [name for name, g in group_mapping.items() if g == group]
            data = []
            for spec in group_spectra:
                spec_data = peaks_df[peaks_df['Spectrum'] == spec]
                if not spec_data.empty:
                    data.extend(spec_data[param].values)
            group_data[group] = data
        
        # Remove empty groups
        group_data = {k: v for k, v in group_data.items() if len(v) > 0}
        
        if len(group_data) < 2:
            continue
        
        # Perform tests
        group_names = list(group_data.keys())
        data_lists = list(group_data.values())
        
        # Check normality for each group
        normality_results = {}
        for gname, data in group_data.items():
            if len(data) > 2:
                stat, p = shapiro(data)
                normality_results[gname] = p > 0.05
        
        # T-test for two groups
        if len(group_data) == 2:
            # Check if data is normally distributed
            normal = all(normality_results.values()) if len(normality_results) == 2 else False
            
            if normal:
                t_stat, p_value = ttest_ind(data_lists[0], data_lists[1])
                test_name = 'T-test'
            else:
                t_stat, p_value = mannwhitneyu(data_lists[0], data_lists[1])
                test_name = 'Mann-Whitney U test'
            
            results[param] = {
                'test': test_name,
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'groups': group_data
            }
        
        # ANOVA for three or more groups
        elif len(group_data) >= 3:
            # Check if all groups are normally distributed
            all_normal = all(normality_results.values()) if len(normality_results) == len(group_data) else False
            
            if all_normal:
                f_stat, p_value = f_oneway(*data_lists)
                test_name = 'ANOVA'
            else:
                # Kruskal-Wallis test (non-parametric)
                from scipy.stats import kruskal
                f_stat, p_value = kruskal(*data_lists)
                test_name = 'Kruskal-Wallis test'
            
            results[param] = {
                'test': test_name,
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'groups': group_data
            }
    
    return results

# NEW FUNCTION: Create box/violin plots for peak parameters
def create_parameter_visualization(peaks_df, parameter, group_mapping=None, plot_type='box'):
    """Create box or violin plot for a peak parameter"""
    
    if peaks_df.empty:
        return None
    
    if group_mapping:
        # Grouped plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = {}
        for spec, group in group_mapping.items():
            spec_data = peaks_df[peaks_df['Spectrum'] == spec]
            if not spec_data.empty:
                if group not in groups:
                    groups[group] = []
                groups[group].extend(spec_data[parameter].values)
        
        # Prepare data for plotting
        data_to_plot = []
        group_labels = []
        for group, data in groups.items():
            if len(data) > 0:
                data_to_plot.append(data)
                group_labels.append(group)
        
        if plot_type == 'box':
            bp = ax.boxplot(data_to_plot, labels=group_labels, patch_artist=True)
            # Color boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for patch, color in zip(bp['boxes'], colors[:len(group_labels)]):
                patch.set_facecolor(color)
        else:  # violin
            import matplotlib.pyplot as plt
            parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            # Color violins
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
        
        ax.set_xlabel('Group', fontsize=11, fontweight='bold')
        ax.set_ylabel(parameter, fontsize=11, fontweight='bold')
        ax.set_title(f'{parameter} Distribution by Group', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    else:
        # Individual plot (all spectra)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by spectrum
        spectra_names = peaks_df['Spectrum'].unique()
        data_to_plot = []
        labels = []
        
        for spec in spectra_names:
            spec_data = peaks_df[peaks_df['Spectrum'] == spec]
            if not spec_data.empty:
                data_to_plot.append(spec_data[parameter].values)
                labels.append(spec)
        
        if plot_type == 'box':
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        else:  # violin
            parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
        
        ax.set_xlabel('Spectrum', fontsize=11, fontweight='bold')
        ax.set_ylabel(parameter, fontsize=11, fontweight='bold')
        ax.set_title(f'{parameter} Distribution by Spectrum', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig

# NEW FUNCTION: Create interactive plot with Plotly
def create_interactive_plot(spectra_dict, ordered_spectra, x_label, y_label, 
                           x_ranges=None, normalize=False, norm_method=None, 
                           norm_range=None, subtract_min=False):
    """Create an interactive plot using Plotly"""
    
    fig = go.Figure()
    
    # Get colors from spectra_dict
    colors = {}
    for name, spec in spectra_dict.items():
        colors[name] = spec['color'] if 'color' in spec else '#1f77b4'
    
    for name in ordered_spectra:
        if name not in spectra_dict:
            continue
        
        data = spectra_dict[name]['data']
        x = data['x'].values
        y = data['y'].values
        
        # Apply normalization if requested
        if normalize:
            y = normalize_spectrum(x, y, norm_method, norm_range)
            if subtract_min:
                y = y - y.min()
        
        # Apply x-range filtering if specified
        if x_ranges:
            for start, end in x_ranges:
                mask = (x >= start) & (x <= end)
                if np.any(mask):
                    fig.add_trace(go.Scatter(
                        x=x[mask], y=y[mask],
                        name=name.replace('.txt', ''),
                        line=dict(color=colors.get(name, '#1f77b4'), width=1.5),
                        mode='lines',
                        hovertemplate='x: %{x:.2f}<br>y: %{y:.4f}<extra></extra>'
                    ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y,
                name=name.replace('.txt', ''),
                line=dict(color=colors.get(name, '#1f77b4'), width=1.5),
                mode='lines',
                hovertemplate='x: %{x:.2f}<br>y: %{y:.4f}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Interactive Spectra Viewer',
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='right',
            x=1.02
        )
    )
    
    return fig

# NEW FUNCTION: Savitzky-Golay filtering
def apply_savgol_filter(x, y, window_length, polyorder):
    """Apply Savitzky-Golay filter to spectrum"""
    if len(y) < window_length:
        return y
    if window_length % 2 == 0:
        window_length += 1  # Make odd
    return savgol_filter(y, window_length, polyorder)

# NEW FUNCTION: Baseline correction using ALS
def als_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric Least Squares baseline correction"""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    
    y = np.array(y, dtype=np.float64)
    L = len(y)
    
    # Initialize weights
    w = np.ones(L)
    
    # Build penalty matrix
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    
    for _ in range(n_iter):
        W = diags(w, 0)
        Z = W + lam * D @ D.T
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    return z

# Function to prepare heatmap data
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
        range_starts = [start for start, end in x_ranges]
        range_ends = [end for start, end in x_ranges]
        x_min = min(range_starts)
        x_max = max(range_ends)
        
        has_data_in_ranges = True
        for name in ordered_spectra:
            if name in spectra_dict:
                x_vals = spectra_dict[name]['data']['x'].values
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
            x_min = max([spectra_dict[name]['data']['x'].min() for name in ordered_spectra if name in spectra_dict])
            x_max = min([spectra_dict[name]['data']['x'].max() for name in ordered_spectra if name in spectra_dict])
    else:
        x_min = max([spectra_dict[name]['data']['x'].min() for name in ordered_spectra if name in spectra_dict])
        x_max = min([spectra_dict[name]['data']['x'].max() for name in ordered_spectra if name in spectra_dict])
    
    if x_min >= x_max:
        return None, None, None, None
    
    common_x = np.linspace(x_min, x_max, 2000)
    
    spectra_matrix = []
    spectra_norm_matrix = []
    y_values = []
    
    for name in ordered_spectra:
        if name not in spectra_dict or name not in heatmap_params:
            continue
        
        data = spectra_dict[name]['data']
        x_orig = data['x'].values
        y_orig = data['y'].values
        
        if x_ranges is not None and len(x_ranges) > 0:
            mask_total = np.zeros_like(x_orig, dtype=bool)
            for start, end in x_ranges:
                mask_range = (x_orig >= start) & (x_orig <= end)
                mask_total = mask_total | mask_range
            
            if np.any(mask_total):
                x_cropped = x_orig[mask_total]
                y_cropped = y_orig[mask_total]
            else:
                x_cropped = x_orig
                y_cropped = y_orig
        else:
            x_cropped = x_orig
            y_cropped = y_orig
        
        if len(x_cropped) == 0:
            continue
        
        y_interp = np.interp(common_x, x_cropped, y_cropped)
        y_norm = normalize_spectrum(common_x, y_interp, norm_method, norm_range, x_ranges)
        
        spectra_matrix.append(y_interp)
        spectra_norm_matrix.append(y_norm)
        param_value = heatmap_params[name]
        y_values.append(param_value)
    
    if not spectra_matrix:
        return None, None, None, None
    
    return np.array(spectra_matrix), np.array(spectra_norm_matrix), common_x, np.array(y_values)

# Main app
def main():
    # Custom header with logo
    import os
    from PIL import Image
    
    # Check if logo exists
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
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
        
        if uploaded_files and st.session_state.get('spectra_loaded', False):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🗑️ Remove all spectra", type="secondary", use_container_width=True):
                    st.session_state.spectra_loaded = False
                    st.session_state.cached_spectra_data = None
                    st.session_state.peak_analysis_triggered = False
                    st.session_state.peak_analysis_results = None
                    st.session_state.correlation_ready = False
                    st.session_state.excluded_peaks = set()
                    st.session_state.heatmap_applied = False
                    st.session_state.heatmap_params = {}
                    st.rerun()
            with col2:
                st.markdown("")
        
        if uploaded_files:
            st.success(f"✅ Loaded {len(uploaded_files)} files")
            
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
                
                selected_spectra = st.multiselect(
                    "Choose spectra to display",
                    options=list(spectra_data.keys()),
                    default=list(spectra_data.keys())
                )
                
                if selected_spectra:
                    ordered_spectra = []
                    for name in selected_spectra:
                        ordered_spectra.append(name)
                    
                    st.markdown("---")
                    st.markdown("### ⚙️ Processing Options")
                    
                    common_x_range = st.checkbox("Align all spectra to common x range", value=False)
                    
                    # NEW: Filtering options
                    st.markdown("#### 🔬 Signal Processing")
                    filter_method = st.selectbox(
                        "Filtering method",
                        ["None", "Savitzky-Golay", "Gaussian"]
                    )
                    
                    filter_window = 11
                    filter_polyorder = 3
                    if filter_method == "Savitzky-Golay":
                        col1, col2 = st.columns(2)
                        with col1:
                            filter_window = st.slider("Window length", min_value=5, max_value=31, value=11, step=2)
                        with col2:
                            filter_polyorder = st.slider("Polynomial order", min_value=2, max_value=5, value=3)
                    
                    # NEW: Baseline correction
                    baseline_correction = st.checkbox("Apply baseline correction (ALS)", value=False)
                    if baseline_correction:
                        col1, col2 = st.columns(2)
                        with col1:
                            baseline_lam = st.slider("Lambda (smoothness)", min_value=1e3, max_value=1e7, value=1e5, step=1e3, format="%.0f")
                        with col2:
                            baseline_p = st.slider("p (asymmetry)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
                    
                    # NEW: Second derivative
                    show_second_derivative = st.checkbox("Show second derivative", value=False)
                    
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
                    
                    fill_area = st.checkbox("Fill area under normalized spectra", value=False)
                    
                    fill_alpha = 0.3
                    if fill_area:
                        fill_alpha = st.slider(
                            "Fill transparency",
                            min_value=0.2,
                            max_value=0.9,
                            value=0.3,
                            step=0.1
                        )
                    
                    subtract_min_intensity = st.checkbox("Subtract minimum intensity (start from zero)", value=False)
                    
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
                    
                    legend_offset = st.slider(
                        "Legend offset from plot (0.5-2.0)",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.02,
                        step=0.02
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
                    
                    heatmap_param_type = st.selectbox(
                        "Parameter type",
                        options=["Temperature (°C)", "Concentration (x)", "Custom"],
                        index=0,
                        key="heatmap_param_type_select"
                    )
                    
                    heatmap_custom_label = ""
                    if heatmap_param_type == "Custom":
                        heatmap_custom_label = st.text_input(
                            "Custom parameter label",
                            value="Parameter",
                            key="heatmap_custom_label"
                        )
                    
                    if heatmap_param_type == "Temperature (°C)":
                        heatmap_y_label = "Temperature (°C)"
                    elif heatmap_param_type == "Concentration (x)":
                        heatmap_y_label = "Concentration (x)"
                    else:
                        heatmap_y_label = heatmap_custom_label if heatmap_custom_label else "Parameter"
                    
                    st.markdown("#### Assign values to spectra:")
                    
                    heatmap_params_temp = {}
                    for name in ordered_spectra:
                        display_name = name.replace('.txt', '')
                        param_key = f"heatmap_{name}"
                        heatmap_params_temp[name] = st.number_input(
                            f"{display_name}",
                            value=st.session_state.heatmap_params.get(name, len(heatmap_params_temp) + 1.0),
                            step=0.1,
                            format="%.1f",
                            key=param_key
                        )
                    
                    st.markdown("#### 🎨 Heatmap Settings")
                    
                    col1, col2 = st.columns(2)
                    with col1:
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
                            index=5,
                            key="heatmap_interpolation_select"
                        )
                    
                    with col2:
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
                    
                    # Interactive plots option
                    st.markdown("#### 🔄 Interactive Plots")
                    interactive_plots = st.checkbox("Enable interactive Plotly plots", value=False)
                    
                    apply_heatmap = st.button(
                        "🔄 Apply for heatmaps",
                        use_container_width=True,
                        key="apply_heatmap_button"
                    )
                    
                    if apply_heatmap:
                        st.session_state.heatmap_params = heatmap_params_temp
                        st.session_state.heatmap_param_type = heatmap_param_type
                        st.session_state.heatmap_interpolation = heatmap_interpolation
                        st.session_state.heatmap_colormap = heatmap_colormap
                        st.session_state.heatmap_y_label = heatmap_y_label
                        st.session_state.heatmap_applied = True
                        st.session_state.heatmap_ordered_names = ordered_spectra
                        st.session_state.heatmap_x_ranges = x_ranges
                        
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
                    
                    # Color Assignment
                    st.markdown("---")
                    st.markdown("### 🎨 Color Assignment")
                    
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
                    
                    for name in ordered_spectra:
                        spectra_data[name]['color'] = colors[name]
                    
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
                        'heatmap_params': st.session_state.heatmap_params,
                        'heatmap_param_type': st.session_state.heatmap_param_type,
                        'heatmap_interpolation': st.session_state.heatmap_interpolation,
                        'heatmap_colormap': st.session_state.heatmap_colormap,
                        'heatmap_y_label': st.session_state.heatmap_y_label,
                        'heatmap_applied': st.session_state.heatmap_applied,
                        'filter_method': filter_method,
                        'filter_window': filter_window,
                        'filter_polyorder': filter_polyorder,
                        'baseline_correction': baseline_correction,
                        'baseline_lam': baseline_lam if baseline_correction else 1e5,
                        'baseline_p': baseline_p if baseline_correction else 0.01,
                        'show_second_derivative': show_second_derivative,
                        'interactive_plots': interactive_plots
                    }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p>🔬 SpectrAnalys v2.0<br>Scientific Spectroscopic Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if uploaded_files and st.session_state.get('spectra_loaded', False) and st.session_state.cached_spectra_data:
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
        
        # NEW: Get additional settings
        filter_method = cached.get('filter_method', 'None')
        filter_window = cached.get('filter_window', 11)
        filter_polyorder = cached.get('filter_polyorder', 3)
        baseline_correction = cached.get('baseline_correction', False)
        baseline_lam = cached.get('baseline_lam', 1e5)
        baseline_p = cached.get('baseline_p', 0.01)
        show_second_derivative = cached.get('show_second_derivative', False)
        interactive_plots = cached.get('interactive_plots', False)
        
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
        
        current_spectra = spectra_data
        if common_x_range:
            current_spectra = align_x_ranges(current_spectra)
        
        filtered_spectra = {name: current_spectra[name] for name in ordered_spectra if name in current_spectra}
        
        # Apply filtering and baseline correction if requested
        processed_spectra = {}
        for name, spec in filtered_spectra.items():
            data = spec['data']
            x = data['x'].values
            y = data['y'].values
            
            # Apply filtering
            if filter_method == "Savitzky-Golay" and len(y) >= filter_window:
                y_filtered = apply_savgol_filter(x, y, filter_window, filter_polyorder)
            elif filter_method == "Gaussian":
                y_filtered = gaussian_filter1d(y, sigma=2.0)
            else:
                y_filtered = y
            
            # Apply baseline correction
            if baseline_correction and len(y_filtered) > 10:
                baseline = als_baseline(y_filtered, lam=baseline_lam, p=baseline_p)
                y_corrected = y_filtered - baseline
            else:
                y_corrected = y_filtered
            
            processed_spectra[name] = {
                'data': pd.DataFrame({'x': x, 'y': y_corrected}),
                'color': spec['color'],
                'original_data': spec['data']
            }
        
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Combined Visualization",
            "🔍 Peak Analysis",
            "📈 Parameter Correlation",
            "🔀 Compare Spectra",
            "📊 Multivariate Analysis",
            "📈 Peak Statistics",
            "📚 Documentation"
        ])
        
        with tab1:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Comprehensive Spectra Analysis")
            st.markdown("*All visualization modes combined for comprehensive spectral comparison*")
            
            # Show second derivative if requested
            if show_second_derivative:
                st.markdown("#### Second Derivative Mode")
                st.info("💡 Displaying second derivative of spectra for enhanced peak detection")
            
            # Prepare normalized spectra
            normalized_spectra = {}
            for name, spec in processed_spectra.items():
                data = spec['data']
                y_norm = normalize_spectrum(
                    data['x'].values,
                    data['y'].values,
                    norm_method,
                    norm_range,
                    x_ranges
                )
                if show_second_derivative:
                    y_norm = np.gradient(np.gradient(y_norm))
                normalized_spectra[name] = {
                    'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
                    'color': spec['color']
                }
            
            if subtract_min_intensity:
                for name in normalized_spectra:
                    y_vals = normalized_spectra[name]['data']['y'].values
                    if len(y_vals) > 0:
                        y_min = y_vals.min()
                        normalized_spectra[name]['data']['y'] = y_vals - y_min
            
            # Use interactive plots if enabled
            if interactive_plots:
                st.markdown("#### Interactive Plot")
                fig_interactive = create_interactive_plot(
                    processed_spectra, ordered_spectra, x_label, y_label,
                    x_ranges, False, None, None, subtract_min_intensity
                )
                st.plotly_chart(fig_interactive, use_container_width=True)
                
                fig_interactive_norm = create_interactive_plot(
                    normalized_spectra, ordered_spectra, x_label, f"Normalized {y_label}",
                    x_ranges, True, norm_method, norm_range, subtract_min_intensity
                )
                st.plotly_chart(fig_interactive_norm, use_container_width=True)
            
            # Define the four visualization configurations
            viz_configs = [
                (processed_spectra, 0, False, False, False, y_label),
                (normalized_spectra, 0, False, True, False, f"Normalized {y_label}"),
                (processed_spectra, raw_offset_step, False, False, True, y_label),
                (normalized_spectra, norm_offset_step, fill_area, True, True, f"Normalized {y_label}")
            ]
            
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
                
                if idx < len(viz_configs) - 1:
                    st.markdown('<div class="separator">****</div>', unsafe_allow_html=True)
            
            if st.session_state.get('heatmap_applied', False):
                st.markdown('<div class="separator">═══════════════════════════════════════════════════</div>', unsafe_allow_html=True)
                st.subheader("🔥 Heatmap Visualization")
                st.markdown("*Spectral evolution heatmaps showing intensity distribution as function of parameter*")
                
                spectra_matrix = st.session_state.get('heatmap_spectra_matrix')
                spectra_norm_matrix = st.session_state.get('heatmap_spectra_norm_matrix')
                x_grid = st.session_state.get('heatmap_x_grid')
                y_values = st.session_state.get('heatmap_y_values')
                heatmap_y_label = st.session_state.get('heatmap_y_label', 'Parameter')
                heatmap_interpolation = st.session_state.get('heatmap_interpolation', 'gaussian')
                heatmap_colormap = st.session_state.get('heatmap_colormap', 'viridis')
                heatmap_x_ranges = st.session_state.get('heatmap_x_ranges', None)
                
                if spectra_matrix is not None and x_grid is not None and y_values is not None:
                    min_val = np.min(spectra_matrix[spectra_matrix > 0]) if np.any(spectra_matrix > 0) else 1
                    max_val = np.max(spectra_matrix)
                    use_log = (max_val / min_val) > 100 if min_val > 0 else False
                    
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
                    
                    param_df = pd.DataFrame({
                        'Spectrum': [name.replace('.txt', '') for name in st.session_state.heatmap_ordered_names if name in st.session_state.heatmap_params],
                        heatmap_y_label: [st.session_state.heatmap_params[name] for name in st.session_state.heatmap_ordered_names if name in st.session_state.heatmap_params]
                    })
                    st.dataframe(param_df, use_container_width=True)
                    
                    st.caption(f"Interpolation: {heatmap_interpolation} | Colormap: {heatmap_colormap} | Log scale: {'Yes' if use_log else 'No'} for intensity, Yes for normalized")
                else:
                    st.warning("⚠️ Heatmap data not available. Please click 'Apply for heatmaps' in the sidebar.")
            else:
                st.info("💡 To generate heatmaps, assign numeric values to spectra in the sidebar and click 'Apply for heatmaps'.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Peak Detection and Analysis")
            st.markdown("*Select spectral range manually for precise peak analysis*")
            
            if analyze_peaks_flag and filtered_spectra:
                st.markdown("#### 📊 Select Analysis Range")
                st.markdown("Use the sliders below to select left and right boundaries for peak analysis")
                
                all_x = []
                for spec in filtered_spectra.values():
                    all_x.extend(spec['data']['x'].values)
                global_min_x = float(np.min(all_x))
                global_max_x = float(np.max(all_x))
                
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
                
                if left_boundary >= right_boundary:
                    st.warning("⚠️ Left boundary must be less than right boundary")
                    manual_range = (None, None)
                else:
                    manual_range = (left_boundary, right_boundary)
                
                fig_range, ax_range = plt.subplots(figsize=(12, 5))
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    ax_range.plot(data['x'].values, data['y'].values, 
                                 color=spec['color'], linewidth=1.5, 
                                 label=name.replace('.txt', ''), alpha=0.7)
                
                if left_boundary < right_boundary:
                    ax_range.axvline(left_boundary, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Left: {left_boundary:.1f}')
                    ax_range.axvline(right_boundary, color='blue', linestyle='-', linewidth=2, alpha=0.7, label=f'Right: {right_boundary:.1f}')
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
                
                if st.session_state.peak_analysis_triggered and st.session_state.peak_analysis_results is not None:
                    peaks_df = st.session_state.peak_analysis_results.copy()
                    
                    if not peaks_df.empty:
                        st.markdown("---")
                        st.subheader("📊 Peak Analysis Results")
                        st.markdown("*Check/Uncheck peaks to include/exclude them from visualization and correlation analysis*")
                        
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
                        
                        peaks_df['temp_id'] = range(len(peaks_df))
                        
                        edited_df = st.data_editor(
                            peaks_df[['Spectrum', 'Peak position', 'Intensity', 'Area', 'FWHM', 'Include', 'temp_id']],
                            column_config={
                                'Include': st.column_config.CheckboxColumn(
                                    "Include?",
                                    help="Check to include this peak in analysis",
                                    default=True
                                ),
                                'temp_id': None
                            },
                            disabled=['Spectrum', 'Peak position', 'Intensity', 'Area', 'FWHM'],
                            hide_index=True,
                            use_container_width=True,
                            key="peak_editor"
                        )
                        
                        if edited_df is not None:
                            include_map = dict(zip(edited_df['temp_id'], edited_df['Include']))
                            peaks_df['Include'] = peaks_df['temp_id'].map(include_map)
                            st.session_state.peak_analysis_results = peaks_df.drop('temp_id', axis=1)
                            peaks_df = peaks_df.drop('temp_id', axis=1)
                        else:
                            peaks_df = peaks_df.drop('temp_id', axis=1)
                        
                        csv = peaks_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download peak analysis as CSV",
                            data=csv,
                            file_name=f"peak_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.markdown("---")
                        st.subheader("🔍 Peak Visualization")
                        st.markdown("*Only checked peaks are shown*")
                        fig_peaks = create_peak_visualization(
                            filtered_spectra, 
                            st.session_state.peak_analysis_x_range,
                            peaks_df
                        )
                        st.pyplot(fig_peaks)
                        
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
                peaks_df = peaks_df[peaks_df['Include'] == True]
                
                if peaks_df.empty:
                    st.warning("⚠️ No peaks are currently included. Please check at least one peak in the Peak Analysis tab.")
                else:
                    param_list = []
                    intensity_list = []
                    area_list = []
                    position_list = []
                    fwhm_list = []
                    
                    for name in ordered_spectra:
                        if name in param_values:
                            spec_peaks = peaks_df[peaks_df['Spectrum'] == name.replace('.txt', '')]
                            if not spec_peaks.empty:
                                main_peak = spec_peaks.loc[spec_peaks['Intensity'].idxmax()]
                                param_list.append(param_values[name])
                                intensity_list.append(main_peak['Intensity'])
                                area_list.append(main_peak['Area'])
                                position_list.append(main_peak['Peak position'])
                                fwhm_list.append(main_peak['FWHM'])
                    
                    if param_list:
                        corr_intensity = pearsonr(param_list, intensity_list)[0] if len(param_list) > 2 else 0
                        corr_area = pearsonr(param_list, area_list)[0] if len(param_list) > 2 else 0
                        corr_position = pearsonr(param_list, position_list)[0] if len(param_list) > 2 else 0
                        corr_fwhm = pearsonr(param_list, fwhm_list)[0] if len(param_list) > 2 else 0
                        
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
                        
                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                        
                        axes[0, 0].scatter(param_list, intensity_list, c='#1f77b4', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[0, 0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[0, 0].set_ylabel("Peak Intensity (a.u.)", fontsize=11, fontweight='bold')
                        axes[0, 0].set_title(f"Intensity vs {param_label}\n(r = {corr_intensity:.3f})", fontsize=12, fontweight='bold')
                        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
                        
                        axes[0, 1].scatter(param_list, area_list, c='#2ca02c', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[0, 1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[0, 1].set_ylabel("Peak Area", fontsize=11, fontweight='bold')
                        axes[0, 1].set_title(f"Area vs {param_label}\n(r = {corr_area:.3f})", fontsize=12, fontweight='bold')
                        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
                        
                        axes[1, 0].scatter(param_list, position_list, c='#d62728', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[1, 0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[1, 0].set_ylabel("Peak Position (cm⁻¹)", fontsize=11, fontweight='bold')
                        axes[1, 0].set_title(f"Position vs {param_label}\n(r = {corr_position:.3f})", fontsize=12, fontweight='bold')
                        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
                        
                        axes[1, 1].scatter(param_list, fwhm_list, c='#9467bd', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[1, 1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[1, 1].set_ylabel("FWHM (cm⁻¹)", fontsize=11, fontweight='bold')
                        axes[1, 1].set_title(f"FWHM vs {param_label}\n(r = {corr_fwhm:.3f})", fontsize=12, fontweight='bold')
                        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
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
        
        with tab4:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("🔀 Spectral Difference Analysis")
            st.markdown("*Compare two spectra to identify differences and visualize them with heatmaps*")
            
            if len(ordered_spectra) >= 2:
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
                
                swap_direction = st.checkbox("Swap difference direction (Sample - Reference)", value=True)
                
                st.markdown("---")
                st.markdown("#### 🎨 Difference Plot Settings")
                
                col1, col2 = st.columns(2)
                with col1:
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
                    smooth_difference = st.checkbox("Apply smoothing to difference profile", value=False)
                    smooth_sigma = 1.0
                    if smooth_difference:
                        smooth_sigma = st.slider(
                            "Smoothing sigma",
                            min_value=0.5,
                            max_value=5.0,
                            value=1.5,
                            step=0.5
                        )
                
                col1, col2 = st.columns(2)
                with col1:
                    symmetric_scale = st.checkbox("Symmetric color scale (centered at zero)", value=True)
                
                with col2:
                    difference_threshold = st.number_input(
                        "Significance threshold (highlight regions with |difference| > threshold)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.01,
                        format="%.3f"
                    )
                
                # NEW: Difference decomposition options
                st.markdown("---")
                st.markdown("#### 🔬 Difference Decomposition Analysis")
                st.markdown("*Decompose the difference into scaling, shift, and other components*")
                perform_decomposition = st.checkbox("Perform difference decomposition analysis", value=False)
                
                st.markdown("---")
                
                if spectrum_a_name and spectrum_b_name:
                    spectrum_a = filtered_spectra[spectrum_a_name]
                    spectrum_b = filtered_spectra[spectrum_b_name]
                    
                    name_a = spectrum_a_name.replace('.txt', '')
                    name_b = spectrum_b_name.replace('.txt', '')
                    
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
                        
                        st.pyplot(fig)
                        
                        col1, col2 = st.columns(2)
                        with col1:
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
                        
                        # Perform decomposition if requested
                        if perform_decomposition:
                            st.markdown("---")
                            st.subheader("🔬 Difference Decomposition Analysis")
                            st.markdown("*Understanding the nature of spectral changes*")
                            
                            with st.spinner("Performing decomposition analysis..."):
                                fig_decomp, decomp_results = create_comparison_decomposition_plot(
                                    spectrum_a, spectrum_b, name_a, name_b,
                                    x_label, y_label, norm_method, norm_range,
                                    show_grid, line_width, fig_width, fig_height,
                                    cached['legend_fontsize'], cached['legend_position'], cached['legend_offset']
                                )
                                
                                # Display decomposition results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Scaling Factor", f"{decomp_results['scaling_factor']:.4f}",
                                             delta=f"{decomp_results['contribution_scaling']:.1f}% contribution")
                                with col2:
                                    st.metric("Spectral Shift", f"{decomp_results['shift']:.3f} cm⁻¹",
                                             delta=f"{decomp_results['contribution_shift']:.1f}% contribution")
                                with col3:
                                    st.metric("Broadening Factor", f"{decomp_results['broadening_factor']:.4f}",
                                             delta=f"{decomp_results['contribution_other']:.1f}% other")
                                
                                # Display contribution bars
                                st.markdown("#### 📊 Contribution Analysis")
                                contributions = {
                                    'Scaling': decomp_results['contribution_scaling'],
                                    'Shift': decomp_results['contribution_shift'],
                                    'Other': decomp_results['contribution_other']
                                }
                                
                                fig_contrib, ax_contrib = plt.subplots(figsize=(8, 4))
                                colors_contrib = ['#2ca02c', '#1f77b4', '#d62728']
                                bars = ax_contrib.bar(contributions.keys(), contributions.values(), 
                                                     color=colors_contrib, alpha=0.7)
                                ax_contrib.set_ylabel('Contribution (%)', fontsize=11, fontweight='bold')
                                ax_contrib.set_title('Components of Spectral Difference', fontsize=12, fontweight='bold')
                                ax_contrib.set_ylim(0, 110)
                                ax_contrib.grid(True, alpha=0.3, linestyle='--')
                                for bar, val in zip(bars, contributions.values()):
                                    ax_contrib.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
                                plt.tight_layout()
                                st.pyplot(fig_contrib)
                                plt.close(fig_contrib)
                                
                                # Display the decomposition plot
                                st.pyplot(fig_decomp)
                                
                                buf = BytesIO()
                                fig_decomp.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                                buf.seek(0)
                                b64 = base64.b64encode(buf.getvalue()).decode()
                                st.markdown(f"""
                                <div style="text-align: center; margin-top: 1rem;">
                                    <a href="data:image/png;base64,{b64}" download="decomposition_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                                        <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                       color: white; border: none; border-radius: 8px; 
                                                       padding: 0.5rem 1rem; cursor: pointer;">
                                            📥 Download Decomposition Plot (PNG)
                                        </button>
                                    </a>
                                </div>
                                """, unsafe_allow_html=True)
                                plt.close(fig_decomp)
            else:
                st.warning("⚠️ Please load at least 2 spectra to use the comparison feature.")
                st.info("Upload multiple .txt files to compare different samples or treatments.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # NEW TAB 5: Multivariate Analysis (PCA and Clustering)
        with tab5:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("📊 Multivariate Spectral Analysis")
            st.markdown("*Principal Component Analysis (PCA) and Cluster Analysis for spectral data*")
            
            if len(ordered_spectra) >= 3:
                st.markdown("#### 🧮 Principal Component Analysis (PCA)")
                st.markdown("*Reduces dimensionality to identify patterns and relationships in spectral data*")
                
                col1, col2 = st.columns(2)
                with col1:
                    n_components = st.slider(
                        "Number of components for PCA",
                        min_value=2,
                        max_value=min(10, len(ordered_spectra)),
                        value=min(3, len(ordered_spectra)),
                        key="pca_components"
                    )
                with col2:
                    normalize_pca = st.checkbox("Normalize data for PCA", value=True, key="pca_normalize")
                
                if st.button("🔬 Run PCA Analysis", key="run_pca"):
                    with st.spinner("Performing PCA..."):
                        fig_pca, pca_results = create_pca_visualization(
                            processed_spectra, ordered_spectra, n_components, normalize_pca
                        )
                        
                        if fig_pca is not None:
                            st.session_state.pca_results = pca_results
                            st.pyplot(fig_pca)
                            
                            buf = BytesIO()
                            fig_pca.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                            buf.seek(0)
                            b64 = base64.b64encode(buf.getvalue()).decode()
                            st.markdown(f"""
                            <div style="text-align: center; margin-top: 1rem;">
                                <a href="data:image/png;base64,{b64}" download="pca_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                   color: white; border: none; border-radius: 8px; 
                                                   padding: 0.5rem 1rem; cursor: pointer;">
                                        📥 Download PCA Analysis (PNG)
                                    </button>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                            plt.close(fig_pca)
                            
                            # Display explained variance table
                            if pca_results:
                                st.markdown("#### 📊 Explained Variance")
                                variance_df = pd.DataFrame({
                                    'Component': [f'PC{i+1}' for i in range(len(pca_results['explained_variance']))],
                                    'Explained Variance (%)': pca_results['explained_variance'] * 100,
                                    'Cumulative (%)': pca_results['cumulative_variance'] * 100
                                })
                                st.dataframe(variance_df, use_container_width=True)
                        else:
                            st.error("❌ PCA failed. Please check your data.")
                
                # Display stored PCA results if available
                if st.session_state.pca_results is not None:
                    st.markdown("---")
                    st.markdown("#### 📈 PCA Results Summary")
                    pca_res = st.session_state.pca_results
                    
                    # Score plot in 2D with interactive selection
                    if 'scores' in pca_res and len(pca_res['scores'][0]) >= 2:
                        fig_scores, ax_scores = plt.subplots(figsize=(10, 6))
                        scores = pca_res['scores']
                        names = pca_res['spectrum_names']
                        ax_scores.scatter(scores[:, 0], scores[:, 1], s=100, alpha=0.7, c=range(len(names)), cmap='viridis')
                        for i, name in enumerate(names):
                            ax_scores.annotate(name, (scores[i, 0], scores[i, 1]), 
                                              fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')
                        ax_scores.set_xlabel(f'PC1 ({pca_res["explained_variance"][0]*100:.1f}%)', fontsize=11, fontweight='bold')
                        ax_scores.set_ylabel(f'PC2 ({pca_res["explained_variance"][1]*100:.1f}%)', fontsize=11, fontweight='bold')
                        ax_scores.set_title('PCA Score Plot', fontsize=12, fontweight='bold')
                        ax_scores.grid(True, alpha=0.3, linestyle='--')
                        ax_scores.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                        ax_scores.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_scores)
                        plt.close(fig_scores)
                
                st.markdown("---")
                st.markdown("#### 🎯 Cluster Analysis")
                st.markdown("*Group spectra based on similarity using clustering algorithms*")
                
                col1, col2 = st.columns(2)
                with col1:
                    cluster_method = st.selectbox(
                        "Clustering method",
                        ["hierarchical", "kmeans"],
                        format_func=lambda x: "Hierarchical" if x == "hierarchical" else "K-Means",
                        key="cluster_method"
                    )
                with col2:
                    n_clusters = st.slider(
                        "Number of clusters",
                        min_value=2,
                        max_value=min(8, len(ordered_spectra)),
                        value=min(3, len(ordered_spectra)),
                        key="n_clusters"
                    )
                
                if st.button("🔬 Run Cluster Analysis", key="run_cluster"):
                    with st.spinner("Performing cluster analysis..."):
                        fig_cluster, labels, cluster_names = create_cluster_visualization(
                            processed_spectra, ordered_spectra, n_clusters, cluster_method
                        )
                        
                        if fig_cluster is not None:
                            st.session_state.cluster_results = (labels, cluster_names)
                            st.pyplot(fig_cluster)
                            
                            buf = BytesIO()
                            fig_cluster.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                            buf.seek(0)
                            b64 = base64.b64encode(buf.getvalue()).decode()
                            st.markdown(f"""
                            <div style="text-align: center; margin-top: 1rem;">
                                <a href="data:image/png;base64,{b64}" download="cluster_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                   color: white; border: none; border-radius: 8px; 
                                                   padding: 0.5rem 1rem; cursor: pointer;">
                                        📥 Download Cluster Analysis (PNG)
                                    </button>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                            plt.close(fig_cluster)
                            
                            # Display cluster assignment
                            if labels is not None and cluster_names is not None:
                                st.markdown("#### 📊 Cluster Assignments")
                                cluster_df = pd.DataFrame({
                                    'Spectrum': cluster_names,
                                    'Cluster': labels + 1
                                })
                                st.dataframe(cluster_df, use_container_width=True)
                        else:
                            st.error("❌ Cluster analysis failed. Please check your data.")
            else:
                st.warning("⚠️ Please load at least 3 spectra for multivariate analysis.")
                st.info("Multivariate analysis requires a minimum of 3 spectra to identify meaningful patterns.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # NEW TAB 6: Peak Statistics (Box/Violin plots)
        with tab6:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("📈 Peak Parameter Statistics")
            st.markdown("*Visualize distribution of peak parameters across spectra*")
            
            if st.session_state.get('correlation_ready', False) and st.session_state.get('correlation_peaks_df') is not None:
                peaks_df = st.session_state.correlation_peaks_df
                peaks_df = peaks_df[peaks_df['Include'] == True]
                
                if not peaks_df.empty:
                    st.markdown("#### 📊 Parameter Selection")
                    
                    parameter = st.selectbox(
                        "Select parameter to visualize",
                        options=['Intensity', 'Area', 'Peak position', 'FWHM'],
                        key="stat_parameter"
                    )
                    
                    plot_type = st.selectbox(
                        "Plot type",
                        options=['box', 'violin'],
                        format_func=lambda x: "Box Plot" if x == 'box' else "Violin Plot",
                        key="stat_plot_type"
                    )
                    
                    # Option to group by user-defined groups
                    use_groups = st.checkbox("Group spectra for comparison", value=False)
                    
                    group_mapping = None
                    if use_groups:
                        st.markdown("#### Assign spectra to groups:")
                        groups = {}
                        for name in ordered_spectra:
                            display_name = name.replace('.txt', '')
                            groups[name] = st.selectbox(
                                f"Group for {display_name}",
                                options=['Group 1', 'Group 2', 'Group 3', 'Group 4'],
                                key=f"group_{name}"
                            )
                        group_mapping = groups
                    
                    if st.button("📊 Generate Statistics Plot", key="gen_stats"):
                        fig_stat = create_parameter_visualization(
                            peaks_df, parameter, group_mapping, plot_type
                        )
                        
                        if fig_stat is not None:
                            st.pyplot(fig_stat)
                            
                            buf = BytesIO()
                            fig_stat.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                            buf.seek(0)
                            b64 = base64.b64encode(buf.getvalue()).decode()
                            st.markdown(f"""
                            <div style="text-align: center; margin-top: 1rem;">
                                <a href="data:image/png;base64,{b64}" download="statistics_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                   color: white; border: none; border-radius: 8px; 
                                                   padding: 0.5rem 1rem; cursor: pointer;">
                                        📥 Download Statistics Plot (PNG)
                                    </button>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                            plt.close(fig_stat)
                    
                    # Statistical tests
                    st.markdown("---")
                    st.markdown("#### 🔬 Statistical Tests")
                    st.markdown("*Perform statistical tests to compare groups*")
                    
                    if use_groups and group_mapping:
                        if st.button("📊 Run Statistical Tests", key="run_stats"):
                            with st.spinner("Running statistical tests..."):
                                stat_results = perform_statistical_tests(peaks_df, group_mapping)
                                
                                if stat_results:
                                    st.session_state.statistical_test_results = stat_results
                                    st.success("✅ Statistical tests completed!")
                                else:
                                    st.warning("⚠️ No valid data for statistical tests.")
                    
                    if st.session_state.statistical_test_results is not None:
                        stat_res = st.session_state.statistical_test_results
                        
                        st.markdown("#### 📊 Test Results")
                        for param, result in stat_res.items():
                            with st.expander(f"📈 {param} - {result['test']}"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Test Statistic", f"{result['statistic']:.4f}")
                                with col2:
                                    st.metric("p-value", f"{result['p_value']:.4f}")
                                with col3:
                                    significance = "✅ Significant (p < 0.05)" if result['significant'] else "❌ Not significant"
                                    st.metric("Significance", significance)
                                
                                # Show group data summary
                                st.markdown("**Group Data Summary:**")
                                for gname, data in result['groups'].items():
                                    if len(data) > 0:
                                        st.write(f"**{gname}:** n={len(data)}, mean={np.mean(data):.4f}, std={np.std(data):.4f}")
                else:
                    st.info("ℹ️ No peaks included. Please include at least one peak in the Peak Analysis tab.")
            else:
                st.info("📊 Please run peak analysis first to obtain peak parameters for statistics.")
                st.info("Go to the Peak Analysis tab and run peak detection.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # NEW TAB 7: Documentation
        with tab7:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.markdown("""
            # 📚 SpectrAnalys Documentation
            
            ## 📖 Getting Started
            
            ### File Format
            Upload `.txt` files with two columns (x y) separated by tabs:


        ### Basic Workflow
        1. **Upload Data** - Select one or more .txt files
        2. **Configure Analysis** - Choose spectra, assign colors, set parameters
        3. **Visualize** - Explore combined spectra visualization
        4. **Analyze Peaks** - Detect and characterize spectral peaks
        5. **Correlate Parameters** - Investigate relationships with experimental parameters
        6. **Compare Spectra** - Analyze differences between two spectra
        7. **Multivariate Analysis** - PCA and cluster analysis
        8. **Export Results** - Download processed data and plots
        
        ## 📊 Visualization Tab
        
        ### Four Visualization Modes
        1. **Raw Spectra** - Original intensity data
        2. **Normalized Spectra** - Intensity normalized by selected method
        3. **Offset Raw Spectra** - Raw spectra with cumulative offsets
        4. **Offset Normalized** - Normalized spectra with cumulative offsets
        
        ### X-axis Ranges (Broken Axis)
        - **Full range** - Display entire spectrum
        - **Custom ranges** - Display multiple x-axis ranges with gaps
        - Example: `100-150, 350-450, 600-800`
        
        ### Normalization Methods
        - **Maximum intensity** - Divide by global maximum
        - **Peak intensity (range)** - Divide by maximum in specified range
        - **Maximum rest intensity** - Divide by maximum in selected ranges
        
        ### Offset Settings
        - Cumulative offset: 1st spectrum at 0, 2nd at +step, 3rd at +2×step
        - Adjust step size for optimal visualization
        
        ## 🔍 Peak Analysis Tab
        
        ### Selecting Analysis Range
        - Use sliders to select left and right boundaries
        - Visual feedback shows selected range on spectra
        
        ### Peak Detection Parameters
        - **Peak width** - Number of points for area calculation
        - Detects peaks based on height and prominence
        
        ### Peak Parameters
        - **Position** - x-coordinate of peak maximum
        - **Intensity** - y-value at peak maximum
        - **Area** - Integral under peak
        - **FWHM** - Full Width at Half Maximum
        
        ### Include/Exclude Peaks
        - Check/uncheck peaks for inclusion in analysis
        - Only included peaks appear in visualizations and correlations
        
        ## 📈 Parameter Correlation Tab
        
        ### Assigning Parameters
        - Enter numeric values for each spectrum in sidebar
        - Parameter can be temperature, concentration, sample number, etc.
        
        ### Correlation Metrics
        - **Pearson correlation coefficient** (r)
          - r > 0.7: Strong positive correlation
          - r < -0.7: Strong negative correlation
          - |r| < 0.3: Weak correlation
        
        ### Visualizations
        - Scatter plots for each peak parameter vs assigned parameter
        - Trend lines and correlation coefficients displayed
        
        ## 🔀 Compare Spectra Tab
        
        ### Difference Analysis
        - Compares two spectra (Reference vs Sample)
        - Difference = Sample - Reference
        - Colormap shows positive (red) and negative (blue) differences
        
        ### Difference Decomposition
        Identifies nature of spectral changes:
        - **Scaling** - Overall intensity change (multiplicative)
        - **Shift** - Peak position change (frequency shift)
        - **Broadening** - Peak width change
        - **Other** - Remaining differences
        
        ### Interpretation
        - **Scaling factor > 1** - Sample has higher intensity
        - **Shift > 0** - Peaks shifted to higher frequencies
        - **Shift < 0** - Peaks shifted to lower frequencies
        - **Broadening > 1** - Peaks wider in sample
        
        ## 📊 Multivariate Analysis Tab
        
        ### Principal Component Analysis (PCA)
        - Reduces dimensionality to identify patterns
        - **Score plot** - Shows grouping/clustering of spectra
        - **Loading plot** - Identifies important spectral regions
        - **Scree plot** - Shows explained variance
        
        ### Cluster Analysis
        - **Hierarchical** - Dendrogram shows similarity hierarchy
        - **K-Means** - Groups spectra into k clusters
        - Optimal cluster number via Elbow method
        
        ## 📈 Peak Statistics Tab
        
        ### Box/Violin Plots
        - Visualize distribution of peak parameters
        - Compare between spectra or user-defined groups
        
        ### Statistical Tests
        - **T-test** - Compare two groups (parametric)
        - **Mann-Whitney U** - Compare two groups (non-parametric)
        - **ANOVA** - Compare three or more groups (parametric)
        - **Kruskal-Wallis** - Compare three or more groups (non-parametric)
        
        ### Interpreting p-values
        - p < 0.05: Statistically significant difference
        - p > 0.05: No significant difference
        
        ## 🎨 Signal Processing Options
        
        ### Filtering
        - **Savitzky-Golay** - Polynomial smoothing preserving peak shape
          - Window length: Number of points (odd)
          - Polynomial order: 2-5
        - **Gaussian** - Simple Gaussian smoothing
        
        ### Baseline Correction
        - **ALS** (Asymmetric Least Squares)
          - Lambda: Smoothness parameter (higher = smoother)
          - p: Asymmetry parameter (0.001-0.1)
        
        ### Second Derivative
        - Enhances peak detection
        - Useful for overlapping peaks
        
        ## 🔧 Troubleshooting
        
        ### Common Issues
        
        **Q: No peaks detected**
        - Adjust range boundaries
        - Lower detection thresholds
        - Check if spectra contain peaks in selected range
        
        **Q: Correlations not showing**
        - Run peak analysis first
        - Ensure peaks are "included" (checkbox)
        - Check that at least 3 spectra have peak data
        
        **Q: Difference decomposition shows NaN**
        - Ensure both spectra are normalized
        - Check for zero values in spectra
        - Try different normalization method
        
        **Q: PCA fails**
        - Need at least 3 spectra
        - Check for constant/near-constant spectra
        - Try normalizing data
        
        ## 💡 Tips
        
        1. **Color assignment** - Use distinct colors for easy identification
        2. **Offset step** - Adjust to prevent spectra overlap
        3. **Normalization** - Try different methods for best results
        4. **X-axis ranges** - Focus on regions of interest
        5. **Peak width** - Adjust based on peak sharpness
        6. **Save plots** - Use download buttons for publication-ready figures
        
        ## ⚡ Keyboard Shortcuts
        
        - **Ctrl+Enter** - Run analysis
        - **Tab** - Navigate between tabs
        - **Space** - Toggle checkboxes
        
        ## 📤 Export Options
        
        - **Raw Data** - CSV with all spectra
        - **Normalized Data** - CSV with normalized spectra
        - **Peak Analysis** - CSV with peak parameters
        - **Plots** - PNG (600 dpi, high quality)
        - **Session Info** - TXT with all settings
        
        ## 🤝 Support
        
        For issues or feature requests, please contact support.
        
        ## 📄 License
        
        SpectrAnalys v2.0 - Scientific Spectroscopic Analysis Platform
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export options section
        st.markdown("---")
        st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
        st.subheader("📤 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
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
        if filtered_spectra:
            export_norm = pd.DataFrame()
            for name, spec in filtered_spectra.items():
                data = spec['data']
                y_norm = normalize_spectrum(
                    data['x'].values, 
                    data['y'].values, 
                    norm_method, 
                    norm_range,
                    x_ranges
                )
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
        Filtering: {filter_method}
        Baseline Correction: {baseline_correction}
        Second Derivative: {show_second_derivative}
        Interactive Plots: {interactive_plots}
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
        7. **Multivariate Analysis** - Perform PCA and cluster analysis to identify patterns
        8. **Statistical Analysis** - Compare peak parameters across groups
        9. **Export Results** - Download processed data, plots, and analysis results
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
        - 📊 **PCA & Cluster Analysis** - Multivariate analysis for pattern detection
        - 📈 **Statistical Tests** - T-test, ANOVA, Mann-Whitney U tests for group comparison
        - 📐 **Signal Processing** - Savitzky-Golay filtering, baseline correction, second derivative
        - 🔄 **Interactive Plots** - Plotly-based interactive visualization with zoom and hover
        - 💾 **Data Export** - Download processed data in CSV format with publication-ready plots
        - 📚 **Comprehensive Documentation** - Detailed guides for all features
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
