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
from scipy.stats import pearsonr, ttest_ind, f_oneway, mannwhitneyu, shapiro, kstest
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, savgol_filter
from scipy.signal import savgol_coeffs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import lstsq
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
if 'statistical_tests_results' not in st.session_state:
    st.session_state.statistical_tests_results = None
if 'difference_decomposition_results' not in st.session_state:
    st.session_state.difference_decomposition_results = None

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
                           legend_fontsize=8, legend_position="right", legend_offset=1.02,
                           savgol_filter_window=None, savgol_filter_order=None,
                           baseline_method=None, baseline_params=None):
    """Create individual scientific plot with download button and advanced processing options"""
    
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
            
            # Apply Savitzky-Golay filter if requested
            if savgol_filter_window is not None and savgol_filter_order is not None:
                if len(y) >= savgol_filter_window:
                    try:
                        y = savgol_filter(y, savgol_filter_window, savgol_filter_order)
                    except:
                        pass  # Keep original if filtering fails
            
            # Apply baseline correction if requested
            if baseline_method is not None and baseline_method != "None":
                if baseline_method == "ALS":
                    # Asymmetric Least Squares baseline
                    if baseline_params is not None:
                        lam = baseline_params.get('lambda', 1e5)
                        p = baseline_params.get('p', 0.01)
                        y = als_baseline_correction(y, lam=lam, p=p)
                elif baseline_method == "Polynomial":
                    # Polynomial baseline with iterative peak removal
                    if baseline_params is not None:
                        degree = baseline_params.get('degree', 3)
                        y = polynomial_baseline_correction(x, y, degree=degree)
                elif baseline_method == "Rolling ball":
                    if baseline_params is not None:
                        window = baseline_params.get('window', 50)
                        y = rolling_ball_baseline_correction(x, y, window=window)
            
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
                
                # Apply Savitzky-Golay filter if requested
                if savgol_filter_window is not None and savgol_filter_order is not None:
                    if len(y_full) >= savgol_filter_window:
                        try:
                            y_full = savgol_filter(y_full, savgol_filter_window, savgol_filter_order)
                        except:
                            pass
                
                # Apply baseline correction if requested
                if baseline_method is not None and baseline_method != "None":
                    if baseline_method == "ALS":
                        if baseline_params is not None:
                            lam = baseline_params.get('lambda', 1e5)
                            p = baseline_params.get('p', 0.01)
                            y_full = als_baseline_correction(y_full, lam=lam, p=p)
                    elif baseline_method == "Polynomial":
                        if baseline_params is not None:
                            degree = baseline_params.get('degree', 3)
                            y_full = polynomial_baseline_correction(x_full, y_full, degree=degree)
                    elif baseline_method == "Rolling ball":
                        if baseline_params is not None:
                            window = baseline_params.get('window', 50)
                            y_full = rolling_ball_baseline_correction(x_full, y_full, window=window)
                
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
                         legend_fontsize=8, legend_position="right", legend_offset=1.02,
                         savgol_filter_window=None, savgol_filter_order=None,
                         baseline_method=None, baseline_params=None):
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
                
                # Apply Savitzky-Golay filter if requested
                if savgol_filter_window is not None and savgol_filter_order is not None:
                    if len(y) >= savgol_filter_window:
                        try:
                            y = savgol_filter(y, savgol_filter_window, savgol_filter_order)
                        except:
                            pass
                
                # Apply baseline correction if requested
                if baseline_method is not None and baseline_method != "None":
                    if baseline_method == "ALS":
                        if baseline_params is not None:
                            lam = baseline_params.get('lambda', 1e5)
                            p = baseline_params.get('p', 0.01)
                            y = als_baseline_correction(y, lam=lam, p=p)
                    elif baseline_method == "Polynomial":
                        if baseline_params is not None:
                            degree = baseline_params.get('degree', 3)
                            y = polynomial_baseline_correction(x, y, degree=degree)
                    elif baseline_method == "Rolling ball":
                        if baseline_params is not None:
                            window = baseline_params.get('window', 50)
                            y = rolling_ball_baseline_correction(x, y, window=window)
                
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
                    
                    # Apply Savitzky-Golay filter if requested
                    if savgol_filter_window is not None and savgol_filter_order is not None:
                        if len(y_full) >= savgol_filter_window:
                            try:
                                y_full = savgol_filter(y_full, savgol_filter_window, savgol_filter_order)
                            except:
                                pass
                    
                    # Apply baseline correction if requested
                    if baseline_method is not None and baseline_method != "None":
                        if baseline_method == "ALS":
                            if baseline_params is not None:
                                lam = baseline_params.get('lambda', 1e5)
                                p = baseline_params.get('p', 0.01)
                                y_full = als_baseline_correction(y_full, lam=lam, p=p)
                        elif baseline_method == "Polynomial":
                            if baseline_params is not None:
                                degree = baseline_params.get('degree', 3)
                                y_full = polynomial_baseline_correction(x_full, y_full, degree=degree)
                        elif baseline_method == "Rolling ball":
                            if baseline_params is not None:
                                window = baseline_params.get('window', 50)
                                y_full = rolling_ball_baseline_correction(x_full, y_full, window=window)
                    
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
    plt.subplots_adjust(hspace=0.4)  # INCREASED from 0.3 to 0.4
    
    # Adjust right margin for legend if needed
    if legend_position == "right":
        right_margin = min(0.92, legend_offset + 0.05)
        plt.subplots_adjust(right=right_margin)
    else:
        plt.tight_layout()
    
    return fig, (mean_diff, max_abs_diff, rms_diff, correlation)

# NEW FUNCTION: Create comparison plot with difference decomposition
def create_comparison_plot_with_decomposition(spectrum_a_data, spectrum_b_data, name_a, name_b,
                                              x_label, y_label, norm_method, norm_range,
                                              offset_step, fill_area, fill_alpha, subtract_min_intensity,
                                              show_grid, line_width, fig_width, fig_height,
                                              legend_fontsize, legend_position, legend_offset,
                                              colormap_name, smooth_difference, smooth_sigma,
                                              symmetric_scale, difference_threshold):
    """Create comparison plot with difference decomposition (scaling, shift, broadening)"""
    
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
    
    # --- DIFFERENCE DECOMPOSITION ---
    
    # 1. Determine optimal scaling factor
    # Minimize MSE: find k such that k * y_a_interp ≈ y_b_interp
    # k_opt = sum(y_a_interp * y_b_interp) / sum(y_a_interp**2)
    if np.sum(y_a_interp**2) > 0:
        k_opt = np.sum(y_a_interp * y_b_interp) / np.sum(y_a_interp**2)
        # Clamp to reasonable range
        k_opt = np.clip(k_opt, 0.1, 10.0)
    else:
        k_opt = 1.0
    
    y_scaled = k_opt * y_a_interp
    
    # 2. Determine shift by cross-correlation
    # Calculate cross-correlation between scaled reference and sample
    correlation = np.correlate(y_b_interp, y_scaled, mode='same')
    shift_idx = np.argmax(correlation) - len(y_scaled)//2
    # Convert to x units
    dx = common_x[1] - common_x[0]
    shift_value = shift_idx * dx
    
    # Apply shift to scaled reference
    if shift_value != 0:
        # Shift using interpolation
        y_shifted = np.interp(common_x, common_x - shift_value, y_scaled, left=0, right=0)
    else:
        y_shifted = y_scaled
    
    # 3. Determine broadening
    # Compare FWHM of peaks in reference and sample
    # Find peaks in both spectra
    peaks_a, _ = find_peaks(y_a_interp, height=np.max(y_a_interp)*0.05, prominence=np.max(y_a_interp)*0.03, distance=5)
    peaks_b, _ = find_peaks(y_b_interp, height=np.max(y_b_interp)*0.05, prominence=np.max(y_b_interp)*0.03, distance=5)
    
    fwhm_ratios = []
    if len(peaks_a) > 0 and len(peaks_b) > 0:
        # Match peaks based on position
        for peak_a_idx in peaks_a:
            peak_a_x = common_x[peak_a_idx]
            # Find nearest peak in sample
            nearest_idx = np.argmin(np.abs(common_x[peaks_b] - peak_a_x))
            if nearest_idx < len(peaks_b):
                peak_b_idx = peaks_b[nearest_idx]
                # Check if peaks are close enough
                if abs(common_x[peak_b_idx] - peak_a_x) < 5.0:  # Within 5 cm^-1
                    fwhm_a = calculate_fwhm(common_x, y_a_interp, peak_a_idx)
                    fwhm_b = calculate_fwhm(common_x, y_b_interp, peak_b_idx)
                    if fwhm_a > 0 and fwhm_b > 0:
                        fwhm_ratios.append(fwhm_b / fwhm_a)
    
    if fwhm_ratios:
        broadening_factor = np.median(fwhm_ratios)
    else:
        broadening_factor = 1.0
    
    # 4. Apply broadening to shifted reference
    # Simple broadening: smooth the shifted reference
    if broadening_factor > 1.0:
        sigma = (broadening_factor - 1.0) * 2.0  # Empirical conversion
        y_broadened = gaussian_filter1d(y_shifted, sigma=min(sigma, 5.0))
    else:
        y_broadened = y_shifted
    
    # 5. Calculate residual difference (unexplained)
    y_residual = y_b_interp - y_broadened
    
    # 6. Calculate variance explained by each component
    total_var = np.var(y_b_interp - y_a_interp)
    if total_var > 0:
        # Variance explained by scaling
        scaled_var = np.var(y_scaled - y_a_interp)
        # Variance explained by shift
        shifted_var = np.var(y_shifted - y_scaled)
        # Variance explained by broadening
        broadened_var = np.var(y_broadened - y_shifted)
        # Residual variance
        residual_var = np.var(y_residual)
        
        # Calculate percentages
        scaling_pct = (scaled_var / total_var) * 100 if total_var > 0 else 0
        shift_pct = (shifted_var / total_var) * 100 if total_var > 0 else 0
        broadening_pct = (broadened_var / total_var) * 100 if total_var > 0 else 0
        residual_pct = (residual_var / total_var) * 100 if total_var > 0 else 0
    else:
        scaling_pct = shift_pct = broadening_pct = residual_pct = 0
    
    # Create figure with 3 rows: original spectra, decomposition, residual
    fig = plt.figure(figsize=(fig_width, fig_height * 1.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.3)
    
    # Row 1: Original spectra with difference
    ax1 = fig.add_subplot(gs[0])
    
    # Plot spectra
    ax1.plot(common_x, y_a_interp, color=spectrum_a_data['color'], linewidth=line_width, 
             label=f'{name_a} (Reference)', alpha=0.7)
    ax1.plot(common_x, y_b_interp, color=spectrum_b_data['color'], linewidth=line_width, 
             label=f'{name_b} (Sample)', alpha=0.7)
    
    # Plot difference
    ax1.fill_between(common_x, 0, y_b_interp - y_a_interp, alpha=0.3, color='gray', label='Difference')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax1.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax1.set_ylabel(f'Normalized {y_label}', fontsize=10, fontweight='bold')
    ax1.legend(loc='best', fontsize=legend_fontsize, frameon=True, edgecolor='black')
    ax1.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Row 2: Decomposition components
    ax2 = fig.add_subplot(gs[1])
    
    # Plot the transformed reference and sample
    ax2.plot(common_x, y_b_interp, color='black', linewidth=line_width, 
             label=f'{name_b} (Sample)', alpha=0.7)
    ax2.plot(common_x, y_broadened, color='red', linewidth=line_width, 
             label='Reconstructed (scaled + shifted + broadened)', alpha=0.7, linestyle='--')
    
    # Show individual components
    ax2.plot(common_x, y_scaled, color='blue', linewidth=1.2, 
             label=f'Scaled (k={k_opt:.3f})', alpha=0.5, linestyle=':')
    ax2.plot(common_x, y_shifted, color='green', linewidth=1.2, 
             label=f'Shifted (Δx={shift_value:.2f} cm⁻¹)', alpha=0.5, linestyle='-.')
    
    ax2.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax2.set_ylabel(f'Normalized {y_label}', fontsize=10, fontweight='bold')
    ax2.legend(loc='best', fontsize=legend_fontsize-1, frameon=True, edgecolor='black')
    ax2.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Row 3: Residual difference
    ax3 = fig.add_subplot(gs[2])
    
    # Plot residual
    ax3.plot(common_x, y_residual, color='purple', linewidth=line_width, label='Residual (unexplained)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Highlight regions with large residuals
    residual_threshold = 0.1 * np.max(np.abs(y_residual)) if np.max(np.abs(y_residual)) > 0 else 0.01
    if residual_threshold > 0:
        large_residual = np.abs(y_residual) > residual_threshold
        if np.any(large_residual):
            # Find contiguous regions
            res_indices = np.where(large_residual)[0]
            if len(res_indices) > 0:
                y_min, y_max = ax3.get_ylim()
                start_idx = res_indices[0]
                for i in range(1, len(res_indices)):
                    if res_indices[i] > res_indices[i-1] + 1:
                        ax3.axvspan(common_x[start_idx], common_x[res_indices[i-1]], alpha=0.2, color='yellow')
                        start_idx = res_indices[i]
                ax3.axvspan(common_x[start_idx], common_x[res_indices[-1]], alpha=0.2, color='yellow')
    
    # Calculate residual statistics
    residual_std = np.std(y_residual)
    max_residual = np.max(np.abs(y_residual))
    
    ax3.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax3.set_ylabel('Residual (a.u.)', fontsize=10, fontweight='bold')
    ax3.legend(loc='best', fontsize=legend_fontsize, frameon=True, edgecolor='black')
    ax3.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = f"""Decomposition Statistics:
    • Scaling factor: {k_opt:.3f}
    • Peak shift: {shift_value:+.2f} cm⁻¹
    • Broadening factor: {broadening_factor:.3f}
    • Residual RMS: {residual_std:.4f}
    
    Variance explained:
    • Scaling: {scaling_pct:.1f}%
    • Shift: {shift_pct:.1f}%
    • Broadening: {broadening_pct:.1f}%
    • Residual: {residual_pct:.1f}%"""
    
    plt.figtext(0.85, 0.25, stats_text, fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Return decomposition results
    decomp_results = {
        'scaling_factor': k_opt,
        'shift_value': shift_value,
        'broadening_factor': broadening_factor,
        'residual_std': residual_std,
        'max_residual': max_residual,
        'scaling_pct': scaling_pct,
        'shift_pct': shift_pct,
        'broadening_pct': broadening_pct,
        'residual_pct': residual_pct
    }
    
    return fig, decomp_results

# NEW FUNCTION: Baseline correction methods
def als_baseline_correction(y, lam=1e5, p=0.01, max_iter=100):
    """Asymmetric Least Squares baseline correction"""
    y = np.array(y, dtype=float)
    L = len(y)
    # Build the difference matrix
    D = np.diff(np.eye(L), 2)
    D = D.astype(float)
    # Solve the system (I + lam * D.T @ D) * z = y
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve
    
    # Use sparse matrices for efficiency
    I = eye(L)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    A = I + lam * D.T @ D
    z = spsolve(A, y)
    
    # Iteratively update weights
    for _ in range(max_iter):
        z_old = z
        w = np.where(y > z, p, 1-p)
        W = diags(w)
        A = W + lam * D.T @ D
        z = spsolve(A, W @ y)
        if np.linalg.norm(z - z_old) < 1e-6:
            break
    
    return y - z

def polynomial_baseline_correction(x, y, degree=3, n_iter=10, threshold=3.0):
    """Polynomial baseline correction with iterative peak removal"""
    y = np.array(y, dtype=float)
    x = np.array(x, dtype=float)
    
    # Initial polynomial fit
    coeffs = np.polyfit(x, y, degree)
    baseline = np.polyval(coeffs, x)
    residual = y - baseline
    
    # Iterative refinement: remove points with positive residuals (peaks)
    mask = np.ones(len(y), dtype=bool)
    for _ in range(n_iter):
        # Fit only points below threshold
        std = np.std(residual[mask]) if np.any(mask) else 1.0
        if std < 1e-6:
            break
        new_mask = residual < threshold * std
        if np.sum(new_mask) < len(y) * 0.3:  # Keep at least 30% of points
            break
        mask = new_mask
        coeffs = np.polyfit(x[mask], y[mask], degree)
        baseline = np.polyval(coeffs, x)
        residual = y - baseline
    
    # Ensure baseline is not above data
    baseline = np.minimum(baseline, y)
    
    return y - baseline

def rolling_ball_baseline_correction(x, y, window=50):
    """Rolling ball baseline correction"""
    y = np.array(y, dtype=float)
    
    # Use minimum filter to find baseline
    from scipy.ndimage import minimum_filter1d
    baseline = minimum_filter1d(y, size=window, mode='nearest')
    
    # Smooth the baseline
    baseline = gaussian_filter1d(baseline, sigma=window/10)
    
    # Ensure baseline is not above data
    baseline = np.minimum(baseline, y)
    
    return y - baseline

# NEW FUNCTION: Peak fitting
def gaussian(x, amplitude, center, sigma, offset=0):
    """Gaussian function for peak fitting"""
    return offset + amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def lorentzian(x, amplitude, center, gamma, offset=0):
    """Lorentzian function for peak fitting"""
    return offset + amplitude * gamma**2 / ((x - center)**2 + gamma**2)

def voigt(x, amplitude, center, sigma, gamma, offset=0):
    """Voigt function (pseudo-Voigt) for peak fitting"""
    # Pseudo-Voigt approximation
    fwhm = np.sqrt(sigma**2 + gamma**2)
    eta = 1 - sigma / fwhm if fwhm > 0 else 0.5
    gauss_part = gaussian(x, amplitude, center, sigma, 0)
    lorentz_part = lorentzian(x, amplitude, center, gamma, 0)
    return offset + (1 - eta) * gauss_part + eta * lorentz_part

def fit_peaks(x, y, peak_positions, model='gaussian', bounds=None):
    """Fit multiple peaks simultaneously"""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Prepare initial parameters
    n_peaks = len(peak_positions)
    if n_peaks == 0:
        return None, None
    
    # Initial guesses
    p0 = []
    lower_bounds = []
    upper_bounds = []
    
    # Add offset
    p0.append(np.min(y))
    lower_bounds.append(-np.inf)
    upper_bounds.append(np.inf)
    
    for pos in peak_positions:
        # Find nearest data point to estimate amplitude and width
        idx = np.argmin(np.abs(x - pos))
        amplitude = y[idx] - np.min(y)
        width = 5.0  # Default width
        
        if model == 'gaussian':
            p0.extend([amplitude, pos, width])
            lower_bounds.extend([0, pos - 10, 0.5])
            upper_bounds.extend([np.inf, pos + 10, 20])
        elif model == 'lorentzian':
            p0.extend([amplitude, pos, width])
            lower_bounds.extend([0, pos - 10, 0.5])
            upper_bounds.extend([np.inf, pos + 10, 20])
        elif model == 'voigt':
            p0.extend([amplitude, pos, width, width/2])
            lower_bounds.extend([0, pos - 10, 0.5, 0.1])
            upper_bounds.extend([np.inf, pos + 10, 20, 20])
    
    # Define model function
    if model == 'gaussian':
        def model_func(x, *params):
            offset = params[0]
            result = np.zeros_like(x) + offset
            for i in range(n_peaks):
                amp = params[1 + i*3]
                center = params[2 + i*3]
                sigma = params[3 + i*3]
                result += gaussian(x, amp, center, sigma, 0)
            return result
    elif model == 'lorentzian':
        def model_func(x, *params):
            offset = params[0]
            result = np.zeros_like(x) + offset
            for i in range(n_peaks):
                amp = params[1 + i*3]
                center = params[2 + i*3]
                gamma = params[3 + i*3]
                result += lorentzian(x, amp, center, gamma, 0)
            return result
    elif model == 'voigt':
        def model_func(x, *params):
            offset = params[0]
            result = np.zeros_like(x) + offset
            for i in range(n_peaks):
                amp = params[1 + i*4]
                center = params[2 + i*4]
                sigma = params[3 + i*4]
                gamma = params[4 + i*4]
                result += voigt(x, amp, center, sigma, gamma, 0)
            return result
    
    try:
        popt, pcov = curve_fit(model_func, x, y, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
        # Calculate R²
        y_fit = model_func(x, *popt)
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        return popt, {'r_squared': r_squared, 'y_fit': y_fit, 'pcov': pcov}
    except:
        return None, None

# NEW FUNCTION: Box plot for peak parameters
def create_peak_parameter_boxplots(peaks_df, param_name='Intensity', groups=None):
    """Create box plot for peak parameters with optional grouping"""
    if peaks_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if groups is None:
        # One box for all spectra
        data = peaks_df[peaks_df['Include'] == True][param_name].values
        ax.boxplot(data, labels=['All spectra'])
        ax.set_ylabel(param_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{param_name} Distribution Across All Spectra', fontsize=12, fontweight='bold')
    else:
        # Grouped box plots
        group_names = list(groups.keys())
        group_data = []
        for group_name in group_names:
            group_spectra = groups[group_name]
            group_vals = []
            for spec in group_spectra:
                spec_data = peaks_df[(peaks_df['Spectrum'] == spec) & (peaks_df['Include'] == True)]
                if not spec_data.empty:
                    group_vals.extend(spec_data[param_name].values)
            group_data.append(group_vals)
        
        # Create box plot
        bp = ax.boxplot(group_data, labels=group_names, patch_artist=True)
        
        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for patch, color in zip(bp['boxes'], colors[:len(group_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add swarm plot overlay
        for i, data in enumerate(group_data):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.3, s=20, color='black')
        
        ax.set_ylabel(param_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{param_name} Distribution by Group', fontsize=12, fontweight='bold')
    
    ax.tick_params(direction='in', length=5, width=1)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    return fig

# NEW FUNCTION: PCA analysis
def perform_pca(spectra_dict, ordered_spectra, n_components=2, x_range=None):
    """Perform PCA on spectra data"""
    # Prepare data matrix
    data_matrix = []
    spectrum_names = []
    x_values = None
    
    for name in ordered_spectra:
        if name in spectra_dict:
            data = spectra_dict[name]['data']
            x = data['x'].values
            y = data['y'].values
            
            # Crop to range if specified
            if x_range is not None and len(x_range) == 2:
                mask = (x >= x_range[0]) & (x <= x_range[1])
                if np.any(mask):
                    x = x[mask]
                    y = y[mask]
            
            if x_values is None:
                x_values = x
            elif len(x) != len(x_values):
                # Interpolate to common grid
                y = np.interp(x_values, x, y)
            
            data_matrix.append(y)
            spectrum_names.append(name.replace('.txt', ''))
    
    if not data_matrix:
        return None, None, None, None
    
    data_matrix = np.array(data_matrix)
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix.T).T
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, data_scaled.shape[0], data_scaled.shape[1]))
    pca_result = pca.fit_transform(data_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    return pca_result, explained_variance, cumulative_variance, pca.components_

# NEW FUNCTION: Perform statistical tests
def perform_statistical_tests(peaks_df, group_mapping, param_name='Intensity'):
    """Perform statistical tests on peak parameters across groups"""
    if peaks_df.empty or not group_mapping:
        return None
    
    # Group data
    groups = {}
    for group_name, spectrum_names in group_mapping.items():
        group_values = []
        for spec in spectrum_names:
            spec_data = peaks_df[(peaks_df['Spectrum'] == spec) & (peaks_df['Include'] == True)]
            if not spec_data.empty:
                group_values.extend(spec_data[param_name].values)
        if group_values:
            groups[group_name] = np.array(group_values)
    
    if len(groups) < 2:
        return None
    
    results = {}
    
    # Check normality for each group
    for group_name, values in groups.items():
        if len(values) >= 3:
            stat, p_value = shapiro(values)
            results[f'{group_name}_normality'] = {'statistic': stat, 'p_value': p_value, 'normal': p_value > 0.05}
    
    # Perform t-test if two groups
    if len(groups) == 2:
        group_names = list(groups.keys())
        g1, g2 = groups[group_names[0]], groups[group_names[1]]
        # Welch's t-test (unequal variance)
        t_stat, p_value = ttest_ind(g1, g2, equal_var=False)
        results['t_test'] = {'statistic': t_stat, 'p_value': p_value, 'significant': p_value < 0.05}
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_mw = mannwhitneyu(g1, g2, alternative='two-sided')
        results['mann_whitney'] = {'statistic': u_stat, 'p_value': p_value_mw, 'significant': p_value_mw < 0.05}
    
    # Perform ANOVA if three or more groups
    if len(groups) >= 3:
        group_values = list(groups.values())
        f_stat, p_value = f_oneway(*group_values)
        results['anova'] = {'statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    
    return results

# NEW FUNCTION: Heatmap from spectra matrix (updated with better colorbar)
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
    
    # Determine number of ticks
    n_ticks = 6
    tick_positions = np.linspace(vmin, vmax, n_ticks)
    
    # Format tick labels
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
    
    # Add border around colorbar
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
    ax.set_xlabel(y_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(x_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Improve tick formatting
    ax.tick_params(direction='in', length=5, width=1)
    
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
        range_starts = [start for start, end in x_ranges]
        range_ends = [end for start, end in x_ranges]
        x_min = min(range_starts)
        x_max = max(range_ends)
        
        # Verify that there is data in these ranges for all spectra
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
    
    # Prepare matrices
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
                st.warning(f"No data points found in specified ranges for {name}. Using full spectrum.")
                x_cropped = x_orig
                y_cropped = y_orig
        else:
            x_cropped = x_orig
            y_cropped = y_orig
        
        if len(x_cropped) == 0:
            st.warning(f"Empty data after cropping for {name}. Skipping.")
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

# NEW FUNCTION: Correlation matrix heatmap
def create_correlation_matrix(spectra_dict, ordered_spectra, x_range=None):
    """Create correlation matrix heatmap for all spectra"""
    # Interpolate all spectra to common grid
    common_x = None
    spectra_data = {}
    
    for name in ordered_spectra:
        if name in spectra_dict:
            data = spectra_dict[name]['data']
            x = data['x'].values
            y = data['y'].values
            
            if x_range is not None and len(x_range) == 2:
                mask = (x >= x_range[0]) & (x <= x_range[1])
                if np.any(mask):
                    x = x[mask]
                    y = y[mask]
            
            if common_x is None:
                common_x = x
            elif len(x) != len(common_x):
                # Interpolate to common grid
                y = np.interp(common_x, x, y)
            
            spectra_data[name.replace('.txt', '')] = y
    
    if not spectra_data or len(spectra_data) < 2:
        return None, None
    
    # Build correlation matrix
    names = list(spectra_data.keys())
    n = len(names)
    corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(spectra_data[names[i]], spectra_data[names[j]])
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
    
    return corr_matrix, names

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
                    st.session_state.pca_results = None
                    st.session_state.cluster_results = None
                    st.session_state.statistical_tests_results = None
                    st.session_state.difference_decomposition_results = None
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
                    
                    st.markdown("#### 🏷️ Axis Labels")
                    x_label = st.text_input("X-axis label", value="Raman shift (cm⁻¹)")
                    y_label = st.text_input("Y-axis label", value="Intensity (a.u.)")
                    
                    # NEW: Advanced processing options
                    st.markdown("#### 🔧 Advanced Processing")
                    
                    with st.expander("📊 Savitzky-Golay Filtering"):
                        enable_savgol = st.checkbox("Enable Savitzky-Golay filter", value=False)
                        savgol_window = 11
                        savgol_order = 3
                        if enable_savgol:
                            col1, col2 = st.columns(2)
                            with col1:
                                savgol_window = st.slider("Window length (odd)", min_value=5, max_value=21, value=11, step=2, key="savgol_window")
                            with col2:
                                savgol_order = st.slider("Polynomial order", min_value=2, max_value=5, value=3, step=1, key="savgol_order")
                                if savgol_order >= savgol_window:
                                    st.warning("Order must be less than window length")
                        else:
                            savgol_window = None
                            savgol_order = None
                    
                    with st.expander("📐 Baseline Correction"):
                        baseline_method = st.selectbox("Baseline correction method", ["None", "ALS", "Polynomial", "Rolling ball"], key="baseline_method")
                        baseline_params = {}
                        if baseline_method == "ALS":
                            lam = st.slider("Lambda (smoothness)", min_value=1e3, max_value=1e7, value=1e5, step=1e3, format="%.0e", key="als_lambda")
                            p = st.slider("p (asymmetry)", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f", key="als_p")
                            baseline_params = {'lambda': lam, 'p': p}
                        elif baseline_method == "Polynomial":
                            degree = st.slider("Polynomial degree", min_value=1, max_value=6, value=3, key="poly_degree")
                            baseline_params = {'degree': degree}
                        elif baseline_method == "Rolling ball":
                            window = st.slider("Window size", min_value=10, max_value=200, value=50, step=5, key="roll_window")
                            baseline_params = {'window': window}
                    
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
                            step=0.1,
                            help="0.2 = more transparent, 0.9 = more opaque"
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
                    
                    # Heatmap Parameters Section
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
                        # NEW: Advanced processing settings
                        'savgol_window': savgol_window if enable_savgol else None,
                        'savgol_order': savgol_order if enable_savgol else None,
                        'baseline_method': baseline_method if baseline_method != "None" else None,
                        'baseline_params': baseline_params if baseline_method != "None" else None
                    }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p>🔬 SpectrAnalys v3.0<br>Scientific Spectroscopic Analysis</p>
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
        savgol_window = cached.get('savgol_window', None)
        savgol_order = cached.get('savgol_order', None)
        baseline_method = cached.get('baseline_method', None)
        baseline_params = cached.get('baseline_params', None)
        
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
        
        # UPDATED Tabs: Added Statistical Analysis, Multivariate Analysis, Documentation
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Combined Spectra Visualization",
            "🔍 Advanced Peak Analysis", 
            "📈 Parameter Correlation",
            "🔀 Compare Two Spectra",
            "📐 Statistical Analysis",
            "🧮 Multivariate Analysis",
            "📚 Documentation"
        ])
        
        with tab1:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Comprehensive Spectra Analysis")
            st.markdown("*All visualization modes combined for comprehensive spectral comparison*")
            
            # Display processing status
            if savgol_window is not None or baseline_method is not None:
                processing_info = []
                if savgol_window is not None:
                    processing_info.append(f"Savitzky-Golay (window={savgol_window}, order={savgol_order})")
                if baseline_method is not None:
                    processing_info.append(f"Baseline: {baseline_method}")
                st.info(f"🔧 Advanced processing active: {', '.join(processing_info)}")
            
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
            
            if subtract_min_intensity:
                for name in normalized_spectra:
                    y_vals = normalized_spectra[name]['data']['y'].values
                    if len(y_vals) > 0:
                        y_min = y_vals.min()
                        normalized_spectra[name]['data']['y'] = y_vals - y_min
                    else:
                        normalized_spectra[name]['data']['y'] = y_vals
            
            viz_configs = [
                (filtered_spectra, 0, False, False, False, y_label),
                (normalized_spectra, 0, False, True, False, f"Normalized {y_label}"),
                (filtered_spectra, raw_offset_step, False, False, True, y_label),
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
                    legend_offset=cached['legend_offset'],
                    savgol_filter_window=savgol_window,
                    savgol_filter_order=savgol_order,
                    baseline_method=baseline_method,
                    baseline_params=baseline_params
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
            
            # Heatmap plots section
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
                        
                        # NEW: Box plot visualization for peak parameters
                        st.markdown("---")
                        st.subheader("📊 Peak Parameter Distribution")
                        st.markdown("*Visualize distribution of peak parameters across spectra*")
                        
                        param_to_plot = st.selectbox(
                            "Select parameter to visualize",
                            options=['Intensity', 'Area', 'FWHM', 'Peak position'],
                            index=0,
                            key="boxplot_param"
                        )
                        
                        # Option to group spectra
                        group_by_spectra = st.checkbox("Group by spectrum", value=True)
                        
                        if group_by_spectra:
                            # Create box plot per spectrum
                            fig_box = create_peak_parameter_boxplots(peaks_df, param_to_plot)
                            if fig_box is not None:
                                st.pyplot(fig_box)
                                plt.close(fig_box)
                            else:
                                st.info("Not enough data for box plot")
                        else:
                            # Allow manual grouping
                            st.info("Manual grouping not implemented in this version")
                        
                        # NEW: Statistical tests on peak parameters
                        st.markdown("---")
                        st.subheader("📐 Statistical Tests on Peak Parameters")
                        st.markdown("*Compare peak parameters between groups*")
                        
                        if len(peaks_df) > 0:
                            # Create groups based on spectra
                            all_spectra = peaks_df['Spectrum'].unique()
                            if len(all_spectra) >= 2:
                                st.info(f"Available spectra: {', '.join(all_spectra)}")
                                
                                # Let user create groups
                                st.markdown("#### Define Groups for Comparison")
                                group1 = st.multiselect(
                                    "Group 1 spectra",
                                    options=all_spectra,
                                    default=all_spectra[:len(all_spectra)//2] if len(all_spectra) > 1 else [],
                                    key="stat_group1"
                                )
                                group2 = st.multiselect(
                                    "Group 2 spectra",
                                    options=all_spectra,
                                    default=all_spectra[len(all_spectra)//2:] if len(all_spectra) > 1 else [],
                                    key="stat_group2"
                                )
                                
                                if group1 and group2:
                                    group_mapping = {'Group 1': group1, 'Group 2': group2}
                                    test_param = st.selectbox(
                                        "Parameter for statistical test",
                                        options=['Intensity', 'Area', 'FWHM', 'Peak position'],
                                        index=0,
                                        key="stat_param"
                                    )
                                    
                                    if st.button("Run Statistical Tests", key="run_stat_tests"):
                                        results = perform_statistical_tests(peaks_df, group_mapping, test_param)
                                        if results:
                                            st.session_state.statistical_tests_results = results
                                            
                                            # Display results
                                            st.markdown("#### Statistical Test Results")
                                            for test_name, test_results in results.items():
                                                if 'normality' in test_name:
                                                    st.write(f"**{test_name}**")
                                                    st.write(f"  Shapiro-Wilk statistic: {test_results['statistic']:.4f}")
                                                    st.write(f"  p-value: {test_results['p_value']:.4f}")
                                                    st.write(f"  Normal distribution: {'✅ Yes' if test_results['normal'] else '❌ No'}")
                                                elif 't_test' in test_name:
                                                    st.write(f"**Welch's t-test**")
                                                    st.write(f"  t-statistic: {test_results['statistic']:.4f}")
                                                    st.write(f"  p-value: {test_results['p_value']:.4f}")
                                                    st.write(f"  Significant difference: {'✅ Yes' if test_results['significant'] else '❌ No'}")
                                                elif 'mann_whitney' in test_name:
                                                    st.write(f"**Mann-Whitney U test**")
                                                    st.write(f"  U-statistic: {test_results['statistic']:.4f}")
                                                    st.write(f"  p-value: {test_results['p_value']:.4f}")
                                                    st.write(f"  Significant difference: {'✅ Yes' if test_results['significant'] else '❌ No'}")
                                                elif 'anova' in test_name:
                                                    st.write(f"**ANOVA**")
                                                    st.write(f"  F-statistic: {test_results['statistic']:.4f}")
                                                    st.write(f"  p-value: {test_results['p_value']:.4f}")
                                                    st.write(f"  Significant difference: {'✅ Yes' if test_results['significant'] else '❌ No'}")
                                        else:
                                            st.warning("Could not perform statistical tests. Check group assignments.")
                                else:
                                    st.info("Select spectra for both groups to run statistical tests")
                            else:
                                st.info("Need at least 2 spectra for statistical comparison")
                        
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
        
        # UPDATED Tab 4: Compare Two Spectra with Decomposition
        with tab4:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("🔀 Spectral Difference Analysis")
            st.markdown("*Compare two spectra to identify differences and understand their nature (scaling, shift, broadening)*")
            
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
                
                # NEW: Difference analysis mode
                analysis_mode = st.radio(
                    "Analysis Mode",
                    ["Standard Difference", "Difference Decomposition (Scaling, Shift, Broadening)"],
                    index=0,
                    key="diff_mode"
                )
                
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
                        index=0,
                        key="diff_colormap"
                    )
                
                with col2:
                    smooth_difference = st.checkbox("Apply smoothing to difference profile", value=False, key="diff_smooth")
                    smooth_sigma = 1.0
                    if smooth_difference:
                        smooth_sigma = st.slider(
                            "Smoothing sigma",
                            min_value=0.5,
                            max_value=5.0,
                            value=1.5,
                            step=0.5,
                            key="diff_sigma"
                        )
                
                col1, col2 = st.columns(2)
                with col1:
                    symmetric_scale = st.checkbox("Symmetric color scale (centered at zero)", value=True, key="diff_sym")
                
                with col2:
                    difference_threshold = st.number_input(
                        "Significance threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.01,
                        format="%.3f",
                        key="diff_threshold"
                    )
                
                st.markdown("---")
                
                if spectrum_a_name and spectrum_b_name:
                    spectrum_a = filtered_spectra[spectrum_a_name]
                    spectrum_b = filtered_spectra[spectrum_b_name]
                    
                    name_a = spectrum_a_name.replace('.txt', '')
                    name_b = spectrum_b_name.replace('.txt', '')
                    
                    with st.spinner("Generating comparison plot..."):
                        if analysis_mode == "Standard Difference":
                            fig, (mean_diff, max_abs_diff, rms_diff, correlation) = create_comparison_plot(
                                spectrum_a, spectrum_b, name_a, name_b,
                                x_label, y_label, norm_method, norm_range,
                                norm_offset_step, fill_area, fill_alpha, subtract_min_intensity,
                                show_grid, line_width, fig_width, fig_height,
                                cached['legend_fontsize'], cached['legend_position'], cached['legend_offset'],
                                selected_colormap, smooth_difference, smooth_sigma,
                                symmetric_scale, difference_threshold
                            )
                            
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
                        else:
                            # NEW: Difference Decomposition mode
                            fig, decomp_results = create_comparison_plot_with_decomposition(
                                spectrum_a, spectrum_b, name_a, name_b,
                                x_label, y_label, norm_method, norm_range,
                                norm_offset_step, fill_area, fill_alpha, subtract_min_intensity,
                                show_grid, line_width, fig_width, fig_height,
                                cached['legend_fontsize'], cached['legend_position'], cached['legend_offset'],
                                selected_colormap, smooth_difference, smooth_sigma,
                                symmetric_scale, difference_threshold
                            )
                            
                            # Display decomposition results
                            st.markdown("#### 📊 Difference Decomposition Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Scaling Factor", f"{decomp_results['scaling_factor']:.3f}")
                            with col2:
                                st.metric("Peak Shift", f"{decomp_results['shift_value']:+.2f} cm⁻¹")
                            with col3:
                                st.metric("Broadening Factor", f"{decomp_results['broadening_factor']:.3f}")
                            
                            # Display variance explained
                            st.markdown("#### 📈 Variance Explained by Each Component")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Scaling", f"{decomp_results['scaling_pct']:.1f}%")
                            with col2:
                                st.metric("Shift", f"{decomp_results['shift_pct']:.1f}%")
                            with col3:
                                st.metric("Broadening", f"{decomp_results['broadening_pct']:.1f}%")
                            with col4:
                                st.metric("Residual", f"{decomp_results['residual_pct']:.1f}%")
                            
                            # Residual statistics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Residual RMS", f"{decomp_results['residual_std']:.4f}")
                            with col2:
                                st.metric("Max Residual", f"{decomp_results['max_residual']:.4f}")
                            
                            # Interpretation guide
                            st.markdown("#### 💡 Interpretation Guide")
                            interpretation = []
                            if decomp_results['scaling_pct'] > 30:
                                interpretation.append("🔹 **Scaling dominated**: Intensity changes significantly across entire spectrum")
                            if decomp_results['shift_pct'] > 20:
                                interpretation.append("🔹 **Shift dominated**: Peaks systematically shifted in frequency")
                            if decomp_results['broadening_pct'] > 20:
                                interpretation.append("🔹 **Broadening dominated**: Peak widths significantly changed")
                            if decomp_results['residual_pct'] > 30:
                                interpretation.append("🔹 **High residual**: Complex changes not captured by simple scaling/shift/broadening")
                            if not interpretation:
                                interpretation.append("🔹 **Simple changes**: Scaling, shift, and broadening explain most variations")
                            
                            for item in interpretation:
                                st.write(item)
                        
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
                            
                            y_diff_exp = y_b_interp_exp - y_a_interp_exp
                            
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
        
        # NEW TAB 5: Statistical Analysis
        with tab5:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("📐 Statistical Analysis")
            st.markdown("*Statistical analysis of spectra and peak parameters*")
            
            # Correlation matrix
            st.markdown("#### 📊 Spectral Correlation Matrix")
            st.markdown("*Visualize correlations between all spectra*")
            
            if len(filtered_spectra) >= 2:
                # Select range for correlation
                corr_x_min = float(np.min([spec['data']['x'].min() for spec in filtered_spectra.values()]))
                corr_x_max = float(np.max([spec['data']['x'].max() for spec in filtered_spectra.values()]))
                
                col1, col2 = st.columns(2)
                with col1:
                    corr_left = st.slider(
                        "Correlation left boundary",
                        min_value=corr_x_min,
                        max_value=corr_x_max,
                        value=corr_x_min,
                        step=(corr_x_max - corr_x_min) / 100,
                        key="corr_left"
                    )
                with col2:
                    corr_right = st.slider(
                        "Correlation right boundary",
                        min_value=corr_x_min,
                        max_value=corr_x_max,
                        value=corr_x_max,
                        step=(corr_x_max - corr_x_min) / 100,
                        key="corr_right"
                    )
                
                if corr_left < corr_right:
                    corr_range = (corr_left, corr_right)
                    
                    if st.button("Generate Correlation Matrix", key="gen_corr_matrix"):
                        corr_matrix, names = create_correlation_matrix(filtered_spectra, ordered_spectra, corr_range)
                        if corr_matrix is not None:
                            # Display heatmap
                            fig, ax = plt.subplots(figsize=(10, 8))
                            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                            
                            ax.set_xticks(range(len(names)))
                            ax.set_yticks(range(len(names)))
                            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
                            ax.set_yticklabels(names, fontsize=8)
                            
                            # Add text annotations
                            for i in range(len(names)):
                                for j in range(len(names)):
                                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                                 ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white",
                                                 fontsize=8)
                            
                            cbar = plt.colorbar(im, ax=ax)
                            cbar.set_label('Correlation coefficient', fontsize=11, fontweight='bold')
                            
                            ax.set_title('Spectral Correlation Matrix', fontsize=12, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Display correlation statistics
                            st.markdown("#### 📈 Correlation Statistics")
                            flat_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Correlation", f"{np.mean(flat_corr):.3f}")
                            with col2:
                                st.metric("Max Correlation", f"{np.max(flat_corr):.3f}")
                            with col3:
                                st.metric("Min Correlation", f"{np.min(flat_corr):.3f}")
                            
                            # Export correlation matrix
                            corr_df = pd.DataFrame(corr_matrix, index=names, columns=names)
                            csv_corr = corr_df.to_csv()
                            st.download_button(
                                label="📥 Download Correlation Matrix (CSV)",
                                data=csv_corr,
                                file_name=f"correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.warning("Left boundary must be less than right boundary")
            else:
                st.info("Need at least 2 spectra for correlation analysis")
            
            # Statistical tests on spectra (comparison of entire spectra)
            st.markdown("---")
            st.markdown("#### 📊 Statistical Comparison of Spectra")
            st.markdown("*Compare entire spectra using statistical tests*")
            
            if len(filtered_spectra) >= 2:
                # Select two spectra to compare statistically
                col1, col2 = st.columns(2)
                with col1:
                    spec_a_test = st.selectbox(
                        "Spectrum A",
                        options=list(filtered_spectra.keys()),
                        index=0,
                        key="spec_a_test"
                    )
                with col2:
                    spec_b_test = st.selectbox(
                        "Spectrum B",
                        options=list(filtered_spectra.keys()),
                        index=min(1, len(filtered_spectra)-1),
                        key="spec_b_test"
                    )
                
                if spec_a_test and spec_b_test and spec_a_test != spec_b_test:
                    if st.button("Compare Spectra Statistically", key="stat_compare_spectra"):
                        data_a = filtered_spectra[spec_a_test]['data']
                        data_b = filtered_spectra[spec_b_test]['data']
                        
                        # Interpolate to common x
                        x_common = np.linspace(
                            max(data_a['x'].min(), data_b['x'].min()),
                            min(data_a['x'].max(), data_b['x'].max()),
                            1000
                        )
                        y_a = np.interp(x_common, data_a['x'].values, data_a['y'].values)
                        y_b = np.interp(x_common, data_b['x'].values, data_b['y'].values)
                        
                        # Perform tests
                        # 1. Kolmogorov-Smirnov test (comparing distributions)
                        ks_stat, ks_p = kstest(y_a, y_b)
                        
                        # 2. Pearson correlation
                        corr, corr_p = pearsonr(y_a, y_b)
                        
                        # 3. Mean difference test (t-test)
                        t_stat, t_p = ttest_ind(y_a, y_b)
                        
                        # Display results
                        st.markdown("#### Statistical Test Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("KS Statistic", f"{ks_stat:.4f}")
                            st.metric("KS p-value", f"{ks_p:.4f}", delta="Significant" if ks_p < 0.05 else "Not significant")
                        with col2:
                            st.metric("Correlation", f"{corr:.4f}")
                            st.metric("Correlation p-value", f"{corr_p:.4f}", delta="Significant" if corr_p < 0.05 else "Not significant")
                        
                        st.metric("t-statistic", f"{t_stat:.4f}")
                        st.metric("t-test p-value", f"{t_p:.4f}", delta="Significant difference" if t_p < 0.05 else "No significant difference")
            else:
                st.info("Need at least 2 spectra for statistical comparison")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # NEW TAB 6: Multivariate Analysis
        with tab6:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("🧮 Multivariate Analysis")
            st.markdown("*PCA, clustering, and advanced multivariate techniques*")
            
            if len(filtered_spectra) >= 3:
                # PCA Analysis
                st.markdown("#### 📊 Principal Component Analysis (PCA)")
                st.markdown("*Dimensionality reduction to visualize spectral patterns*")
                
                # PCA range selection
                pca_x_min = float(np.min([spec['data']['x'].min() for spec in filtered_spectra.values()]))
                pca_x_max = float(np.max([spec['data']['x'].max() for spec in filtered_spectra.values()]))
                
                col1, col2 = st.columns(2)
                with col1:
                    pca_left = st.slider(
                        "PCA left boundary",
                        min_value=pca_x_min,
                        max_value=pca_x_max,
                        value=pca_x_min,
                        step=(pca_x_max - pca_x_min) / 100,
                        key="pca_left"
                    )
                with col2:
                    pca_right = st.slider(
                        "PCA right boundary",
                        min_value=pca_x_min,
                        max_value=pca_x_max,
                        value=pca_x_max,
                        step=(pca_x_max - pca_x_min) / 100,
                        key="pca_right"
                    )
                
                n_components = st.slider(
                    "Number of principal components",
                    min_value=2,
                    max_value=min(10, len(filtered_spectra)),
                    value=2,
                    key="pca_components"
                )
                
                if pca_left < pca_right:
                    pca_range = (pca_left, pca_right)
                    
                    if st.button("Run PCA", key="run_pca"):
                        with st.spinner("Performing PCA..."):
                            pca_result, explained_var, cum_var, components = perform_pca(
                                filtered_spectra, ordered_spectra, n_components, pca_range
                            )
                            
                            if pca_result is not None:
                                st.session_state.pca_results = {
                                    'scores': pca_result,
                                    'explained_variance': explained_var,
                                    'cumulative_variance': cum_var,
                                    'components': components,
                                    'names': [name.replace('.txt', '') for name in ordered_spectra if name in filtered_spectra]
                                }
                                
                                # Display explained variance
                                st.markdown("#### 📈 Explained Variance")
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                
                                # Scree plot
                                ax1.bar(range(1, len(explained_var)+1), explained_var * 100, alpha=0.7, color='#1f77b4')
                                ax1.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
                                ax1.set_ylabel('Explained Variance (%)', fontsize=11, fontweight='bold')
                                ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
                                ax1.grid(True, alpha=0.3, linestyle='--')
                                
                                # Cumulative variance
                                ax2.plot(range(1, len(cum_var)+1), cum_var * 100, 'bo-', linewidth=2, markersize=8)
                                ax2.set_xlabel('Number of Components', fontsize=11, fontweight='bold')
                                ax2.set_ylabel('Cumulative Variance (%)', fontsize=11, fontweight='bold')
                                ax2.set_title('Cumulative Variance', fontsize=12, fontweight='bold')
                                ax2.grid(True, alpha=0.3, linestyle='--')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Score plot (PC1 vs PC2)
                                if pca_result.shape[1] >= 2:
                                    st.markdown("#### 🎯 Score Plot")
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    names = st.session_state.pca_results['names']
                                    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                                                       s=100, c=range(len(names)), cmap='viridis',
                                                       alpha=0.7, edgecolors='black', linewidth=1.5)
                                    
                                    # Add labels
                                    for i, name in enumerate(names):
                                        ax.annotate(name, (pca_result[i, 0], pca_result[i, 1]),
                                                   fontsize=8, ha='center', va='bottom')
                                    
                                    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=11, fontweight='bold')
                                    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=11, fontweight='bold')
                                    ax.set_title('PCA Score Plot', fontsize=12, fontweight='bold')
                                    ax.grid(True, alpha=0.3, linestyle='--')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    
                                    # Loading plot (PC1)
                                    st.markdown("#### 📊 Loading Plot (PC1)")
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # Use common x grid
                                    x_common = np.linspace(pca_range[0], pca_range[1], len(components[0]))
                                    ax.plot(x_common, components[0], linewidth=2, color='#1f77b4')
                                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                                    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
                                    ax.set_ylabel('PC1 Loading', fontsize=11, fontweight='bold')
                                    ax.set_title('PC1 Loadings', fontsize=12, fontweight='bold')
                                    ax.grid(True, alpha=0.3, linestyle='--')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                            else:
                                st.error("PCA failed. Check data quality.")
                else:
                    st.warning("Left boundary must be less than right boundary")
                
                # Cluster Analysis
                st.markdown("---")
                st.markdown("#### 📊 Cluster Analysis")
                st.markdown("*Group spectra based on similarity*")
                
                cluster_method = st.selectbox(
                    "Clustering method",
                    options=["Hierarchical", "K-means"],
                    index=0,
                    key="cluster_method"
                )
                
                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=min(5, len(filtered_spectra)-1),
                    value=2,
                    key="n_clusters"
                )
                
                if st.button("Run Clustering", key="run_clustering"):
                    with st.spinner("Performing clustering..."):
                        # Prepare data matrix
                        data_matrix = []
                        names = []
                        for name in ordered_spectra:
                            if name in filtered_spectra:
                                data = filtered_spectra[name]['data']
                                y = data['y'].values
                                data_matrix.append(y)
                                names.append(name.replace('.txt', ''))
                        
                        if data_matrix:
                            data_matrix = np.array(data_matrix)
                            
                            if cluster_method == "Hierarchical":
                                # Hierarchical clustering
                                linkage_matrix = linkage(data_matrix, method='ward')
                                fig, ax = plt.subplots(figsize=(12, 6))
                                dendrogram(linkage_matrix, labels=names, ax=ax,
                                          leaf_rotation=45, leaf_font_size=8)
                                ax.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
                                ax.set_xlabel('Spectra', fontsize=11, fontweight='bold')
                                ax.set_ylabel('Distance', fontsize=11, fontweight='bold')
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Get cluster assignments
                                from scipy.cluster.hierarchy import fcluster
                                clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                            else:
                                # K-means clustering
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                clusters = kmeans.fit_predict(data_matrix)
                                
                                # Visualize clusters in PC space if PCA was run
                                if st.session_state.pca_results is not None:
                                    pca_scores = st.session_state.pca_results['scores']
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    scatter = ax.scatter(pca_scores[:, 0], pca_scores[:, 1],
                                                       c=clusters, cmap='viridis', s=100,
                                                       alpha=0.7, edgecolors='black', linewidth=1.5)
                                    
                                    for i, name in enumerate(names):
                                        ax.annotate(name, (pca_scores[i, 0], pca_scores[i, 1]),
                                                   fontsize=8, ha='center', va='bottom')
                                    
                                    ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
                                    ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
                                    ax.set_title(f'K-means Clusters (k={n_clusters})', fontsize=12, fontweight='bold')
                                    ax.grid(True, alpha=0.3, linestyle='--')
                                    plt.colorbar(scatter, ax=ax, label='Cluster')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                            
                            # Display cluster assignments
                            st.markdown("#### 🏷️ Cluster Assignments")
                            cluster_df = pd.DataFrame({
                                'Spectrum': names,
                                'Cluster': clusters
                            })
                            st.dataframe(cluster_df, use_container_width=True)
                            
                            # Export cluster data
                            csv_cluster = cluster_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Cluster Assignments (CSV)",
                                data=csv_cluster,
                                file_name=f"cluster_assignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
            else:
                st.info("Need at least 3 spectra for multivariate analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # NEW TAB 7: Documentation
        with tab7:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("📚 Documentation")
            st.markdown("*Complete guide to SpectrAnalys features and workflows*")
            
            st.markdown("""
            ## 📖 Table of Contents
            1. [Getting Started](#getting-started)
            2. [Data Import and Processing](#data-import)
            3. [Visualization](#visualization)
            4. [Peak Analysis](#peak-analysis)
            5. [Correlation Analysis](#correlation)
            6. [Spectral Comparison](#comparison)
            7. [Statistical Analysis](#statistical)
            8. [Multivariate Analysis](#multivariate)
            9. [Advanced Processing](#advanced)
            10. [Frequently Asked Questions](#faq)
            11. [Troubleshooting](#troubleshooting)
            
            ---
            
            <a name="getting-started"></a>
            ## 1. Getting Started

            ### Basic Workflow
            1. **Upload Files**: Use the sidebar to upload one or more .txt files
            2. **Select Spectra**: Choose which spectra to display and analyze
            3. **Configure Settings**: Set normalization, offset, colors, and processing options
            4. **Visualize**: Explore spectra in the combined visualization tab
            5. **Analyze**: Use peak analysis, correlation, comparison, or multivariate tools
            6. **Export**: Download data, plots, and results in various formats
            
            ---
            
            <a name="data-import"></a>
            ## 2. Data Import and Processing
            
            ### Uploading Files
            - Files must be in .txt format with tab-separated columns
            - First column: x-axis values (e.g., Raman shift)
            - Second column: y-axis values (e.g., intensity)
            - Multiple files can be uploaded simultaneously
            
            ### Spectrum Selection
            - Use the multi-select box to choose which spectra to display
            - Spectra can be reordered by selecting in desired order
            - Each spectrum can be assigned a custom color
            
            ### X-Axis Ranges
            - **Full range**: Display the entire spectrum
            - **Custom ranges**: Define multiple ranges (e.g., 100-150, 350-450)
            - Multiple ranges are displayed as separate segments with gaps
            
            ### Normalization Methods
            1. **Maximum intensity**: Normalize to the highest intensity peak
            2. **Peak intensity (range)**: Normalize to the maximum in a user-defined range
            3. **Maximum rest intensity**: Normalize to the maximum in specified ranges (useful for spectra with large background)
            
            ### Offset Settings
            - **Raw spectra offset**: Add vertical offset to raw spectra for clear visualization
            - **Normalized spectra offset**: Add vertical offset to normalized spectra
            - Offset increases cumulatively: 1st spectrum at 0, 2nd at +step, 3rd at +2×step, etc.
            
            ---
            
            <a name="visualization"></a>
            ## 3. Visualization
            
            ### Four Visualization Modes
            1. **Raw spectra**: Original intensity values
            2. **Normalized spectra**: Intensity normalized to 0-1 scale
            3. **Raw spectra with offsets**: Raw spectra with cumulative vertical offset
            4. **Normalized spectra with offsets**: Normalized spectra with cumulative vertical offset
            
            ### Customization Options
            - **Fill area**: Fill the area under normalized curves
            - **Fill transparency**: Adjust fill opacity (0.2-0.9)
            - **Subtract minimum intensity**: Remove baseline offset
            - **Grid**: Toggle grid lines
            - **Line thickness**: Adjust spectrum line width
            
            ### Legend Settings
            - Position: Right, Best, Upper right, Upper left, Lower left, Lower right
            - Font size: 4-16 points
            - Offset: Adjust legend distance from plot
            
            ### Heatmap Visualization
            - Requires assigning numeric values to spectra (temperature, concentration, etc.)
            - Interpolation methods: None, Bilinear, Bicubic, Spline16, Spline36, Gaussian, Lanczos
            - Color palettes: Viridis, Plasma, Inferno, Magma, Cividis, Spectral, Cool-Warm, etc.
            - Log scale available for intensity data
            
            ---
            
            <a name="peak-analysis"></a>
            ## 4. Peak Analysis
            
            ### Detection and Analysis
            1. **Select range**: Use sliders to choose analysis region
            2. **Run analysis**: Click "Analyze" to detect peaks
            3. **Review results**: Check/uncheck peaks to include/exclude
            4. **Visualize**: View peaks on the spectrum
            
            ### Peak Parameters
            - **Position**: Peak center (cm⁻¹)
            - **Intensity**: Peak height (a.u.)
            - **Area**: Integrated peak area
            - **FWHM**: Full Width at Half Maximum
            
            ### Include/Exclude Peaks
            - Use checkboxes to include/exclude peaks from analysis
            - Only included peaks appear in visualization and correlation analysis
            - Excluded peaks are hidden but not deleted
            
            ### Statistical Tests on Peaks
            - **t-test**: Compare peak parameters between two groups
            - **ANOVA**: Compare peak parameters among three or more groups
            - **Mann-Whitney U test**: Non-parametric comparison
            - **Shapiro-Wilk test**: Test for normal distribution
            
            ---
            
            <a name="correlation"></a>
            ## 5. Correlation Analysis
            
            ### Setup
            1. Enable correlation analysis in sidebar
            2. Assign numeric values to each spectrum
            3. Run peak analysis first (required)
            4. Select spectra and parameters
            
            ### Correlation Plots
            - **Intensity vs parameter**: Scatter plot with correlation coefficient
            - **Area vs parameter**: Correlation of peak area
            - **Position vs parameter**: Correlation of peak position
            - **FWHM vs parameter**: Correlation of peak width
            
            ### Interpretation
            - **r > 0.7**: Strong positive correlation
            - **r < -0.7**: Strong negative correlation
            - **|r| < 0.3**: Weak or no correlation
            - **p < 0.05**: Statistically significant
            
            ---
            
            <a name="comparison"></a>
            ## 6. Spectral Comparison
            
            ### Standard Difference Analysis
            - Compare two spectra with difference profile
            - Colored heatmap shows difference intensity
            - Highlight regions with significant differences
            
            ### Difference Decomposition (NEW)
            Identifies the nature of differences:
            
            1. **Scaling**: Overall intensity change (multiplication factor)
            2. **Shift**: Peak position change (frequency shift)
            3. **Broadening**: Peak width change (FWHM change)
            
            **Interpretation Guide:**
            - **Scaling > 30%**: Intensity changes across entire spectrum
            - **Shift > 20%**: Systematic peak frequency shift
            - **Broadening > 20%**: Significant peak width changes
            - **Residual > 30%**: Complex changes not explained by simple effects
            
            ---
            
            <a name="statistical"></a>
            ## 7. Statistical Analysis
            
            ### Correlation Matrix
            - Visualize correlations between all spectra
            - Red: Negative correlation, Blue: Positive correlation
            - Export matrix as CSV
            
            ### Spectral Comparison Tests
            - **KS test**: Compare spectral distributions
            - **Pearson correlation**: Linear relationship between spectra
            - **t-test**: Mean difference between spectra
            
            ---
            
            <a name="multivariate"></a>
            ## 8. Multivariate Analysis
            
            ### Principal Component Analysis (PCA)
            - **Score plot**: Visualize spectral patterns
            - **Loading plot**: Identify important spectral regions
            - **Scree plot**: Determine optimal number of components
            - **Explained variance**: Understand data structure
            
            ### Cluster Analysis
            - **Hierarchical clustering**: Dendrogram visualization
            - **K-means clustering**: Group spectra by similarity
            - **Cluster assignments**: View which spectra belong to which group
            
            ---
            
            <a name="advanced"></a>
            ## 9. Advanced Processing
            
            ### Savitzky-Golay Filter
            - Smooth spectra while preserving peak shapes
            - **Window length**: Odd number (5-21)
            - **Polynomial order**: 2-5
            - Higher order preserves details but may amplify noise
            
            ### Baseline Correction
            1. **ALS (Asymmetric Least Squares)**:
               - Lambda: Smoothness parameter (higher = smoother)
               - p: Asymmetry parameter (higher = more aggressive)
            
            2. **Polynomial**:
               - Degree: 1-6
               - Iteratively removes peaks for baseline estimation
            
            3. **Rolling Ball**:
               - Window size: 10-200 points
               - Simulates a ball rolling over the spectrum
            
            ---
            
            <a name="faq"></a>
            ## 10. Frequently Asked Questions
            
            ### Q: What file formats are supported?
            A: .txt files with two tab-separated columns (x, y).
            
            ### Q: How many spectra can I analyze?
            A: There's no hard limit, but performance may decrease with >50 spectra.
            
            ### Q: What normalization method should I use?
            A: Use "Maximum intensity" for most cases. Use "Peak intensity (range)" if you want to normalize to a specific reference peak.
            
            ### Q: Why aren't any peaks detected?
            A: Try adjusting the analysis range or peak detection parameters. Peaks must have sufficient height and prominence.
            
            ### Q: What does the difference decomposition show?
            A: It breaks down differences into scaling (intensity change), shift (frequency change), and broadening (width change) to help understand the nature of changes.
            
            ### Q: Can I export plots for publication?
            A: Yes, plots can be downloaded as high-resolution PNG (600 dpi).
            
            ---
            
            <a name="troubleshooting"></a>
            ## 11. Troubleshooting
            
            ### Common Issues
            
            **No data displayed:**
            - Check file format (two columns, tab-separated)
            - Verify data contains numeric values
            - Ensure spectra are selected in the sidebar
            
            **Slow performance:**
            - Reduce number of displayed spectra
            - Use fewer interpolation points
            - Disable animations if enabled
            
            **PCA fails:**
            - Ensure at least 3 spectra are loaded
            - Check that spectra have sufficient data points
            - Try reducing the spectral range
            
            **Peak analysis shows no results:**
            - Adjust detection parameters (height, prominence)
            - Try a different spectral range
            - Check that spectra are properly normalized
            
            **Heatmap not appearing:**
            - Assign numeric values to all spectra
            - Click "Apply for heatmaps" button
            - Ensure heatmap parameters are set correctly
            
            ### Getting Help
            - Check the documentation above
            - Review the data format requirements
            - Ensure all inputs are valid
            
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
                Savitzky-Golay Filter: {'Enabled' if savgol_window else 'Disabled'}
                Baseline Correction: {baseline_method if baseline_method else 'None'}
                """
                st.download_button(
                    label="📄 Export Session Info",
                    data=session_info,
                    file_name=f"spectranalys_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
