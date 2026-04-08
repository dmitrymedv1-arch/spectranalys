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

# ============================================================
# INITIALIZE SESSION STATE (сохраняем состояние между вкладками)
# ============================================================

def init_session_state():
    """Initialize all session state variables"""
    
    # Data state
    if 'spectra_data' not in st.session_state:
        st.session_state.spectra_data = {}
    if 'uploaded_files_loaded' not in st.session_state:
        st.session_state.uploaded_files_loaded = False
    if 'ordered_spectra' not in st.session_state:
        st.session_state.ordered_spectra = []
    if 'selected_spectra' not in st.session_state:
        st.session_state.selected_spectra = []
    
    # Processing options state
    if 'common_x_range' not in st.session_state:
        st.session_state.common_x_range = False
    if 'x_range_option' not in st.session_state:
        st.session_state.x_range_option = "Full range"
    if 'x_ranges_input' not in st.session_state:
        st.session_state.x_ranges_input = ""
    if 'x_ranges' not in st.session_state:
        st.session_state.x_ranges = None
    if 'x_label' not in st.session_state:
        st.session_state.x_label = "Raman shift (cm⁻¹)"
    if 'y_label' not in st.session_state:
        st.session_state.y_label = "Intensity (a.u.)"
    
    # Figure aspect ratio
    if 'aspect_ratio_option' not in st.session_state:
        st.session_state.aspect_ratio_option = "3×4"
    
    # Normalization options
    if 'norm_method' not in st.session_state:
        st.session_state.norm_method = "Maximum intensity"
    if 'norm_range_input' not in st.session_state:
        st.session_state.norm_range_input = ""
    if 'norm_range' not in st.session_state:
        st.session_state.norm_range = None
    if 'subtract_min_normalized' not in st.session_state:
        st.session_state.subtract_min_normalized = False
    
    # Offset options
    if 'raw_offset_step' not in st.session_state:
        st.session_state.raw_offset_step = 1000.0
    if 'norm_offset_step' not in st.session_state:
        st.session_state.norm_offset_step = 0.5
    
    # Fill area options
    if 'fill_area' not in st.session_state:
        st.session_state.fill_area = False
    if 'fill_type' not in st.session_state:
        st.session_state.fill_type = "Semi-transparent fill"
    
    # Peak analysis options
    if 'analyze_peaks_flag' not in st.session_state:
        st.session_state.analyze_peaks_flag = False
    if 'peak_width' not in st.session_state:
        st.session_state.peak_width = 20
    if 'peak_prominence_factor' not in st.session_state:
        st.session_state.peak_prominence_factor = 5
    if 'peak_height_factor' not in st.session_state:
        st.session_state.peak_height_factor = 10
    
    # Peak analysis region (manual selection)
    if 'x_min_selected' not in st.session_state:
        st.session_state.x_min_selected = None
    if 'x_max_selected' not in st.session_state:
        st.session_state.x_max_selected = None
    if 'peaks_df' not in st.session_state:
        st.session_state.peaks_df = pd.DataFrame()
    if 'peak_analysis_performed' not in st.session_state:
        st.session_state.peak_analysis_performed = False
    
    # Parameter correlation
    if 'param_correlation' not in st.session_state:
        st.session_state.param_correlation = False
    if 'param_values' not in st.session_state:
        st.session_state.param_values = {}
    if 'param_label' not in st.session_state:
        st.session_state.param_label = "Temperature (°C)"
    
    # Color assignments
    if 'colors' not in st.session_state:
        st.session_state.colors = {}
    
    # Current tab tracking
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Combined Spectra Visualization"

# Call initialization
init_session_state()

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
def normalize_spectrum(x, y, norm_method, norm_range=None, subtract_min=False):
    """Normalize spectrum using different methods"""
    if norm_method == "Maximum intensity":
        y_norm = y / y.max() if y.max() != 0 else y
    elif norm_method == "Area":
        area = simpson(y, x)
        y_norm = y / area if area != 0 else y
    elif norm_method == "Peak intensity (range)":
        if norm_range is not None:
            mask = (x >= norm_range[0]) & (x <= norm_range[1])
            if np.any(mask):
                max_in_range = y[mask].max()
                y_norm = y / max_in_range if max_in_range != 0 else y
            else:
                y_norm = y / y.max() if y.max() != 0 else y
        else:
            y_norm = y / y.max() if y.max() != 0 else y
    else:
        y_norm = y
    
    # Apply subtract minimum if requested
    if subtract_min:
        y_norm = y_norm - y_norm.min()
    
    return y_norm

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

# Function to create gradient fill between y and baseline
def gradient_fill(ax, x, y, baseline, color, alpha=0.3):
    """Create gradient fill from baseline to y curve"""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as patches
    from matplotlib.path import Path
    
    # Create a colormap that fades to transparent at the bottom
    cmap = LinearSegmentedColormap.from_list('gradient', [color, (1, 1, 1, 0)], N=100)
    
    # Create polygon for fill
    verts = np.vstack([np.stack([x, y], axis=1),
                       np.stack([x[::-1], np.full_like(x[::-1], baseline)], axis=1)])
    verts = np.vstack([verts, verts[0]])
    
    # Create path and patch
    path = Path(verts)
    patch = patches.PathPatch(path, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(patch)

# Function to create individual plot (single visualization type)
def create_individual_plot(spectra_dict, x_label, y_label, title,
                           offset_step, fill_area, normalized, use_offset,
                           norm_method, x_ranges=None, subtract_min=False,
                           gradient_fill_enabled=False, figure_size=(10, 6)):
    """Create individual scientific plot for a single visualization type"""
    
    # Prepare data with normalization if needed
    if normalized:
        plot_spectra = {}
        for name, spec in spectra_dict.items():
            data = spec['data']
            y_norm = normalize_spectrum(
                data['x'].values,
                data['y'].values,
                norm_method,
                None,
                subtract_min
            )
            plot_spectra[name] = {
                'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
                'color': spec['color']
            }
    else:
        plot_spectra = spectra_dict
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Store handles and labels for legend
    handles = []
    labels = []
    
    spectra_items = list(plot_spectra.items())
    
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
            
            if fill_area and normalized and use_offset:
                # Check if gradient fill is enabled
                if gradient_fill_enabled:
                    gradient_fill(ax, x, y_plot, offset, color, alpha=0.3)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
                else:
                    ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
            elif fill_area and normalized:
                if gradient_fill_enabled:
                    gradient_fill(ax, x, y_plot, 0, color, alpha=0.3)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
                else:
                    ax.fill_between(x, 0, y_plot, alpha=0.3, color=color)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
            else:
                line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
            
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
                
                # Apply cumulative offset if requested
                if use_offset:
                    offset = idx * offset_step
                else:
                    offset = 0
                
                y_plot = y + offset
                
                # Plot
                if fill_area and normalized and use_offset:
                    if gradient_fill_enabled:
                        gradient_fill(ax, x, y_plot, offset, color, alpha=0.3)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    else:
                        ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                elif fill_area and normalized:
                    if gradient_fill_enabled:
                        gradient_fill(ax, x, y_plot, 0, color, alpha=0.3)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    else:
                        ax.fill_between(x, 0, y_plot, alpha=0.3, color=color)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                else:
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                
                # Add to handles only for first range
                if range_idx == 0 and idx == 0:
                    handles.append(line_handle[0])
                    labels.append(display_name)
                elif range_idx == 0:
                    handles.append(line_handle[0])
                    labels.append(display_name)
            
            # Add vertical line for range boundaries
            ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    
    # Add legend outside the plot to the right
    if handles:
        if use_offset:
            reversed_handles = list(reversed(handles))
            reversed_labels = list(reversed(labels))
            legend = ax.legend(reversed_handles, reversed_labels, 
                              loc='center left', 
                              bbox_to_anchor=(1.02, 0.5),
                              fontsize=8,
                              frameon=True, 
                              edgecolor='black', 
                              prop={'weight': 'bold'})
            for text, handle in zip(legend.get_texts(), reversed_handles):
                text.set_color(handle.get_color())
        else:
            legend = ax.legend(handles, labels, 
                              loc='center left', 
                              bbox_to_anchor=(1.02, 0.5),
                              fontsize=8,
                              frameon=True, 
                              edgecolor='black', 
                              prop={'weight': 'bold'})
            for text, handle in zip(legend.get_texts(), handles):
                text.set_color(handle.get_color())
    
    ax.tick_params(direction='in', length=5, width=1)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    return fig


# Function to create combined plot with all four visualization types (vertical layout) with aspect ratio control
def create_combined_plot(spectra_dict, x_label, y_label, title,
                         raw_offset_step, norm_offset_step, fill_area,
                         norm_method, x_ranges=None, figure_aspect_ratio=(12, 18),
                         subtract_min_normalized=False, gradient_fill_enabled=False):
    """Create scientific plot with all four visualization types in vertical subplots"""
    
    # Prepare normalized spectra with subtract_min option
    normalized_spectra = {}
    for name, spec in spectra_dict.items():
        data = spec['data']
        y_norm = normalize_spectrum(
            data['x'].values,
            data['y'].values,
            norm_method,
            None,
            subtract_min_normalized
        )
        normalized_spectra[name] = {
            'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
            'color': spec['color']
        }
    
    # Create figure with 4 subplots vertically (4 rows, 1 column) using custom aspect ratio
    fig, axes = plt.subplots(4, 1, figsize=figure_aspect_ratio)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Define the four visualization types
    viz_configs = [
        (axes[0], "Raw Spectra", spectra_dict, 0, False, False, False, x_label, y_label),
        (axes[1], f"Normalized Spectra ({norm_method})", normalized_spectra, 0, False, True, False, x_label, f"Normalized Intensity ({norm_method})"),
        (axes[2], f"Raw Spectra + Offset (step = {raw_offset_step})", spectra_dict, raw_offset_step, False, False, True, x_label, y_label),
        (axes[3], f"Normalized Spectra + Offset (step = {norm_offset_step})", normalized_spectra, norm_offset_step, fill_area, True, True, x_label, f"Normalized Intensity ({norm_method})")
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
                    # Check if gradient fill is enabled for the normalized+offset plot
                    if gradient_fill_enabled and subplot_title.startswith("Normalized Spectra + Offset"):
                        # Apply gradient fill
                        gradient_fill(ax, x, y_plot, offset, color, alpha=0.3)
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
                    else:
                        # Regular semi-transparent fill
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
                        # Check if gradient fill is enabled for the normalized+offset plot
                        if gradient_fill_enabled and subplot_title.startswith("Normalized Spectra + Offset"):
                            gradient_fill(ax, x, y_plot, offset, color, alpha=0.3)
                            line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                        else:
                            ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                            line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    elif fill and normalized:
                        if gradient_fill_enabled and subplot_title.startswith("Normalized Spectra + Offset"):
                            gradient_fill(ax, x, y_plot, 0, color, alpha=0.3)
                            line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                        else:
                            ax.fill_between(x, 0, y_plot, alpha=0.3, color=color)
                            line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    else:
                        line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    
                    # Add to handles only for first range
                    if range_idx == 0 and idx == 0:
                        handles.append(line_handle[0])
                        labels.append(display_name)
                    elif range_idx == 0:
                        handles.append(line_handle[0])
                        labels.append(display_name)
                
                # Add vertical line for range boundaries
                ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
                ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            
            ax.set_xlabel(xl, fontsize=10, fontweight='bold')
            ax.set_ylabel(yl, fontsize=10, fontweight='bold')
            ax.set_title(subplot_title, fontsize=11, fontweight='bold')
        
        # Add legend outside the plot to the right
        if handles:
            # For offset plots, reverse the legend order so top curve appears first
            if use_offset:
                # Create reversed lists for legend
                reversed_handles = list(reversed(handles))
                reversed_labels = list(reversed(labels))
                
                # Place legend outside the plot - to the right
                legend = ax.legend(reversed_handles, reversed_labels, 
                                  loc='center left', 
                                  bbox_to_anchor=(1.02, 0.5),
                                  fontsize=8,
                                  frameon=True, 
                                  edgecolor='black', 
                                  prop={'weight': 'bold'})
                
                # Set legend text colors to match line colors using reversed handles
                for text, handle in zip(legend.get_texts(), reversed_handles):
                    text.set_color(handle.get_color())
            else:
                # Place legend outside the plot - to the right
                legend = ax.legend(handles, labels, 
                                  loc='center left', 
                                  bbox_to_anchor=(1.02, 0.5),
                                  fontsize=8,
                                  frameon=True, 
                                  edgecolor='black', 
                                  prop={'weight': 'bold'})
                
                # Set legend text colors to match line colors
                for text, handle in zip(legend.get_texts(), handles):
                    text.set_color(handle.get_color())
        
        ax.tick_params(direction='in', length=5, width=1)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4, right=0.85)
    
    return fig

# Function to calculate FWHM for a peak
def calculate_fwhm(x, y, peak_idx, peak_prominence=0.5):
    """Calculate Full Width at Half Maximum for a peak"""
    peak_y = y[peak_idx]
    half_max = peak_y * peak_prominence
    
    # Find left half-max point
    left_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if y[i] <= half_max:
            left_idx = i
            break
    
    # Find right half-max point
    right_idx = peak_idx
    for i in range(peak_idx + 1, len(y)):
        if y[i] <= half_max:
            right_idx = i
            break
    
    # Interpolate for more accurate FWHM
    if left_idx > 0 and right_idx < len(x) - 1:
        # Linear interpolation for left side
        x_left = x[left_idx] + (x[left_idx + 1] - x[left_idx]) * (half_max - y[left_idx]) / (y[left_idx + 1] - y[left_idx])
        # Linear interpolation for right side
        x_right = x[right_idx] + (x[right_idx + 1] - x[right_idx]) * (half_max - y[right_idx]) / (y[right_idx + 1] - y[right_idx])
        fwhm = x_right - x_left
    else:
        fwhm = (x[right_idx] - x[left_idx]) if right_idx > left_idx else 0
    
    return fwhm

# Function for peak analysis with region selection
def analyze_peaks_in_region(spectra_dict, x_min, x_max, peak_width=20, peak_prominence_factor=0.05, peak_height_factor=0.1):
    """Analyze peaks in a specific x-axis region"""
    results = []
    
    for name, spec in spectra_dict.items():
        data = spec['data']
        x = data['x'].values
        y = data['y'].values
        
        # Crop to selected region
        mask = (x >= x_min) & (x <= x_max)
        if not np.any(mask):
            continue
        
        x_cropped = x[mask]
        y_cropped = y[mask]
        
        if len(x_cropped) == 0:
            continue
        
        # Find peaks
        peaks, properties = find_peaks(y_cropped, 
                                       height=np.max(y_cropped) * peak_height_factor, 
                                       prominence=np.max(y_cropped) * peak_prominence_factor)
        
        for peak_idx in peaks:
            peak_x = x_cropped[peak_idx]
            peak_y = y_cropped[peak_idx]
            
            # Calculate area around peak
            left_idx = max(0, peak_idx - peak_width)
            right_idx = min(len(x_cropped), peak_idx + peak_width)
            area = simpson(y_cropped[left_idx:right_idx+1], x_cropped[left_idx:right_idx+1])
            
            # Calculate FWHM
            fwhm = calculate_fwhm(x_cropped, y_cropped, peak_idx)
            
            results.append({
                'Spectrum': name.replace('.txt', ''),
                'Peak position (cm⁻¹)': peak_x,
                'Intensity (a.u.)': peak_y,
                'Area': area,
                'FWHM (cm⁻¹)': fwhm
            })
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# Function to get global x range for sliders
def get_global_x_range(spectra_dict):
    """Get global min and max x values across all spectra"""
    if not spectra_dict:
        return 0, 100
    
    min_x = float('inf')
    max_x = float('-inf')
    
    for spec in spectra_dict.values():
        x = spec['data']['x'].values
        min_x = min(min_x, x.min())
        max_x = max(max_x, x.max())
    
    return min_x, max_x

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
        
        if uploaded_files:
            # Load data only when new files are uploaded
            if not st.session_state.uploaded_files_loaded or len(uploaded_files) != len(st.session_state.spectra_data):
                st.session_state.spectra_data = {}
                for file in uploaded_files:
                    data = load_spectrum(file)
                    if data is not None:
                        st.session_state.spectra_data[file.name] = {
                            'data': data,
                            'color': None
                        }
                st.session_state.uploaded_files_loaded = True
                st.session_state.ordered_spectra = list(st.session_state.spectra_data.keys())
                st.session_state.selected_spectra = list(st.session_state.spectra_data.keys())
            
            if st.session_state.spectra_data:
                st.success(f"✅ Loaded {len(st.session_state.spectra_data)} files")
                
                st.markdown("---")
                st.markdown("### 📋 Spectrum Selection")
                
                # Select and order spectra with session state
                selected_spectra = st.multiselect(
                    "Choose spectra to display",
                    options=list(st.session_state.spectra_data.keys()),
                    default=st.session_state.selected_spectra,
                    key="multiselect_spectra"
                )
                st.session_state.selected_spectra = selected_spectra
                
                if selected_spectra:
                    # Order spectra
                    ordered_spectra = []
                    for name in selected_spectra:
                        ordered_spectra.append(name)
                    st.session_state.ordered_spectra = ordered_spectra
                    
                    # Assign colors with default distinct colors
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
                            if name not in st.session_state.colors:
                                st.session_state.colors[name] = default_color
                            color_val = st.color_picker(
                                f"Color {i+1}",
                                value=st.session_state.colors[name],
                                key=f"color_{name}"
                            )
                            st.session_state.colors[name] = color_val
                    
                    # Update spectra data with colors
                    for name in ordered_spectra:
                        st.session_state.spectra_data[name]['color'] = st.session_state.colors[name]
                    
                    st.markdown("---")
                    st.markdown("### ⚙️ Processing Options")
                    
                    # Common x range option
                    common_x_range = st.checkbox("Align all spectra to common x range", 
                                                  value=st.session_state.common_x_range,
                                                  key="common_x_range_checkbox")
                    st.session_state.common_x_range = common_x_range
                    
                    # X-axis ranges
                    st.markdown("#### 📊 X-axis Ranges")
                    x_range_option = st.radio(
                        "Select range mode",
                        ["Full range", "Custom ranges (multiple)"],
                        index=0 if st.session_state.x_range_option == "Full range" else 1,
                        key="x_range_option_radio"
                    )
                    st.session_state.x_range_option = x_range_option
                    
                    x_ranges = None
                    if x_range_option == "Custom ranges (multiple)":
                        range_input = st.text_area(
                            "Enter ranges (e.g., 100-150, 350-450, 600-800)",
                            placeholder="100-150, 350-450, 600-800",
                            value=st.session_state.x_ranges_input,
                            help="Each range will be displayed as a separate segment on the same graph",
                            key="x_ranges_input_area"
                        )
                        st.session_state.x_ranges_input = range_input
                        if range_input:
                            x_ranges = parse_x_ranges(range_input)
                            if x_ranges:
                                st.session_state.x_ranges = x_ranges
                                st.info(f"📌 Selected {len(x_ranges)} ranges: {', '.join([f'{r[0]:.0f}-{r[1]:.0f}' for r in x_ranges])}")
                            else:
                                st.session_state.x_ranges = None
                        else:
                            st.session_state.x_ranges = None
                    else:
                        st.session_state.x_ranges = None
                    
                    # Axis labels
                    st.markdown("#### 🏷️ Axis Labels")
                    x_label = st.text_input("X-axis label", 
                                            value=st.session_state.x_label,
                                            key="x_label_input")
                    y_label = st.text_input("Y-axis label", 
                                            value=st.session_state.y_label,
                                            key="y_label_input")
                    st.session_state.x_label = x_label
                    st.session_state.y_label = y_label
                    
                    # Figure aspect ratio control
                    st.markdown("#### 📐 Figure Aspect Ratio")
                    aspect_ratio_option = st.selectbox(
                        "Graph area size (height × width, excluding legend)",
                        ["3×3", "4×3", "5×3", "6×3", "7×3", "9×3"],
                        index=["3×3", "4×3", "5×3", "6×3", "7×3", "9×3"].index(st.session_state.aspect_ratio_option) if st.session_state.aspect_ratio_option in ["3×3", "4×3", "5×3", "6×3", "7×3", "9×3"] else 1,
                        help="Controls the size of the actual plot area (axes with numbers and labels)",
                        key="aspect_ratio_select"
                    )
                    st.session_state.aspect_ratio_option = aspect_ratio_option
                    
                    # Convert aspect ratio to figsize
                    aspect_map = {
                        "3×3": (8, 8),
                        "4×3": (8, 10.67),
                        "5×3": (8, 13.33),
                        "6×3": (8, 16),
                        "7×3": (8, 18.67),
                        "9×3": (8, 24)
                    }
                    figure_aspect_ratio = aspect_map[aspect_ratio_option]
                    
                    # Normalization options
                    st.markdown("#### 📐 Normalization")
                    norm_method = st.selectbox(
                        "Normalization method",
                        ["Maximum intensity", "Area", "Peak intensity (range)"],
                        index=["Maximum intensity", "Area", "Peak intensity (range)"].index(st.session_state.norm_method),
                        key="norm_method_select"
                    )
                    st.session_state.norm_method = norm_method
                    
                    norm_range = None
                    if norm_method == "Peak intensity (range)":
                        norm_range_input = st.text_input(
                            "Peak range for normalization (e.g., 800-1000)",
                            placeholder="800-1000",
                            value=st.session_state.norm_range_input,
                            key="norm_range_input"
                        )
                        st.session_state.norm_range_input = norm_range_input
                        if norm_range_input:
                            try:
                                start, end = norm_range_input.split('-')
                                norm_range = (float(start), float(end))
                                st.session_state.norm_range = norm_range
                            except:
                                st.warning("Invalid range format")
                                st.session_state.norm_range = None
                        else:
                            st.session_state.norm_range = None
                    
                    # Subtract minimum for normalized spectra
                    st.markdown("#### 🔧 Normalized Spectra Adjustments")
                    subtract_min_normalized = st.checkbox(
                        "Subtract minimum intensity from normalized spectra",
                        value=st.session_state.subtract_min_normalized,
                        help="Subtracts the minimum intensity value from each normalized spectrum, shifting baseline to zero",
                        key="subtract_min_checkbox"
                    )
                    st.session_state.subtract_min_normalized = subtract_min_normalized
                    
                    # Offset options
                    st.markdown("#### 📈 Offset Settings")
                    col1, col2 = st.columns(2)
                    with col1:
                        raw_offset_step = st.slider(
                            "Raw spectra offset step",
                            min_value=0.0,
                            max_value=50000.0,
                            value=st.session_state.raw_offset_step,
                            step=100.0,
                            key="raw_offset_step_slider"
                        )
                        st.session_state.raw_offset_step = raw_offset_step
                    with col2:
                        norm_offset_step = st.slider(
                            "Normalized spectra offset step",
                            min_value=0.0,
                            max_value=5.0,
                            value=st.session_state.norm_offset_step,
                            step=0.05,
                            key="norm_offset_step_slider"
                        )
                        st.session_state.norm_offset_step = norm_offset_step
                    
                    # Fill area option with gradient choice
                    st.markdown("#### 🎨 Fill Area Settings")
                    fill_area = st.checkbox("Fill area under normalized spectra", 
                                            value=st.session_state.fill_area,
                                            key="fill_area_checkbox")
                    st.session_state.fill_area = fill_area
                    
                    gradient_fill_enabled = False
                    if fill_area:
                        fill_type = st.radio(
                            "Fill type for normalized spectra+offset",
                            ["Semi-transparent fill", "Gradient fill (fades to transparent)"],
                            index=0 if st.session_state.fill_type == "Semi-transparent fill" else 1,
                            help="Gradient fill creates a smooth fade from opaque at the top to transparent at the bottom",
                            key="fill_type_radio"
                        )
                        st.session_state.fill_type = fill_type
                        gradient_fill_enabled = (fill_type == "Gradient fill (fades to transparent)")
                    
                    # Peak analysis options
                    st.markdown("---")
                    st.markdown("### 🔍 Peak Analysis Settings")
                    analyze_peaks_flag = st.checkbox("Enable advanced peak analysis", 
                                                     value=st.session_state.analyze_peaks_flag,
                                                     key="analyze_peaks_checkbox")
                    st.session_state.analyze_peaks_flag = analyze_peaks_flag
                    
                    if analyze_peaks_flag:
                        peak_width = st.slider(
                            "Peak width for area calculation (points)",
                            min_value=5,
                            max_value=100,
                            value=st.session_state.peak_width,
                            step=5,
                            key="peak_width_slider"
                        )
                        st.session_state.peak_width = peak_width
                        
                        peak_prominence_factor = st.slider(
                            "Peak prominence factor (% of max)",
                            min_value=1,
                            max_value=20,
                            value=st.session_state.peak_prominence_factor,
                            step=1,
                            key="peak_prominence_slider"
                        ) / 100.0
                        st.session_state.peak_prominence_factor = peak_prominence_factor * 100
                        
                        peak_height_factor = st.slider(
                            "Peak height threshold (% of max)",
                            min_value=1,
                            max_value=30,
                            value=st.session_state.peak_height_factor,
                            step=1,
                            key="peak_height_slider"
                        ) / 100.0
                        st.session_state.peak_height_factor = peak_height_factor * 100
                    
                    # Parameter correlation
                    st.markdown("---")
                    st.markdown("### 📊 Parameter Correlation")
                    param_correlation = st.checkbox("Enable correlation analysis", 
                                                    value=st.session_state.param_correlation,
                                                    key="param_correlation_checkbox")
                    st.session_state.param_correlation = param_correlation
                    
                    if param_correlation:
                        st.info("💡 Assign numeric values to each spectrum for correlation analysis")
                        param_values = {}
                        for name in st.session_state.ordered_spectra:
                            if name not in st.session_state.param_values:
                                st.session_state.param_values[name] = float(len(st.session_state.param_values) + 1)
                            param_values[name] = st.number_input(
                                f"Value for {name.replace('.txt', '')}",
                                value=st.session_state.param_values[name],
                                step=1.0,
                                key=f"param_{name}"
                            )
                            st.session_state.param_values[name] = param_values[name]
                        
                        param_label = st.text_input("Parameter label", 
                                                    value=st.session_state.param_label,
                                                    key="param_label_input")
                        st.session_state.param_label = param_label
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar footer with info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p>🔬 SpectrAnalys v2.0<br>Scientific Spectroscopic Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.spectra_data and st.session_state.selected_spectra:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.ordered_spectra)}</div>
                <div class="metric-label">Spectra Loaded</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.session_state.x_ranges:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(st.session_state.x_ranges)}</div>
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
                <div class="metric-value">{st.session_state.norm_method[:10]}</div>
                <div class="metric-label">Normalization</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'✓' if st.session_state.analyze_peaks_flag else '✗'}</div>
                <div class="metric-label">Peak Analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Apply common x range if selected
        current_spectra = st.session_state.spectra_data
        if st.session_state.common_x_range:
            current_spectra = align_x_ranges(current_spectra)
        
        # Filter spectra based on selection
        filtered_spectra = {name: current_spectra[name] for name in st.session_state.ordered_spectra if name in current_spectra}
        
        # Get global x range for peak analysis region sliders
        global_x_min, global_x_max = get_global_x_range(filtered_spectra)
        
        # Initialize x_min_selected and x_max_selected if not set
        if st.session_state.x_min_selected is None:
            st.session_state.x_min_selected = global_x_min
        if st.session_state.x_max_selected is None:
            st.session_state.x_max_selected = global_x_max
        
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs([
            "📊 Combined Spectra Visualization",
            "🔍 Advanced Peak Analysis", 
            "📈 Parameter Correlation"
        ])
        
        with tab1:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Comprehensive Spectra Analysis")
            st.markdown("*Each visualization mode is displayed independently for separate export*")
            
            if filtered_spectra:
                # Define configurations for all four independent plots
                plot_configs = [
                    {
                        "title": "Raw Spectra",
                        "normalized": False,
                        "use_offset": False,
                        "offset_step": 0,
                        "fill_area": False,
                        "ylabel": st.session_state.y_label,
                        "key": "raw"
                    },
                    {
                        "title": f"Normalized Spectra ({st.session_state.norm_method})",
                        "normalized": True,
                        "use_offset": False,
                        "offset_step": 0,
                        "fill_area": False,
                        "ylabel": f"Normalized Intensity ({st.session_state.norm_method})",
                        "key": "norm"
                    },
                    {
                        "title": f"Raw Spectra + Offset (step = {st.session_state.raw_offset_step})",
                        "normalized": False,
                        "use_offset": True,
                        "offset_step": st.session_state.raw_offset_step,
                        "fill_area": False,
                        "ylabel": st.session_state.y_label,
                        "key": "raw_offset"
                    },
                    {
                        "title": f"Normalized Spectra + Offset (step = {st.session_state.norm_offset_step})",
                        "normalized": True,
                        "use_offset": True,
                        "offset_step": st.session_state.norm_offset_step,
                        "fill_area": st.session_state.fill_area,
                        "ylabel": f"Normalized Intensity ({st.session_state.norm_method})",
                        "key": "norm_offset"
                    }
                ]
                
                # Display each plot in a separate row with its own download button
                for config in plot_configs:
                    st.markdown(f"#### {config['title']}")
                    
                    col_plot, col_btn = st.columns([4, 1])
                    
                    with col_plot:
                        # Create plot for display (standard DPI for quick rendering)
                        fig_display = create_individual_plot(
                            filtered_spectra, st.session_state.x_label, config['ylabel'], config['title'],
                            config['offset_step'], config['fill_area'], config['normalized'],
                            config['use_offset'], st.session_state.norm_method, st.session_state.x_ranges,
                            st.session_state.subtract_min_normalized, 
                            (st.session_state.fill_type == "Gradient fill (fades to transparent)") and config['key'] == "norm_offset",
                            (figure_aspect_ratio[0], figure_aspect_ratio[1] // 2)
                        )
                        st.pyplot(fig_display)
                        plt.close(fig_display)
                    
                    with col_btn:
                        st.markdown("<br>", unsafe_allow_html=True)
                        # Create high DPI version for download
                        fig_high_dpi = create_individual_plot(
                            filtered_spectra, st.session_state.x_label, config['ylabel'], config['title'],
                            config['offset_step'], config['fill_area'], config['normalized'],
                            config['use_offset'], st.session_state.norm_method, st.session_state.x_ranges,
                            st.session_state.subtract_min_normalized,
                            (st.session_state.fill_type == "Gradient fill (fades to transparent)") and config['key'] == "norm_offset",
                            (figure_aspect_ratio[0], figure_aspect_ratio[1] // 2)
                        )
                        
                        # Save to buffer with 600 DPI
                        buf = BytesIO()
                        fig_high_dpi.savefig(buf, format='png', dpi=600, bbox_inches='tight', facecolor='white')
                        buf.seek(0)
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        
                        # Create safe filename
                        safe_title = config['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('+', 'plus')
                        filename = f"spectra_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 1rem;">
                            <a href="data:image/png;base64,{b64}" download="{filename}">
                                <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                               color: white; border: none; border-radius: 8px; 
                                               padding: 0.5rem 1rem; cursor: pointer; width: 100%;">
                                    📥 Download PNG<br><span style="font-size: 0.7rem;">600 DPI</span>
                                </button>
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        plt.close(fig_high_dpi)
                    
                    st.markdown("---")
                
                # Optional: Combined plot download (kept for backward compatibility)
                with st.expander("📥 Download Combined Plot (Legacy - 4-in-1 view)"):
                    fig_combined = create_combined_plot(
                        filtered_spectra, st.session_state.x_label, st.session_state.y_label,
                        "SpectrAnalys - Multi-Mode Spectral Visualization",
                        st.session_state.raw_offset_step, st.session_state.norm_offset_step, st.session_state.fill_area,
                        st.session_state.norm_method, st.session_state.x_ranges, figure_aspect_ratio,
                        st.session_state.subtract_min_normalized,
                        (st.session_state.fill_type == "Gradient fill (fades to transparent)")
                    )
                    buf_combined = BytesIO()
                    fig_combined.savefig(buf_combined, format='png', dpi=600, bbox_inches='tight', facecolor='white')
                    buf_combined.seek(0)
                    b64_combined = base64.b64encode(buf_combined.getvalue()).decode()
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <a href="data:image/png;base64,{b64_combined}" download="spectra_combined_all_modes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                            <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                           color: white; border: none; border-radius: 8px; 
                                           padding: 0.5rem 1rem; cursor: pointer;">
                                📥 Download Combined 4-in-1 Plot (600 DPI)
                            </button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                    plt.close(fig_combined)
                
            else:
                st.warning("No spectra selected for visualization")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Peak Detection and Analysis")
            
            if st.session_state.analyze_peaks_flag and filtered_spectra:
                # Region selection sliders
                st.markdown("### 🎯 Select Analysis Region")
                st.markdown("*Move the sliders to select the x-axis region for peak detection. Click 'Apply & Analyze' to update peak analysis.*")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_min_selected = st.slider(
                        "Left boundary (cm⁻¹)",
                        min_value=float(global_x_min),
                        max_value=float(global_x_max),
                        value=float(st.session_state.x_min_selected),
                        step=(global_x_max - global_x_min) / 100,
                        key="x_min_slider_tab2"
                    )
                with col2:
                    x_max_selected = st.slider(
                        "Right boundary (cm⁻¹)",
                        min_value=float(global_x_min),
                        max_value=float(global_x_max),
                        value=float(st.session_state.x_max_selected),
                        step=(global_x_max - global_x_min) / 100,
                        key="x_max_slider_tab2"
                    )
                
                # Ensure x_min < x_max
                if x_min_selected >= x_max_selected:
                    st.warning("⚠️ Left boundary must be less than right boundary. Adjust the sliders.")
                    x_min_selected, x_max_selected = x_max_selected, x_min_selected
                
                # Store in session state
                st.session_state.x_min_selected = x_min_selected
                st.session_state.x_max_selected = x_max_selected
                
                # Display selected region info
                st.info(f"📊 Selected region: {x_min_selected:.1f} - {x_max_selected:.1f} cm⁻¹ (width: {x_max_selected - x_min_selected:.1f} cm⁻¹)")
                
                # Button to apply changes and analyze peaks
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    analyze_button = st.button("🔍 Apply & Analyze Peaks", use_container_width=True)
                
                # Visualize the selected region on a spectrum (always shown)
                st.markdown("---")
                st.markdown("### 📈 Region Preview with Floating Boundaries")
                
                # Create preview plot with highlighted region
                fig_preview, ax_preview = plt.subplots(figsize=(10, 5))
                
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    x = data['x'].values
                    y = data['y'].values
                    color = spec['color']
                    
                    ax_preview.plot(x, y, color=color, linewidth=1.5, label=name.replace('.txt', ''), alpha=0.7)
                
                # Highlight the selected region
                ax_preview.axvspan(x_min_selected, x_max_selected, alpha=0.2, color='green', label='Selected region')
                # Add vertical lines at boundaries
                ax_preview.axvline(x_min_selected, color='green', linestyle='--', linewidth=2, alpha=0.8)
                ax_preview.axvline(x_max_selected, color='green', linestyle='--', linewidth=2, alpha=0.8)
                
                ax_preview.set_xlabel(st.session_state.x_label, fontsize=11, fontweight='bold')
                ax_preview.set_ylabel(st.session_state.y_label, fontsize=11, fontweight='bold')
                ax_preview.set_title("Spectra with Selected Analysis Region (Floating Boundaries)", fontsize=12, fontweight='bold')
                ax_preview.legend(loc='best', fontsize=9, frameon=True, edgecolor='black')
                ax_preview.tick_params(direction='in', length=5, width=1)
                ax_preview.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                st.pyplot(fig_preview)
                plt.close()
                
                # Perform peak analysis only when button is clicked
                if analyze_button or st.session_state.peak_analysis_performed:
                    st.session_state.peak_analysis_performed = True
                    
                    st.markdown("---")
                    st.markdown("### 🔬 Peak Analysis Results")
                    
                    # Run analysis
                    peaks_df = analyze_peaks_in_region(
                        filtered_spectra, 
                        x_min_selected, 
                        x_max_selected, 
                        st.session_state.peak_width, 
                        st.session_state.peak_prominence_factor / 100.0, 
                        st.session_state.peak_height_factor / 100.0
                    )
                    
                    st.session_state.peaks_df = peaks_df
                    
                    if not peaks_df.empty:
                        # Display peak statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Peaks Detected", len(peaks_df))
                        with col2:
                            st.metric("Unique Spectra", peaks_df['Spectrum'].nunique())
                        with col3:
                            st.metric("Avg Peak Intensity", f"{peaks_df['Intensity (a.u.)'].mean():.2f}")
                        with col4:
                            st.metric("Avg FWHM", f"{peaks_df['FWHM (cm⁻¹)'].mean():.2f} cm⁻¹")
                        
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
                        
                        # Visualize peaks in the selected region
                        st.markdown("---")
                        st.subheader("Peak Visualization in Selected Region")
                        fig_peaks, ax_peaks = plt.subplots(figsize=(12, 6))
                        
                        for name, spec in filtered_spectra.items():
                            data = spec['data']
                            x = data['x'].values
                            y = data['y'].values
                            
                            # Crop to selected region for visualization
                            mask = (x >= x_min_selected) & (x <= x_max_selected)
                            x_cropped = x[mask]
                            y_cropped = y[mask]
                            
                            if len(x_cropped) == 0:
                                continue
                            
                            ax_peaks.plot(x_cropped, y_cropped, color=spec['color'], linewidth=1.5, label=name.replace('.txt', ''), alpha=0.7)
                            
                            # Mark peaks for this spectrum
                            spec_peaks = peaks_df[peaks_df['Spectrum'] == name.replace('.txt', '')]
                            for _, peak in spec_peaks.iterrows():
                                ax_peaks.axvline(peak['Peak position (cm⁻¹)'], color=spec['color'], 
                                               linestyle='--', alpha=0.5, linewidth=1)
                                ax_peaks.text(peak['Peak position (cm⁻¹)'], peak['Intensity (a.u.)'] * 0.8, 
                                           f"{peak['Peak position (cm⁻¹)']:.1f}", 
                                           fontsize=8, ha='center', fontweight='bold')
                        
                        # Add region boundaries
                        ax_peaks.axvline(x_min_selected, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax_peaks.axvline(x_max_selected, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                        
                        ax_peaks.set_xlabel(st.session_state.x_label, fontsize=11, fontweight='bold')
                        ax_peaks.set_ylabel(st.session_state.y_label, fontsize=11, fontweight='bold')
                        ax_peaks.set_title(f"Detected Peaks in Region {x_min_selected:.1f} - {x_max_selected:.1f} cm⁻¹", fontsize=12, fontweight='bold')
                        ax_peaks.legend(loc='best', fontsize=10, frameon=True, edgecolor='black', prop={'weight': 'bold'})
                        ax_peaks.tick_params(direction='in', length=5, width=1)
                        ax_peaks.grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        st.pyplot(fig_peaks)
                        plt.close()
                    else:
                        st.info("ℹ️ No peaks detected in the selected region. Try adjusting the region boundaries or peak detection parameters.")
                else:
                    st.info("👆 Click 'Apply & Analyze Peaks' to detect peaks in the selected region.")
            else:
                st.info("🔍 Enable advanced peak analysis in the sidebar to detect and analyze peaks in your spectra.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Parameter Correlation Analysis")
            st.markdown("*Analyze relationships between experimental parameters and spectral features (Intensity, Area, Position, FWHM)*")
            
            if st.session_state.param_correlation and filtered_spectra and st.session_state.param_values:
                # Use saved peaks_df from session state
                if st.session_state.analyze_peaks_flag and not st.session_state.peaks_df.empty:
                    peaks_df_corr = st.session_state.peaks_df
                else:
                    peaks_df_corr = pd.DataFrame()
                
                if not peaks_df_corr.empty:
                    # Prepare data for correlation - collect all peaks for each spectrum
                    param_list = []
                    intensity_list = []
                    area_list = []
                    position_list = []
                    fwhm_list = []
                    spectrum_names_list = []
                    
                    for name in st.session_state.ordered_spectra:
                        if name in st.session_state.param_values:
                            spec_peaks = peaks_df_corr[peaks_df_corr['Spectrum'] == name.replace('.txt', '')]
                            if not spec_peaks.empty:
                                # Take the most intense peak
                                main_peak = spec_peaks.loc[spec_peaks['Intensity (a.u.)'].idxmax()]
                                param_list.append(st.session_state.param_values[name])
                                intensity_list.append(main_peak['Intensity (a.u.)'])
                                area_list.append(main_peak['Area'])
                                position_list.append(main_peak['Peak position (cm⁻¹)'])
                                fwhm_list.append(main_peak['FWHM (cm⁻¹)'])
                                spectrum_names_list.append(name.replace('.txt', ''))
                    
                    if param_list and len(param_list) > 1:
                        # Calculate correlation coefficients for all four parameters
                        corr_intensity = pearsonr(param_list, intensity_list)[0] if len(param_list) > 2 else 0
                        corr_area = pearsonr(param_list, area_list)[0] if len(param_list) > 2 else 0
                        corr_position = pearsonr(param_list, position_list)[0] if len(param_list) > 2 else 0
                        corr_fwhm = pearsonr(param_list, fwhm_list)[0] if len(param_list) > 2 else 0
                        
                        # Display correlation metrics in 2x2 grid
                        st.markdown("#### 📊 Correlation Coefficients")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Intensity Correlation", f"{corr_intensity:.4f}", 
                                     delta="strong" if abs(corr_intensity) > 0.7 else "weak")
                            st.metric("Area Correlation", f"{corr_area:.4f}",
                                     delta="strong" if abs(corr_area) > 0.7 else "weak")
                        with col2:
                            st.metric("Position (Wavenumber) Correlation", f"{corr_position:.4f}",
                                     delta="strong" if abs(corr_position) > 0.7 else "weak")
                            st.metric("FWHM Correlation", f"{corr_fwhm:.4f}",
                                     delta="strong" if abs(corr_fwhm) > 0.7 else "weak")
                        
                        st.markdown("---")
                        
                        # Create correlation plots for all four parameters (2x2 grid)
                        st.subheader("📈 Correlation Plots")
                        fig_corr, axes_corr = plt.subplots(2, 2, figsize=(14, 12))
                        
                        # Intensity correlation
                        axes_corr[0, 0].scatter(param_list, intensity_list, c='#1f77b4', alpha=0.6, s=100, edgecolors='white', linewidth=2)
                        axes_corr[0, 0].set_xlabel(st.session_state.param_label, fontsize=11, fontweight='bold')
                        axes_corr[0, 0].set_ylabel("Peak Intensity (a.u.)", fontsize=11, fontweight='bold')
                        axes_corr[0, 0].set_title(f"Intensity vs {st.session_state.param_label}\n(r = {corr_intensity:.4f})", fontsize=12, fontweight='bold')
                        axes_corr[0, 0].grid(True, alpha=0.3, linestyle='--')
                        
                        # Area correlation
                        axes_corr[0, 1].scatter(param_list, area_list, c='#2ca02c', alpha=0.6, s=100, edgecolors='white', linewidth=2)
                        axes_corr[0, 1].set_xlabel(st.session_state.param_label, fontsize=11, fontweight='bold')
                        axes_corr[0, 1].set_ylabel("Peak Area", fontsize=11, fontweight='bold')
                        axes_corr[0, 1].set_title(f"Area vs {st.session_state.param_label}\n(r = {corr_area:.4f})", fontsize=12, fontweight='bold')
                        axes_corr[0, 1].grid(True, alpha=0.3, linestyle='--')
                        
                        # Position (Wavenumber) correlation
                        axes_corr[1, 0].scatter(param_list, position_list, c='#d62728', alpha=0.6, s=100, edgecolors='white', linewidth=2)
                        axes_corr[1, 0].set_xlabel(st.session_state.param_label, fontsize=11, fontweight='bold')
                        axes_corr[1, 0].set_ylabel("Peak Position (cm⁻¹)", fontsize=11, fontweight='bold')
                        axes_corr[1, 0].set_title(f"Position vs {st.session_state.param_label}\n(r = {corr_position:.4f})", fontsize=12, fontweight='bold')
                        axes_corr[1, 0].grid(True, alpha=0.3, linestyle='--')
                        
                        # FWHM correlation
                        axes_corr[1, 1].scatter(param_list, fwhm_list, c='#9467bd', alpha=0.6, s=100, edgecolors='white', linewidth=2)
                        axes_corr[1, 1].set_xlabel(st.session_state.param_label, fontsize=11, fontweight='bold')
                        axes_corr[1, 1].set_ylabel("FWHM (cm⁻¹)", fontsize=11, fontweight='bold')
                        axes_corr[1, 1].set_title(f"FWHM vs {st.session_state.param_label}\n(r = {corr_fwhm:.4f})", fontsize=12, fontweight='bold')
                        axes_corr[1, 1].grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        st.pyplot(fig_corr)
                        plt.close()
                        
                        # Show correlation data table
                        st.markdown("---")
                        st.subheader("📋 Correlation Data Table")
                        corr_data = pd.DataFrame({
                            'Spectrum': spectrum_names_list,
                            st.session_state.param_label: param_list,
                            'Intensity (a.u.)': intensity_list,
                            'Area': area_list,
                            'Position (cm⁻¹)': position_list,
                            'FWHM (cm⁻¹)': fwhm_list
                        })
                        st.dataframe(corr_data, use_container_width=True)
                        
                        # Download button for correlation data
                        csv_corr = corr_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download correlation data as CSV",
                            data=csv_corr,
                            file_name=f"correlation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        if len(param_list) <= 1:
                            st.warning("⚠️ Need at least 2 data points for correlation analysis. Please add more spectra or adjust peak detection.")
                        else:
                            st.info("ℹ️ Enable peak analysis and ensure peaks are detected in the selected region to perform correlation analysis.")
                else:
                    st.info("ℹ️ No peak analysis data available. Please go to the 'Advanced Peak Analysis' tab, select a region, and click 'Apply & Analyze Peaks' first.")
            else:
                st.info("📊 Enable parameter correlation in the sidebar and assign numeric values to spectra for correlation analysis.")
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
                    y_norm = normalize_spectrum(data['x'].values, data['y'].values, 
                                                st.session_state.norm_method, 
                                                st.session_state.norm_range, 
                                                st.session_state.subtract_min_normalized)
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
Spectra Files: {', '.join(st.session_state.ordered_spectra)}
Normalization Method: {st.session_state.norm_method}
X-axis Ranges: {st.session_state.x_ranges if st.session_state.x_ranges else 'Full range'}
Raw Offset Step: {st.session_state.raw_offset_step}
Normalized Offset Step: {st.session_state.norm_offset_step}
Fill Area: {st.session_state.fill_area}
Gradient Fill: {st.session_state.fill_type}
Subtract Min Normalized: {st.session_state.subtract_min_normalized}
Figure Aspect Ratio: {st.session_state.aspect_ratio_option}
Peak Analysis: {st.session_state.analyze_peaks_flag}
Peak Analysis Region: {st.session_state.x_min_selected} - {st.session_state.x_max_selected} cm⁻¹
Correlation Analysis: {st.session_state.param_correlation}
Correlation Parameter: {st.session_state.param_label}
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
