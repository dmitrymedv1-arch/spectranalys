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
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as transforms

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

# Function to calculate FWHM
def calculate_fwhm(x, y, peak_idx):
    """Calculate Full Width at Half Maximum for a peak"""
    peak_y = y[peak_idx]
    half_max = peak_y / 2
    
    # Find left crossing
    left_idx = peak_idx
    for i in range(peak_idx, 0, -1):
        if y[i] <= half_max:
            left_idx = i
            break
    
    # Find right crossing
    right_idx = peak_idx
    for i in range(peak_idx, len(y)-1):
        if y[i] <= half_max:
            right_idx = i
            break
    
    if left_idx < right_idx:
        # Linear interpolation for more accurate FWHM
        if left_idx > 0 and y[left_idx] < half_max:
            slope = (y[left_idx+1] - y[left_idx]) / (x[left_idx+1] - x[left_idx])
            if slope != 0:
                left_x = x[left_idx] + (half_max - y[left_idx]) / slope
            else:
                left_x = x[left_idx]
        else:
            left_x = x[left_idx]
        
        if right_idx < len(y)-1 and y[right_idx] < half_max:
            slope = (y[right_idx+1] - y[right_idx]) / (x[right_idx+1] - x[right_idx])
            if slope != 0:
                right_x = x[right_idx] + (half_max - y[right_idx]) / slope
            else:
                right_x = x[right_idx]
        else:
            right_x = x[right_idx]
        
        return right_x - left_x
    return 0

# Function to fit peak with Gaussian or Lorentzian
def fit_peak_gaussian(x, y, peak_idx):
    """Fit Gaussian function to peak for accurate FWHM"""
    try:
        peak_x = x[peak_idx]
        peak_y = y[peak_idx]
        
        # Define Gaussian function
        def gaussian(x, amp, center, sigma, offset):
            return amp * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset
        
        # Select region around peak (± 50 points or ± 100 cm⁻¹)
        region_size = min(50, len(x)//4)
        start_idx = max(0, peak_idx - region_size)
        end_idx = min(len(x)-1, peak_idx + region_size)
        x_region = x[start_idx:end_idx+1]
        y_region = y[start_idx:end_idx+1]
        
        # Initial guess
        sigma_guess = (x_region[-1] - x_region[0]) / 6
        offset_guess = np.min(y_region)
        
        # Fit
        popt, _ = curve_fit(gaussian, x_region, y_region, 
                           p0=[peak_y - offset_guess, peak_x, sigma_guess, offset_guess],
                           maxfev=5000)
        
        # FWHM = 2.35482 * sigma
        fwhm = 2.35482 * abs(popt[2])
        return fwhm, popt
    except:
        return calculate_fwhm(x, y, peak_idx), None

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

# Function to subtract baseline (minimum intensity)
def subtract_baseline(y):
    """Subtract minimum intensity from spectrum"""
    min_y = y.min()
    return y - min_y

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

# Function to create individual plot with specified size
def create_individual_plot(spectra_dict, x_label, y_label, title,
                           offset_step, fill_area, normalized, 
                           norm_method, x_ranges=None, subtract_min=False, 
                           fill_type="solid", apply_offset=True):
    """Create individual scientific plot"""
    
    fig, ax = plt.subplots(figsize=(3, 3))  # Default size, will be updated by caller
    
    handles = []
    labels = []
    
    spectra_items = list(spectra_dict.items())
    
    if x_ranges is None or len(x_ranges) == 0:
        # Simple plot without broken axis
        for idx, (name, spec) in enumerate(spectra_items):
            data = spec['data']
            x = data['x'].values
            y = data['y'].values
            
            if subtract_min and normalized:
                y = subtract_baseline(y)
            
            color = spec['color']
            display_name = name.replace('.txt', '')
            
            # Apply cumulative offset if requested
            if apply_offset:
                offset = idx * offset_step
            else:
                offset = 0
            
            y_plot = y + offset
            
            if fill_area and normalized:
                if fill_type == "solid":
                    ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                elif fill_type == "gradient":
                    # Create gradient fill using polygon
                    verts = np.column_stack([x, y_plot])
                    verts = np.vstack([verts, [x[-1], offset], [x[0], offset]])
                    polygon = Polygon(verts, closed=True, facecolor=color, alpha=0.3, linewidth=0)
                    ax.add_patch(polygon)
                    
                    # Add additional polygons for gradient effect
                    for i in range(5):
                        alpha_value = 0.3 * (1 - i/5) + 0.1
                        y_offset = offset + (y_plot - offset) * (i/5)
                        verts_grad = np.column_stack([x, y_offset])
                        verts_grad = np.vstack([verts_grad, [x[-1], offset], [x[0], offset]])
                        polygon_grad = Polygon(verts_grad, closed=True, facecolor=color, alpha=alpha_value/3, linewidth=0)
                        ax.add_patch(polygon_grad)
            
            line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
            handles.append(line_handle[0])
            labels.append(display_name)
        
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        
    else:
        # Broken axis plot with multiple x-ranges
        n_ranges = len(x_ranges)
        
        for range_idx, (start, end) in enumerate(x_ranges):
            for idx, (name, spec) in enumerate(spectra_items):
                data = spec['data']
                x_full = data['x'].values
                y_full = data['y'].values
                
                if subtract_min and normalized:
                    y_full = subtract_baseline(y_full)
                
                color = spec['color']
                display_name = name.replace('.txt', '')
                
                # Crop to current range
                mask = (x_full >= start) & (x_full <= end)
                if not np.any(mask):
                    continue
                
                x = x_full[mask]
                y = y_full[mask]
                
                # Apply cumulative offset if requested
                if apply_offset:
                    offset = idx * offset_step
                else:
                    offset = 0
                
                y_plot = y + offset
                
                # Plot
                if fill_area and normalized:
                    if fill_type == "solid":
                        ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                    elif fill_type == "gradient":
                        verts = np.column_stack([x, y_plot])
                        verts = np.vstack([verts, [x[-1], offset], [x[0], offset]])
                        polygon = Polygon(verts, closed=True, facecolor=color, alpha=0.3, linewidth=0)
                        ax.add_patch(polygon)
                        
                        for i in range(5):
                            alpha_value = 0.3 * (1 - i/5) + 0.1
                            y_offset = offset + (y_plot - offset) * (i/5)
                            verts_grad = np.column_stack([x, y_offset])
                            verts_grad = np.vstack([verts_grad, [x[-1], offset], [x[0], offset]])
                            polygon_grad = Polygon(verts_grad, closed=True, facecolor=color, alpha=alpha_value/3, linewidth=0)
                            ax.add_patch(polygon_grad)
                
                line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                
                if range_idx == 0:
                    handles.append(line_handle[0])
                    labels.append(display_name)
            
            # Add vertical line for range boundaries
            ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Add legend outside the plot to the right
    if handles:
        if apply_offset:
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
    
    return fig, ax

# Function to create combined plot with all four visualization types (vertical layout)
def create_combined_plot(spectra_dict, x_label, y_label, title,
                         raw_offset_step, norm_offset_step, fill_area,
                         norm_method, x_ranges=None, subtract_min=False, 
                         fill_type="solid", figure_size=(12, 18)):
    """Create scientific plot with all four visualization types in vertical subplots"""
    
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
    
    # Create figure with 4 subplots vertically (4 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=figure_size)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Define the four visualization types
    viz_configs = [
        (axes[0], "Raw Spectra", spectra_dict, 0, False, False, False, x_label, y_label, False, "solid", True),
        (axes[1], f"Normalized Spectra ({norm_method})", normalized_spectra, 0, False, True, False, x_label, f"Normalized Intensity ({norm_method})", False, "solid", True),
        (axes[2], f"Raw Spectra + Offset (step = {raw_offset_step})", spectra_dict, raw_offset_step, False, False, True, x_label, y_label, False, "solid", True),
        (axes[3], f"Normalized Spectra + Offset (step = {norm_offset_step})", normalized_spectra, norm_offset_step, fill_area, True, True, x_label, f"Normalized Intensity ({norm_method})", subtract_min, fill_type, True)
    ]
    
    for ax, subplot_title, spectra, offset_step, fill, normalized, use_offset, xl, yl, sub_min, f_type, apply_off in viz_configs:
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
                
                if sub_min and normalized:
                    y = subtract_baseline(y)
                
                color = spec['color']
                display_name = name.replace('.txt', '')
                
                # Apply cumulative offset if requested
                if use_offset:
                    offset = idx * offset_step
                else:
                    offset = 0
                
                y_plot = y + offset
                
                if fill and normalized:
                    if f_type == "solid":
                        ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                    elif f_type == "gradient":
                        # Create gradient fill
                        verts = np.column_stack([x, y_plot])
                        verts = np.vstack([verts, [x[-1], offset], [x[0], offset]])
                        polygon = Polygon(verts, closed=True, facecolor=color, alpha=0.3, linewidth=0)
                        ax.add_patch(polygon)
                        
                        for i in range(5):
                            alpha_value = 0.3 * (1 - i/5) + 0.1
                            y_offset = offset + (y_plot - offset) * (i/5)
                            verts_grad = np.column_stack([x, y_offset])
                            verts_grad = np.vstack([verts_grad, [x[-1], offset], [x[0], offset]])
                            polygon_grad = Polygon(verts_grad, closed=True, facecolor=color, alpha=alpha_value/3, linewidth=0)
                            ax.add_patch(polygon_grad)
                
                line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
                handles.append(line_handle[0])
                labels.append(display_name)
            
            ax.set_xlabel(xl, fontsize=10, fontweight='bold')
            ax.set_ylabel(yl, fontsize=10, fontweight='bold')
            ax.set_title(subplot_title, fontsize=11, fontweight='bold')
            
        else:
            # Broken axis plot with multiple x-ranges
            for range_idx, (start, end) in enumerate(x_ranges):
                for idx, (name, spec) in enumerate(spectra_items):
                    data = spec['data']
                    x_full = data['x'].values
                    y_full = data['y'].values
                    
                    if sub_min and normalized:
                        y_full = subtract_baseline(y_full)
                    
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
                    if fill and normalized:
                        if f_type == "solid":
                            ax.fill_between(x, offset, y_plot, alpha=0.3, color=color)
                        elif f_type == "gradient":
                            verts = np.column_stack([x, y_plot])
                            verts = np.vstack([verts, [x[-1], offset], [x[0], offset]])
                            polygon = Polygon(verts, closed=True, facecolor=color, alpha=0.3, linewidth=0)
                            ax.add_patch(polygon)
                            
                            for i in range(5):
                                alpha_value = 0.3 * (1 - i/5) + 0.1
                                y_offset = offset + (y_plot - offset) * (i/5)
                                verts_grad = np.column_stack([x, y_offset])
                                verts_grad = np.vstack([verts_grad, [x[-1], offset], [x[0], offset]])
                                polygon_grad = Polygon(verts_grad, closed=True, facecolor=color, alpha=alpha_value/3, linewidth=0)
                                ax.add_patch(polygon_grad)
                    
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name if range_idx == 0 else "")
                    
                    if range_idx == 0:
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
    plt.subplots_adjust(top=0.95, hspace=0.4, right=0.85)
    
    return fig

# Function for peak analysis with region selection
def analyze_peaks_region(spectra_dict, x_range_peak, peak_width=20, fit_method="gaussian"):
    """Analyze peaks in selected region"""
    results = []
    
    for name, spec in spectra_dict.items():
        data = spec['data']
        x = data['x'].values
        y = data['y'].values
        
        # Crop to selected region
        if x_range_peak is not None and len(x_range_peak) == 2:
            mask = (x >= x_range_peak[0]) & (x <= x_range_peak[1])
            x = x[mask]
            y = y[mask]
        
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
            
            # Calculate FWHM
            if fit_method == "gaussian":
                fwhm, fit_params = fit_peak_gaussian(x, y, peak_idx)
            else:
                fwhm = calculate_fwhm(x, y, peak_idx)
            
            results.append({
                'Spectrum': name.replace('.txt', ''),
                'Peak position': peak_x,
                'Intensity': peak_y,
                'Area': area,
                'FWHM': fwhm
            })
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# Function for correlation analysis with temperature/parameter
def correlation_analysis(peaks_df, param_values, param_label):
    """Perform correlation analysis between parameter and peak properties"""
    if peaks_df.empty or not param_values:
        return None, None
    
    results = {}
    
    # Group by spectrum
    for spectrum in peaks_df['Spectrum'].unique():
        if spectrum in param_values:
            spec_peaks = peaks_df[peaks_df['Spectrum'] == spectrum]
            if not spec_peaks.empty:
                # Get main peak (highest intensity)
                main_peak = spec_peaks.loc[spec_peaks['Intensity'].idxmax()]
                
                if spectrum not in results:
                    results[spectrum] = {
                        'param': param_values[spectrum],
                        'intensity': main_peak['Intensity'],
                        'area': main_peak['Area'],
                        'position': main_peak['Peak position'],
                        'fwhm': main_peak['FWHM']
                    }
    
    if not results:
        return None, None
    
    # Convert to DataFrame
    corr_df = pd.DataFrame.from_dict(results, orient='index')
    corr_df.reset_index(drop=True, inplace=True)
    
    # Calculate correlations
    if len(corr_df) > 2:
        corr_intensity = pearsonr(corr_df['param'], corr_df['intensity'])[0]
        corr_area = pearsonr(corr_df['param'], corr_df['area'])[0]
        corr_position = pearsonr(corr_df['param'], corr_df['position'])[0]
        corr_fwhm = pearsonr(corr_df['param'], corr_df['fwhm'])[0]
    else:
        corr_intensity = corr_area = corr_position = corr_fwhm = 0
    
    return corr_df, {
        'intensity': corr_intensity,
        'area': corr_area,
        'position': corr_position,
        'fwhm': corr_fwhm
    }

# Main app
def main():
    # Initialize session state for independent component management
    if 'spectra_loaded' not in st.session_state:
        st.session_state.spectra_loaded = False
        st.session_state.spectra_data = {}
        st.session_state.ordered_spectra = []
        st.session_state.peak_analysis_done = False
        st.session_state.peaks_df = pd.DataFrame()
        st.session_state.analysis_region = [None, None]
        st.session_state.individual_figs = {}
    
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
                st.session_state.spectra_data = spectra_data
                st.session_state.spectra_loaded = True
                
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
                    
                    st.session_state.ordered_spectra = ordered_spectra
                    
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
                        st.session_state.spectra_data[name]['color'] = colors[name]
                    
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
                    
                    # NEW: Subtract baseline checkbox for normalized spectra
                    subtract_minimum = st.checkbox("Subtract minimum intensity (baseline correction)", value=False)
                    
                    # NEW: Fill type selection
                    fill_type = "solid"
                    if fill_area:
                        fill_type = st.radio(
                            "Fill type for normalized spectra",
                            ["Solid (semi-transparent)", "Gradient (fading to base)"],
                            index=0
                        )
                        fill_type = "solid" if fill_type == "Solid (semi-transparent)" else "gradient"
                    
                    # NEW: Figure size selector for combined plot
                    st.markdown("#### 📐 Plot Size (Combined View)")
                    fig_size_options = {
                        "3×3": (3, 3),
                        "3×4": (3, 4),
                        "3×5": (3, 5),
                        "3×6": (3, 6),
                        "3×7": (3, 7),
                        "3×9": (3, 9)
                    }
                    selected_size = st.selectbox(
                        "Select figure size (width × height in inches)",
                        options=list(fig_size_options.keys()),
                        index=2  # Default 3×5
                    )
                    figure_size = fig_size_options[selected_size]
                    # Scale for combined plot (4 subplots vertically)
                    combined_figure_size = (figure_size[0] * 1.5, figure_size[1] * 4)
                    
                    # Peak analysis options
                    st.markdown("---")
                    st.markdown("### 🔍 Peak Analysis")
                    
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
                        
                        param_label = st.text_input("Parameter label", value="Temperature (°C)")
                    
                    # Store all parameters in session state for independent updates
                    st.session_state.processing_params = {
                        'common_x_range': common_x_range,
                        'x_ranges': x_ranges,
                        'x_label': x_label,
                        'y_label': y_label,
                        'norm_method': norm_method,
                        'norm_range': norm_range,
                        'raw_offset_step': raw_offset_step,
                        'norm_offset_step': norm_offset_step,
                        'fill_area': fill_area,
                        'subtract_minimum': subtract_minimum,
                        'fill_type': fill_type,
                        'figure_size': combined_figure_size,
                        'param_correlation': param_correlation,
                        'param_values': param_values if param_correlation else None,
                        'param_label': param_label if param_correlation else "Parameter"
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
    if st.session_state.spectra_loaded and st.session_state.spectra_data and st.session_state.ordered_spectra:
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
            x_ranges = st.session_state.processing_params.get('x_ranges', None)
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
            norm_method = st.session_state.processing_params.get('norm_method', "Maximum intensity")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{norm_method[:10]}</div>
                <div class="metric-label">Normalization</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            peak_analysis_enabled = st.session_state.peak_analysis_done
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'✓' if peak_analysis_enabled else '✗'}</div>
                <div class="metric-label">Peak Analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Apply common x range if selected
        current_spectra = st.session_state.spectra_data
        if st.session_state.processing_params.get('common_x_range', False):
            current_spectra = align_x_ranges(current_spectra)
        
        # Filter spectra based on selection
        filtered_spectra = {name: current_spectra[name] for name in st.session_state.ordered_spectra if name in current_spectra}
        
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
                # Get parameters from session state
                params = st.session_state.processing_params
                
                fig = create_combined_plot(
                    filtered_spectra, 
                    params['x_label'], 
                    params['y_label'],
                    "SpectrAnalys - Multi-Mode Spectral Visualization",
                    params['raw_offset_step'], 
                    params['norm_offset_step'], 
                    params['fill_area'],
                    params['norm_method'], 
                    params['x_ranges'],
                    params['subtract_minimum'],
                    params['fill_type'],
                    params['figure_size']
                )
                st.pyplot(fig)
                
                # Download button for combined plot
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode()
                st.markdown(f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <a href="data:image/png;base64,{b64}" download="spectra_combined_plot.png">
                        <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       color: white; border: none; border-radius: 8px; 
                                       padding: 0.5rem 1rem; cursor: pointer;">
                            📥 Download Combined Plot (PNG, 600 DPI)
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
                plt.close()
                
                # NEW: Individual plots with separate downloads
                st.markdown("---")
                st.subheader("Individual Plots (Separate Downloads)")
                st.markdown("*Each visualization can be downloaded independently*")
                
                # Prepare data for individual plots
                normalized_spectra_ind = {}
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    y_norm = normalize_spectrum(
                        data['x'].values,
                        data['y'].values,
                        params['norm_method'],
                        None
                    )
                    normalized_spectra_ind[name] = {
                        'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
                        'color': spec['color']
                    }
                
                # Define individual plot configurations
                individual_configs = [
                    ("Raw Spectra", filtered_spectra, 0, False, False, params['x_label'], params['y_label'], False, "solid", False),
                    (f"Normalized Spectra ({params['norm_method']})", normalized_spectra_ind, 0, False, True, params['x_label'], f"Normalized Intensity ({params['norm_method']})", False, "solid", False),
                    (f"Raw Spectra + Offset (step = {params['raw_offset_step']})", filtered_spectra, params['raw_offset_step'], False, False, params['x_label'], params['y_label'], False, "solid", True),
                    (f"Normalized Spectra + Offset (step = {params['norm_offset_step']})", normalized_spectra_ind, params['norm_offset_step'], params['fill_area'], True, params['x_label'], f"Normalized Intensity ({params['norm_method']})", params['subtract_minimum'], params['fill_type'], True)
                ]
                
                # Create grid for individual plots (4 rows, 1 column)
                for plot_idx, (title, spectra, offset_step, fill, normalized, xl, yl, sub_min, f_type, use_offset) in enumerate(individual_configs):
                    st.markdown(f"**{title}**")
                    
                    # Create figure with selected size
                    fig_ind, ax_ind = create_individual_plot(
                        spectra, xl, yl, title, offset_step, fill, 
                        normalized, params['norm_method'], params['x_ranges'],
                        sub_min, f_type, use_offset  # 12 аргументов, apply_offset=use_offset
                    )
                    # Resize figure to selected dimensions
                    fig_ind.set_size_inches(params['figure_size'][0]/1.5, params['figure_size'][1])
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.pyplot(fig_ind)
                    
                    # Download button for individual plot
                    buf_ind = BytesIO()
                    fig_ind.savefig(buf_ind, format='png', dpi=600, bbox_inches='tight')
                    buf_ind.seek(0)
                    b64_ind = base64.b64encode(buf_ind.getvalue()).decode()
                    
                    with col2:
                        st.markdown(f"""
                        <a href="data:image/png;base64,{b64_ind}" download="spectra_plot_{plot_idx}.png">
                            <button style="background: #3498db; color: white; border: none; border-radius: 5px; 
                                           padding: 0.3rem 0.8rem; cursor: pointer; font-size: 0.8rem;">
                                📥 PNG
                            </button>
                        </a>
                        """, unsafe_allow_html=True)
                    
                    plt.close(fig_ind)
                
            else:
                st.warning("No spectra selected for visualization")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Peak Detection and Analysis")
            st.markdown("*Select region of interest, then click 'Run Peak Analysis'*")
            
            if filtered_spectra:
                # Get full x-range for slider
                all_x = []
                for spec in filtered_spectra.values():
                    all_x.extend(spec['data']['x'].values)
                if all_x:
                    global_min_x = min(all_x)
                    global_max_x = max(all_x)
                    
                    # NEW: Region selection sliders
                    st.markdown("#### 📍 Select Analysis Region")
                    col1, col2 = st.columns(2)
                    with col1:
                        left_bound = st.slider(
                            "Left boundary (cm⁻¹)",
                            min_value=float(global_min_x),
                            max_value=float(global_max_x),
                            value=float(global_min_x + (global_max_x - global_min_x) * 0.3),
                            step=1.0,
                            key="peak_left_bound"
                        )
                    with col2:
                        right_bound = st.slider(
                            "Right boundary (cm⁻¹)",
                            min_value=float(global_min_x),
                            max_value=float(global_max_x),
                            value=float(global_min_x + (global_max_x - global_min_x) * 0.7),
                            step=1.0,
                            key="peak_right_bound"
                        )
                    
                    # Ensure left < right
                    if left_bound >= right_bound:
                        st.warning("⚠️ Left boundary must be less than right boundary")
                        analysis_region = None
                    else:
                        analysis_region = (left_bound, right_bound)
                        st.session_state.analysis_region = analysis_region
                    
                    # Display region on spectra
                    st.markdown("#### 📊 Spectra with Selected Region")
                    fig_region, ax_region = plt.subplots(figsize=(12, 6))
                    
                    for name, spec in filtered_spectra.items():
                        data = spec['data']
                        x = data['x'].values
                        y = data['y'].values
                        ax_region.plot(x, y, color=spec['color'], linewidth=1.5, 
                                     label=name.replace('.txt', ''), alpha=0.7)
                    
                    # Highlight selected region
                    if analysis_region:
                        ax_region.axvspan(analysis_region[0], analysis_region[1], 
                                         alpha=0.2, color='yellow', label='Analysis Region')
                        ax_region.axvline(analysis_region[0], color='red', linestyle='--', alpha=0.5)
                        ax_region.axvline(analysis_region[1], color='red', linestyle='--', alpha=0.5)
                    
                    ax_region.set_xlabel(st.session_state.processing_params['x_label'], fontsize=11, fontweight='bold')
                    ax_region.set_ylabel(st.session_state.processing_params['y_label'], fontsize=11, fontweight='bold')
                    ax_region.set_title("Spectra with Selected Analysis Region", fontsize=12, fontweight='bold')
                    ax_region.legend(loc='best', fontsize=10)
                    ax_region.grid(True, alpha=0.3, linestyle='--')
                    
                    st.pyplot(fig_region)
                    plt.close()
                    
                    # Peak analysis parameters
                    st.markdown("#### ⚙️ Peak Analysis Parameters")
                    col1, col2 = st.columns(2)
                    with col1:
                        peak_width = st.slider(
                            "Peak width for area calculation (points)",
                            min_value=5,
                            max_value=100,
                            value=20,
                            step=5,
                            key="peak_width_analysis"
                        )
                    with col2:
                        fit_method = st.selectbox(
                            "FWHM calculation method",
                            ["gaussian", "direct"],
                            index=0,
                            help="Gaussian fitting (more accurate) or direct calculation (faster)"
                        )
                    
                    # Run analysis button
                    if st.button("🚀 Run Peak Analysis", key="run_peak_analysis"):
                        with st.spinner("Analyzing peaks..."):
                            peaks_df = analyze_peaks_region(
                                filtered_spectra, 
                                analysis_region, 
                                peak_width, 
                                fit_method
                            )
                            st.session_state.peaks_df = peaks_df
                            st.session_state.peak_analysis_done = True
                    
                    # Display results if analysis has been run
                    if st.session_state.peak_analysis_done and not st.session_state.peaks_df.empty:
                        peaks_df = st.session_state.peaks_df
                        
                        # Display peak statistics
                        st.markdown("---")
                        st.markdown("#### 📊 Peak Analysis Results")
                        
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
                            
                            if analysis_region:
                                mask = (x >= analysis_region[0]) & (x <= analysis_region[1])
                                x = x[mask]
                                y = y[mask]
                            
                            if len(x) == 0:
                                continue
                            
                            ax.plot(x, y, color=spec['color'], linewidth=1.5, 
                                   label=name.replace('.txt', ''), alpha=0.7)
                            
                            # Mark peaks
                            spec_peaks = peaks_df[peaks_df['Spectrum'] == name.replace('.txt', '')]
                            for _, peak in spec_peaks.iterrows():
                                ax.axvline(peak['Peak position'], color=spec['color'], 
                                          linestyle='--', alpha=0.5, linewidth=1)
                                ax.text(peak['Peak position'], peak['Intensity']*0.8, 
                                       f"{peak['Peak position']:.1f}\nFWHM:{peak['FWHM']:.1f}", 
                                       fontsize=7, ha='center', fontweight='bold')
                        
                        ax.set_xlabel(st.session_state.processing_params['x_label'], fontsize=11, fontweight='bold')
                        ax.set_ylabel(st.session_state.processing_params['y_label'], fontsize=11, fontweight='bold')
                        ax.set_title("Detected Peaks with FWHM", fontsize=12, fontweight='bold')
                        ax.legend(loc='best', fontsize=10, frameon=True, edgecolor='black', prop={'weight': 'bold'})
                        ax.tick_params(direction='in', length=5, width=1)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    elif st.session_state.peak_analysis_done:
                        st.warning("No peaks detected in the selected region. Try adjusting the region boundaries.")
                    else:
                        st.info("👈 Select analysis region and click 'Run Peak Analysis' to detect peaks")
                
            else:
                st.warning("No spectra available for peak analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Parameter Correlation Analysis")
            st.markdown("*Correlate spectral features (Intensity, Area, Position, FWHM) with experimental parameters*")
            
            if st.session_state.processing_params.get('param_correlation', False) and st.session_state.peak_analysis_done:
                param_values = st.session_state.processing_params.get('param_values', {})
                param_label = st.session_state.processing_params.get('param_label', "Parameter")
                
                if param_values and not st.session_state.peaks_df.empty:
                    # Convert param_values keys to match spectrum names
                    param_dict = {}
                    for spec_name in st.session_state.ordered_spectra:
                        if spec_name in param_values:
                            param_dict[spec_name.replace('.txt', '')] = param_values[spec_name]
                    
                    # Perform correlation analysis
                    corr_df, correlations = correlation_analysis(
                        st.session_state.peaks_df, 
                        param_dict, 
                        param_label
                    )
                    
                    if corr_df is not None and not corr_df.empty:
                        # Display correlation metrics
                        st.markdown("#### 📈 Correlation Coefficients")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Intensity vs Parameter", f"{correlations['intensity']:.3f}", 
                                     delta="strong" if abs(correlations['intensity']) > 0.7 else "weak")
                        with col2:
                            st.metric("Area vs Parameter", f"{correlations['area']:.3f}",
                                     delta="strong" if abs(correlations['area']) > 0.7 else "weak")
                        with col3:
                            st.metric("Position vs Parameter", f"{correlations['position']:.3f}",
                                     delta="strong" if abs(correlations['position']) > 0.7 else "weak")
                        with col4:
                            st.metric("FWHM vs Parameter", f"{correlations['fwhm']:.3f}",
                                     delta="strong" if abs(correlations['fwhm']) > 0.7 else "weak")
                        
                        st.markdown("---")
                        
                        # Create correlation plots (4 plots now including FWHM)
                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                        
                        # Intensity plot
                        axes[0, 0].scatter(corr_df['param'], corr_df['intensity'], 
                                         c='#1f77b4', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[0, 0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[0, 0].set_ylabel("Peak Intensity (a.u.)", fontsize=11, fontweight='bold')
                        axes[0, 0].set_title(f"Intensity vs {param_label}\n(r = {correlations['intensity']:.3f})", 
                                           fontsize=12, fontweight='bold')
                        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
                        
                        # Area plot
                        axes[0, 1].scatter(corr_df['param'], corr_df['area'], 
                                         c='#2ca02c', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[0, 1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[0, 1].set_ylabel("Peak Area", fontsize=11, fontweight='bold')
                        axes[0, 1].set_title(f"Area vs {param_label}\n(r = {correlations['area']:.3f})", 
                                           fontsize=12, fontweight='bold')
                        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
                        
                        # Position plot
                        axes[1, 0].scatter(corr_df['param'], corr_df['position'], 
                                         c='#d62728', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[1, 0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[1, 0].set_ylabel("Peak Position (cm⁻¹)", fontsize=11, fontweight='bold')
                        axes[1, 0].set_title(f"Position vs {param_label}\n(r = {correlations['position']:.3f})", 
                                           fontsize=12, fontweight='bold')
                        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
                        
                        # FWHM plot (NEW)
                        axes[1, 1].scatter(corr_df['param'], corr_df['fwhm'], 
                                         c='#9467bd', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                        axes[1, 1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                        axes[1, 1].set_ylabel("FWHM (cm⁻¹)", fontsize=11, fontweight='bold')
                        axes[1, 1].set_title(f"FWHM vs {param_label}\n(r = {correlations['fwhm']:.3f})", 
                                           fontsize=12, fontweight='bold')
                        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Show correlation data table
                        st.markdown("---")
                        st.subheader("Correlation Data Table")
                        st.dataframe(corr_df, use_container_width=True)
                        
                        # Download button
                        csv = corr_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download correlation data as CSV",
                            data=csv,
                            file_name=f"correlation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Insufficient data for correlation analysis. Need at least 3 data points.")
                else:
                    if st.session_state.peaks_df.empty:
                        st.info("🔍 Please run peak analysis in the 'Advanced Peak Analysis' tab first.")
                    else:
                        st.info("📊 Please assign parameter values in the sidebar for correlation analysis.")
            else:
                if not st.session_state.processing_params.get('param_correlation', False):
                    st.info("📊 Enable parameter correlation in the sidebar to analyze relationships between spectral features and experimental parameters.")
                elif not st.session_state.peak_analysis_done:
                    st.info("🔍 Please run peak analysis in the 'Advanced Peak Analysis' tab first.")
            
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
                                               st.session_state.processing_params['norm_method'], 
                                               st.session_state.processing_params.get('norm_range', None))
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
Normalization Method: {st.session_state.processing_params.get('norm_method', 'N/A')}
X-axis Ranges: {st.session_state.processing_params.get('x_ranges', 'Full range')}
Raw Offset Step: {st.session_state.processing_params.get('raw_offset_step', 'N/A')}
Normalized Offset Step: {st.session_state.processing_params.get('norm_offset_step', 'N/A')}
Fill Area: {st.session_state.processing_params.get('fill_area', False)}
Subtract Minimum: {st.session_state.processing_params.get('subtract_minimum', False)}
Fill Type: {st.session_state.processing_params.get('fill_type', 'solid')}
Peak Analysis Completed: {st.session_state.peak_analysis_done}
Correlation Analysis: {st.session_state.processing_params.get('param_correlation', False)}
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
