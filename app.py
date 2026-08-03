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

# ============================================================================
# PART 1: SESSION STATE INITIALIZATION (STRUCTURED BY MODULES)
# ============================================================================

# Data Manager
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = {
        'raw_spectra': {},
        'file_names': [],
        'loaded': False,
        'last_update': None
    }

# Visualizer Module
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = {
        'params': {
            'x_label': 'Raman shift (cm⁻¹)',
            'y_label': 'Intensity (a.u.)',
            'norm_method': 'Maximum intensity',
            'norm_range': None,
            'raw_offset_step': 1000.0,
            'norm_offset_step': 0.5,
            'fill_area': False,
            'fill_alpha': 0.3,
            'subtract_min_intensity': False,
            'show_grid': True,
            'line_width': 1.5,
            'fig_width': 5,
            'fig_height': 3,
            'x_ranges': None,
            'common_x_range': False,
            'legend_fontsize': 8,
            'legend_position': 'right',
            'legend_offset': 1.02,
            'selected_spectra': [],
            'colors': {}
        },
        'computed_data': {
            'filtered_spectra': None,
            'normalized_spectra': None,
            'plots': []
        },
        'ready': False,
        'last_update': None
    }

# Peak Analyzer Module
if 'peak_analyzer' not in st.session_state:
    st.session_state.peak_analyzer = {
        'params': {
            'left_boundary': None,
            'right_boundary': None,
            'peak_width': 20
        },
        'results': {
            'peaks_df': None,
            'x_range': (None, None)
        },
        'ready': False,
        'last_update': None,
        'excluded_peaks': set()
    }

# Correlation Module
if 'correlation' not in st.session_state:
    st.session_state.correlation = {
        'params': {
            'param_values': {},
            'param_label': 'Sample number',
            'enabled': False
        },
        'results': None,
        'ready': False,
        'last_update': None
    }

# Comparator Module
if 'comparator' not in st.session_state:
    st.session_state.comparator = {
        'params': {
            'ref_spectrum': None,
            'sample_spectrum': None,
            'smooth_difference': False,
            'smooth_sigma': 1.5,
            'symmetric_scale': True,
            'difference_threshold': 0.1,
            'selected_colormap': 'RdBu_r',
            'swap_direction': True
        },
        'results': None,
        'ready': False,
        'last_update': None
    }

# Heatmap Module
if 'heatmap' not in st.session_state:
    st.session_state.heatmap = {
        'params': {
            'param_type': 'Temperature (°C)',
            'custom_label': 'Parameter',
            'param_values': {},
            'interpolation': 'gaussian',
            'colormap': 'viridis',
            'ordered_names': []
        },
        'data': {
            'spectra_matrix': None,
            'spectra_norm_matrix': None,
            'x_grid': None,
            'y_values': None,
            'x_ranges': None
        },
        'ready': False,
        'last_update': None
    }

# ============================================================================
# PART 2: CORE FUNCTIONS (UNCHANGED - PRESERVED)
# ============================================================================

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

@st.cache_data
def load_all_spectra(uploaded_files):
    """Load all spectra with caching"""
    spectra_data = {}
    for file in uploaded_files:
        data = load_spectrum(file)
        if data is not None:
            spectra_data[file.name] = {
                'data': data,
                'color': None
            }
    return spectra_data

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
    
    # NEW METHOD: Maximum rest intensity (normalize only within visible ranges when custom ranges are active)
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
    
    # Find common x range - skip if any spectrum has no data
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
    
    # Create vertices for polygon
    verts = []
    for i in range(len(x)):
        verts.append((x[i], y[i] + offset))
    for i in range(len(x)-1, -1, -1):
        verts.append((x[i], offset))
    
    # Create polygon
    poly = plt.Polygon(verts, closed=True, facecolor=color, alpha=0.3, edgecolor='none')
    ax.add_patch(poly)
    
    # Create gradient by overlaying semi-transparent polygons
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
    ax_bottom.set_ylabel('Intensity Difference (a.u.)', fontsize=10, fontweight='bold')
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
    
    data_clean = data_matrix_transposed[np.isfinite(data_matrix_transposed)]
    if len(data_clean) > 0:
        vmin = np.percentile(data_clean, 1)  # Use 1st percentile to avoid outliers
        vmax = np.percentile(data_clean, 99)  # Use 99th percentile to avoid outliers
    else:
        vmin = np.nanmin(data_matrix_transposed)
        vmax = np.nanmax(data_matrix_transposed)
    
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
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=11, fontweight='bold')
    
    # Set labels
    ax.set_xlabel(y_label, fontsize=11, fontweight='bold')
    ax.set_ylabel(x_label, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
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

# ============================================================================
# PART 3: MAIN APPLICATION
# ============================================================================

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
    
    # ========================================================================
    # SIDEBAR - DATA MANAGER (Only file loading and basic management)
    # ========================================================================
    with st.sidebar:
        st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Data Import")
        
        uploaded_files = st.file_uploader(
            "Upload spectra files (.txt format, tab-separated)",
            type=['txt'],
            accept_multiple_files=True,
            key="file_uploader_main"
        )
        
        # Remove all spectra button
        if uploaded_files and st.session_state.data_manager['loaded']:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🗑️ Remove all spectra", type="secondary", use_container_width=True):
                    # Clear all spectrum-related session state but preserve UI settings
                    st.session_state.data_manager = {
                        'raw_spectra': {},
                        'file_names': [],
                        'loaded': False,
                        'last_update': None
                    }
                    st.session_state.visualizer['ready'] = False
                    st.session_state.peak_analyzer['ready'] = False
                    st.session_state.correlation['ready'] = False
                    st.session_state.comparator['ready'] = False
                    st.session_state.heatmap['ready'] = False
                    st.rerun()
            with col2:
                st.markdown("")
        
        # Load data if files are uploaded
        if uploaded_files:
            if not st.session_state.data_manager['loaded'] or \
               st.session_state.data_manager['file_names'] != [f.name for f in uploaded_files]:
                with st.spinner("Loading spectra..."):
                    spectra_data = load_all_spectra(uploaded_files)
                    if spectra_data:
                        st.session_state.data_manager['raw_spectra'] = spectra_data
                        st.session_state.data_manager['file_names'] = [f.name for f in uploaded_files]
                        st.session_state.data_manager['loaded'] = True
                        st.session_state.data_manager['last_update'] = datetime.now()
                        st.success(f"✅ Loaded {len(spectra_data)} files")
        
        # Display loaded files info
        if st.session_state.data_manager['loaded']:
            file_list = st.session_state.data_manager['file_names']
            st.info(f"📄 {len(file_list)} spectra loaded: {', '.join(file_list[:3])}{'...' if len(file_list) > 3 else ''}")
            if st.session_state.data_manager['last_update']:
                st.caption(f"Last update: {st.session_state.data_manager['last_update'].strftime('%H:%M:%S')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar footer with info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p>🔬 SpectrAnalys v3.0<br>Scientific Spectroscopic Analysis</p>
            <p style="font-size: 0.7rem;">Independent Modules Architecture</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN CONTENT - Only if data is loaded
    # ========================================================================
    if st.session_state.data_manager['loaded']:
        # Get raw spectra data
        spectra_data = st.session_state.data_manager['raw_spectra']
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(spectra_data)}</div>
                <div class="metric-label">Spectra Loaded</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            viz_ready = st.session_state.visualizer['ready']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'✓' if viz_ready else '⏳'}</div>
                <div class="metric-label">Visualizer {'Ready' if viz_ready else 'Not initialized'}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            peak_ready = st.session_state.peak_analyzer['ready']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'✓' if peak_ready else '⏳'}</div>
                <div class="metric-label">Peak Analyzer {'Ready' if peak_ready else 'Not initialized'}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            heatmap_ready = st.session_state.heatmap['ready']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'✓' if heatmap_ready else '⏳'}</div>
                <div class="metric-label">Heatmap {'Ready' if heatmap_ready else 'Not initialized'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Combined Spectra Visualization",
            "🔍 Advanced Peak Analysis", 
            "📈 Parameter Correlation",
            "🔀 Compare Two Spectra"
        ])
        
        # ========================================================================
        # TAB 1: SPECTRA VISUALIZER (Independent Module)
        # ========================================================================
        with tab1:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Comprehensive Spectra Analysis")
            st.markdown("*All visualization modes combined for comprehensive spectral comparison*")
            
            # Get spectra list
            spectra_names = list(spectra_data.keys())
            
            # Create form for visualization settings
            with st.form("visualizer_settings"):
                st.markdown("#### 📋 Spectrum Selection")
                
                # Select and order spectra
                selected_spectra = st.multiselect(
                    "Choose spectra to display",
                    options=spectra_names,
                    default=spectra_names if not st.session_state.visualizer['params']['selected_spectra'] 
                            else st.session_state.visualizer['params']['selected_spectra'],
                    key="viz_select_spectra"
                )
                
                if selected_spectra:
                    st.markdown("---")
                    st.markdown("#### ⚙️ Processing Options")
                    
                    # Common x range option
                    common_x_range = st.checkbox(
                        "Align all spectra to common x range",
                        value=st.session_state.visualizer['params']['common_x_range'],
                        key="viz_common_x"
                    )
                    
                    # X-axis ranges
                    st.markdown("#### 📊 X-axis Ranges")
                    x_range_option = st.radio(
                        "Select range mode",
                        ["Full range", "Custom ranges (multiple)"],
                        key="viz_range_mode"
                    )
                    
                    x_ranges = None
                    if x_range_option == "Custom ranges (multiple)":
                        range_input = st.text_area(
                            "Enter ranges (e.g., 100-150, 350-450, 600-800)",
                            placeholder="100-150, 350-450, 600-800",
                            help="Each range will be displayed as a separate segment on the same graph",
                            key="viz_ranges_input"
                        )
                        if range_input:
                            x_ranges = parse_x_ranges(range_input)
                            if x_ranges:
                                st.info(f"📌 Selected {len(x_ranges)} ranges: {', '.join([f'{r[0]:.0f}-{r[1]:.0f}' for r in x_ranges])}")
                    
                    # Axis labels
                    st.markdown("#### 🏷️ Axis Labels")
                    x_label = st.text_input(
                        "X-axis label",
                        value=st.session_state.visualizer['params']['x_label'],
                        key="viz_x_label"
                    )
                    y_label = st.text_input(
                        "Y-axis label",
                        value=st.session_state.visualizer['params']['y_label'],
                        key="viz_y_label"
                    )
                    
                    # Normalization options
                    st.markdown("#### 📐 Normalization")
                    
                    # Build normalization options based on x_range_option
                    if x_range_option == "Custom ranges (multiple)":
                        norm_options = ["Maximum intensity", "Peak intensity (range)", "Maximum rest intensity"]
                    else:
                        norm_options = ["Maximum intensity", "Peak intensity (range)"]
                    
                    norm_method = st.selectbox(
                        "Normalization method",
                        norm_options,
                        index=norm_options.index(st.session_state.visualizer['params']['norm_method']) 
                               if st.session_state.visualizer['params']['norm_method'] in norm_options else 0,
                        key="viz_norm_method"
                    )
                    
                    norm_range = None
                    if norm_method == "Peak intensity (range)":
                        norm_range_input = st.text_input(
                            "Peak range for normalization (e.g., 800-1000)",
                            placeholder="800-1000",
                            key="viz_norm_range"
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
                            value=st.session_state.visualizer['params']['raw_offset_step'],
                            step=100.0,
                            key="viz_raw_offset"
                        )
                    with col2:
                        norm_offset_step = st.slider(
                            "Normalized spectra offset step",
                            min_value=0.0,
                            max_value=5.0,
                            value=st.session_state.visualizer['params']['norm_offset_step'],
                            step=0.05,
                            key="viz_norm_offset"
                        )
                    
                    # Fill area option
                    fill_area = st.checkbox(
                        "Fill area under normalized spectra",
                        value=st.session_state.visualizer['params']['fill_area'],
                        key="viz_fill_area"
                    )
                    
                    fill_alpha = 0.3
                    if fill_area:
                        fill_alpha = st.slider(
                            "Fill transparency",
                            min_value=0.2,
                            max_value=0.9,
                            value=st.session_state.visualizer['params']['fill_alpha'],
                            step=0.1,
                            help="0.2 = more transparent, 0.9 = more opaque",
                            key="viz_fill_alpha"
                        )
                    
                    # Subtract minimum intensity checkbox
                    subtract_min_intensity = st.checkbox(
                        "Subtract minimum intensity (start from zero)",
                        value=st.session_state.visualizer['params']['subtract_min_intensity'],
                        key="viz_subtract_min"
                    )
                    
                    # Plot settings
                    st.markdown("#### 🎨 Plot Settings")
                    col1, col2 = st.columns(2)
                    with col1:
                        show_grid = st.checkbox(
                            "Show grid on plots",
                            value=st.session_state.visualizer['params']['show_grid'],
                            key="viz_grid"
                        )
                    with col2:
                        line_width = st.slider(
                            "Spectrum line thickness",
                            min_value=0.5,
                            max_value=3.0,
                            value=st.session_state.visualizer['params']['line_width'],
                            step=0.1,
                            key="viz_linewidth"
                        )
                    
                    # Figure size selector
                    st.markdown("#### 📐 Plot Size (Width × Height)")
                    fig_size_options = {
                        "3×3": (3, 3),
                        "3×4": (4, 3),
                        "3×5": (5, 3),
                        "3×6": (6, 3),
                        "3×7": (7, 3),
                        "3×9": (9, 3)
                    }
                    current_size = f"{st.session_state.visualizer['params']['fig_width']}×{st.session_state.visualizer['params']['fig_height']}"
                    default_index = list(fig_size_options.keys()).index(current_size) if current_size in fig_size_options else 2
                    selected_size = st.selectbox(
                        "Select plot dimensions (width × height in inches)",
                        options=list(fig_size_options.keys()),
                        index=default_index,
                        key="viz_size"
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
                            value=st.session_state.visualizer['params']['legend_fontsize'],
                            step=1,
                            key="viz_legend_fontsize"
                        )
                    with col2:
                        legend_position = st.selectbox(
                            "Legend position",
                            options=["right", "best", "upper right", "upper left", "lower left", "lower right"],
                            index=["right", "best", "upper right", "upper left", "lower left", "lower right"]
                                 .index(st.session_state.visualizer['params']['legend_position']),
                            key="viz_legend_position"
                        )
                    
                    legend_offset = st.slider(
                        "Legend offset from plot (0.5-2.0)",
                        min_value=0.5,
                        max_value=2.0,
                        value=st.session_state.visualizer['params']['legend_offset'],
                        step=0.02,
                        help="Higher value moves legend further right (for right position)",
                        key="viz_legend_offset"
                    )
                    
                    # Color Assignment
                    st.markdown("---")
                    st.markdown("### 🎨 Color Assignment")
                    
                    # Define default color palette
                    default_colors = [
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
                    ]
                    
                    colors = {}
                    for i, name in enumerate(selected_spectra):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{name.replace('.txt', '')}**")
                        with col2:
                            default_color = default_colors[i % len(default_colors)]
                            current_color = st.session_state.visualizer['params']['colors'].get(name, default_color)
                            colors[name] = st.color_picker(
                                f"Color {i+1}",
                                value=current_color,
                                key=f"viz_color_{name}"
                            )
                    
                    # SUBMIT BUTTON
                    st.markdown("---")
                    apply_viz = st.form_submit_button(
                        "🔄 Apply Visualization",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    if apply_viz and selected_spectra:
                        with st.spinner("Generating visualizations..."):
                            # Update parameters in session state
                            st.session_state.visualizer['params']['x_label'] = x_label
                            st.session_state.visualizer['params']['y_label'] = y_label
                            st.session_state.visualizer['params']['norm_method'] = norm_method
                            st.session_state.visualizer['params']['norm_range'] = norm_range
                            st.session_state.visualizer['params']['raw_offset_step'] = raw_offset_step
                            st.session_state.visualizer['params']['norm_offset_step'] = norm_offset_step
                            st.session_state.visualizer['params']['fill_area'] = fill_area
                            st.session_state.visualizer['params']['fill_alpha'] = fill_alpha
                            st.session_state.visualizer['params']['subtract_min_intensity'] = subtract_min_intensity
                            st.session_state.visualizer['params']['show_grid'] = show_grid
                            st.session_state.visualizer['params']['line_width'] = line_width
                            st.session_state.visualizer['params']['fig_width'] = fig_width
                            st.session_state.visualizer['params']['fig_height'] = fig_height
                            st.session_state.visualizer['params']['x_ranges'] = x_ranges
                            st.session_state.visualizer['params']['common_x_range'] = common_x_range
                            st.session_state.visualizer['params']['legend_fontsize'] = legend_fontsize
                            st.session_state.visualizer['params']['legend_position'] = legend_position
                            st.session_state.visualizer['params']['legend_offset'] = legend_offset
                            st.session_state.visualizer['params']['selected_spectra'] = selected_spectra
                            st.session_state.visualizer['params']['colors'] = colors
                            
                            # Update spectra data with colors
                            for name in selected_spectra:
                                if name in spectra_data:
                                    spectra_data[name]['color'] = colors[name]
                            
                            # Apply common x range if selected
                            current_spectra = spectra_data
                            if common_x_range:
                                current_spectra = align_x_ranges(current_spectra)
                            
                            # Filter spectra based on selection
                            filtered_spectra = {name: current_spectra[name] for name in selected_spectra if name in current_spectra}
                            
                            # Prepare normalized spectra
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
                            
                            # Store computed data
                            st.session_state.visualizer['computed_data']['filtered_spectra'] = filtered_spectra
                            st.session_state.visualizer['computed_data']['normalized_spectra'] = normalized_spectra
                            st.session_state.visualizer['ready'] = True
                            st.session_state.visualizer['last_update'] = datetime.now()
                            
                            st.success(f"✅ Visualizations updated! ({len(selected_spectra)} spectra)")
                            st.rerun()
            
            # Display visualizations if ready
            if st.session_state.visualizer['ready']:
                filtered_spectra = st.session_state.visualizer['computed_data']['filtered_spectra']
                normalized_spectra = st.session_state.visualizer['computed_data']['normalized_spectra']
                params = st.session_state.visualizer['params']
                
                if filtered_spectra:
                    # Display last update time
                    if st.session_state.visualizer['last_update']:
                        st.caption(f"🕐 Last updated: {st.session_state.visualizer['last_update'].strftime('%H:%M:%S')}")
                    
                    # Define the four visualization configurations
                    viz_configs = [
                        (filtered_spectra, 0, False, False, False, params['y_label']),
                        (normalized_spectra, 0, False, True, False, f"Normalized {params['y_label']}"),
                        (filtered_spectra, params['raw_offset_step'], False, False, True, params['y_label']),
                        (normalized_spectra, params['norm_offset_step'], params['fill_area'], True, True, f"Normalized {params['y_label']}")
                    ]
                    
                    # Create and display individual plots
                    for idx, (spectra, offset_step, fill, normalized, use_offset, yl) in enumerate(viz_configs):
                        fig = create_individual_plot(
                            spectra, params['x_label'], yl, "",
                            offset_step, fill, normalized, use_offset,
                            params['x_ranges'], params['subtract_min_intensity'], params['fill_alpha'],
                            params['show_grid'], params['line_width'], params['fig_width'], params['fig_height'],
                            legend_fontsize=params['legend_fontsize'],
                            legend_position=params['legend_position'],
                            legend_offset=params['legend_offset']
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
                    
                    # Heatmap section (independent)
                    st.markdown('<div class="separator">═══════════════════════════════════════════════════</div>', unsafe_allow_html=True)
                    st.subheader("🔥 Heatmap Visualization")
                    st.markdown("*Spectral evolution heatmaps showing intensity distribution as function of parameter*")
                    
                    # Heatmap settings form
                    with st.form("heatmap_settings"):
                        st.markdown("#### 📊 Heatmap Parameters")
                        st.markdown("*Assign numeric values (temperature, concentration, etc.) to each spectrum for heatmap visualization*")
                        
                        # Parameter type selection
                        heatmap_param_type = st.selectbox(
                            "Parameter type",
                            options=["Temperature (°C)", "Concentration (x)", "Custom"],
                            index=["Temperature (°C)", "Concentration (x)", "Custom"]
                                 .index(st.session_state.heatmap['params']['param_type']),
                            key="hm_param_type"
                        )
                        
                        # Custom label if Custom is selected
                        heatmap_custom_label = ""
                        if heatmap_param_type == "Custom":
                            heatmap_custom_label = st.text_input(
                                "Custom parameter label",
                                value=st.session_state.heatmap['params']['custom_label'],
                                key="hm_custom_label"
                            )
                        
                        # Determine the label for the heatmap y-axis
                        if heatmap_param_type == "Temperature (°C)":
                            heatmap_y_label = "Temperature (°C)"
                        elif heatmap_param_type == "Concentration (x)":
                            heatmap_y_label = "Concentration (x)"
                        else:
                            heatmap_y_label = heatmap_custom_label if heatmap_custom_label else "Parameter"
                        
                        # Get selected spectra for heatmap
                        hm_spectra = st.session_state.visualizer['params']['selected_spectra']
                        if not hm_spectra:
                            hm_spectra = list(spectra_data.keys())
                        
                        # Create input fields for each spectrum
                        st.markdown("#### Assign values to spectra:")
                        heatmap_params_temp = {}
                        for name in hm_spectra:
                            display_name = name.replace('.txt', '')
                            current_val = st.session_state.heatmap['params']['param_values'].get(name, len(heatmap_params_temp) + 1.0)
                            heatmap_params_temp[name] = st.number_input(
                                f"{display_name}",
                                value=current_val,
                                step=0.1,
                                format="%.1f",
                                key=f"hm_val_{name}"
                            )
                        
                        # Heatmap visualization settings
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
                                index=list(interpolation_options.keys()).index(st.session_state.heatmap['params']['interpolation'])
                                    if st.session_state.heatmap['params']['interpolation'] in interpolation_options else 5,
                                key="hm_interpolation"
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
                                index=list(colormap_options.keys()).index(st.session_state.heatmap['params']['colormap'])
                                    if st.session_state.heatmap['params']['colormap'] in colormap_options else 0,
                                key="hm_colormap"
                            )
                        
                        # Submit button
                        generate_heatmap = st.form_submit_button(
                            "🔥 Generate Heatmap",
                            use_container_width=True,
                            type="primary"
                        )
                        
                        if generate_heatmap and hm_spectra:
                            with st.spinner("Generating heatmap..."):
                                # Store heatmap parameters
                                st.session_state.heatmap['params']['param_type'] = heatmap_param_type
                                st.session_state.heatmap['params']['custom_label'] = heatmap_custom_label
                                st.session_state.heatmap['params']['param_values'] = heatmap_params_temp
                                st.session_state.heatmap['params']['interpolation'] = heatmap_interpolation
                                st.session_state.heatmap['params']['colormap'] = heatmap_colormap
                                st.session_state.heatmap['params']['ordered_names'] = hm_spectra
                                
                                # Prepare heatmap data
                                spectra_matrix, spectra_norm_matrix, x_grid, y_values = prepare_heatmap_data(
                                    spectra_data, hm_spectra, heatmap_params_temp,
                                    params['norm_method'], params['norm_range'], params['x_ranges']
                                )
                                
                                if spectra_matrix is not None:
                                    st.session_state.heatmap['data']['spectra_matrix'] = spectra_matrix
                                    st.session_state.heatmap['data']['spectra_norm_matrix'] = spectra_norm_matrix
                                    st.session_state.heatmap['data']['x_grid'] = x_grid
                                    st.session_state.heatmap['data']['y_values'] = y_values
                                    st.session_state.heatmap['data']['x_ranges'] = params['x_ranges']
                                    st.session_state.heatmap['ready'] = True
                                    st.session_state.heatmap['last_update'] = datetime.now()
                                    st.success(f"✅ Heatmap generated! {len(y_values)} spectra, {len(x_grid)} points each.")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to prepare heatmap data. Check that all spectra are valid.")
                    
                    # Display heatmap if ready
                    if st.session_state.heatmap['ready']:
                        # Get heatmap data from session state
                        spectra_matrix = st.session_state.heatmap['data']['spectra_matrix']
                        spectra_norm_matrix = st.session_state.heatmap['data']['spectra_norm_matrix']
                        x_grid = st.session_state.heatmap['data']['x_grid']
                        y_values = st.session_state.heatmap['data']['y_values']
                        heatmap_y_label = st.session_state.heatmap['params']['param_type']
                        if heatmap_y_label == "Custom":
                            heatmap_y_label = st.session_state.heatmap['params']['custom_label']
                        heatmap_interpolation = st.session_state.heatmap['params']['interpolation']
                        heatmap_colormap = st.session_state.heatmap['params']['colormap']
                        
                        if spectra_matrix is not None and x_grid is not None and y_values is not None:
                            # Show last update time
                            if st.session_state.heatmap['last_update']:
                                st.caption(f"🕐 Heatmap updated: {st.session_state.heatmap['last_update'].strftime('%H:%M:%S')}")
                            
                            # Determine if we should use log scale
                            min_val = np.min(spectra_matrix[spectra_matrix > 0]) if np.any(spectra_matrix > 0) else 1
                            max_val = np.max(spectra_matrix)
                            use_log = (max_val / min_val) > 100 if min_val > 0 else False
                            
                            # Create heatmap for raw intensity
                            fig_heatmap = create_heatmap(
                                spectra_matrix, x_grid, y_values,
                                params['x_label'], heatmap_y_label,
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
                                params['x_label'], heatmap_y_label,
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
                                'Spectrum': [name.replace('.txt', '') for name in st.session_state.heatmap['params']['ordered_names']],
                                heatmap_y_label: [st.session_state.heatmap['params']['param_values'][name] 
                                                 for name in st.session_state.heatmap['params']['ordered_names']]
                            })
                            st.dataframe(param_df, use_container_width=True)
                            
                            # Add interpolation and colormap info
                            st.caption(f"Interpolation: {heatmap_interpolation} | Colormap: {heatmap_colormap} | Log scale: {'Yes' if use_log else 'No'} for intensity, Yes for normalized")
                        else:
                            st.warning("⚠️ Heatmap data not available. Please generate heatmap using the form above.")
                    else:
                        st.info("💡 Configure heatmap parameters in the form above and click 'Generate Heatmap'.")
                else:
                    st.warning("⚠️ No spectra selected. Please select at least one spectrum in the settings form.")
            else:
                st.info("💡 Configure visualization settings in the form above and click 'Apply Visualization'.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========================================================================
        # TAB 2: PEAK ANALYZER (Independent Module)
        # ========================================================================
        with tab2:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Peak Detection and Analysis")
            st.markdown("*Select spectral range manually for precise peak analysis*")
            
            # Check if visualizer has selected spectra
            selected_spectra = st.session_state.visualizer['params']['selected_spectra']
            if not selected_spectra:
                st.warning("⚠️ Please select spectra in the Visualization tab first.")
            else:
                # Get spectra data
                spectra_data_local = st.session_state.data_manager['raw_spectra']
                filtered_spectra = {name: spectra_data_local[name] for name in selected_spectra if name in spectra_data_local}
                
                if not filtered_spectra:
                    st.warning("⚠️ No spectra available for analysis.")
                else:
                    # Peak analysis settings form
                    with st.form("peak_settings"):
                        st.markdown("#### 📊 Select Analysis Range")
                        
                        # Get global x range
                        all_x = []
                        for spec in filtered_spectra.values():
                            all_x.extend(spec['data']['x'].values)
                        global_min_x = float(np.min(all_x))
                        global_max_x = float(np.max(all_x))
                        
                        # Get current values from session state
                        current_left = st.session_state.peak_analyzer['params']['left_boundary']
                        current_right = st.session_state.peak_analyzer['params']['right_boundary']
                        if current_left is None:
                            current_left = global_min_x
                        if current_right is None:
                            current_right = global_max_x
                        
                        # Create range sliders
                        col1, col2 = st.columns(2)
                        with col1:
                            left_boundary = st.slider(
                                "Left boundary (cm⁻¹)",
                                min_value=global_min_x,
                                max_value=global_max_x,
                                value=current_left,
                                step=(global_max_x - global_min_x) / 100,
                                key="peak_left"
                            )
                        with col2:
                            right_boundary = st.slider(
                                "Right boundary (cm⁻¹)",
                                min_value=global_min_x,
                                max_value=global_max_x,
                                value=current_right,
                                step=(global_max_x - global_min_x) / 100,
                                key="peak_right"
                            )
                        
                        # Peak width
                        peak_width = st.slider(
                            "Peak width for area calculation (points)",
                            min_value=5,
                            max_value=100,
                            value=st.session_state.peak_analyzer['params']['peak_width'],
                            step=5,
                            key="peak_width"
                        )
                        
                        # Submit button
                        analyze_peaks = st.form_submit_button(
                            "🔍 Detect Peaks",
                            use_container_width=True,
                            type="primary"
                        )
                        
                        if analyze_peaks:
                            if left_boundary < right_boundary:
                                with st.spinner("Analyzing peaks..."):
                                    manual_range = (left_boundary, right_boundary)
                                    peaks_df = analyze_peaks_manual_range(
                                        filtered_spectra, 
                                        manual_range, 
                                        peak_width
                                    )
                                    st.session_state.peak_analyzer['params']['left_boundary'] = left_boundary
                                    st.session_state.peak_analyzer['params']['right_boundary'] = right_boundary
                                    st.session_state.peak_analyzer['params']['peak_width'] = peak_width
                                    st.session_state.peak_analyzer['results']['peaks_df'] = peaks_df
                                    st.session_state.peak_analyzer['results']['x_range'] = manual_range
                                    st.session_state.peak_analyzer['ready'] = True
                                    st.session_state.peak_analyzer['last_update'] = datetime.now()
                                    st.session_state.peak_analyzer['excluded_peaks'] = set()
                                    st.success(f"✅ Peak analysis complete! Found {len(peaks_df)} peaks total.")
                                    st.rerun()
                            else:
                                st.error("Please select a valid range (left < right)")
                    
                    # Display full spectra with range boundaries (always visible)
                    st.markdown("#### 📈 Full Spectra with Selected Range")
                    
                    # Get current range values
                    left_val = st.session_state.peak_analyzer['params']['left_boundary']
                    right_val = st.session_state.peak_analyzer['params']['right_boundary']
                    
                    fig_range, ax_range = plt.subplots(figsize=(12, 5))
                    for name, spec in filtered_spectra.items():
                        color = st.session_state.visualizer['params']['colors'].get(name, '#1f77b4')
                        data = spec['data']
                        ax_range.plot(data['x'].values, data['y'].values, 
                                     color=color, linewidth=1.5, 
                                     label=name.replace('.txt', ''), alpha=0.7)
                    
                    # Add range boundaries
                    if left_val is not None and right_val is not None and left_val < right_val:
                        ax_range.axvline(left_val, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Left: {left_val:.1f}')
                        ax_range.axvline(right_val, color='blue', linestyle='-', linewidth=2, alpha=0.7, label=f'Right: {right_val:.1f}')
                        ax_range.axvspan(left_val, right_val, alpha=0.2, color='gray')
                    
                    ax_range.set_xlabel(st.session_state.visualizer['params']['x_label'], fontsize=11, fontweight='bold')
                    ax_range.set_ylabel(st.session_state.visualizer['params']['y_label'], fontsize=11, fontweight='bold')
                    ax_range.set_title("Full Spectra with Selected Analysis Range", fontsize=12, fontweight='bold')
                    ax_range.legend(loc='best', fontsize=9)
                    ax_range.tick_params(direction='in', length=5, width=1)
                    ax_range.grid(True, alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig_range)
                    plt.close()
                    
                    # Display results if analysis has been run
                    if st.session_state.peak_analyzer['ready']:
                        peaks_df = st.session_state.peak_analyzer['results']['peaks_df']
                        
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
                                    'temp_id': None
                                },
                                disabled=['Spectrum', 'Peak position', 'Intensity', 'Area', 'FWHM'],
                                hide_index=True,
                                use_container_width=True,
                                key="peak_editor"
                            )
                            
                            # Update the Include column based on user edits
                            if edited_df is not None:
                                include_map = dict(zip(edited_df['temp_id'], edited_df['Include']))
                                peaks_df['Include'] = peaks_df['temp_id'].map(include_map)
                                st.session_state.peak_analyzer['results']['peaks_df'] = peaks_df.drop('temp_id', axis=1)
                                peaks_df = peaks_df.drop('temp_id', axis=1)
                            else:
                                peaks_df = peaks_df.drop('temp_id', axis=1)
                            
                            # Download button for peak analysis
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
                            
                            # Get colors for filtered spectra
                            for name in filtered_spectra:
                                filtered_spectra[name]['color'] = st.session_state.visualizer['params']['colors'].get(name, '#1f77b4')
                            
                            fig_peaks = create_peak_visualization(
                                filtered_spectra, 
                                st.session_state.peak_analyzer['results']['x_range'],
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
                            st.session_state.correlation['ready'] = True
                        else:
                            st.info("ℹ️ No peaks detected in the selected range. Try adjusting the range boundaries.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========================================================================
        # TAB 3: CORRELATION ANALYZER (Independent Module)
        # ========================================================================
        with tab3:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("Parameter Correlation Analysis")
            st.markdown("*Correlate spectral features (Intensity, Area, Position, FWHM) with experimental parameters*")
            
            # Check if peak analysis is ready
            if not st.session_state.peak_analyzer['ready']:
                st.info("📊 Please run peak analysis in the 'Advanced Peak Analysis' tab first to obtain peak data for correlation")
            else:
                peaks_df = st.session_state.peak_analyzer['results']['peaks_df']
                
                if peaks_df is None or peaks_df.empty:
                    st.warning("⚠️ No peaks found. Please run peak analysis first.")
                else:
                    # Filter to only include peaks marked as Include=True
                    peaks_df = peaks_df[peaks_df['Include'] == True]
                    
                    if peaks_df.empty:
                        st.warning("⚠️ No peaks are currently included. Please check at least one peak in the Peak Analysis tab.")
                    else:
                        # Get spectra names
                        spectra_names = list(st.session_state.visualizer['params']['selected_spectra'])
                        if not spectra_names:
                            spectra_names = list(st.session_state.data_manager['raw_spectra'].keys())
                        
                        # Correlation settings form
                        with st.form("correlation_settings"):
                            st.markdown("#### 📊 Parameter Values")
                            
                            param_values = {}
                            for name in spectra_names:
                                display_name = name.replace('.txt', '')
                                current_val = st.session_state.correlation['params']['param_values'].get(name, len(param_values) + 1.0)
                                param_values[name] = st.number_input(
                                    f"Value for {display_name}",
                                    value=float(current_val),
                                    step=1.0,
                                    key=f"corr_param_{name}"
                                )
                            
                            param_label = st.text_input(
                                "Parameter label",
                                value=st.session_state.correlation['params']['param_label'],
                                key="corr_label"
                            )
                            
                            # Submit button
                            calculate_corr = st.form_submit_button(
                                "📊 Calculate Correlations",
                                use_container_width=True,
                                type="primary"
                            )
                            
                            if calculate_corr:
                                with st.spinner("Calculating correlations..."):
                                    st.session_state.correlation['params']['param_values'] = param_values
                                    st.session_state.correlation['params']['param_label'] = param_label
                                    st.session_state.correlation['params']['enabled'] = True
                                    
                                    # Prepare data for correlation
                                    param_list = []
                                    intensity_list = []
                                    area_list = []
                                    position_list = []
                                    fwhm_list = []
                                    
                                    for name in spectra_names:
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
                                        
                                        st.session_state.correlation['results'] = {
                                            'param_list': param_list,
                                            'intensity_list': intensity_list,
                                            'area_list': area_list,
                                            'position_list': position_list,
                                            'fwhm_list': fwhm_list,
                                            'corr_intensity': corr_intensity,
                                            'corr_area': corr_area,
                                            'corr_position': corr_position,
                                            'corr_fwhm': corr_fwhm
                                        }
                                        st.session_state.correlation['last_update'] = datetime.now()
                                        st.success("✅ Correlation calculations complete!")
                                        st.rerun()
                                    else:
                                        st.error("❌ No matching peaks found for correlation analysis.")
                        
                        # Display results if available
                        if st.session_state.correlation['results'] is not None:
                            results = st.session_state.correlation['results']
                            
                            # Display correlation metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Intensity Correlation", f"{results['corr_intensity']:.3f}", 
                                         delta="strong" if abs(results['corr_intensity']) > 0.7 else "weak")
                            with col2:
                                st.metric("Area Correlation", f"{results['corr_area']:.3f}",
                                         delta="strong" if abs(results['corr_area']) > 0.7 else "weak")
                            with col3:
                                st.metric("Position Correlation", f"{results['corr_position']:.3f}",
                                         delta="strong" if abs(results['corr_position']) > 0.7 else "weak")
                            with col4:
                                st.metric("FWHM Correlation", f"{results['corr_fwhm']:.3f}",
                                         delta="strong" if abs(results['corr_fwhm']) > 0.7 else "weak")
                            
                            st.markdown("---")
                            
                            # Create correlation plots (4 plots: Intensity, Area, Position, FWHM)
                            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                            
                            param_label = st.session_state.correlation['params']['param_label']
                            
                            # Intensity plot
                            axes[0, 0].scatter(results['param_list'], results['intensity_list'], c='#1f77b4', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                            axes[0, 0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                            axes[0, 0].set_ylabel("Peak Intensity (a.u.)", fontsize=11, fontweight='bold')
                            axes[0, 0].set_title(f"Intensity vs {param_label}\n(r = {results['corr_intensity']:.3f})", fontsize=12, fontweight='bold')
                            axes[0, 0].grid(True, alpha=0.3, linestyle='--')
                            
                            # Area plot
                            axes[0, 1].scatter(results['param_list'], results['area_list'], c='#2ca02c', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                            axes[0, 1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                            axes[0, 1].set_ylabel("Peak Area", fontsize=11, fontweight='bold')
                            axes[0, 1].set_title(f"Area vs {param_label}\n(r = {results['corr_area']:.3f})", fontsize=12, fontweight='bold')
                            axes[0, 1].grid(True, alpha=0.3, linestyle='--')
                            
                            # Position plot
                            axes[1, 0].scatter(results['param_list'], results['position_list'], c='#d62728', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                            axes[1, 0].set_xlabel(param_label, fontsize=11, fontweight='bold')
                            axes[1, 0].set_ylabel("Peak Position (cm⁻¹)", fontsize=11, fontweight='bold')
                            axes[1, 0].set_title(f"Position vs {param_label}\n(r = {results['corr_position']:.3f})", fontsize=12, fontweight='bold')
                            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
                            
                            # FWHM plot
                            axes[1, 1].scatter(results['param_list'], results['fwhm_list'], c='#9467bd', alpha=0.6, s=80, edgecolors='white', linewidth=2)
                            axes[1, 1].set_xlabel(param_label, fontsize=11, fontweight='bold')
                            axes[1, 1].set_ylabel("FWHM (cm⁻¹)", fontsize=11, fontweight='bold')
                            axes[1, 1].set_title(f"FWHM vs {param_label}\n(r = {results['corr_fwhm']:.3f})", fontsize=12, fontweight='bold')
                            axes[1, 1].grid(True, alpha=0.3, linestyle='--')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Show correlation table
                            st.markdown("---")
                            st.subheader("Correlation Data Table")
                            corr_data = pd.DataFrame({
                                'Spectrum': [name.replace('.txt', '') for name in spectra_names if name in st.session_state.correlation['params']['param_values']],
                                param_label: results['param_list'],
                                'Intensity': results['intensity_list'],
                                'Area': results['area_list'],
                                'Position': results['position_list'],
                                'FWHM': results['fwhm_list']
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
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========================================================================
        # TAB 4: SPECTRA COMPARATOR (Independent Module)
        # ========================================================================
        with tab4:
            st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
            st.subheader("🔀 Spectral Difference Analysis")
            st.markdown("*Compare two spectra to identify differences and visualize them with heatmaps*")
            
            # Get spectra names
            spectra_names = list(st.session_state.data_manager['raw_spectra'].keys())
            
            if len(spectra_names) >= 2:
                # Comparison settings form
                with st.form("comparison_settings"):
                    st.markdown("#### Select spectra for comparison")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        ref_index = 0
                        if st.session_state.comparator['params']['ref_spectrum'] in spectra_names:
                            ref_index = spectra_names.index(st.session_state.comparator['params']['ref_spectrum'])
                        spectrum_a_name = st.selectbox(
                            "Reference Spectrum",
                            options=spectra_names,
                            index=ref_index,
                            key="comp_ref"
                        )
                    
                    with col2:
                        sample_index = min(1, len(spectra_names)-1)
                        if st.session_state.comparator['params']['sample_spectrum'] in spectra_names:
                            sample_index = spectra_names.index(st.session_state.comparator['params']['sample_spectrum'])
                        spectrum_b_name = st.selectbox(
                            "Sample Spectrum",
                            options=spectra_names,
                            index=sample_index,
                            key="comp_sample"
                        )
                    
                    # Option to swap difference direction
                    swap_direction = st.checkbox(
                        "Swap difference direction (Sample - Reference)",
                        value=st.session_state.comparator['params']['swap_direction'],
                        key="comp_swap"
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
                        current_cmap = st.session_state.comparator['params']['selected_colormap']
                        selected_colormap = st.selectbox(
                            "Color palette for difference heatmap",
                            options=list(colormap_options.keys()),
                            format_func=lambda x: colormap_options[x],
                            index=list(colormap_options.keys()).index(current_cmap) if current_cmap in colormap_options else 0,
                            key="comp_cmap"
                        )
                    
                    with col2:
                        smooth_difference = st.checkbox(
                            "Apply smoothing to difference profile",
                            value=st.session_state.comparator['params']['smooth_difference'],
                            key="comp_smooth"
                        )
                        smooth_sigma = 1.5
                        if smooth_difference:
                            smooth_sigma = st.slider(
                                "Smoothing sigma",
                                min_value=0.5,
                                max_value=5.0,
                                value=st.session_state.comparator['params']['smooth_sigma'],
                                step=0.5,
                                help="Higher values produce smoother difference profiles",
                                key="comp_sigma"
                            )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        symmetric_scale = st.checkbox(
                            "Symmetric color scale (centered at zero)",
                            value=st.session_state.comparator['params']['symmetric_scale'],
                            key="comp_sym"
                        )
                    
                    with col2:
                        difference_threshold = st.number_input(
                            "Significance threshold (highlight regions with |difference| > threshold)",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.comparator['params']['difference_threshold'],
                            step=0.01,
                            format="%.3f",
                            help="Regions with absolute difference exceeding this value will be highlighted",
                            key="comp_threshold"
                        )
                    
                    st.markdown("---")
                    
                    # Submit button
                    compare_spectra = st.form_submit_button(
                        "🔀 Compare Spectra",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    if compare_spectra and spectrum_a_name and spectrum_b_name:
                        if spectrum_a_name != spectrum_b_name:
                            with st.spinner("Generating comparison plot..."):
                                # Store parameters
                                st.session_state.comparator['params']['ref_spectrum'] = spectrum_a_name
                                st.session_state.comparator['params']['sample_spectrum'] = spectrum_b_name
                                st.session_state.comparator['params']['swap_direction'] = swap_direction
                                st.session_state.comparator['params']['selected_colormap'] = selected_colormap
                                st.session_state.comparator['params']['smooth_difference'] = smooth_difference
                                st.session_state.comparator['params']['smooth_sigma'] = smooth_sigma
                                st.session_state.comparator['params']['symmetric_scale'] = symmetric_scale
                                st.session_state.comparator['params']['difference_threshold'] = difference_threshold
                                
                                # Get selected spectra data
                                spectra_data_local = st.session_state.data_manager['raw_spectra']
                                spectrum_a = spectra_data_local[spectrum_a_name]
                                spectrum_b = spectra_data_local[spectrum_b_name]
                                
                                # Add colors
                                spectrum_a['color'] = st.session_state.visualizer['params']['colors'].get(spectrum_a_name, '#1f77b4')
                                spectrum_b['color'] = st.session_state.visualizer['params']['colors'].get(spectrum_b_name, '#ff7f0e')
                                
                                name_a = spectrum_a_name.replace('.txt', '')
                                name_b = spectrum_b_name.replace('.txt', '')
                                
                                # Get visualization parameters
                                viz_params = st.session_state.visualizer['params']
                                
                                # Create comparison plot
                                fig, (mean_diff, max_abs_diff, rms_diff, correlation) = create_comparison_plot(
                                    spectrum_a, spectrum_b, name_a, name_b,
                                    viz_params['x_label'], viz_params['y_label'],
                                    viz_params['norm_method'], viz_params['norm_range'],
                                    viz_params['norm_offset_step'], viz_params['fill_area'],
                                    viz_params['fill_alpha'], viz_params['subtract_min_intensity'],
                                    viz_params['show_grid'], viz_params['line_width'],
                                    viz_params['fig_width'], viz_params['fig_height'],
                                    viz_params['legend_fontsize'], viz_params['legend_position'],
                                    viz_params['legend_offset'],
                                    selected_colormap, smooth_difference, smooth_sigma,
                                    symmetric_scale, difference_threshold
                                )
                                
                                # Store results
                                st.session_state.comparator['results'] = {
                                    'fig': fig,
                                    'mean_diff': mean_diff,
                                    'max_abs_diff': max_abs_diff,
                                    'rms_diff': rms_diff,
                                    'correlation': correlation,
                                    'name_a': name_a,
                                    'name_b': name_b
                                }
                                st.session_state.comparator['ready'] = True
                                st.session_state.comparator['last_update'] = datetime.now()
                                st.success("✅ Comparison complete!")
                                st.rerun()
                        else:
                            st.error("Please select two different spectra for comparison.")
                
                # Display results if ready
                if st.session_state.comparator['ready'] and st.session_state.comparator['results'] is not None:
                    results = st.session_state.comparator['results']
                    
                    # Display statistics
                    st.markdown("#### 📊 Difference Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Difference", f"{results['mean_diff']:.4f}")
                    with col2:
                        st.metric("Max |Difference|", f"{results['max_abs_diff']:.4f}")
                    with col3:
                        st.metric("RMS Difference", f"{results['rms_diff']:.4f}")
                    with col4:
                        st.metric("Spectral Correlation", f"{results['correlation']:.4f}", 
                                 delta="strong" if abs(results['correlation']) > 0.7 else "weak")
                    
                    if st.session_state.comparator['last_update']:
                        st.caption(f"🕐 Last updated: {st.session_state.comparator['last_update'].strftime('%H:%M:%S')}")
                    
                    # Display the plot
                    st.pyplot(results['fig'])
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        # Download plot as PNG
                        buf = BytesIO()
                        results['fig'].savefig(buf, format='png', dpi=600, bbox_inches='tight')
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
                        spectra_data_local = st.session_state.data_manager['raw_spectra']
                        spectrum_a = spectra_data_local[st.session_state.comparator['params']['ref_spectrum']]
                        spectrum_b = spectra_data_local[st.session_state.comparator['params']['sample_spectrum']]
                        
                        x_a = spectrum_a['data']['x'].values
                        y_a_raw = spectrum_a['data']['y'].values
                        y_a_norm = normalize_spectrum(x_a, y_a_raw, 
                                                     st.session_state.visualizer['params']['norm_method'],
                                                     st.session_state.visualizer['params']['norm_range'])
                        
                        x_b = spectrum_b['data']['x'].values
                        y_b_raw = spectrum_b['data']['y'].values
                        y_b_norm = normalize_spectrum(x_b, y_b_raw,
                                                     st.session_state.visualizer['params']['norm_method'],
                                                     st.session_state.visualizer['params']['norm_range'])
                        
                        common_x_min = max(x_a.min(), x_b.min())
                        common_x_max = min(x_a.max(), x_b.max())
                        common_x_exp = np.linspace(common_x_min, common_x_max, 2000)
                        y_a_interp_exp = np.interp(common_x_exp, x_a, y_a_norm)
                        y_b_interp_exp = np.interp(common_x_exp, x_b, y_b_norm)
                        
                        if st.session_state.visualizer['params']['subtract_min_intensity']:
                            y_a_interp_exp = y_a_interp_exp - y_a_interp_exp.min()
                            y_b_interp_exp = y_b_interp_exp - y_b_interp_exp.min()
                        
                        if st.session_state.comparator['params']['swap_direction']:
                            y_diff_exp = y_b_interp_exp - y_a_interp_exp
                        else:
                            y_diff_exp = y_a_interp_exp - y_b_interp_exp
                        
                        diff_df = pd.DataFrame({
                            'x': common_x_exp,
                            f'{results["name_a"]}_normalized': y_a_interp_exp,
                            f'{results["name_b"]}_normalized': y_b_interp_exp,
                            'difference': y_diff_exp
                        })
                        
                        csv_diff = diff_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Export Difference Data (CSV)",
                            data=csv_diff,
                            file_name=f"difference_data_{results['name_a']}_{results['name_b']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    plt.close(results['fig'])
            else:
                st.warning("⚠️ Please load at least 2 spectra to use the comparison feature.")
                st.info("Upload multiple .txt files to compare different samples or treatments.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========================================================================
        # EXPORT SECTION (Global)
        # ========================================================================
        st.markdown("---")
        st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
        st.subheader("📤 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        # Get filtered spectra for export
        if st.session_state.visualizer['ready']:
            filtered_spectra = st.session_state.visualizer['computed_data']['filtered_spectra']
        else:
            filtered_spectra = st.session_state.data_manager['raw_spectra']
        
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
            if st.session_state.visualizer['ready']:
                normalized_spectra = st.session_state.visualizer['computed_data']['normalized_spectra']
                if normalized_spectra:
                    export_norm = pd.DataFrame()
                    for name, spec in normalized_spectra.items():
                        data = spec['data']
                        temp_df = pd.DataFrame({
                            f"{name.replace('.txt', '')}_x": data['x'].values,
                            f"{name.replace('.txt', '')}_y_norm": data['y'].values
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
Spectra Files: {', '.join(list(st.session_state.data_manager['raw_spectra'].keys()))}
Normalization Method: {st.session_state.visualizer['params']['norm_method']}
X-axis Ranges: {st.session_state.visualizer['params']['x_ranges'] if st.session_state.visualizer['params']['x_ranges'] else 'Full range'}
Raw Offset Step: {st.session_state.visualizer['params']['raw_offset_step']}
Normalized Offset Step: {st.session_state.visualizer['params']['norm_offset_step']}
Fill Area: {st.session_state.visualizer['params']['fill_area']}
Fill Transparency: {st.session_state.visualizer['params']['fill_alpha']}
Subtract Minimum Intensity: {st.session_state.visualizer['params']['subtract_min_intensity']}
Grid Enabled: {st.session_state.visualizer['params']['show_grid']}
Line Width: {st.session_state.visualizer['params']['line_width']}
Peak Analysis Ready: {st.session_state.peak_analyzer['ready']}
Correlation Ready: {st.session_state.correlation['ready']}
Heatmap Ready: {st.session_state.heatmap['ready']}
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
        st.markdown("## 🎯 Welcome to SpectrAnalys v3.0")
        st.markdown("Advanced spectroscopic data analysis platform for researchers and scientists")
        st.markdown("*New: Independent module architecture for faster, more responsive workflows*")
        
        st.markdown("### 📖 Quick Start Guide:")
        st.markdown("""
        1. **Upload Data** - Select one or more .txt files with two columns (x y, tab-separated)
        2. **Configure Visualization** - In the Visualization tab, select spectra, assign colors, set normalization and offset parameters, then click "Apply Visualization"
        3. **Analyze Peaks** - In the Peak Analysis tab, select range and click "Detect Peaks"
        4. **Correlate Parameters** - In the Correlation tab, assign parameter values and click "Calculate Correlations"
        5. **Compare Spectra** - In the Comparison tab, select two spectra and click "Compare Spectra"
        6. **Generate Heatmaps** - In the Visualization tab, configure heatmap parameters and click "Generate Heatmap"
        7. **Export Results** - Download processed data, plots, and analysis results
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
        - 💾 **Data Export** - Download processed data in CSV format with publication-ready plots
        - 📐 **Multiple Normalization Methods** - Maximum intensity or custom peak range normalization
        - 📏 **Cumulative Offset** - Add offsets to spectra for clear visualization (1st: 0, 2nd: +step, 3rd: +2×step)
        - 🎛️ **Adjustable Transparency** - Control fill opacity from 0.2 to 0.9
        - ⚙️ **Grid & Line Thickness** - Customize plot appearance
        - ⚡ **Independent Modules** - Each analysis tab works independently with its own Apply button
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
        <p>🔬 SpectrAnalys v3.0 | Independent Modules Architecture | Scientific Spectroscopic Analysis Platform</p>
        <p style="font-size: 0.75rem;">© 2024 SpectrAnalys - Advanced Spectroscopy Data Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
