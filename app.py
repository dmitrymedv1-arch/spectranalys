import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simpson
from scipy.special import wofz
from io import StringIO, BytesIO
import re
from datetime import datetime
import base64
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection

try:
    from lmfit import Parameters
    from lmfit.models import VoigtModel
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False

try:
    import pybaselines
    from pybaselines import whittaker as pb_whittaker
    from pybaselines import polynomial as pb_polynomial
    PYBASELINES_AVAILABLE = True
except ImportError:
    PYBASELINES_AVAILABLE = False

st.set_page_config(
    page_title="SpectrAnalys",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

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

if 'spectral_markers' not in st.session_state:
    st.session_state.spectral_markers = []
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
if 'spectral_markers_update_needed' not in st.session_state:
    st.session_state.spectral_markers_update_needed = False
if 'pending_marker_index' not in st.session_state:
    st.session_state.pending_marker_index = None

if 'enable_3d_graphs' not in st.session_state:
    st.session_state.enable_3d_graphs = False
if 'graph3d_x_label' not in st.session_state:
    st.session_state.graph3d_x_label = 'Raman shift (cm⁻¹)'
if 'graph3d_y_label' not in st.session_state:
    st.session_state.graph3d_y_label = 'Parameter'
if 'graph3d_z_label' not in st.session_state:
    st.session_state.graph3d_z_label = 'Intensity (a.u.)'
if 'graph3d_azimuth' not in st.session_state:
    st.session_state.graph3d_azimuth = -60
if 'graph3d_elevation' not in st.session_state:
    st.session_state.graph3d_elevation = 30
if 'graph3d_params' not in st.session_state:
    st.session_state.graph3d_params = {}
if 'graph3d_created' not in st.session_state:
    st.session_state.graph3d_created = False

if 'gradient_palette' not in st.session_state:
    st.session_state.gradient_palette = 'RdYlBu'
if 'gradient_applied' not in st.session_state:
    st.session_state.gradient_applied = False
if 'gradient_colors_cache' not in st.session_state:
    st.session_state.gradient_colors_cache = {}
if 'gradient_version' not in st.session_state:
    st.session_state.gradient_version = 0

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Combined Spectra Visualization"

if 'peak_enabled' not in st.session_state:
    st.session_state.peak_enabled = False
if 'correlation_enabled' not in st.session_state:
    st.session_state.correlation_enabled = False

# Preprocessing global state
if 'preproc_despike_enabled' not in st.session_state:
    st.session_state.preproc_despike_enabled = False
if 'preproc_despike_kernel' not in st.session_state:
    st.session_state.preproc_despike_kernel = 3
if 'preproc_despike_threshold' not in st.session_state:
    st.session_state.preproc_despike_threshold = 8.0
if 'preproc_denoise_enabled' not in st.session_state:
    st.session_state.preproc_denoise_enabled = False
if 'preproc_denoise_method' not in st.session_state:
    st.session_state.preproc_denoise_method = 'Savitzky-Golay'
if 'preproc_denoise_window' not in st.session_state:
    st.session_state.preproc_denoise_window = 11
if 'preproc_denoise_polyorder' not in st.session_state:
    st.session_state.preproc_denoise_polyorder = 3
if 'preproc_denoise_sigma' not in st.session_state:
    st.session_state.preproc_denoise_sigma = 1.0
if 'preproc_baseline_enabled' not in st.session_state:
    st.session_state.preproc_baseline_enabled = False
if 'preproc_baseline_method' not in st.session_state:
    st.session_state.preproc_baseline_method = 'asPLS'
if 'preproc_baseline_lam' not in st.session_state:
    st.session_state.preproc_baseline_lam = 1e5
if 'preproc_baseline_p' not in st.session_state:
    st.session_state.preproc_baseline_p = 1e-2
if 'preproc_baseline_diff_order' not in st.session_state:
    st.session_state.preproc_baseline_diff_order = 2
if 'preproc_baseline_max_iter' not in st.session_state:
    st.session_state.preproc_baseline_max_iter = 50
if 'preproc_baseline_tol' not in st.session_state:
    st.session_state.preproc_baseline_tol = 1e-3
if 'preproc_baseline_polyorder' not in st.session_state:
    st.session_state.preproc_baseline_polyorder = 3
if 'preproc_applied' not in st.session_state:
    st.session_state.preproc_applied = False
if 'preproc_cache_key' not in st.session_state:
    st.session_state.preproc_cache_key = None
if 'preproc_cached_spectra' not in st.session_state:
    st.session_state.preproc_cached_spectra = None
if 'preproc_version' not in st.session_state:
    st.session_state.preproc_version = 0

# Peak detection global parameters
if 'peak_det_height_factor' not in st.session_state:
    st.session_state.peak_det_height_factor = 0.05
if 'peak_det_prominence_factor' not in st.session_state:
    st.session_state.peak_det_prominence_factor = 0.03
if 'peak_det_min_distance' not in st.session_state:
    st.session_state.peak_det_min_distance = 5
if 'peak_det_min_height_abs' not in st.session_state:
    st.session_state.peak_det_min_height_abs = 2.0

# Voigt fitting state
if 'voigt_fit_triggered' not in st.session_state:
    st.session_state.voigt_fit_triggered = False
if 'voigt_fit_results' not in st.session_state:
    st.session_state.voigt_fit_results = None
if 'voigt_fit_components_df' not in st.session_state:
    st.session_state.voigt_fit_components_df = None
if 'voigt_fit_spectrum_name' not in st.session_state:
    st.session_state.voigt_fit_spectrum_name = None
if 'voigt_fit_x_range' not in st.session_state:
    st.session_state.voigt_fit_x_range = (None, None)
if 'voigt_use_normalized' not in st.session_state:
    st.session_state.voigt_use_normalized = False
if 'voigt_fit_log' not in st.session_state:
    st.session_state.voigt_fit_log = None

def load_instruction_html():
    try:
        if os.path.exists('instruction.html'):
            with open('instruction.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            if os.path.exists('logo.png'):
                with open('logo.png', 'rb') as f:
                    logo_data = f.read()
                    logo_base64 = base64.b64encode(logo_data).decode('utf-8')
                html_content = html_content.replace('{logo_base64}', logo_base64)
            else:
                html_content = html_content.replace('{logo_base64}', '')
                html_content = html_content.replace('<div style="display: flex; justify-content: center; margin-bottom: 10px;">',
                                                   '<div style="display: flex; justify-content: center; margin-bottom: 10px; font-size: 48px;">🔬')
            return html_content
        else:
            return None
    except Exception as e:
        print(f"Error loading instruction: {e}")
        return None

st.markdown("""
<style>
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --background-color: #f8f9fa;
        --card-background: #ffffff;
        --text-color: #2c3e50;
        --border-color: #e0e0e0;
    }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 2rem; color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .custom-header h1 { margin: 0; font-size: 2.5rem; font-weight: 700; }
    .custom-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem; }
    .scientific-card {
        background: var(--card-background); border-radius: 12px; padding: 1.5rem;
        margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color); transition: transform 0.2s, box-shadow 0.2s;
    }
    .scientific-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
    .card-title {
        font-size: 1.25rem; font-weight: 600; color: var(--primary-color);
        margin-bottom: 1rem; padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--secondary-color); display: inline-block;
    }
    .css-1d391kg { background-color: var(--background-color); }
    .sidebar .sidebar-content { background-color: var(--background-color); }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem;
        font-weight: 500; transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px; padding: 1rem; color: white; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.875rem; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; border-radius: 8px; padding: 0.5rem 1rem;
        font-weight: 500; transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: rgba(102, 126, 234, 0.1); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    .streamlit-expanderHeader { background-color: var(--background-color); border-radius: 8px; font-weight: 500; }
    .dataframe { border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stAlert { border-radius: 8px; border-left: 4px solid; }
    .footer {
        text-align: center; padding: 2rem; margin-top: 2rem;
        border-top: 1px solid var(--border-color); color: #666; font-size: 0.875rem;
    }
    .separator {
        text-align: center; margin: 20px 0; font-size: 20px;
        font-weight: bold; color: #667eea;
    }
    div[data-testid="stPills"] button {
        background: linear-gradient(135deg, #e8f0fe 0%, #d4e4fc 100%);
        color: #2c3e50 !important;
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin: 0.2rem !important;
        transition: all 0.25s !important;
    }
    div[data-testid="stPills"] button:hover {
        background: linear-gradient(135deg, #d4e4fc 0%, #b8d4fa 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(102,126,234,0.25);
    }
    div[data-testid="stPills"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid #764ba2 !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 11, 'axes.labelweight': 'bold',
    'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.facecolor': '#f8f9fa', 'axes.edgecolor': '#2c3e50', 'axes.linewidth': 1.2,
    'xtick.color': '#2c3e50', 'ytick.color': '#2c3e50',
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 6, 'xtick.minor.size': 3,
    'ytick.major.size': 6, 'ytick.minor.size': 3,
    'xtick.major.width': 1, 'ytick.major.width': 1,
    'legend.fontsize': 10, 'legend.frameon': True, 'legend.framealpha': 0.95,
    'legend.edgecolor': '#2c3e50', 'legend.fancybox': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'figure.facecolor': 'white',
    'lines.linewidth': 0.6, 'lines.markersize': 5, 'errorbar.capsize': 3,
})

@st.cache_data
def load_spectrum(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        data = pd.read_csv(StringIO(content), sep='\s+', header=None, names=['x', 'y'])
        data = data.dropna()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data = data.sort_values('x').reset_index(drop=True)
        return data
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {e}")
        return None

def _despike_whitaker_hayes(y, kernel_size=3, threshold=8.0):
    import copy
    spectrum = copy.deepcopy(np.asarray(y, dtype=float))
    if len(spectrum) < 4:
        return spectrum
    mad = np.median(np.abs(spectrum - np.median(spectrum)))
    if mad == 0:
        return spectrum
    mz = np.abs(0.6745 * (np.diff(spectrum) - np.median(np.diff(spectrum))) / (np.median(np.abs(np.diff(spectrum) - np.median(np.diff(spectrum)))) + 1e-12))
    spikes = np.concatenate([[False], mz > threshold])
    max_iter = 10
    it = 0
    while np.any(spikes) and it < max_iter:
        changes = False
        for i in range(len(spikes)):
            if spikes[i]:
                lo = max(0, i - kernel_size)
                hi = min(len(spectrum), i + kernel_size + 1)
                nb = np.arange(lo, hi)
                good = nb[~spikes[nb]]
                if len(good) == 0:
                    continue
                spectrum[i] = np.mean(spectrum[good])
                spikes[i] = False
                changes = True
        if not changes:
            break
        it += 1
    return spectrum

def _denoise_savgol(y, window_length=11, polyorder=3):
    y = np.asarray(y, dtype=float)
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    if wl <= polyorder:
        wl = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    if wl >= len(y):
        wl = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
    if wl < 3:
        return y
    return savgol_filter(y, window_length=wl, polyorder=polyorder)

def _denoise_gaussian(y, sigma=1.0):
    return gaussian_filter(np.asarray(y, dtype=float), sigma=sigma)

def _denoise_whittaker(y, lam=1e3, d=2):
    y = np.asarray(y, dtype=float)
    I = np.eye(len(y))
    D = np.diff(I, d).T
    return y @ np.linalg.inv(I + lam * D.T @ D).T

def _baseline_fallback_poly(y, polyorder=3):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    try:
        coef = np.polyfit(x, y, polyorder)
        return np.polyval(coef, x)
    except Exception:
        return np.zeros_like(y)

def _baseline_als_fallback(y, lam=1e5, p=1e-2, diff_order=2, max_iter=50, tol=1e-3):
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = np.zeros(L)
    for _ in range(max_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z_new = spsolve(Z, w * y)
        if np.linalg.norm(z_new - z) / (np.linalg.norm(z_new) + 1e-12) < tol:
            z = z_new
            break
        z = z_new
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def _apply_baseline_method(y, method, lam, p, diff_order, max_iter, tol, polyorder):
    y = np.asarray(y, dtype=float)
    if PYBASELINES_AVAILABLE:
        try:
            if method == 'asPLS':
                baseline, _ = pybaselines.whittaker.aspls(y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
                return baseline
            elif method == 'airPLS':
                baseline, _ = pybaselines.whittaker.airpls(y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
                return baseline
            elif method == 'arPLS':
                baseline, _ = pybaselines.whittaker.arpls(y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
                return baseline
            elif method == 'iarPLS':
                baseline, _ = pybaselines.whittaker.iarpls(y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
                return baseline
            elif method == 'drPLS':
                baseline, _ = pybaselines.whittaker.drpls(y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
                return baseline
            elif method == 'IAsLS':
                baseline, _ = pybaselines.whittaker.iasls(y, lam=lam, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol)
                return baseline
            elif method == 'AsLS':
                baseline, _ = pybaselines.whittaker.asls(y, lam=lam, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol)
                return baseline
            elif method == 'ModPoly':
                baseline, _ = pybaselines.polynomial.modpoly(y, poly_order=polyorder, tol=tol, max_iter=max_iter)
                return baseline
            elif method == 'IModPoly':
                baseline, _ = pybaselines.polynomial.imodpoly(y, poly_order=polyorder, tol=tol, max_iter=max_iter)
                return baseline
            elif method == 'Poly':
                baseline, _ = pybaselines.polynomial.poly(y, poly_order=polyorder)
                return baseline
        except Exception:
            pass
    if method == 'Poly':
        return _baseline_fallback_poly(y, polyorder)
    return _baseline_als_fallback(y, lam=lam, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol)

@st.cache_data
def preprocess_spectra_cached(spectra_pickle_key, spectra_pickle, despike_enabled, despike_kernel, despike_threshold,
                              denoise_enabled, denoise_method, denoise_window, denoise_polyorder, denoise_sigma,
                              baseline_enabled, baseline_method, baseline_lam, baseline_p, baseline_diff_order,
                              baseline_max_iter, baseline_tol, baseline_polyorder):
    result = {}
    for name, spec in spectra_pickle.items():
        data = spec['data']
        x = data['x'].values.copy()
        y = data['y'].values.copy()
        if despike_enabled:
            y = _despike_whitaker_hayes(y, kernel_size=despike_kernel, threshold=despike_threshold)
        if denoise_enabled:
            if denoise_method == 'Savitzky-Golay':
                y = _denoise_savgol(y, window_length=denoise_window, polyorder=denoise_polyorder)
            elif denoise_method == 'Gaussian':
                y = _denoise_gaussian(y, sigma=denoise_sigma)
            elif denoise_method == 'Whittaker':
                y = _denoise_whittaker(y, lam=1e3, d=2)
        if baseline_enabled:
            baseline = _apply_baseline_method(y, baseline_method, baseline_lam, baseline_p,
                                              baseline_diff_order, baseline_max_iter, baseline_tol, baseline_polyorder)
            y = y - baseline
        result[name] = {
            'data': pd.DataFrame({'x': x, 'y': y}),
            'original_data': spec['data'],
            'color': spec['color']
        }
    return result

@st.cache_data
def normalize_spectrum(x, y, norm_method, norm_range=None, x_ranges_for_rest=None):
    if len(y) == 0:
        return y
    y = np.array(y)
    mask = np.isfinite(y)
    if not np.any(mask):
        return y
    y_clean = y[mask]

    if norm_method == "Maximum intensity":
        max_val = y_clean.max()
        if max_val != 0 and np.isfinite(max_val):
            result = y / max_val
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
            y_in_ranges = []
            for start, end in x_ranges_for_rest:
                mask_range = (x >= start) & (x <= end)
                if np.any(mask_range):
                    y_in_range = y[mask_range]
                    y_in_range_clean = y_in_range[np.isfinite(y_in_range)]
                    if len(y_in_range_clean) > 0:
                        y_in_ranges.extend(y_in_range_clean)
            if len(y_in_ranges) > 0:
                max_in_ranges = np.max(y_in_ranges)
                if max_in_ranges != 0 and np.isfinite(max_in_ranges):
                    result = y / max_in_ranges
                    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                    return result
            max_val = y_clean.max()
            if max_val != 0 and np.isfinite(max_val):
                result = y / max_val
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                return result
            return y
        else:
            max_val = y_clean.max()
            if max_val != 0 and np.isfinite(max_val):
                result = y / max_val
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                return result
            return y

    return y

@st.cache_data
def align_x_ranges(spectra_dict):
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
    if ranges is None:
        return [(x, y)]
    segments = []
    for start, end in ranges:
        mask = (x >= start) & (x <= end)
        if np.any(mask):
            segments.append((x[mask], y[mask]))
    return segments

def gradient_fill(ax, x, y, color, offset=0):
    verts = []
    for i in range(len(x)):
        verts.append((x[i], y[i] + offset))
    for i in range(len(x)-1, -1, -1):
        verts.append((x[i], offset))
    poly = plt.Polygon(verts, closed=True, facecolor=color, alpha=0.3, edgecolor='none')
    ax.add_patch(poly)
    n_layers = 20
    for i in range(n_layers):
        alpha_i = 0.3 + (i / n_layers) * 0.6
        y_level = offset + (y + offset - offset) * (i / n_layers)
        for j in range(len(x)-1):
            ax.fill_between(x[j:j+2], offset, y_level[j:j+2],
                           color=color, alpha=0.03, linewidth=0)

def create_individual_plot(spectra_dict, x_label, y_label, title,
                           offset_step, fill_area, normalized, use_offset,
                           x_ranges, subtract_min_intensity, fill_alpha,
                           show_grid, line_width, fig_width, fig_height,
                           legend_fontsize=8, legend_position="right", legend_offset=1.02):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    handles = []
    labels = []
    spectra_items = list(spectra_dict.items())

    if x_ranges is None or len(x_ranges) == 0:
        for idx, (name, spec) in enumerate(spectra_items):
            data = spec['data']
            x = data['x'].values
            y = data['y'].values
            color = spec['color']
            display_name = name.replace('.txt', '')
            if subtract_min_intensity and normalized:
                if len(y) > 0:
                    y = y - y.min()
                else:
                    y = y
            if use_offset:
                offset = idx * offset_step
            else:
                offset = 0
            y_plot = y + offset
            if fill_area and normalized:
                if len(x) > 0 and len(y_plot) > 0:
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
        for range_idx, (start, end) in enumerate(x_ranges):
            for idx, (name, spec) in enumerate(spectra_items):
                data = spec['data']
                x_full = data['x'].values
                y_full = data['y'].values
                color = spec['color']
                display_name = name.replace('.txt', '')
                mask = (x_full >= start) & (x_full <= end)
                if not np.any(mask):
                    continue
                x = x_full[mask]
                y = y_full[mask]
                if subtract_min_intensity and normalized:
                    y = y - y.min()
                if use_offset:
                    offset = idx * offset_step
                else:
                    offset = 0
                y_plot = y + offset
                if fill_area and normalized:
                    ax.fill_between(x, offset, y_plot, alpha=fill_alpha, color=color)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                else:
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                if range_idx == 0:
                    handles.append(line_handle[0])
                    labels.append(display_name)
            ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold')

    if handles:
        if legend_position == "right":
            bbox_anchor = (legend_offset, 0.5)
            loc = 'center left'
        elif legend_position == "best":
            bbox_anchor = None
            loc = 'best'
        else:
            bbox_anchor = None
            loc = legend_position
        if use_offset and legend_position == "right":
            reversed_handles = list(reversed(handles))
            reversed_labels = list(reversed(labels))
            legend = ax.legend(reversed_handles, reversed_labels,
                              loc=loc, bbox_to_anchor=bbox_anchor,
                              fontsize=legend_fontsize, frameon=True, edgecolor='black',
                              prop={'weight': 'bold'})
            for text, handle in zip(legend.get_texts(), reversed_handles):
                text.set_color(handle.get_color())
        else:
            legend = ax.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_anchor,
                              fontsize=legend_fontsize, frameon=True, edgecolor='black',
                              prop={'weight': 'bold'})
            if legend_position == "right":
                for text, handle in zip(legend.get_texts(), handles):
                    text.set_color(handle.get_color())
        if len(handles) > 10:
            legend._legend_box.align = "left"
            if len(handles) > 15:
                legend._ncol = 2

    ax.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    else:
        ax.grid(False)

    if legend_position == "right" and len(handles) > 0:
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
    normalized_spectra = {}
    for name, spec in spectra_dict.items():
        data = spec['data']
        y_norm = normalize_spectrum(
            data['x'].values, data['y'].values, norm_method, None, x_ranges
        )
        normalized_spectra[name] = {
            'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
            'color': spec['color']
        }
    fig, axes = plt.subplots(4, 1, figsize=(12, 18))
    viz_configs = [
        (axes[0], spectra_dict, 0, False, False, False, x_label, y_label),
        (axes[1], normalized_spectra, 0, False, True, False, x_label, f"Normalized {y_label}"),
        (axes[2], spectra_dict, raw_offset_step, False, False, True, x_label, y_label),
        (axes[3], normalized_spectra, norm_offset_step, fill_area, True, True, x_label, f"Normalized {y_label}")
    ]
    for ax, spectra, offset_step, fill, normalized, use_offset, xl, yl in viz_configs:
        handles = []
        labels = []
        spectra_items = list(spectra.items())
        if x_ranges is None or len(x_ranges) == 0:
            for idx, (name, spec) in enumerate(spectra_items):
                data = spec['data']
                x = data['x'].values
                y = data['y'].values
                color = spec['color']
                display_name = name.replace('.txt', '')
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
            for range_idx, (start, end) in enumerate(x_ranges):
                for idx, (name, spec) in enumerate(spectra_items):
                    data = spec['data']
                    x_full = data['x'].values
                    y_full = data['y'].values
                    color = spec['color']
                    display_name = name.replace('.txt', '')
                    mask = (x_full >= start) & (x_full <= end)
                    if not np.any(mask):
                        continue
                    x = x_full[mask]
                    y = y_full[mask]
                    if use_offset:
                        offset = idx * offset_step
                    else:
                        offset = 0
                    y_plot = y + offset
                    if fill and normalized and use_offset:
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
                    if range_idx == 0:
                        handles.append(line_handle[0])
                        labels.append(display_name)
                ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
                ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.set_xlabel(xl, fontsize=10, fontweight='bold')
            ax.set_ylabel(yl, fontsize=10, fontweight='bold')
        if handles:
            if legend_position == "right":
                bbox_anchor = (legend_offset, 0.5)
                loc = 'center left'
            elif legend_position == "best":
                bbox_anchor = None
                loc = 'best'
            else:
                bbox_anchor = None
                loc = legend_position
            if use_offset and legend_position == "right":
                reversed_handles = list(reversed(handles))
                reversed_labels = list(reversed(labels))
                legend = ax.legend(reversed_handles, reversed_labels, loc=loc,
                                  bbox_to_anchor=bbox_anchor, fontsize=legend_fontsize,
                                  frameon=True, edgecolor='black', prop={'weight': 'bold'})
                for text, handle in zip(legend.get_texts(), reversed_handles):
                    text.set_color(handle.get_color())
            else:
                legend = ax.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_anchor,
                                  fontsize=legend_fontsize, frameon=True, edgecolor='black',
                                  prop={'weight': 'bold'})
                if legend_position == "right":
                    for text, handle in zip(legend.get_texts(), handles):
                        text.set_color(handle.get_color())
            if len(handles) > 10:
                legend._legend_box.align = "left"
                if len(handles) > 15:
                    legend._ncol = 2
        ax.tick_params(direction='in', length=5, width=1)
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.grid(False)
    if legend_position == "right":
        right_margin = min(0.92, legend_offset + 0.05)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4, right=right_margin)
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4)
    return fig

def calculate_fwhm(x, y, peak_idx):
    peak_y = y[peak_idx]
    half_max = peak_y / 2
    x_left = x[0]
    x_right = x[-1]
    left_idx = peak_idx
    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1
    if left_idx > 0 and y[left_idx] < half_max:
        x_left = np.interp(half_max, [y[left_idx], y[left_idx+1]], [x[left_idx], x[left_idx+1]])
    elif left_idx == 0:
        if y[0] < half_max:
            x_left = np.interp(half_max, [y[0], y[1]], [x[0], x[1]])
        else:
            x_left = x[0]
    right_idx = peak_idx
    while right_idx < len(y) - 1 and y[right_idx] > half_max:
        right_idx += 1
    if right_idx < len(y) - 1 and y[right_idx] < half_max:
        x_right = np.interp(half_max, [y[right_idx], y[right_idx-1]], [x[right_idx], x[right_idx-1]])
    elif right_idx == len(y) - 1:
        if y[-1] < half_max:
            x_right = np.interp(half_max, [y[-2], y[-1]], [x[-2], x[-1]])
        else:
            x_right = x[-1]
    fwhm = x_right - x_left
    return max(fwhm, 0)

def voigt_height(amplitude, sigma, gamma):
    return (amplitude / (sigma * np.sqrt(2 * np.pi))) * \
           np.real(wofz((1j * gamma) / (sigma * np.sqrt(2))))

def voigt_amplitude_from_height(height, sigma, gamma):
    return height * sigma * np.sqrt(2 * np.pi) / \
        np.real(wofz((1j * gamma) / (sigma * np.sqrt(2))))

def _peak_detection_params():
    return {
        'height_factor': st.session_state.peak_det_height_factor,
        'prominence_factor': st.session_state.peak_det_prominence_factor,
        'min_distance': st.session_state.peak_det_min_distance,
        'min_height_abs': st.session_state.peak_det_min_height_abs,
    }

@st.cache_data
def analyze_peaks_manual_range_cached(spectra_key, x_range, peak_width, spectra_data_pickle,
                                      height_factor, prominence_factor, min_distance):
    spectra_dict = spectra_data_pickle
    results = []
    for name, spec in spectra_dict.items():
        data = spec['data']
        x_full = data['x'].values
        y_full = data['y'].values
        if x_range[0] is not None and x_range[1] is not None:
            mask = (x_full >= x_range[0]) & (x_full <= x_range[1])
            x = x_full[mask]
            y = y_full[mask]
        else:
            x = x_full
            y = y_full
        if len(x) == 0:
            continue
        peaks, properties = find_peaks(y, height=np.max(y)*height_factor,
                                       prominence=np.max(y)*prominence_factor,
                                       distance=min_distance)
        for peak_idx in peaks:
            peak_x = x[peak_idx]
            peak_y = y[peak_idx]
            left_idx = max(0, peak_idx - peak_width)
            right_idx = min(len(x), peak_idx + peak_width)
            area = simpson(y[left_idx:right_idx+1], x[left_idx:right_idx+1])
            fwhm = calculate_fwhm(x, y, peak_idx)
            results.append({
                'Spectrum': name.replace('.txt', ''),
                'Peak position': peak_x,
                'Intensity': peak_y,
                'Area': area,
                'FWHM': fwhm,
                'Include': True
            })
    return pd.DataFrame(results) if results else pd.DataFrame()

def analyze_peaks_manual_range(spectra_dict, x_range, peak_width=20,
                                height_factor=0.05, prominence_factor=0.03, min_distance=5):
    results = []
    for name, spec in spectra_dict.items():
        data = spec['data']
        x_full = data['x'].values
        y_full = data['y'].values
        if x_range[0] is not None and x_range[1] is not None:
            mask = (x_full >= x_range[0]) & (x_full <= x_range[1])
            x = x_full[mask]
            y = y_full[mask]
        else:
            x = x_full
            y = y_full
        if len(x) == 0:
            continue
        peaks, properties = find_peaks(y, height=np.max(y)*height_factor,
                                       prominence=np.max(y)*prominence_factor,
                                       distance=min_distance)
        for peak_idx in peaks:
            peak_x = x[peak_idx]
            peak_y = y[peak_idx]
            left_idx = max(0, peak_idx - peak_width)
            right_idx = min(len(x), peak_idx + peak_width)
            area = simpson(y[left_idx:right_idx+1], x[left_idx:right_idx+1])
            fwhm = calculate_fwhm(x, y, peak_idx)
            results.append({
                'Spectrum': name.replace('.txt', ''),
                'Peak position': peak_x,
                'Intensity': peak_y,
                'Area': area,
                'FWHM': fwhm,
                'Include': True
            })
    return pd.DataFrame(results) if results else pd.DataFrame()

def create_peak_visualization(spectra_dict, x_range, peaks_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    included_peaks = peaks_df[peaks_df['Include'] == True]
    for name, spec in spectra_dict.items():
        data = spec['data']
        x = data['x'].values
        y = data['y'].values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            continue
        color = spec['color']
        ax.plot(x, y, color=color, linewidth=1.5, label=name.replace('.txt', ''), alpha=0.7)
        spec_peaks = included_peaks[included_peaks['Spectrum'] == name.replace('.txt', '')]
        for _, peak in spec_peaks.iterrows():
            ax.axvline(peak['Peak position'], color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax.text(peak['Peak position'], peak['Intensity']*0.8,
                   f"{peak['Peak position']:.1f}", fontsize=8, ha='center', fontweight='bold')
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
    x_a = spectrum_a_data['data']['x'].values
    y_a_raw = spectrum_a_data['data']['y'].values
    x_b = spectrum_b_data['data']['x'].values
    y_b_raw = spectrum_b_data['data']['y'].values
    if not isinstance(x_a, np.ndarray):
        x_a = np.array(x_a)
    if not isinstance(x_b, np.ndarray):
        x_b = np.array(x_b)
    if not isinstance(y_a_raw, np.ndarray):
        y_a_raw = np.array(y_a_raw)
    if not isinstance(y_b_raw, np.ndarray):
        y_b_raw = np.array(y_b_raw)
    y_a_norm = normalize_spectrum(x_a, y_a_raw, norm_method, norm_range)
    y_b_norm = normalize_spectrum(x_b, y_b_raw, norm_method, norm_range)
    common_x_min = max(x_a.min(), x_b.min())
    common_x_max = min(x_a.max(), x_b.max())
    common_x = np.linspace(common_x_min, common_x_max, 2000)
    y_a_interp = np.interp(common_x, x_a, y_a_norm)
    y_b_interp = np.interp(common_x, x_b, y_b_norm)
    if subtract_min_intensity:
        y_a_interp = y_a_interp - y_a_interp.min()
        y_b_interp = y_b_interp - y_b_interp.min()
    y_diff = y_b_interp - y_a_interp
    if smooth_difference:
        y_diff_smoothed = gaussian_filter1d(y_diff, sigma=smooth_sigma)
    else:
        y_diff_smoothed = y_diff
    mean_diff = np.mean(y_diff_smoothed)
    max_abs_diff = np.max(np.abs(y_diff_smoothed))
    rms_diff = np.sqrt(np.mean(y_diff_smoothed**2))
    correlation = pearsonr(y_a_interp, y_b_interp)[0]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(fig_width, fig_height * 1.2),
                                             gridspec_kw={'height_ratios': [1, 0.6]})
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

    if symmetric_scale:
        y_limit = max_abs_diff * 1.1
        ax_bottom.set_ylim(-y_limit, y_limit)
    else:
        y_min = np.min(y_diff_smoothed)
        y_max = np.max(y_diff_smoothed)
        y_margin = (y_max - y_min) * 0.1
        ax_bottom.set_ylim(y_min - y_margin, y_max + y_margin)
    cmap = plt.get_cmap(colormap_name)
    if symmetric_scale:
        norm = plt.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)
    else:
        norm = plt.Normalize(vmin=np.min(y_diff_smoothed), vmax=np.max(y_diff_smoothed))
    ax_bottom.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

    verts = []
    colors_list = []
    for i in range(len(common_x) - 1):
        y1 = y_diff_smoothed[i]
        y2 = y_diff_smoothed[i+1]
        if y1 * y2 >= 0:
            verts.append([(common_x[i], 0), (common_x[i], y1),
                          (common_x[i+1], y2), (common_x[i+1], 0)])
            colors_list.append(cmap(norm((y1 + y2) / 2)))
        else:
            t = -y1 / (y2 - y1)
            x_zero = common_x[i] + t * (common_x[i+1] - common_x[i])
            verts.append([(common_x[i], 0), (common_x[i], y1), (x_zero, 0)])
            colors_list.append(cmap(norm(y1/2)))
            verts.append([(x_zero, 0), (common_x[i+1], y2), (common_x[i+1], 0)])
            colors_list.append(cmap(norm(y2/2)))
    if len(verts) > 0:
        pc = PolyCollection(verts, facecolors=colors_list, alpha=0.6,
                           edgecolors='none', zorder=2)
        ax_bottom.add_collection(pc)

    ax_bottom.plot(common_x, y_diff_smoothed, color='black', linewidth=1.2,
                  alpha=0.8, zorder=3, label='Difference profile')
    if difference_threshold > 0:
        above_threshold = np.abs(y_diff_smoothed) > difference_threshold
        if np.any(above_threshold):
            diff_indices = np.where(above_threshold)[0]
            regions = []
            start_idx = diff_indices[0]
            for i in range(1, len(diff_indices)):
                if diff_indices[i] > diff_indices[i-1] + 1:
                    regions.append((common_x[start_idx], common_x[diff_indices[i-1]]))
                    start_idx = diff_indices[i]
            regions.append((common_x[start_idx], common_x[diff_indices[-1]]))
            y_min, y_max = ax_bottom.get_ylim()
            for start, end in regions:
                ax_bottom.axvspan(start, end, alpha=0.2, color='yellow', zorder=0)
    ax_bottom.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax_bottom.set_ylabel('Intensity Diff.', fontsize=10, fontweight='bold')
    ax_bottom.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax_bottom.grid(True, alpha=0.3, linestyle='--')
    else:
        ax_bottom.grid(False)
    ax_bottom.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    if symmetric_scale:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff))
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(y_diff_smoothed), vmax=np.max(y_diff_smoothed)))
    sm.set_array([])
    pos = ax_bottom.get_position()
    colorbar_offset = legend_offset + 0.50
    cbar_ax = fig.add_axes([pos.x1 + (colorbar_offset - 1.0) * 0.1, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Amplitude', fontsize=9, fontweight='bold')

    if difference_threshold > 0 and np.any(above_threshold):
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='yellow', alpha=0.3, label=f'|Difference| > {difference_threshold:.3f}')]
        ax_bottom.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.subplots_adjust(hspace=0.4)
    if legend_position == "right":
        right_margin = min(0.92, legend_offset + 0.05)
        plt.subplots_adjust(right=right_margin)
    else:
        plt.tight_layout()

    return fig, (mean_diff, max_abs_diff, rms_diff, correlation)

def create_heatmap(spectra_matrix, x_grid, y_values, x_label, y_label,
                   colorbar_label, colormap, interpolation, title,
                   fig_width=10, fig_height=8, log_scale=False, show_grid=True):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    data_matrix = np.array(spectra_matrix)
    if log_scale:
        data_matrix = np.maximum(data_matrix, 1e-10)
        data_matrix = np.log(data_matrix)
        colorbar_label = f"log ({colorbar_label})"
    extent = [y_values[0], y_values[-1], x_grid[0], x_grid[-1]]
    data_matrix_transposed = data_matrix.T
    data_clean = data_matrix_transposed[np.isfinite(data_matrix_transposed)]
    if len(data_clean) > 0:
        vmin = np.nanmin(data_matrix_transposed)
        vmax = np.nanmax(data_matrix_transposed)
    else:
        vmin = 0
        vmax = 1
    im = ax.imshow(data_matrix_transposed,
                   extent=extent,
                   aspect='auto',
                   origin='lower',
                   cmap=colormap,
                   interpolation=interpolation,
                   interpolation_stage='data' if interpolation != 'none' else None,
                   vmin=vmin,
                   vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=20, fontweight='bold')
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
    cbar.ax.tick_params(labelsize=16)
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
    ax.set_xlabel(y_label, fontsize=20, fontweight='bold')
    ax.set_ylabel(x_label, fontsize=20, fontweight='bold')
    ax.tick_params(direction='in', length=5, width=1, labelsize=15)
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    else:
        ax.grid(False)
    plt.tight_layout()
    return fig

@st.cache_data
def prepare_heatmap_data_cached(spectra_names, spectra_hash, spectra_data_pickle, ordered_spectra, heatmap_params, norm_method, norm_range, x_ranges):
    spectra_dict = spectra_data_pickle
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

def prepare_heatmap_data(spectra_dict, ordered_spectra, heatmap_params, norm_method, norm_range, x_ranges):
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

@st.fragment
def marker_controls_fragment(global_min_x, global_max_x):
    pending_idx = st.session_state.pending_marker_index
    pending_line = None
    if pending_idx is not None and 0 <= pending_idx < len(st.session_state.spectral_markers):
        candidate = st.session_state.spectral_markers[pending_idx]
        if candidate.get('pending', False):
            pending_line = candidate

    st.markdown("---")
    st.markdown("#### ✏️ Add New Marker")

    if pending_line is None:
        st.info("📍 **Step 1: Add a vertical line**")
        col1, col2 = st.columns(2)
        with col1:
            marker_position = st.slider(
                "Line position (X value)",
                min_value=global_min_x, max_value=global_max_x,
                value=(global_min_x + global_max_x) / 2,
                step=(global_max_x - global_min_x) / 500,
                key="marker_position_slider"
            )
            st.session_state.spectral_markers_preview_position = marker_position
        with col2:
            marker_color = st.color_picker(
                "Marker color",
                value=st.session_state.spectral_markers_temp_color,
                key="marker_color_picker"
            )
            st.session_state.spectral_markers_temp_color = marker_color
        marker_name = st.text_input(
            "Marker name (optional)",
            value=st.session_state.spectral_markers_temp_name,
            placeholder="e.g., I, Peak 1, Marker A",
            key="marker_name_input"
        )
        st.session_state.spectral_markers_temp_name = marker_name

        if st.button("➕ Add Line", use_container_width=True):
            new_marker = {
                'type': 'line',
                'position': marker_position,
                'width': 0,
                'name': marker_name if marker_name else "",
                'color': marker_color,
                'pending': True
            }
            st.session_state.spectral_markers.append(new_marker)
            st.session_state.pending_marker_index = len(st.session_state.spectral_markers) - 1
            st.session_state.spectral_markers_temp_name = ""
            st.session_state.spectral_markers_update_needed = True
            st.success(f"✅ Line added at position {marker_position:.1f}. Now you can expand it to a region!")
            st.rerun(scope="fragment")

        if st.session_state.spectral_markers:
            confirmed_count = len([m for m in st.session_state.spectral_markers if not m.get('pending', False)])
            st.info(f"ℹ️ You have {confirmed_count} confirmed markers. Add a new line above.")
    else:
        st.info(f"📍 **Step 2: Expand line at {pending_line['position']:.1f} to a region**")
        st.markdown("*Use the slider below to set the region width (half-width on each side)*")
        region_width = st.slider(
            "Region half-width",
            min_value=0.1, max_value=(global_max_x - global_min_x) / 4,
            value=min(5.0, (global_max_x - global_min_x) / 20),
            step=0.1,
            key="region_width_slider"
        )
        st.session_state.spectral_markers_preview_width = region_width
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Region", use_container_width=True):
                idx = st.session_state.pending_marker_index
                if idx is not None and 0 <= idx < len(st.session_state.spectral_markers):
                    st.session_state.spectral_markers[idx]['type'] = 'region'
                    st.session_state.spectral_markers[idx]['width'] = region_width
                    st.session_state.spectral_markers[idx]['pending'] = False
                    st.session_state.spectral_markers_update_needed = True
                    st.success(f"✅ Region added: center={pending_line['position']:.1f}, half-width={region_width:.1f}")
                    st.session_state.spectral_markers_preview_width = 0
                    st.session_state.pending_marker_index = None
                    st.rerun(scope="fragment")
        with col2:
            if st.button("❌ Cancel Region", use_container_width=True):
                idx = st.session_state.pending_marker_index
                if idx is not None and 0 <= idx < len(st.session_state.spectral_markers):
                    st.session_state.spectral_markers[idx]['pending'] = False
                    st.session_state.spectral_markers[idx]['type'] = 'line'
                    st.session_state.spectral_markers_update_needed = True
                    st.info(f"⏹️ Line kept at position {pending_line['position']:.1f} (not expanded to region)")
                    st.session_state.spectral_markers_preview_width = 0
                    st.session_state.pending_marker_index = None
                    st.rerun(scope="fragment")

def create_spectral_markers_plot(spectra_dict, x_label, y_label, offset_step,
                                  fill_area, normalized, use_offset, x_ranges,
                                  subtract_min_intensity, fill_alpha, show_grid,
                                  line_width, fig_width, fig_height, legend_fontsize,
                                  legend_position, legend_offset, markers,
                                  preview_position, preview_width, show_x_values,
                                  is_region_mode=False, show_preview=True):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    handles = []
    labels = []
    spectra_items = list(spectra_dict.items())

    if x_ranges is None or len(x_ranges) == 0:
        for idx, (name, spec) in enumerate(spectra_items):
            data = spec['data']
            x = data['x'].values
            y = data['y'].values
            color = spec['color']
            display_name = name.replace('.txt', '')
            if subtract_min_intensity and normalized:
                if len(y) > 0:
                    y = y - y.min()
                else:
                    y = y
            if use_offset:
                offset = idx * offset_step
            else:
                offset = 0
            y_plot = y + offset
            if fill_area and normalized:
                if len(x) > 0 and len(y_plot) > 0:
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
        for range_idx, (start, end) in enumerate(x_ranges):
            for idx, (name, spec) in enumerate(spectra_items):
                data = spec['data']
                x_full = data['x'].values
                y_full = data['y'].values
                color = spec['color']
                display_name = name.replace('.txt', '')
                mask = (x_full >= start) & (x_full <= end)
                if not np.any(mask):
                    continue
                x = x_full[mask]
                y = y_full[mask]
                if subtract_min_intensity and normalized:
                    y = y - y.min()
                if use_offset:
                    offset = idx * offset_step
                else:
                    offset = 0
                y_plot = y + offset
                if fill_area and normalized:
                    ax.fill_between(x, offset, y_plot, alpha=fill_alpha, color=color)
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                else:
                    line_handle = ax.plot(x, y_plot, color=color, linewidth=line_width, label=display_name if range_idx == 0 else "")
                if range_idx == 0:
                    handles.append(line_handle[0])
                    labels.append(display_name)
            ax.axvline(start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold')

    if handles:
        if legend_position == "right":
            bbox_anchor = (legend_offset, 0.5)
            loc = 'center left'
        elif legend_position == "best":
            bbox_anchor = None
            loc = 'best'
        else:
            bbox_anchor = None
            loc = legend_position
        if use_offset and legend_position == "right":
            reversed_handles = list(reversed(handles))
            reversed_labels = list(reversed(labels))
            legend = ax.legend(reversed_handles, reversed_labels, loc=loc,
                              bbox_to_anchor=bbox_anchor, fontsize=legend_fontsize,
                              frameon=True, edgecolor='black', prop={'weight': 'bold'})
            for text, handle in zip(legend.get_texts(), reversed_handles):
                text.set_color(handle.get_color())
        else:
            legend = ax.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_anchor,
                              fontsize=legend_fontsize, frameon=True, edgecolor='black',
                              prop={'weight': 'bold'})
            if legend_position == "right":
                for text, handle in zip(legend.get_texts(), handles):
                    text.set_color(handle.get_color())
        if len(handles) > 10:
            legend._legend_box.align = "left"
            if len(handles) > 15:
                legend._ncol = 2

    y_min, y_max = ax.get_ylim()
    y_text_position = y_max + (y_max - y_min) * 0.08
    x_visible_min, x_visible_max = ax.get_xlim()

    for marker in markers:
        if marker.get('pending', False):
            continue
        marker_type = marker.get('type', 'line')
        position = marker.get('position')
        width = marker.get('width', 0)
        name = marker.get('name', '')
        color = marker.get('color', '#000000')
        if marker_type == 'line':
            ax.axvline(x=position, color=color, linestyle='--', linewidth=0.6, alpha=1.0)
            if show_x_values and x_visible_min <= position <= x_visible_max:
                label_text = f"{position:.1f}" if not name else f"{name} {position:.1f}"
                ax.text(position, y_text_position, label_text,
                       ha='center', va='bottom', fontsize=8,
                       color=color, fontweight='bold',
                       rotation=45, rotation_mode='anchor')
        elif marker_type == 'region':
            half_width = width / 2
            x_start = position - half_width
            x_end = position + half_width
            ax.axvspan(x_start, x_end, alpha=0.3, color=color)
            ax.axvline(x=x_start, color=color, linestyle='--', linewidth=0.4, alpha=0.7)
            ax.axvline(x=x_end, color=color, linestyle='--', linewidth=0.4, alpha=0.7)
            ax.axvline(x=position, color=color, linestyle='--', linewidth=0.5, alpha=0.5)
            if show_x_values and (x_visible_min <= position <= x_visible_max):
                label_text = f"{position:.1f} ± {half_width:.1f}" if not name else f"{name} {position:.1f} ± {half_width:.1f}"
                ax.text(position, y_text_position, label_text,
                       ha='center', va='bottom', fontsize=8,
                       color=color, fontweight='bold',
                       rotation=45, rotation_mode='anchor')

    if show_preview and preview_position is not None:
        if is_region_mode and preview_width > 0:
            half_width = preview_width / 2
            x_start = preview_position - half_width
            x_end = preview_position + half_width
            ax.axvspan(x_start, x_end, alpha=0.2, color='gray')
            ax.axvline(x=preview_position, color='gray', linestyle='-.', linewidth=1.0, alpha=0.5)
            ax.axvline(x=x_start, color='gray', linestyle='-.', linewidth=0.5, alpha=0.3)
            ax.axvline(x=x_end, color='gray', linestyle='-.', linewidth=0.5, alpha=0.3)
            if show_x_values:
                preview_text = f"Preview: {preview_position:.1f} ± {half_width:.1f}"
                ax.text(preview_position, y_text_position, preview_text,
                       ha='center', va='bottom', fontsize=8,
                       color='gray', fontweight='bold', alpha=0.7,
                       rotation=45, rotation_mode='anchor')
        else:
            ax.axvline(x=preview_position, color='gray', linestyle='-.', linewidth=1.0, alpha=0.5)
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

    if legend_position == "right" and len(handles) > 0:
        estimated_legend_width = min(0.35, 0.15 + (len(handles) * legend_fontsize / 300))
        right_margin = min(0.95, legend_offset + estimated_legend_width)
        plt.tight_layout()
        plt.subplots_adjust(right=right_margin, top=0.92)
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

    return fig

@st.cache_data
def prepare_marker_spectra(filtered_spectra, norm_method, norm_range, x_ranges, subtract_min_intensity):
    normalized_spectra = {}
    for name, spec in filtered_spectra.items():
        data = spec['data']
        y_norm = normalize_spectrum(
            data['x'].values, data['y'].values, norm_method, norm_range, x_ranges
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
    return normalized_spectra

@st.cache_data
def prepare_3d_data(spectra_dict, ordered_spectra, graph3d_params, x_ranges, norm_method, norm_range):
    all_x = []
    for name in ordered_spectra:
        if name in spectra_dict:
            x_vals = spectra_dict[name]['data']['x'].values
            if len(x_vals) > 0:
                all_x.extend(x_vals)
    if not all_x:
        return None, None, None, None, None
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
        return None, None, None, None, None
    common_x = np.linspace(x_min, x_max, 500)
    spectra_matrix = []
    spectra_norm_matrix = []
    y_values = []
    colors = []
    for name in ordered_spectra:
        if name not in spectra_dict or name not in graph3d_params:
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
        param_value = graph3d_params[name]
        y_values.append(param_value)
        colors.append(spectra_dict[name]['color'])
    if not spectra_matrix:
        return None, None, None, None, None
    return np.array(spectra_matrix), np.array(spectra_norm_matrix), common_x, np.array(y_values), colors

def create_3d_surface_plot(X, Y, Z, x_label, y_label, z_label, colormap,
                           azimuth, elevation, show_grid=True, fig_width=10, fig_height=8,
                           title=""):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, projection='3d')
    X_grid, Y_grid = np.meshgrid(X, Y)
    surf = ax.plot_surface(X_grid, Y_grid, Z, cmap=colormap,
                          linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.view_init(elev=elevation, azim=azimuth)
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    else:
        ax.grid(False)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label(z_label, fontsize=11, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_3d_line_plot(X, Y, Z, colors, x_label, y_label, z_label,
                        azimuth, elevation, line_width=1.5, show_grid=True,
                        fig_width=10, fig_height=8, title=""):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(Z)):
        y_pos = np.full_like(X, Y[i])
        ax.plot(X, y_pos, Z[i], color=colors[i], linewidth=line_width, alpha=0.8)
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.view_init(elev=elevation, azim=azimuth)
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    else:
        ax.grid(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def apply_gradient_colors(ordered_spectra, palette_name):
    if not ordered_spectra:
        return {}
    gradient_colors = {}
    try:
        cmap = plt.get_cmap(palette_name)
        n_spectra = len(ordered_spectra)
        if n_spectra == 1:
            colors_rgba = [cmap(0.5)]
        else:
            colors_rgba = [cmap(i / (n_spectra - 1)) for i in range(n_spectra)]
        for i, name in enumerate(ordered_spectra):
            rgba = colors_rgba[i % len(colors_rgba)]
            r = int(rgba[0] * 255)
            g = int(rgba[1] * 255)
            b = int(rgba[2] * 255)
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            gradient_colors[name] = hex_color
        return gradient_colors
    except Exception as e:
        st.error(f"Error applying gradient colors: {e}")
        return {}

def extract_number_from_filename(filename):
    name = filename.replace('.txt', '')
    numbers = re.findall(r'[-+]?\d*\.?\d+', name)
    if numbers:
        return float(numbers[0])
    return None

def _dataframe_hash(df):
    if df is None:
        return 0
    try:
        return hash((tuple(df['x'].values.tobytes()), tuple(df['y'].values.tobytes())))
    except Exception:
        return hash(str(df.shape))

def _spectra_dict_hash(spectra_dict):
    try:
        items = []
        for name in sorted(spectra_dict.keys()):
            spec = spectra_dict[name]
            df = spec['data']
            items.append((name, _dataframe_hash(df), spec.get('color', '')))
        return hash(tuple(items))
    except Exception:
        return hash(str(len(spectra_dict)))

def fit_voigt_decomposition(xs, ys, threshold, min_dist, min_height, max_iterations,
                             center_vary, fit_method='leastsq', reduce_factor=None,
                             progress_callback=None):
    if not LMFIT_AVAILABLE:
        return None, "lmfit library is not available. Please install: pip install lmfit"
    if len(xs) < 10:
        return None, "Not enough data points for fitting."
    if reduce_factor and reduce_factor > 1:
        xs_fit = xs[::reduce_factor]
        ys_fit = ys[::reduce_factor]
    else:
        xs_fit = xs
        ys_fit = ys
    peak_indices, properties = find_peaks(ys_fit, height=threshold, distance=min_dist)
    num_peaks = len(peak_indices)
    if num_peaks == 0:
        return None, "No peaks detected with the given threshold."
    peak_positions = xs_fit[peak_indices]
    peak_heights = ys_fit[peak_indices]
    results_half = peak_widths(ys_fit, peak_indices, rel_height=0.5)
    width_in_samples = results_half[0]
    dx = np.diff(xs_fit).mean() if len(xs_fit) > 1 else 1.0
    fwhms = width_in_samples * dx
    sigmas = fwhms / (2 * np.sqrt(2 * np.log(2)))
    gammas = fwhms / 2
    model = None
    params = Parameters()
    min_sigma = 1e-6
    min_gamma = 1e-6
    x_span = xs_fit.max() - xs_fit.min()
    max_amplitude = ys_fit.max() * np.pi * x_span / 2
    max_sigma = x_span / 10
    max_gamma = x_span / 10
    for i in range(num_peaks):
        prefix = f'p{i}_'
        voigt = VoigtModel(prefix=prefix)
        pars = voigt.make_params()
        pos = peak_positions[i]
        height = peak_heights[i]
        sigma_guess = sigmas[i] if len(sigmas) > i and np.isfinite(sigmas[i]) and sigmas[i] > 0 else x_span / 50
        gamma_guess = gammas[i] if len(gammas) > i and np.isfinite(gammas[i]) and gammas[i] > 0 else x_span / 50
        sigma_guess = np.clip(sigma_guess, min_sigma, max_sigma)
        gamma_guess = np.clip(gamma_guess, min_gamma, max_gamma)
        try:
            amplitude_guess = voigt_amplitude_from_height(height, sigma_guess, gamma_guess)
        except Exception:
            amplitude_guess = height * 10
        if not np.isfinite(amplitude_guess):
            amplitude_guess = height * 10
        amplitude_guess = np.clip(amplitude_guess, None, max_amplitude)
        try:
            min_amplitude = voigt_amplitude_from_height(min_height, sigma_guess, gamma_guess)
        except Exception:
            min_amplitude = 0
        if not np.isfinite(min_amplitude):
            min_amplitude = 0
        pars[prefix + 'amplitude'].set(value=amplitude_guess, min=min_amplitude, max=max_amplitude, vary=True)
        pars[prefix + 'center'].set(value=pos, min=pos - center_vary, max=pos + center_vary, vary=True)
        pars[prefix + 'sigma'].set(value=sigma_guess, min=min_sigma, max=max_sigma, vary=True)
        pars[prefix + 'gamma'].set(value=gamma_guess, min=min_gamma, max=max_gamma, vary=True)
        if model is None:
            model = voigt
        else:
            model += voigt
        params.update(pars)
    try:
        final_result = model.fit(ys_fit, params, x=xs_fit, method=fit_method)
    except Exception as e:
        return None, f"Initial fitting failed: {e}"
    params = final_result.params.copy()
    negligible_peaks = []
    for param_name in final_result.params:
        if 'amplitude' in param_name:
            prefix = param_name.split('amplitude')[0]
            amplitude = final_result.params[prefix + 'amplitude'].value
            sigma = final_result.params[prefix + 'sigma'].value
            gamma = final_result.params[prefix + 'gamma'].value
            try:
                h = voigt_height(amplitude, sigma, gamma)
            except Exception:
                h = 0
            if not np.isfinite(h) or h < min_height * 1.1:
                negligible_peaks.append(prefix)
    if negligible_peaks:
        for prefix in negligible_peaks:
            for par in list(params.keys()):
                if par.startswith(prefix):
                    del params[par]
        model = None
        prefixes_added = set()
        for param_name in params:
            parts = param_name.split('_')
            if len(parts) < 2:
                continue
            prefix = '_'.join(parts[:-1]) + '_'
            if prefix not in prefixes_added:
                voigt = VoigtModel(prefix=prefix)
                if model is None:
                    model = voigt
                else:
                    model += voigt
                prefixes_added.add(prefix)
        if model is not None:
            try:
                final_result = model.fit(ys_fit, params, x=xs_fit, method=fit_method)
                params = final_result.params.copy()
            except Exception:
                pass
    iteration = 0
    log_lines = []
    log_lines.append(f"Initial fit: {num_peaks} peaks, {len(negligible_peaks)} removed")
    while iteration < max_iterations:
        try:
            residuals = ys_fit - final_result.best_fit
        except Exception:
            break
        residual_peak_indices, _ = find_peaks(residuals, height=threshold, distance=min_dist)
        if len(residual_peak_indices) == 0:
            log_lines.append(f"Iter {iteration+1}: no residual peaks, stop.")
            break
        residual_widths, _, _, _ = peak_widths(residuals, residual_peak_indices, rel_height=0.5)
        residual_fwhms = residual_widths * dx
        model_new = None
        params_new = Parameters()
        for idx, fwhm in zip(residual_peak_indices, residual_fwhms):
            pos = xs_fit[idx]
            height = residuals[idx]
            if height < min_height:
                continue
            sigma_guess = fwhm / (2 * np.sqrt(2 * np.log(2))) if fwhm > 0 else x_span / 50
            gamma_guess = fwhm / 2 if fwhm > 0 else x_span / 50
            sigma_guess = np.clip(sigma_guess, min_sigma, max_sigma)
            gamma_guess = np.clip(gamma_guess, min_gamma, max_gamma)
            try:
                amplitude_guess = voigt_amplitude_from_height(height, sigma_guess, gamma_guess)
            except Exception:
                continue
            if not np.isfinite(amplitude_guess) or amplitude_guess <= 0:
                continue
            amplitude_guess = np.clip(amplitude_guess, None, max_amplitude)
            try:
                min_amplitude = voigt_amplitude_from_height(min_height, sigma_guess, gamma_guess)
            except Exception:
                min_amplitude = 0
            prefix = f'iter{iteration+1}_p{idx}_'
            voigt = VoigtModel(prefix=prefix)
            pars = voigt.make_params()
            pars[prefix + 'amplitude'].set(value=amplitude_guess, min=min_amplitude, max=max_amplitude, vary=True)
            pars[prefix + 'center'].set(value=pos, min=pos - center_vary, max=pos + center_vary, vary=True)
            pars[prefix + 'sigma'].set(value=sigma_guess, min=min_sigma, max=max_sigma, vary=True)
            pars[prefix + 'gamma'].set(value=gamma_guess, min=min_gamma, max=max_gamma, vary=True)
            if model_new is None:
                model_new = voigt
            else:
                model_new += voigt
            params_new.update(pars)
        if model_new is None:
            log_lines.append(f"Iter {iteration+1}: no valid residual peaks, stop.")
            break
        model += model_new
        params.update(params_new)
        params.update(final_result.params)
        try:
            final_result = model.fit(ys_fit, params, x=xs_fit, method=fit_method)
            params = final_result.params.copy()
        except Exception as e:
            log_lines.append(f"Iter {iteration+1}: fitting failed: {e}")
            break
        negligible_peaks = []
        for param_name in final_result.params:
            if 'amplitude' in param_name:
                prefix = param_name.split('amplitude')[0]
                amplitude = final_result.params[prefix + 'amplitude'].value
                sigma = final_result.params[prefix + 'sigma'].value
                gamma = final_result.params[prefix + 'gamma'].value
                try:
                    h = voigt_height(amplitude, sigma, gamma)
                except Exception:
                    h = 0
                if not np.isfinite(h) or h < min_height * 1.1:
                    negligible_peaks.append(prefix)
        if negligible_peaks:
            for prefix in negligible_peaks:
                for par in list(params.keys()):
                    if par.startswith(prefix):
                        del params[par]
            model = None
            prefixes_added = set()
            for param_name in params:
                parts = param_name.split('_')
                if len(parts) < 2:
                    continue
                prefix = '_'.join(parts[:-1]) + '_'
                if prefix not in prefixes_added:
                    voigt = VoigtModel(prefix=prefix)
                    if model is None:
                        model = voigt
                    else:
                        model += voigt
                    prefixes_added.add(prefix)
            if model is not None:
                params.update(final_result.params)
                try:
                    final_result = model.fit(ys_fit, params, x=xs_fit, method=fit_method)
                    params = final_result.params.copy()
                except Exception:
                    pass
        log_lines.append(f"Iter {iteration+1}: added {len(params_new)} residual peaks, removed {len(negligible_peaks)} negligible.")
        if progress_callback is not None:
            try:
                progress_callback(iteration + 1, max_iterations, log_lines[-1])
            except Exception:
                pass
        iteration += 1
    components = []
    seen_prefixes = set()
    for param_name in final_result.params:
        if 'amplitude' in param_name:
            prefix = param_name.split('amplitude')[0]
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            try:
                amplitude = final_result.params[prefix + 'amplitude'].value
                center = final_result.params[prefix + 'center'].value
                sigma = final_result.params[prefix + 'sigma'].value
                gamma = final_result.params[prefix + 'gamma'].value
                amplitude_err = final_result.params[prefix + 'amplitude'].stderr
                center_err = final_result.params[prefix + 'center'].stderr
                sigma_err = final_result.params[prefix + 'sigma'].stderr
                gamma_err = final_result.params[prefix + 'gamma'].stderr
            except Exception:
                continue
            try:
                height = voigt_height(amplitude, sigma, gamma)
            except Exception:
                height = 0
            fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * sigma
            fwhm_lorentz = 2 * gamma
            fwhm_voigt = 0.5346 * fwhm_lorentz + np.sqrt(0.2166 * fwhm_lorentz ** 2 + fwhm_gauss ** 2)
            area = amplitude
            eta = fwhm_lorentz / (fwhm_lorentz + fwhm_gauss) if (fwhm_lorentz + fwhm_gauss) > 0 else 0
            components.append({
                'Component': prefix.rstrip('_'),
                'Center (cm⁻¹)': center,
                'Height': height,
                'Amplitude': amplitude,
                'Sigma (cm⁻¹)': sigma,
                'Gamma (cm⁻¹)': gamma,
                'FWHM (cm⁻¹)': fwhm_voigt,
                'Area': area,
                'Lorentz fraction': eta,
                'Center err': center_err if center_err is not None else np.nan,
                'Height err': amplitude_err if amplitude_err is not None else np.nan,
                'FWHM err': sigma_err if sigma_err is not None else np.nan,
            })
    components_df = pd.DataFrame(components)
    if not components_df.empty:
        components_df = components_df.sort_values('Center (cm⁻¹)').reset_index(drop=True)
    fit_info = {
        'n_components': len(components_df),
        'reduced_chi_square': final_result.redchi if hasattr(final_result, 'redchi') else None,
        'aic': final_result.aic if hasattr(final_result, 'aic') else None,
        'bic': final_result.bic if hasattr(final_result, 'bic') else None,
        'nfev': final_result.nfev if hasattr(final_result, 'nfev') else None,
        'success': final_result.success if hasattr(final_result, 'success') else None,
        'message': final_result.message if hasattr(final_result, 'message') else '',
        'log': log_lines,
        'xs': xs_fit,
        'ys': ys_fit,
        'best_fit': final_result.best_fit,
        'best_fit_full': final_result,
    }
    return (final_result, components_df, fit_info), None

def plot_voigt_fit(xs, ys, best_fit_result, components_df, title, x_label, y_label,
                   show_components=True, show_residuals=True, show_grid=True,
                   fig_width=14, fig_height=9):
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height),
                                       gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax2 = None
    ax1.plot(xs, ys, lw=1.2, label='Data', color='black', alpha=0.85)
    ax1.plot(xs, best_fit_result.best_fit, label='Voigt fit', color='red', lw=2, ls='--')
    if show_components:
        try:
            comps = best_fit_result.eval_components(x=xs)
            cmap = plt.get_cmap('tab20')
            i = 0
            for key, comp_y in comps.items():
                color = cmap(i % 20)
                ax1.plot(xs, comp_y, ls='-', alpha=0.7, lw=1.0, color=color, label=f'  {key.rstrip("_")}')
                i += 1
        except Exception:
            pass
    ax1.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8, frameon=True, edgecolor='black', prop={'weight': 'bold'}, ncol=2)
    ax1.tick_params(direction='in', length=5, width=1)
    if show_grid:
        ax1.grid(True, alpha=0.3, linestyle='--')
    if ax2 is not None:
        residuals = ys - best_fit_result.best_fit
        ax2.plot(xs, residuals, color='blue', lw=1.0, label='Residuals')
        ax2.axhline(0, color='black', linestyle='--', lw=0.8)
        ax2.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax2.tick_params(direction='in', length=5, width=1)
        if show_grid:
            ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=9)
    plt.tight_layout()
    return fig

@st.fragment
def render_combined_tab(cached, filtered_spectra, x_ranges):
    st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
    st.subheader("Comprehensive Spectra Analysis")
    st.markdown("*All visualization modes combined for comprehensive spectral comparison*")

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
    fig_width = cached['fig_width']
    fig_height = cached['fig_height']

    normalized_spectra = {}
    for name, spec in filtered_spectra.items():
        data = spec['data']
        y_norm = normalize_spectrum(
            data['x'].values, data['y'].values, norm_method, norm_range, x_ranges
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

    st.info("💡 For heatmap visualization, use the **Heatmaps** tab.")

    st.markdown('</div>', unsafe_allow_html=True)

@st.fragment
def render_peak_tab(cached, filtered_spectra):
    st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
    st.subheader("Peak Detection and Analysis")
    st.markdown("*Select spectral range manually for precise peak analysis*")

    peak_enabled = st.checkbox(
        "Enable advanced peak analysis",
        value=st.session_state.peak_enabled,
        key="peak_enabled_checkbox"
    )
    st.session_state.peak_enabled = peak_enabled

    if not peak_enabled:
        st.info("🔍 Enable advanced peak analysis above to detect and analyze peaks in your spectra.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if not filtered_spectra:
        st.warning("No spectra loaded.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    peak_width = st.slider(
        "Peak width for area calculation (points)",
        min_value=5, max_value=100, value=20, step=5,
        key="peak_width_slider"
    )

    x_label = cached['x_label']
    y_label = cached['y_label']

    st.markdown("#### 📊 Select Analysis Range")
    st.markdown("Use the sliders below to select left and right boundaries for peak analysis")
    st.caption(f"Peak detection parameters (from sidebar): height={st.session_state.peak_det_height_factor:.3f}×max, "
               f"prominence={st.session_state.peak_det_prominence_factor:.3f}×max, distance={st.session_state.peak_det_min_distance}")

    all_x = []
    for spec in filtered_spectra.values():
        all_x.extend(spec['data']['x'].values)
    global_min_x = float(np.min(all_x))
    global_max_x = float(np.max(all_x))

    col1, col2 = st.columns(2)
    with col1:
        left_boundary = st.slider(
            "Left boundary (cm⁻¹)",
            min_value=global_min_x, max_value=global_max_x,
            value=global_min_x,
            step=(global_max_x - global_min_x) / 100,
            key="left_boundary"
        )
    with col2:
        right_boundary = st.slider(
            "Right boundary (cm⁻¹)",
            min_value=global_min_x, max_value=global_max_x,
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
                    filtered_spectra, manual_range, peak_width,
                    height_factor=st.session_state.peak_det_height_factor,
                    prominence_factor=st.session_state.peak_det_prominence_factor,
                    min_distance=st.session_state.peak_det_min_distance,
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
                        "Include?", help="Check to include this peak in analysis", default=True
                    ),
                    'temp_id': None
                },
                disabled=['Spectrum', 'Peak position', 'Intensity', 'Area', 'FWHM'],
                hide_index=True, use_container_width=True, key="peak_editor"
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
            fig_peaks = create_peak_visualization(filtered_spectra, st.session_state.peak_analysis_x_range, peaks_df)
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

    st.markdown('</div>', unsafe_allow_html=True)

@st.fragment
def render_correlation_tab(cached, filtered_spectra, ordered_spectra):
    st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
    st.subheader("Parameter Correlation Analysis")
    st.markdown("*Correlate spectral features (Intensity, Area, Position, FWHM) with experimental parameters*")

    correlation_enabled = st.checkbox(
        "Enable correlation analysis",
        value=st.session_state.correlation_enabled,
        key="correlation_enabled_checkbox"
    )
    st.session_state.correlation_enabled = correlation_enabled

    if not correlation_enabled:
        st.info("📊 Enable correlation analysis above to assign parameter values and analyze correlations.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if not ordered_spectra:
        st.warning("No spectra loaded.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("#### 📊 Assign parameter values")
    param_values = {}
    for name in ordered_spectra:
        param_values[name] = st.number_input(
            f"Value for {name.replace('.txt', '')}",
            value=float(len(param_values) + 1),
            step=1.0,
            key=f"corr_param_{name}"
        )
    param_label = st.text_input("Parameter label", value="Sample number", key="corr_param_label")

    if not (st.session_state.get('correlation_ready', False) and st.session_state.get('correlation_peaks_df') is not None):
        st.info("📊 Please run peak analysis in the 'Advanced Peak Analysis' tab first to obtain peak data for correlation")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    peaks_df = st.session_state.correlation_peaks_df
    peaks_df = peaks_df[peaks_df['Include'] == True]

    if peaks_df.empty:
        st.warning("⚠️ No peaks are currently included. Please check at least one peak in the Peak Analysis tab.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

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

    if not param_list:
        st.info("ℹ️ No matching peaks found for correlation analysis. Make sure peaks are detected and included.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

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

    st.markdown('</div>', unsafe_allow_html=True)

@st.fragment
def render_compare_tab(cached, filtered_spectra, ordered_spectra):
    st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
    st.subheader("🔀 Spectral Difference Analysis")
    st.markdown("*Compare two spectra to identify differences and visualize them with heatmaps*")

    if len(ordered_spectra) < 2:
        st.warning("⚠️ Please load at least 2 spectra to use the comparison feature.")
        st.info("Upload multiple .txt files to compare different samples or treatments.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2)
    with col1:
        spectrum_a_name = st.selectbox("Reference Spectrum", options=ordered_spectra, index=0, key="ref_spectrum")
    with col2:
        spectrum_b_name = st.selectbox("Sample Spectrum", options=ordered_spectra, index=min(1, len(ordered_spectra)-1), key="sample_spectrum")

    swap_direction = st.checkbox("Swap difference direction (Sample - Reference)", value=True, key="swap_dir")

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
            index=0, key="diff_colormap"
        )
    with col2:
        smooth_difference = st.checkbox("Apply smoothing to difference profile", value=False, key="smooth_diff")
        smooth_sigma = 1.0
        if smooth_difference:
            smooth_sigma = st.slider("Smoothing sigma", min_value=0.5, max_value=5.0, value=1.5, step=0.5, key="smooth_sigma")

    col1, col2 = st.columns(2)
    with col1:
        symmetric_scale = st.checkbox("Symmetric color scale (centered at zero)", value=True, key="sym_scale")
    with col2:
        difference_threshold = st.number_input(
            "Significance threshold",
            min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.3f",
            help="Regions with absolute difference exceeding this value will be highlighted",
            key="diff_threshold"
        )

    st.markdown("---")

    if spectrum_a_name and spectrum_b_name:
        spectrum_a = filtered_spectra[spectrum_a_name]
        spectrum_b = filtered_spectra[spectrum_b_name]
        name_a = spectrum_a_name.replace('.txt', '')
        name_b = spectrum_b_name.replace('.txt', '')

        with st.spinner("Generating comparison plot..."):
            fig, (mean_diff, max_abs_diff, rms_diff, correlation) = create_comparison_plot(
                spectrum_a, spectrum_b, name_a, name_b,
                cached['x_label'], cached['y_label'], cached['norm_method'], cached['norm_range'],
                cached['norm_offset_step'], cached['fill_area'], cached['fill_alpha'],
                cached['subtract_min_intensity'], cached['show_grid'], cached['line_width'],
                cached['fig_width'], cached['fig_height'],
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
                y_a_norm = normalize_spectrum(x_a, y_a_raw, cached['norm_method'], cached['norm_range'])
                x_b = spectrum_b['data']['x'].values
                y_b_raw = spectrum_b['data']['y'].values
                y_b_norm = normalize_spectrum(x_b, y_b_raw, cached['norm_method'], cached['norm_range'])
                common_x_min = max(x_a.min(), x_b.min())
                common_x_max = min(x_a.max(), x_b.max())
                common_x_exp = np.linspace(common_x_min, common_x_max, 2000)
                y_a_interp_exp = np.interp(common_x_exp, x_a, y_a_norm)
                y_b_interp_exp = np.interp(common_x_exp, x_b, y_b_norm)
                if cached['subtract_min_intensity']:
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

    st.markdown('</div>', unsafe_allow_html=True)

@st.fragment
def render_markers_plot_fragment(spectra, selected_offset_step, selected_fill,
                                  selected_normalized, selected_use_offset, selected_yl,
                                  cached, x_ranges, show_values, show_preview, is_region_mode):
    fig_markers = create_spectral_markers_plot(
        spectra, cached['x_label'], selected_yl,
        selected_offset_step, selected_fill, selected_normalized,
        selected_use_offset, x_ranges, cached['subtract_min_intensity'],
        cached['fill_alpha'], cached['show_grid'], cached['line_width'],
        cached['fig_width'], cached['fig_height'],
        cached['legend_fontsize'], cached['legend_position'], cached['legend_offset'],
        st.session_state.spectral_markers,
        st.session_state.spectral_markers_preview_position,
        st.session_state.spectral_markers_preview_width,
        show_values,
        is_region_mode,
        show_preview
    )
    st.pyplot(fig_markers)
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

@st.fragment
def render_markers_tab(cached, filtered_spectra):
    st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
    st.subheader("📏 Spectral Markers")
    st.markdown("*Add vertical lines or regions to track spectral features across different conditions*")

    if not filtered_spectra:
        st.warning("⚠️ No spectra loaded. Please load spectra in the sidebar first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    x_ranges = cached['x_ranges']
    normalized_spectra = prepare_marker_spectra(
        filtered_spectra, cached['norm_method'], cached['norm_range'], x_ranges, cached['subtract_min_intensity']
    )

    viz_configs = [
        (filtered_spectra, 0, False, False, False, cached['y_label']),
        (normalized_spectra, 0, False, True, False, f"Normalized {cached['y_label']}"),
        (filtered_spectra, cached['raw_offset_step'], False, False, True, cached['y_label']),
        (normalized_spectra, cached['norm_offset_step'], cached['fill_area'], True, True, f"Normalized {cached['y_label']}")
    ]

    plot_labels = [
        "1. Raw Spectra (No Offset)",
        "2. Normalized Spectra (No Offset)",
        "3. Raw Spectra (With Offset)",
        "4. Normalized Spectra (With Offset)"
    ]

    st.markdown("#### Select plot for marker placement")
    selected_plot_idx = st.radio(
        "Choose a plot type:",
        options=range(len(plot_labels)),
        format_func=lambda x: plot_labels[x],
        index=st.session_state.spectral_markers_selected_plot,
        key="markers_plot_select"
    )
    st.session_state.spectral_markers_selected_plot = selected_plot_idx

    selected_spectra, selected_offset_step, selected_fill, selected_normalized, selected_use_offset, selected_yl = viz_configs[selected_plot_idx]

    all_x = []
    for spec in selected_spectra.values():
        all_x.extend(spec['data']['x'].values)
    if not all_x:
        st.warning("No data available in selected spectra")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    global_min_x = float(np.min(all_x))
    global_max_x = float(np.max(all_x))

    pending_idx = st.session_state.pending_marker_index
    pending_line = None
    if pending_idx is not None and 0 <= pending_idx < len(st.session_state.spectral_markers):
        candidate = st.session_state.spectral_markers[pending_idx]
        if candidate.get('pending', False):
            pending_line = candidate

    st.markdown("---")
    st.markdown("#### ✏️ Add New Marker")

    if pending_line is None:
        st.info("📍 **Step 1: Add a vertical line** — move the slider to preview the position, then click **Add Line**.")
        col1, col2 = st.columns(2)
        with col1:
            marker_position = st.slider(
                "Line position (X value)",
                min_value=global_min_x, max_value=global_max_x,
                value=(global_min_x + global_max_x) / 2,
                step=(global_max_x - global_min_x) / 500,
                key="marker_position_slider"
            )
            st.session_state.spectral_markers_preview_position = marker_position
            st.session_state.spectral_markers_preview_width = 0
        with col2:
            marker_color = st.color_picker(
                "Marker color",
                value=st.session_state.spectral_markers_temp_color,
                key="marker_color_picker"
            )
            st.session_state.spectral_markers_temp_color = marker_color
        marker_name = st.text_input(
            "Marker name (optional)",
            value=st.session_state.spectral_markers_temp_name,
            placeholder="e.g., I, Peak 1, Marker A",
            key="marker_name_input"
        )
        st.session_state.spectral_markers_temp_name = marker_name

        if st.button("➕ Add Line", use_container_width=True, key="add_line_btn"):
            new_marker = {
                'type': 'line',
                'position': marker_position,
                'width': 0,
                'name': marker_name if marker_name else "",
                'color': marker_color,
                'pending': True
            }
            st.session_state.spectral_markers.append(new_marker)
            st.session_state.pending_marker_index = len(st.session_state.spectral_markers) - 1
            st.session_state.spectral_markers_temp_name = ""
            st.session_state.spectral_markers_update_needed = True
            st.rerun(scope="fragment")

        if st.session_state.spectral_markers:
            confirmed_count = len([m for m in st.session_state.spectral_markers if not m.get('pending', False)])
            st.info(f"ℹ️ You have {confirmed_count} confirmed markers.")
    else:
        st.info(f"📍 **Step 2: Expand line at {pending_line['position']:.1f} to a region** — move the slider to preview the region width.")
        region_width = st.slider(
            "Region half-width",
            min_value=0.1, max_value=(global_max_x - global_min_x) / 4,
            value=min(5.0, (global_max_x - global_min_x) / 20),
            step=0.1,
            key="region_width_slider"
        )
        st.session_state.spectral_markers_preview_position = pending_line['position']
        st.session_state.spectral_markers_preview_width = region_width

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Region", use_container_width=True, key="confirm_region_btn"):
                idx = st.session_state.pending_marker_index
                if idx is not None and 0 <= idx < len(st.session_state.spectral_markers):
                    st.session_state.spectral_markers[idx]['type'] = 'region'
                    st.session_state.spectral_markers[idx]['width'] = region_width
                    st.session_state.spectral_markers[idx]['pending'] = False
                    st.session_state.spectral_markers_update_needed = True
                    st.session_state.spectral_markers_preview_width = 0
                    st.session_state.pending_marker_index = None
                    st.rerun(scope="fragment")
        with col2:
            if st.button("❌ Cancel Region", use_container_width=True, key="cancel_region_btn"):
                idx = st.session_state.pending_marker_index
                if idx is not None and 0 <= idx < len(st.session_state.spectral_markers):
                    st.session_state.spectral_markers[idx]['pending'] = False
                    st.session_state.spectral_markers[idx]['type'] = 'line'
                    st.session_state.spectral_markers_update_needed = True
                    st.session_state.spectral_markers_preview_width = 0
                    st.session_state.pending_marker_index = None
                    st.rerun(scope="fragment")

    st.markdown("---")
    st.markdown("#### 📊 Plot with Markers")
    st.markdown("*Preview (gray dashed) updates live as you move the slider. Confirmed markers are shown in their assigned colors.*")

    col1, col2 = st.columns(2)
    with col1:
        show_values = st.checkbox(
            "Show X values on plot",
            value=st.session_state.spectral_markers_show_values,
            key="markers_show_values"
        )
        st.session_state.spectral_markers_show_values = show_values
    with col2:
        show_preview = st.checkbox(
            "Show preview line/region (during editing)",
            value=st.session_state.spectral_markers_show_preview,
            key="markers_show_preview"
        )
        st.session_state.spectral_markers_show_preview = show_preview
        if not show_preview:
            st.caption("🔒 Preview hidden. All markers are confirmed.")

    is_region_mode = pending_line is not None
    preview_position = st.session_state.spectral_markers_preview_position
    preview_width = st.session_state.spectral_markers_preview_width

    if pending_line is not None:
        preview_position = pending_line['position']
    elif preview_position is None:
        preview_position = (global_min_x + global_max_x) / 2

    fig_markers = create_spectral_markers_plot(
        selected_spectra, cached['x_label'], selected_yl,
        selected_offset_step, selected_fill, selected_normalized,
        selected_use_offset, x_ranges, cached['subtract_min_intensity'],
        cached['fill_alpha'], cached['show_grid'], cached['line_width'],
        cached['fig_width'], cached['fig_height'],
        cached['legend_fontsize'], cached['legend_position'], cached['legend_offset'],
        st.session_state.spectral_markers,
        preview_position,
        preview_width,
        show_values,
        is_region_mode,
        show_preview
    )
    st.pyplot(fig_markers)
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

    if st.session_state.spectral_markers_update_needed:
        st.session_state.spectral_markers_update_needed = False

    if st.session_state.spectral_markers:
        st.markdown("---")
        st.markdown("#### 📋 Existing Markers")
        markers_data = []
        for i, marker in enumerate(st.session_state.spectral_markers):
            if marker.get('pending', False):
                continue
            marker_type = marker['type']
            position = marker['position']
            width = marker['width'] if marker_type == 'region' else '-'
            name = marker['name'] if marker['name'] else ''
            color = marker['color']
            markers_data.append({
                '#': i + 1, 'Name': name, 'Type': marker_type.capitalize(),
                'Position': f"{position:.1f}",
                'Width': f"{width:.1f}" if width != '-' else '-',
                'Color': color, 'index': i
            })
        if markers_data:
            df_markers = pd.DataFrame(markers_data)
            with st.container():
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
                        st.markdown(f"<div style='width:20px;height:20px;background-color:{row['Color']};border:1px solid #ccc;border-radius:3px;'></div>", unsafe_allow_html=True)
                    with col7:
                        if st.button("🗑️", key=f"delete_marker_{row['index']}"):
                            st.session_state.spectral_markers.pop(row['index'])
                            st.session_state.pending_marker_index = None
                            st.session_state.spectral_markers_update_needed = True
                            st.rerun(scope="fragment")
                if st.button("🗑️ Clear All Markers", use_container_width=True, key="clear_all_markers_btn"):
                    st.session_state.spectral_markers = []
                    st.session_state.spectral_markers_preview_position = None
                    st.session_state.spectral_markers_preview_width = 0
                    st.session_state.pending_marker_index = None
                    st.session_state.spectral_markers_update_needed = True
                    st.rerun(scope="fragment")
        else:
            st.info("No confirmed markers yet. Use the controls above to add lines or regions.")
    else:
        st.info("No markers added yet. Use the controls above to add lines or regions.")

    st.markdown('</div>', unsafe_allow_html=True)

@st.fragment
def render_3d_tab(cached, filtered_spectra, ordered_spectra):
    st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
    st.subheader("🎲 3D Visualization")
    st.markdown("*Three-dimensional representation of spectral evolution as function of parameter*")

    enable_3d = st.checkbox(
        "3D Graphs - Create 3D graphs",
        value=st.session_state.enable_3d_graphs,
        key="enable_3d_checkbox"
    )
    st.session_state.enable_3d_graphs = enable_3d

    if not enable_3d:
        st.info("🎲 Enable 3D graphs above to configure and generate 3D visualizations.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("#### 📐 3D Axis Labels")
    col1, col2, col3 = st.columns(3)
    with col1:
        graph3d_x_label = st.text_input("X-axis label (Raman shift)",
                                        value=st.session_state.graph3d_x_label, key="graph3d_x_label_input")
        st.session_state.graph3d_x_label = graph3d_x_label
    with col2:
        graph3d_y_label = st.text_input("Y-axis label (Parameter)",
                                        value=st.session_state.graph3d_y_label, key="graph3d_y_label_input")
        st.session_state.graph3d_y_label = graph3d_y_label
    with col3:
        graph3d_z_label = st.text_input("Z-axis label (Intensity)",
                                        value=st.session_state.graph3d_z_label, key="graph3d_z_label_input")
        st.session_state.graph3d_z_label = graph3d_z_label

    st.markdown("#### 📍 Parameter Values for Y-axis")
    st.markdown("*Assign numeric values to each spectrum for the Y-axis*")

    use_filename_values_3d = st.checkbox(
        "📄 Extract numbers from filenames for Y-axis",
        value=False, key="use_filename_values_3d",
        help="Automatically extract numeric values from filenames"
    )

    if use_filename_values_3d:
        for name in ordered_spectra:
            extracted_value = extract_number_from_filename(name)
            if extracted_value is not None:
                st.session_state.graph3d_params[name] = extracted_value
            elif name not in st.session_state.graph3d_params:
                st.session_state.graph3d_params[name] = 0.0

    graph3d_params_temp = {}
    for name in ordered_spectra:
        display_name = name.replace('.txt', '')
        default_value = st.session_state.graph3d_params.get(name, len(graph3d_params_temp) + 1.0)
        graph3d_params_temp[name] = st.number_input(
            f"{display_name}", value=default_value, step=0.01, format="%.2f", key=f"graph3d_{name}"
        )

    if use_filename_values_3d:
        st.markdown("#### 📋 Extracted values from filenames:")
        extracted_data = []
        for name in ordered_spectra:
            value = extract_number_from_filename(name)
            if value is not None:
                extracted_data.append({'Filename': name, 'Extracted Value': value})
            else:
                extracted_data.append({'Filename': name, 'Extracted Value': '❌ No number found'})
        extracted_df = pd.DataFrame(extracted_data)
        st.dataframe(extracted_df, use_container_width=True, hide_index=True)

    st.session_state.graph3d_params = graph3d_params_temp

    st.markdown("#### 🎯 Viewing Angle Settings")
    col1, col2 = st.columns(2)
    with col1:
        graph3d_azimuth = st.slider(
            "Azimuth (rotation)", min_value=-180, max_value=180,
            value=st.session_state.graph3d_azimuth, step=5, key="graph3d_azimuth_slider"
        )
        st.session_state.graph3d_azimuth = graph3d_azimuth
    with col2:
        graph3d_elevation = st.slider(
            "Elevation (height)", min_value=0, max_value=90,
            value=st.session_state.graph3d_elevation, step=5, key="graph3d_elevation_slider"
        )
        st.session_state.graph3d_elevation = graph3d_elevation

    if st.button("🚀 Create Graphs", use_container_width=True, key="create_3d_graphs_button"):
        with st.spinner("Generating 3D graphs..."):
            spectra_matrix, spectra_norm_matrix, x_grid, y_values, colors = prepare_3d_data(
                filtered_spectra, ordered_spectra, graph3d_params_temp,
                cached['x_ranges'], cached['norm_method'], cached['norm_range']
            )
            if spectra_matrix is not None and len(y_values) > 0:
                sorted_data = sorted(
                    [(y_values[i], i) for i in range(len(y_values))], key=lambda x: x[0]
                )
                sorted_indices = [item[1] for item in sorted_data]
                spectra_matrix = spectra_matrix[sorted_indices]
                spectra_norm_matrix = spectra_norm_matrix[sorted_indices]
                y_values = np.array([y_values[i] for i in sorted_indices])
                colors = [colors[i] for i in sorted_indices]
                st.session_state.graph3d_created = True
                st.session_state.graph3d_spectra_matrix = spectra_matrix
                st.session_state.graph3d_spectra_norm_matrix = spectra_norm_matrix
                st.session_state.graph3d_x_grid = x_grid
                st.session_state.graph3d_y_values = y_values
                st.session_state.graph3d_colors = colors
                st.success(f"✅ 3D graphs generated! {len(y_values)} spectra prepared.")
            else:
                st.error("❌ Failed to prepare data for 3D graphs. Check that all spectra are valid.")
                st.session_state.graph3d_created = False
        st.rerun(scope="fragment")

    if st.session_state.get('graph3d_created', False):
        spectra_matrix = st.session_state.get('graph3d_spectra_matrix')
        spectra_norm_matrix = st.session_state.get('graph3d_spectra_norm_matrix')
        x_grid = st.session_state.get('graph3d_x_grid')
        y_values = st.session_state.get('graph3d_y_values')
        colors = st.session_state.get('graph3d_colors')
        graph3d_x_label = st.session_state.get('graph3d_x_label', 'Raman shift (cm⁻¹)')
        graph3d_y_label = st.session_state.get('graph3d_y_label', 'Parameter')
        graph3d_z_label = st.session_state.get('graph3d_z_label', 'Intensity (a.u.)')
        graph3d_azimuth = st.session_state.get('graph3d_azimuth', -60)
        graph3d_elevation = st.session_state.get('graph3d_elevation', 30)
        heatmap_colormap = st.session_state.get('heatmap_colormap', 'viridis')

        if spectra_matrix is not None and x_grid is not None and y_values is not None:
            st.markdown("#### 3D Surface Plot (Original Intensity)")
            fig_surface_orig = create_3d_surface_plot(
                x_grid, y_values, spectra_matrix,
                graph3d_x_label, graph3d_y_label, graph3d_z_label,
                heatmap_colormap, graph3d_azimuth, graph3d_elevation,
                show_grid=cached['show_grid'],
                fig_width=cached['fig_width'] + 2, fig_height=cached['fig_height'] + 1,
                title="3D Surface Plot - Original Intensity"
            )
            st.pyplot(fig_surface_orig)
            buf = BytesIO()
            fig_surface_orig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem;">
                <a href="data:image/png;base64,{b64}" download="3d_surface_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   color: white; border: none; border-radius: 8px; 
                                   padding: 0.4rem 1rem; cursor: pointer; font-size: 0.9rem;">
                        📥 Download Surface Plot (Original) (PNG, 600 dpi)
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)
            plt.close(fig_surface_orig)

            st.markdown("#### 3D Surface Plot (Normalized Intensity)")
            fig_surface_norm = create_3d_surface_plot(
                x_grid, y_values, spectra_norm_matrix,
                graph3d_x_label, graph3d_y_label, f"Normalized {graph3d_z_label}",
                heatmap_colormap, graph3d_azimuth, graph3d_elevation,
                show_grid=cached['show_grid'],
                fig_width=cached['fig_width'] + 2, fig_height=cached['fig_height'] + 1,
                title="3D Surface Plot - Normalized Intensity"
            )
            st.pyplot(fig_surface_norm)
            buf = BytesIO()
            fig_surface_norm.savefig(buf, format='png', dpi=600, bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem;">
                <a href="data:image/png;base64,{b64}" download="3d_surface_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   color: white; border: none; border-radius: 8px; 
                                   padding: 0.4rem 1rem; cursor: pointer; font-size: 0.9rem;">
                        📥 Download Surface Plot (Normalized) (PNG, 600 dpi)
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)
            plt.close(fig_surface_norm)

            st.markdown("#### 3D Line Plot (Original Intensity)")
            fig_line_orig = create_3d_line_plot(
                x_grid, y_values, spectra_matrix, colors,
                graph3d_x_label, graph3d_y_label, graph3d_z_label,
                graph3d_azimuth, graph3d_elevation,
                line_width=cached['line_width'], show_grid=cached['show_grid'],
                fig_width=cached['fig_width'] + 2, fig_height=cached['fig_height'] + 1,
                title="3D Line Plot - Original Intensity"
            )
            st.pyplot(fig_line_orig)
            buf = BytesIO()
            fig_line_orig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem;">
                <a href="data:image/png;base64,{b64}" download="3d_line_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   color: white; border: none; border-radius: 8px; 
                                   padding: 0.4rem 1rem; cursor: pointer; font-size: 0.9rem;">
                        📥 Download Line Plot (Original) (PNG, 600 dpi)
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)
            plt.close(fig_line_orig)

            st.markdown("#### 3D Line Plot (Normalized Intensity)")
            fig_line_norm = create_3d_line_plot(
                x_grid, y_values, spectra_norm_matrix, colors,
                graph3d_x_label, graph3d_y_label, f"Normalized {graph3d_z_label}",
                graph3d_azimuth, graph3d_elevation,
                line_width=cached['line_width'], show_grid=cached['show_grid'],
                fig_width=cached['fig_width'] + 2, fig_height=cached['fig_height'] + 1,
                title="3D Line Plot - Normalized Intensity"
            )
            st.pyplot(fig_line_norm)
            buf = BytesIO()
            fig_line_norm.savefig(buf, format='png', dpi=600, bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem;">
                <a href="data:image/png;base64,{b64}" download="3d_line_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   color: white; border: none; border-radius: 8px; 
                                   padding: 0.4rem 1rem; cursor: pointer; font-size: 0.9rem;">
                        📥 Download Line Plot (Normalized) (PNG, 600 dpi)
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)
            plt.close(fig_line_norm)

            st.markdown("---")
            st.markdown("#### 📊 3D Graph Parameters Used:")
            param_3d_df = pd.DataFrame({
                'Spectrum': [name.replace('.txt', '') for name in ordered_spectra if name in st.session_state.graph3d_params],
                graph3d_y_label: [st.session_state.graph3d_params[name] for name in ordered_spectra if name in st.session_state.graph3d_params]
            })
            st.dataframe(param_3d_df, use_container_width=True)
            st.caption(f"Viewing angle: Azimuth={graph3d_azimuth}°, Elevation={graph3d_elevation}° | Colormap: {heatmap_colormap}")
        else:
            st.warning("⚠️ 3D graph data not available. Please click 'Create Graphs' above.")

    st.markdown('</div>', unsafe_allow_html=True)

@st.fragment
def render_heatmap_tab(cached, filtered_spectra, ordered_spectra):
    st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
    st.subheader("🔥 Heatmap Visualization")
    st.markdown("*Spectral evolution heatmaps showing intensity distribution as function of parameter*")

    if not ordered_spectra:
        st.warning("No spectra loaded.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("### 📊 Heatmap Parameters")
    st.markdown("*Assign numeric values (temperature, concentration, etc.) to each spectrum for heatmap visualization*")

    heatmap_param_type = st.selectbox(
        "Parameter type",
        options=["Temperature (°C)", "Concentration (x)", "Custom"],
        index=0, key="heatmap_param_type_select"
    )
    heatmap_custom_label = ""
    if heatmap_param_type == "Custom":
        heatmap_custom_label = st.text_input("Custom parameter label", value="Parameter", key="heatmap_custom_label")

    if heatmap_param_type == "Temperature (°C)":
        heatmap_y_label = "Temperature (°C)"
    elif heatmap_param_type == "Concentration (x)":
        heatmap_y_label = "Concentration (x)"
    else:
        heatmap_y_label = heatmap_custom_label if heatmap_custom_label else "Parameter"

    st.markdown("#### Assign values to spectra:")
    use_filename_values = st.checkbox(
        "📄 Assign as the file name (extract numbers from filenames)",
        value=False,
        help="Automatically extract numeric values from filenames (e.g., '0.05.txt' → 0.05, 'temp_100.txt' → 100)"
    )

    if use_filename_values:
        for name in ordered_spectra:
            extracted_value = extract_number_from_filename(name)
            if extracted_value is not None:
                st.session_state.heatmap_params[name] = extracted_value
            elif name not in st.session_state.heatmap_params:
                st.session_state.heatmap_params[name] = 0.0

    heatmap_params_temp = {}
    for name in ordered_spectra:
        display_name = name.replace('.txt', '')
        param_key = f"heatmap_{name}"
        default_value = st.session_state.heatmap_params.get(name, len(heatmap_params_temp) + 1.0)
        heatmap_params_temp[name] = st.number_input(
            f"{display_name}", value=default_value, step=0.01, format="%.2f", key=param_key
        )

    if use_filename_values:
        st.markdown("---")
        st.markdown("#### 📋 Extracted values from filenames:")
        extracted_data = []
        for name in ordered_spectra:
            value = extract_number_from_filename(name)
            if value is not None:
                extracted_data.append({'Filename': name, 'Extracted Value': value})
            else:
                extracted_data.append({'Filename': name, 'Extracted Value': '❌ No number found'})
        extracted_df = pd.DataFrame(extracted_data)
        st.dataframe(extracted_df, use_container_width=True, hide_index=True)

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
            index=5, key="heatmap_interpolation_select"
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
            index=0, key="heatmap_colormap_select"
        )

    if st.button("🔄 Apply for heatmaps", use_container_width=True, key="apply_heatmap_button"):
        if use_filename_values:
            for name in ordered_spectra:
                extracted_value = extract_number_from_filename(name)
                if extracted_value is not None:
                    heatmap_params_temp[name] = extracted_value
        st.session_state.heatmap_params = heatmap_params_temp
        st.session_state.heatmap_param_type = heatmap_param_type
        st.session_state.heatmap_interpolation = heatmap_interpolation
        st.session_state.heatmap_colormap = heatmap_colormap
        st.session_state.heatmap_y_label = heatmap_y_label
        st.session_state.heatmap_applied = True
        st.session_state.heatmap_ordered_names = ordered_spectra
        st.session_state.heatmap_x_ranges = cached['x_ranges']

        spectra_matrix, spectra_norm_matrix, x_grid, y_values = prepare_heatmap_data(
            filtered_spectra, ordered_spectra, heatmap_params_temp,
            cached['norm_method'], cached['norm_range'], cached['x_ranges']
        )
        if spectra_matrix is not None and len(y_values) > 0:
            sorted_data = sorted(
                [(y_values[i], i) for i in range(len(y_values))], key=lambda x: x[0]
            )
            sorted_indices = [item[1] for item in sorted_data]
            spectra_matrix = spectra_matrix[sorted_indices]
            spectra_norm_matrix = spectra_norm_matrix[sorted_indices]
            y_values = np.array([y_values[i] for i in sorted_indices])
            st.session_state.heatmap_spectra_matrix = spectra_matrix
            st.session_state.heatmap_spectra_norm_matrix = spectra_norm_matrix
            st.session_state.heatmap_y_values = y_values
        if spectra_matrix is not None:
            st.session_state.heatmap_spectra_matrix = spectra_matrix
            st.session_state.heatmap_spectra_norm_matrix = spectra_norm_matrix
            st.session_state.heatmap_x_grid = x_grid
            st.session_state.heatmap_y_values = y_values
            st.success(f"✅ Heatmap data prepared! {len(y_values)} spectra with {len(x_grid)} points each.")
        else:
            st.error("❌ Failed to prepare heatmap data. Check that all spectra are valid.")
            st.session_state.heatmap_applied = False
        st.rerun(scope="fragment")

    if st.session_state.get('heatmap_applied', False):
        st.markdown('<div class="separator">═══════════════════════════════════════════════════</div>', unsafe_allow_html=True)
        st.markdown("### 🔥 Heatmap Visualization")

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
                cached['x_label'], heatmap_y_label, "Intensity (a.u.)",
                heatmap_colormap, heatmap_interpolation,
                f"Intensity Heatmap: {heatmap_y_label} vs Raman shift",
                fig_width=12, fig_height=8, log_scale=use_log
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
                cached['x_label'], heatmap_y_label, "Normalized Intensity (a.u.)",
                heatmap_colormap, heatmap_interpolation,
                f"Normalized Intensity Heatmap: {heatmap_y_label} vs Raman shift",
                fig_width=12, fig_height=8, log_scale=True
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

            st.markdown("#### 📊 Heatmap Parameters Used:")
            param_df = pd.DataFrame({
                'Spectrum': [name.replace('.txt', '') for name in st.session_state.heatmap_ordered_names if name in st.session_state.heatmap_params],
                heatmap_y_label: [st.session_state.heatmap_params[name] for name in st.session_state.heatmap_ordered_names if name in st.session_state.heatmap_params]
            })
            st.dataframe(param_df, use_container_width=True)
            st.caption(f"Interpolation: {heatmap_interpolation} | Colormap: {heatmap_colormap} | Log scale: {'Yes' if use_log else 'No'} for intensity, Yes for normalized")
        else:
            st.warning("⚠️ Heatmap data not available. Please click 'Apply for heatmaps' above.")

    st.markdown('</div>', unsafe_allow_html=True)

@st.fragment
def render_voigt_tab(cached, filtered_spectra, ordered_spectra):
    st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
    st.subheader("🧬 Spectral Decomposition (Voigt Fitting)")
    st.markdown("*Decompose complex spectra into individual Voigt components with iterative residual analysis*")

    if not LMFIT_AVAILABLE:
        st.error("❌ `lmfit` library is not installed. Please run: `pip install lmfit`")
        st.info("Voigt fitting requires lmfit for nonlinear least-squares optimization.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if not filtered_spectra:
        st.warning("⚠️ No spectra loaded. Please load spectra in the sidebar first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    x_label = cached['x_label']
    y_label = cached['y_label']

    st.markdown("#### 📊 Select Spectrum for Fitting")
    spectrum_to_fit = st.selectbox(
        "Choose a spectrum:",
        options=ordered_spectra,
        index=0,
        key="voigt_spectrum_select"
    )
    st.session_state.voigt_fit_spectrum_name = spectrum_to_fit

    st.markdown("#### 📐 Fitting Range")
    st.markdown("*Uses the range from Peak Analysis if available, or select manually*")

    spec_data = filtered_spectra[spectrum_to_fit]['data']
    all_x = spec_data['x'].values
    global_min_x = float(np.min(all_x))
    global_max_x = float(np.max(all_x))

    inherited_range = st.session_state.peak_analysis_x_range
    inherited_active = (
        inherited_range is not None
        and inherited_range[0] is not None
        and inherited_range[1] is not None
        and inherited_range[0] < inherited_range[1]
    )

    use_inherited = st.checkbox(
        "Use range from Peak Analysis",
        value=inherited_active,
        key="voigt_use_inherited_range",
        disabled=not inherited_active
    )

    if inherited_active and use_inherited:
        left_boundary = inherited_range[0]
        right_boundary = inherited_range[1]
        st.info(f"📌 Using inherited range: {left_boundary:.1f} – {right_boundary:.1f} cm⁻¹")
    else:
        col1, col2 = st.columns(2)
        with col1:
            left_boundary = st.slider(
                "Left boundary (cm⁻¹)",
                min_value=global_min_x, max_value=global_max_x,
                value=global_min_x,
                step=(global_max_x - global_min_x) / 200,
                key="voigt_left_boundary"
            )
        with col2:
            right_boundary = st.slider(
                "Right boundary (cm⁻¹)",
                min_value=global_min_x, max_value=global_max_x,
                value=global_max_x,
                step=(global_max_x - global_min_x) / 200,
                key="voigt_right_boundary"
            )

    if left_boundary >= right_boundary:
        st.warning("⚠️ Left boundary must be less than right boundary")
        fit_range = (None, None)
    else:
        fit_range = (left_boundary, right_boundary)

    st.session_state.voigt_fit_x_range = fit_range

    use_normalized = st.checkbox(
        "Use normalized data for fitting",
        value=st.session_state.voigt_use_normalized,
        key="voigt_use_norm_checkbox",
        help="Normalization changes physical meaning of amplitudes. Keep off for physical quantities."
    )
    st.session_state.voigt_use_normalized = use_normalized

    st.markdown("---")
    st.markdown("#### ⚙️ Fitting Parameters")
    st.caption(f"Peak detection inherited from sidebar: height={st.session_state.peak_det_height_factor:.3f}×max, "
               f"prominence={st.session_state.peak_det_prominence_factor:.3f}×max, distance={st.session_state.peak_det_min_distance}")

    col1, col2 = st.columns(2)
    with col1:
        voigt_threshold = st.number_input(
            "Peak detection threshold (absolute)",
            min_value=0.0, value=float(np.max(spec_data['y'].values) * st.session_state.peak_det_height_factor),
            step=0.01, format="%.4f", key="voigt_threshold",
            help="Minimum height for peaks in data and residuals"
        )
        voigt_min_dist = st.number_input(
            "Minimum distance between peaks (points)",
            min_value=1, value=int(st.session_state.peak_det_min_distance),
            step=1, key="voigt_min_dist"
        )
        voigt_min_height = st.number_input(
            "Minimum peak height for keeping component",
            min_value=0.0, value=float(st.session_state.peak_det_min_height_abs),
            step=0.01, format="%.4f", key="voigt_min_height",
            help="Components with height below this value are removed"
        )
    with col2:
        voigt_max_iter = st.number_input(
            "Maximum iterations",
            min_value=0, max_value=20, value=5, step=1, key="voigt_max_iter",
            help="Number of residual-analysis iterations to add new components"
        )
        voigt_center_vary = st.number_input(
            "Center variation (± cm⁻¹)",
            min_value=0.1, value=5.0, step=0.5, format="%.2f", key="voigt_center_vary"
        )
        voigt_reduce = st.number_input(
            "Reduce data points factor",
            min_value=1, max_value=50, value=1, step=1, key="voigt_reduce",
            help="Use every N-th point to speed up fitting. 1 = use all points"
        )

    voigt_fit_method = st.selectbox(
        "Optimization method",
        options=["leastsq", "least_squares", "nelder", "lbfgsb"],
        index=0, key="voigt_fit_method"
    )

    st.markdown("---")

    if st.button("🚀 Run Voigt Decomposition", use_container_width=True, key="run_voigt_fit"):
        if fit_range[0] is None or fit_range[1] is None:
            st.error("Please select a valid range (left < right)")
        else:
            spec_data_local = filtered_spectra[spectrum_to_fit]['data']
            x_full = spec_data_local['x'].values
            y_full = spec_data_local['y'].values
            if use_normalized:
                y_full = normalize_spectrum(x_full, y_full, cached['norm_method'], cached['norm_range'], cached['x_ranges'])
            mask = (x_full >= fit_range[0]) & (x_full <= fit_range[1])
            x_fit = x_full[mask]
            y_fit = y_full[mask]
            if len(x_fit) < 10:
                st.error("Not enough data points in selected range.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def _progress_cb(iter_num, max_iter, msg):
                    try:
                        progress_bar.progress(min(1.0, iter_num / max(1, max_iter)))
                        status_text.info(f"Iteration {iter_num}/{max_iter}: {msg}")
                    except Exception:
                        pass

                with st.spinner("Running Voigt decomposition (may take a while)..."):
                    result, error = fit_voigt_decomposition(
                        x_fit, y_fit,
                        threshold=voigt_threshold,
                        min_dist=int(voigt_min_dist),
                        min_height=voigt_min_height,
                        max_iterations=int(voigt_max_iter),
                        center_vary=voigt_center_vary,
                        fit_method=voigt_fit_method,
                        reduce_factor=int(voigt_reduce) if voigt_reduce > 1 else None,
                        progress_callback=_progress_cb
                    )
                progress_bar.empty()
                status_text.empty()
                if error:
                    st.error(f"❌ Fitting failed: {error}")
                    st.session_state.voigt_fit_triggered = False
                    st.session_state.voigt_fit_results = None
                else:
                    best_result, components_df, fit_info = result
                    st.session_state.voigt_fit_triggered = True
                    st.session_state.voigt_fit_results = fit_info
                    st.session_state.voigt_fit_components_df = components_df
                    st.session_state.voigt_fit_log = fit_info.get('log', [])
                    st.success(f"✅ Voigt decomposition completed! {fit_info['n_components']} components fitted.")

    if st.session_state.get('voigt_fit_triggered', False) and st.session_state.get('voigt_fit_results') is not None:
        fit_info = st.session_state.voigt_fit_results
        components_df = st.session_state.voigt_fit_components_df

        st.markdown("---")
        st.markdown("#### 📊 Fit Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Components", fit_info['n_components'])
        with col2:
            rchi = fit_info.get('reduced_chi_square')
            st.metric("Reduced χ²", f"{rchi:.4e}" if rchi is not None else "N/A")
        with col3:
            aic = fit_info.get('aic')
            st.metric("AIC", f"{aic:.2f}" if aic is not None else "N/A")
        with col4:
            bic = fit_info.get('bic')
            st.metric("BIC", f"{bic:.2f}" if bic is not None else "N/A")

        with st.expander("📋 Fit log (iterations)", expanded=False):
            for line in st.session_state.get('voigt_fit_log', []):
                st.text(line)

        st.markdown("---")
        st.markdown("#### 📈 Voigt Fit Visualization")
        show_components = st.checkbox("Show components", value=True, key="voigt_show_components")
        show_residuals = st.checkbox("Show residuals panel", value=True, key="voigt_show_residuals")

        fig_voigt = plot_voigt_fit(
            fit_info['xs'], fit_info['ys'], fit_info['best_fit_full'],
            components_df,
            title=f"Voigt Decomposition: {spectrum_to_fit.replace('.txt', '')}",
            x_label=x_label, y_label=y_label,
            show_components=show_components, show_residuals=show_residuals,
            show_grid=cached['show_grid'],
            fig_width=cached['fig_width'] + 4,
            fig_height=cached['fig_height'] + 3
        )
        st.pyplot(fig_voigt)
        buf = BytesIO()
        fig_voigt.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f"""
        <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 1rem;">
            <a href="data:image/png;base64,{b64}" download="voigt_decomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png">
                <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               color: white; border: none; border-radius: 8px; 
                               padding: 0.4rem 1rem; cursor: pointer; font-size: 0.9rem;">
                    📥 Download Voigt Fit (PNG, 600 dpi)
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        plt.close(fig_voigt)

        st.markdown("---")
        st.markdown("#### 📋 Component Parameters Table")
        st.markdown("*Each row corresponds to one Voigt component. Error columns show 1σ standard errors from the fit.*")
        if components_df is not None and not components_df.empty:
            st.dataframe(components_df, use_container_width=True, hide_index=True)
            csv_components = components_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Components (CSV)",
                data=csv_components,
                file_name=f"voigt_components_{spectrum_to_fit.replace('.txt','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="voigt_components_download"
            )
        else:
            st.info("No components in the final fit.")

        with st.expander("📄 Full lmfit report", expanded=False):
            try:
                st.text(fit_info['best_fit_full'].fit_report())
            except Exception as e:
                st.warning(f"Could not generate full report: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

def main():
    import os
    from PIL import Image

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

    with st.sidebar:
        st.markdown('<div class="scientific-card">', unsafe_allow_html=True)

        st.markdown("### 📚 Documentation")
        instruction_content = load_instruction_html()
        if instruction_content:
            b64_instruction = base64.b64encode(instruction_content.encode('utf-8')).decode()
            st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <a href="data:text/html;base64,{b64_instruction}" download="SpectrAnalys_Instruction.html" 
                   style="display: inline-block; 
                          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          color: white; 
                          padding: 8px 16px; 
                          border-radius: 8px; 
                          text-decoration: none;
                          font-weight: 500;
                          font-size: 0.9rem;
                          width: 100%;
                          text-align: center;
                          transition: all 0.3s;">
                    📖 Download Full Instruction (HTML)
                </a>
            </div>
            """, unsafe_allow_html=True)
            st.caption("💡 Complete user guide with detailed workflow")
        else:
            st.warning("⚠️ Instruction file not found. Please ensure 'instruction.html' is in the root folder.")

        st.markdown("---")
        st.markdown("### 📁 Data Import")
        uploaded_files = st.file_uploader(
            "Upload spectra files (.txt format, tab-separated)",
            type=['txt'],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}"
        )

        if uploaded_files and st.session_state.get('spectra_loaded', False):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🗑️ Remove all spectra", type="secondary", use_container_width=True):
                    st.session_state.uploader_key = st.session_state.uploader_key + 1
                    keys_to_keep = ['uploader_key']
                    for key in list(st.session_state.keys()):
                        if key not in keys_to_keep:
                            del st.session_state[key]
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

                    with st.expander("🧹 Preprocessing (Despike / Denoise / Baseline)", expanded=False):
                        st.markdown("*Applied to all spectra before normalization. Order: Despike → Denoise → Baseline.*")
                        preproc_despike_enabled = st.checkbox(
                            "Enable despiking (Whitaker-Hayes)",
                            value=st.session_state.preproc_despike_enabled,
                            key="preproc_despike_enabled_cb"
                        )
                        st.session_state.preproc_despike_enabled = preproc_despike_enabled
                        if preproc_despike_enabled:
                            col1, col2 = st.columns(2)
                            with col1:
                                preproc_despike_kernel = st.number_input(
                                    "Kernel size", min_value=1, max_value=20,
                                    value=st.session_state.preproc_despike_kernel, step=1,
                                    key="preproc_despike_kernel_inp"
                                )
                                st.session_state.preproc_despike_kernel = int(preproc_despike_kernel)
                            with col2:
                                preproc_despike_threshold = st.number_input(
                                    "Z-score threshold", min_value=1.0, max_value=50.0,
                                    value=float(st.session_state.preproc_despike_threshold),
                                    step=0.5, format="%.1f", key="preproc_despike_threshold_inp"
                                )
                                st.session_state.preproc_despike_threshold = float(preproc_despike_threshold)

                        preproc_denoise_enabled = st.checkbox(
                            "Enable denoising",
                            value=st.session_state.preproc_denoise_enabled,
                            key="preproc_denoise_enabled_cb"
                        )
                        st.session_state.preproc_denoise_enabled = preproc_denoise_enabled
                        if preproc_denoise_enabled:
                            preproc_denoise_method = st.selectbox(
                                "Denoising method",
                                options=["Savitzky-Golay", "Gaussian", "Whittaker"],
                                index=["Savitzky-Golay", "Gaussian", "Whittaker"].index(st.session_state.preproc_denoise_method)
                                if st.session_state.preproc_denoise_method in ["Savitzky-Golay", "Gaussian", "Whittaker"] else 0,
                                key="preproc_denoise_method_sel"
                            )
                            st.session_state.preproc_denoise_method = preproc_denoise_method
                            if preproc_denoise_method == "Savitzky-Golay":
                                col1, col2 = st.columns(2)
                                with col1:
                                    preproc_denoise_window = st.number_input(
                                        "Window length (odd)", min_value=3, max_value=101,
                                        value=int(st.session_state.preproc_denoise_window), step=2,
                                        key="preproc_denoise_window_inp"
                                    )
                                    st.session_state.preproc_denoise_window = int(preproc_denoise_window)
                                with col2:
                                    preproc_denoise_polyorder = st.number_input(
                                        "Polynomial order", min_value=1, max_value=10,
                                        value=int(st.session_state.preproc_denoise_polyorder), step=1,
                                        key="preproc_denoise_polyorder_inp"
                                    )
                                    st.session_state.preproc_denoise_polyorder = int(preproc_denoise_polyorder)
                            elif preproc_denoise_method == "Gaussian":
                                preproc_denoise_sigma = st.number_input(
                                    "Gaussian sigma", min_value=0.1, max_value=20.0,
                                    value=float(st.session_state.preproc_denoise_sigma),
                                    step=0.1, format="%.1f", key="preproc_denoise_sigma_inp"
                                )
                                st.session_state.preproc_denoise_sigma = float(preproc_denoise_sigma)

                        preproc_baseline_enabled = st.checkbox(
                            "Enable baseline correction",
                            value=st.session_state.preproc_baseline_enabled,
                            key="preproc_baseline_enabled_cb"
                        )
                        st.session_state.preproc_baseline_enabled = preproc_baseline_enabled
                        if preproc_baseline_enabled:
                            if not PYBASELINES_AVAILABLE:
                                st.caption("⚠️ `pybaselines` not installed — fallback algorithms will be used. Install: `pip install pybaselines`")
                            baseline_methods = ['asPLS', 'airPLS', 'arPLS', 'iarPLS', 'drPLS', 'IAsLS', 'AsLS',
                                                'ModPoly', 'IModPoly', 'Poly']
                            preproc_baseline_method = st.selectbox(
                                "Baseline method",
                                options=baseline_methods,
                                index=baseline_methods.index(st.session_state.preproc_baseline_method)
                                if st.session_state.preproc_baseline_method in baseline_methods else 0,
                                key="preproc_baseline_method_sel"
                            )
                            st.session_state.preproc_baseline_method = preproc_baseline_method
                            if preproc_baseline_method in ['Poly', 'ModPoly', 'IModPoly']:
                                preproc_baseline_polyorder = st.number_input(
                                    "Polynomial order", min_value=1, max_value=10,
                                    value=int(st.session_state.preproc_baseline_polyorder), step=1,
                                    key="preproc_baseline_polyorder_inp"
                                )
                                st.session_state.preproc_baseline_polyorder = int(preproc_baseline_polyorder)
                            if preproc_baseline_method not in ['Poly']:
                                col1, col2 = st.columns(2)
                                with col1:
                                    preproc_baseline_lam = st.number_input(
                                        "Smoothness λ", min_value=1.0, max_value=1e12,
                                        value=float(st.session_state.preproc_baseline_lam),
                                        step=1e4, format="%.2e", key="preproc_baseline_lam_inp"
                                    )
                                    st.session_state.preproc_baseline_lam = float(preproc_baseline_lam)
                                with col2:
                                    preproc_baseline_diff_order = st.number_input(
                                        "Difference order", min_value=1, max_value=5,
                                        value=int(st.session_state.preproc_baseline_diff_order), step=1,
                                        key="preproc_baseline_diff_order_inp"
                                    )
                                    st.session_state.preproc_baseline_diff_order = int(preproc_baseline_diff_order)
                                if preproc_baseline_method in ['AsLS', 'IAsLS']:
                                    preproc_baseline_p = st.number_input(
                                        "Asymmetry p", min_value=1e-6, max_value=0.5,
                                        value=float(st.session_state.preproc_baseline_p),
                                        step=1e-3, format="%.4f", key="preproc_baseline_p_inp"
                                    )
                                    st.session_state.preproc_baseline_p = float(preproc_baseline_p)
                                col1, col2 = st.columns(2)
                                with col1:
                                    preproc_baseline_max_iter = st.number_input(
                                        "Max iterations", min_value=1, max_value=500,
                                        value=int(st.session_state.preproc_baseline_max_iter), step=1,
                                        key="preproc_baseline_max_iter_inp"
                                    )
                                    st.session_state.preproc_baseline_max_iter = int(preproc_baseline_max_iter)
                                with col2:
                                    preproc_baseline_tol = st.number_input(
                                        "Tolerance", min_value=1e-10, max_value=1.0,
                                        value=float(st.session_state.preproc_baseline_tol),
                                        step=1e-4, format="%.4e", key="preproc_baseline_tol_inp"
                                    )
                                    st.session_state.preproc_baseline_tol = float(preproc_baseline_tol)

                        if st.button("🧹 Apply Preprocessing", use_container_width=True, key="apply_preprocessing_btn"):
                            spectra_for_preproc = {}
                            for name in ordered_spectra:
                                if name in spectra_data:
                                    spectra_for_preproc[name] = spectra_data[name]
                            try:
                                processed = preprocess_spectra_cached(
                                    "current", spectra_for_preproc,
                                    st.session_state.preproc_despike_enabled,
                                    st.session_state.preproc_despike_kernel,
                                    st.session_state.preproc_despike_threshold,
                                    st.session_state.preproc_denoise_enabled,
                                    st.session_state.preproc_denoise_method,
                                    st.session_state.preproc_denoise_window,
                                    st.session_state.preproc_denoise_polyorder,
                                    st.session_state.preproc_denoise_sigma,
                                    st.session_state.preproc_baseline_enabled,
                                    st.session_state.preproc_baseline_method,
                                    st.session_state.preproc_baseline_lam,
                                    st.session_state.preproc_baseline_p,
                                    st.session_state.preproc_baseline_diff_order,
                                    st.session_state.preproc_baseline_max_iter,
                                    st.session_state.preproc_baseline_tol,
                                    st.session_state.preproc_baseline_polyorder
                                )
                                st.session_state.preproc_cached_spectra = processed
                                st.session_state.preproc_applied = True
                                st.session_state.preproc_version += 1
                                st.success(f"✅ Preprocessing applied to {len(processed)} spectra.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Preprocessing failed: {e}")

                        if st.session_state.preproc_applied:
                            st.info(f"✅ Preprocessing active (version {st.session_state.preproc_version})")
                            if st.button("↩️ Disable Preprocessing", use_container_width=True, key="disable_preprocessing_btn"):
                                st.session_state.preproc_cached_spectra = None
                                st.session_state.preproc_applied = False
                                st.session_state.preproc_version += 1
                                st.rerun()

                    with st.expander("🎯 Peak Detection Parameters", expanded=False):
                        st.markdown("*Global parameters for `find_peaks`. Used in Peak Analysis and Voigt Fitting.*")
                        col1, col2 = st.columns(2)
                        with col1:
                            peak_det_height_factor = st.number_input(
                                "Height factor (× max intensity)",
                                min_value=0.001, max_value=1.0,
                                value=float(st.session_state.peak_det_height_factor),
                                step=0.001, format="%.3f", key="peak_det_height_inp"
                            )
                            st.session_state.peak_det_height_factor = float(peak_det_height_factor)
                            peak_det_min_distance = st.number_input(
                                "Minimum distance (points)", min_value=1, max_value=200,
                                value=int(st.session_state.peak_det_min_distance), step=1,
                                key="peak_det_min_dist_inp"
                            )
                            st.session_state.peak_det_min_distance = int(peak_det_min_distance)
                        with col2:
                            peak_det_prominence_factor = st.number_input(
                                "Prominence factor (× max intensity)",
                                min_value=0.001, max_value=1.0,
                                value=float(st.session_state.peak_det_prominence_factor),
                                step=0.001, format="%.3f", key="peak_det_prom_inp"
                            )
                            st.session_state.peak_det_prominence_factor = float(peak_det_prominence_factor)
                            peak_det_min_height_abs = st.number_input(
                                "Absolute min height (for Voigt)", min_value=0.0, max_value=1e9,
                                value=float(st.session_state.peak_det_min_height_abs),
                                step=0.01, format="%.3f", key="peak_det_min_h_inp"
                            )
                            st.session_state.peak_det_min_height_abs = float(peak_det_min_height_abs)

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
                            min_value=0.0, max_value=50000.0, value=1000.0, step=100.0,
                            key="raw_offset_step"
                        )
                    with col2:
                        norm_offset_step = st.slider(
                            "Normalized spectra offset step",
                            min_value=0.0, max_value=5.0, value=0.5, step=0.05,
                            key="norm_offset_step"
                        )

                    fill_area = st.checkbox("Fill area under normalized spectra", value=False)

                    fill_alpha = 0.3
                    if fill_area:
                        fill_alpha = st.slider(
                            "Fill transparency",
                            min_value=0.2, max_value=0.9, value=0.3, step=0.1,
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
                            min_value=0.5, max_value=3.0, value=1.5, step=0.1
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
                            min_value=4, max_value=16, value=8, step=1,
                            key="legend_fontsize"
                        )
                    with col2:
                        legend_position = st.selectbox(
                            "Legend position",
                            options=["right", "best", "upper right", "upper left", "lower left", "lower right"],
                            index=0, key="legend_position"
                        )
                    legend_offset = st.slider(
                        "Legend offset from plot (0.5-2.0)",
                        min_value=0.5, max_value=2.0, value=1.02, step=0.02,
                        help="Higher value moves legend further right (for right position)"
                    )

                    st.markdown("---")
                    st.markdown("### 🎨 Gradient Color Assignment")
                    st.markdown("*Automatically assign colors to spectra using a gradient palette*")

                    gradient_palettes = {
                        'Reds_r': 'Red gradient (reversed)',
                        'RdBu_r': 'Red-Blue (diverging, reversed)',
                        'turbo': 'Turbo (improved jet)',
                        'twilight_shifted': 'Twilight Shifted (blue-red-purple)',
                        'Spectral': 'Spectral (rainbow)',
                        'coolwarm': 'Cool-Warm (diverging)',
                        'viridis': 'Viridis (perceptual)',
                        'plasma': 'Plasma (perceptual)',
                        'inferno': 'Inferno (perceptual)',
                        'magma': 'Magma (perceptual)',
                        'BrBG': 'Brown-Blue-Green (diverging)',
                    }

                    selected_gradient_palette = st.selectbox(
                        "Select gradient palette",
                        options=list(gradient_palettes.keys()),
                        format_func=lambda x: gradient_palettes[x],
                        index=3, key="gradient_palette_select"
                    )
                    st.session_state.gradient_palette = selected_gradient_palette

                    if st.button("🎨 Apply gradient colors", use_container_width=True, key="apply_gradient_colors"):
                        with st.spinner("Applying gradient colors..."):
                            gradient_colors = apply_gradient_colors(ordered_spectra, selected_gradient_palette)
                            if gradient_colors:
                                st.session_state.gradient_colors_cache = gradient_colors
                                st.session_state.gradient_applied = True
                                st.session_state.gradient_version = st.session_state.get('gradient_version', 0) + 1
                                for name, hex_color in gradient_colors.items():
                                    if name in spectra_data:
                                        spectra_data[name]['color'] = hex_color
                                if st.session_state.cached_spectra_data is not None:
                                    st.session_state.cached_spectra_data['spectra_data'] = spectra_data
                                    st.session_state.cached_spectra_data['gradient_applied'] = True
                                    st.session_state.cached_spectra_data['gradient_palette'] = selected_gradient_palette
                                    st.session_state.cached_spectra_data['gradient_colors_cache'] = gradient_colors
                                    st.session_state.cached_spectra_data['gradient_version'] = st.session_state.gradient_version
                                st.success(f"✅ Gradient colors applied using {gradient_palettes[selected_gradient_palette]}!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to apply gradient colors. Please try again.")

                    default_colors = [
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
                    ]

                    for i, name in enumerate(ordered_spectra):
                        if name in spectra_data and spectra_data[name].get('color') is None:
                            spectra_data[name]['color'] = default_colors[i % len(default_colors)]

                    if st.session_state.gradient_applied and st.session_state.gradient_colors_cache:
                        for name in ordered_spectra:
                            if name in spectra_data and name in st.session_state.gradient_colors_cache:
                                spectra_data[name]['color'] = st.session_state.gradient_colors_cache[name]

                    st.markdown("---")
                    st.markdown("### 🎨 Color Assignment")
                    st.markdown("*Manually adjust individual spectrum colors (after applying gradient if needed)*")

                    default_colors = [
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
                    ]

                    if st.session_state.gradient_applied and st.session_state.gradient_colors_cache:
                        for name in ordered_spectra:
                            if name in st.session_state.gradient_colors_cache:
                                spectra_data[name]['color'] = st.session_state.gradient_colors_cache[name]

                    colors = {}
                    gradient_version = st.session_state.get('gradient_version', 0)

                    for i, name in enumerate(ordered_spectra):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{name.replace('.txt', '')}**")
                        with col2:
                            if st.session_state.gradient_applied and name in st.session_state.gradient_colors_cache:
                                current_color = st.session_state.gradient_colors_cache[name]
                            else:
                                current_color = spectra_data.get(name, {}).get('color', default_colors[i % len(default_colors)])
                            color_key = f"color_{name}_v{gradient_version}"
                            colors[name] = st.color_picker(
                                f"Color {i+1}",
                                value=current_color,
                                key=color_key
                            )

                    for i, name in enumerate(ordered_spectra):
                        if name in colors:
                            spectra_data[name]['color'] = colors[name]
                            if name in st.session_state.gradient_colors_cache:
                                st.session_state.gradient_colors_cache[name] = colors[name]
                        else:
                            default_color = default_colors[i % len(default_colors)]
                            spectra_data[name]['color'] = default_color
                            if st.session_state.cached_spectra_data is not None:
                                st.session_state.cached_spectra_data['spectra_data'] = spectra_data
                                if 'gradient_version' not in st.session_state.cached_spectra_data:
                                    st.session_state.cached_spectra_data['gradient_version'] = st.session_state.gradient_version

                    if st.session_state.cached_spectra_data is not None:
                        if st.session_state.gradient_applied and st.session_state.gradient_colors_cache:
                            for name in ordered_spectra:
                                if name in st.session_state.gradient_colors_cache:
                                    spectra_data[name]['color'] = st.session_state.gradient_colors_cache[name]
                                    st.session_state.cached_spectra_data['spectra_data'][name]['color'] = st.session_state.gradient_colors_cache[name]
                        else:
                            st.session_state.cached_spectra_data['spectra_data'] = spectra_data

                    if st.session_state.preproc_applied and st.session_state.preproc_cached_spectra is not None:
                        spectra_data_effective = st.session_state.preproc_cached_spectra
                    else:
                        spectra_data_effective = spectra_data

                    st.session_state.spectra_loaded = True
                    st.session_state.cached_spectra_data = {
                        'spectra_data': spectra_data_effective,
                        'spectra_data_raw': spectra_data,
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
                        'legend_fontsize': legend_fontsize,
                        'legend_position': legend_position,
                        'legend_offset': legend_offset,
                        'gradient_palette': st.session_state.gradient_palette,
                        'gradient_applied': st.session_state.gradient_applied,
                        'gradient_colors_cache': st.session_state.gradient_colors_cache,
                        'preproc_applied': st.session_state.preproc_applied,
                        'preproc_version': st.session_state.preproc_version,
                    }

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p>🔬 SpectrAnalys v2.1<br>Scientific Spectroscopic Analysis</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_files and st.session_state.get('spectra_loaded', False) and st.session_state.cached_spectra_data:
        cached = st.session_state.cached_spectra_data
        spectra_data = cached['spectra_data']
        ordered_spectra = cached['ordered_spectra']
        x_ranges = cached['x_ranges']
        common_x_range = cached['common_x_range']
        norm_method = cached['norm_method']

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
            preproc_label = "On" if st.session_state.get('preproc_applied', False) else "Off"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{preproc_label}</div>
                <div class="metric-label">Preprocessing</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        current_spectra = spectra_data
        if common_x_range:
            current_spectra = align_x_ranges(current_spectra)

        filtered_spectra = {name: current_spectra[name] for name in ordered_spectra if name in current_spectra}

        tab_names = [
            "Combined Spectra Visualization",
            "Advanced Peak Analysis",
            "Spectral Decomposition",
            "Parameter Correlation",
            "Compare Two Spectra",
            "Spectral Markers",
            "3D Graphs",
            "Heatmaps"
        ]

        current_tab = st.session_state.active_tab
        if current_tab not in tab_names:
            current_tab = tab_names[0]
            st.session_state.active_tab = current_tab

        selected_tab = st.pills(
            "Select analysis view",
            options=tab_names,
            selection_mode="single",
            default=current_tab,
            key="active_tab_pills",
            label_visibility="collapsed"
        )
        if selected_tab is None:
            selected_tab = current_tab
        st.session_state.active_tab = selected_tab

        if selected_tab == "Combined Spectra Visualization":
            render_combined_tab(cached, filtered_spectra, x_ranges)
        elif selected_tab == "Advanced Peak Analysis":
            render_peak_tab(cached, filtered_spectra)
        elif selected_tab == "Spectral Decomposition":
            render_voigt_tab(cached, filtered_spectra, ordered_spectra)
        elif selected_tab == "Parameter Correlation":
            render_correlation_tab(cached, filtered_spectra, ordered_spectra)
        elif selected_tab == "Compare Two Spectra":
            render_compare_tab(cached, filtered_spectra, ordered_spectra)
        elif selected_tab == "Spectral Markers":
            render_markers_tab(cached, filtered_spectra)
        elif selected_tab == "3D Graphs":
            render_3d_tab(cached, filtered_spectra, ordered_spectra)
        elif selected_tab == "Heatmaps":
            render_heatmap_tab(cached, filtered_spectra, ordered_spectra)

        st.markdown("---")
        st.markdown('<div class="scientific-card">', unsafe_allow_html=True)
        st.subheader("📤 Data Export")

        col1, col2, col3, col4 = st.columns(4)

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
                    label="📥 Export Preprocessed Data (CSV)",
                    data=csv,
                    file_name=f"preprocessed_spectra_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.session_state.get('spectra_data_raw') or cached.get('spectra_data_raw'):
                raw_dict = cached.get('spectra_data_raw', {})
                export_raw = pd.DataFrame()
                for name in ordered_spectra:
                    if name in raw_dict:
                        data = raw_dict[name]['data']
                        temp_df = pd.DataFrame({
                            f"{name.replace('.txt', '')}_x": data['x'].values,
                            f"{name.replace('.txt', '')}_y_raw": data['y'].values
                        })
                        if export_raw.empty:
                            export_raw = temp_df
                        else:
                            export_raw = pd.concat([export_raw, temp_df], axis=1)
                if not export_raw.empty:
                    csv_raw = export_raw.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Raw Data (CSV, original)",
                        data=csv_raw,
                        file_name=f"raw_spectra_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

        with col3:
            if filtered_spectra:
                export_norm = pd.DataFrame()
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    y_norm = normalize_spectrum(
                        data['x'].values, data['y'].values,
                        cached['norm_method'], cached['norm_range'], cached['x_ranges']
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

        with col4:
            session_info = f"""SpectrAnalys Analysis Session
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Spectra Files: {', '.join(ordered_spectra)}
Normalization Method: {cached['norm_method']}
X-axis Ranges: {cached['x_ranges'] if cached['x_ranges'] else 'Full range'}
Raw Offset Step: {cached['raw_offset_step']}
Normalized Offset Step: {cached['norm_offset_step']}
Fill Area: {cached['fill_area']}
Fill Transparency: {cached['fill_alpha']}
Subtract Minimum Intensity: {cached['subtract_min_intensity']}
Grid Enabled: {cached['show_grid']}
Line Width: {cached['line_width']}
Preprocessing Applied: {st.session_state.get('preproc_applied', False)}
Preprocessing Version: {st.session_state.get('preproc_version', 0)}
Despike: {st.session_state.get('preproc_despike_enabled', False)}
Denoise: {st.session_state.get('preproc_denoise_enabled', False)} ({st.session_state.get('preproc_denoise_method', 'N/A')})
Baseline: {st.session_state.get('preproc_baseline_enabled', False)} ({st.session_state.get('preproc_baseline_method', 'N/A')})
Peak Detection: height={st.session_state.get('peak_det_height_factor', 0.05)}, prominence={st.session_state.get('peak_det_prominence_factor', 0.03)}, distance={st.session_state.get('peak_det_min_distance', 5)}
Peak Analysis: {st.session_state.get('peak_enabled', False)}
Voigt Fitting: {st.session_state.get('voigt_fit_triggered', False)}
Correlation Analysis: {st.session_state.get('correlation_enabled', False)}
Spectral Markers: {len(st.session_state.spectral_markers)} markers
3D Graphs Enabled: {st.session_state.get('enable_3d_graphs', False)}
3D Graphs Created: {st.session_state.get('graph3d_created', False)}
Gradient Colors Applied: {st.session_state.get('gradient_applied', False)}
Gradient Palette: {st.session_state.get('gradient_palette', 'Not applied')}
"""
            st.download_button(
                label="📄 Export Session Info",
                data=session_info,
                file_name=f"spectranalys_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("## 🎯 Welcome to SpectrAnalys")
        st.markdown("Advanced spectroscopic data analysis platform for researchers and scientists")

        st.markdown("### 📖 Quick Start Guide:")
        st.markdown("""
        1. **Upload Data** - Select one or more .txt files with two columns (x y, tab-separated)
        2. **Configure Processing** - Preprocessing (despike, denoise, baseline), normalization, colors, offsets
        3. **Visualize** - Explore combined spectra visualization with multiple display modes
        4. **Analyze Peaks** - Detect and characterize spectral peaks automatically
        5. **Voigt Decomposition** - Decompose complex spectra into individual Voigt components
        6. **Correlate Parameters** - Investigate relationships between spectral features and experimental parameters
        7. **Compare Spectra** - Analyze differences between two spectra with heatmap visualization
        8. **Add Markers** - Place vertical lines and regions to track spectral features
        9. **3D Visualization** - Explore spectra in 3D with surface and line plots
        10. **Heatmaps** - Generate heatmaps for temperature/concentration dependence
        11. **Export Results** - Download processed data, plots, and analysis results
        """)

        st.markdown("### ✨ Key Features:")
        st.markdown("""
        - 🧹 **Preprocessing Pipeline** - Despiking (Whitaker-Hayes), denoising (SavGol/Gaussian/Whittaker), baseline correction (10+ methods via pybaselines)
        - 🎯 **Global Peak Detection** - Configurable height/prominence/distance used across all analyses
        - 🔬 **Multi-Mode Visualization** - Raw, normalized, and offset spectra in one comprehensive view
        - 📊 **Broken Axis Support** - Display multiple x-axis ranges with gaps between them
        - 🎨 **Customizable Colors** - Individual color assignment for each spectrum
        - 🧬 **Voigt Decomposition** - Iterative residual-based fitting with lmfit; full parameter table with errors
        - 📈 **Automatic Peak Detection** - Find peaks, calculate areas, and analyze intensities
        - 🔗 **Parameter Correlation** - Correlate spectral features with experimental parameters
        - 🔀 **Spectral Comparison** - Compare two spectra with difference analysis and heatmap visualization
        - 🔥 **Heatmap Generation** - Visualize spectral evolution as function of temperature or concentration
        - 📏 **Spectral Markers** - Add vertical lines and regions to track peak positions across conditions
        - 🎲 **3D Visualization** - Surface and line plots for comprehensive spectral analysis
        - 💾 **Data Export** - Download raw, preprocessed, and normalized data in CSV format
        - 📐 **Multiple Normalization Methods** - Maximum intensity or custom peak range normalization
        - ⚡ **Fragment Isolation** - Each tab runs independently for fast interactions
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

    st.markdown("""
    <div class="footer">
        <p>🔬 SpectrAnalys | Scientific Spectroscopic Analysis Platform | Built with Streamlit & Python</p>
        <p style="font-size: 0.75rem;">© 2026 SpectrAnalys | © developed by daM - Advanced Spectroscopy Data Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
