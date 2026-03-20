import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson
from io import StringIO
import re
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Spectra Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply scientific plotting style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
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
        return y / y.max()
    
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
    if not range_str or range_str == "Full range":
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

# Function to crop spectrum to ranges
def crop_to_ranges(x, y, ranges):
    """Crop spectrum to specified ranges"""
    if ranges is None:
        return x, y
    
    mask = np.zeros(len(x), dtype=bool)
    for start, end in ranges:
        mask |= (x >= start) & (x <= end)
    
    return x[mask], y[mask]

# Function to create plot
def create_plot(spectra_dict, x_label, y_label, title, offset=0, fill=False, 
                x_range=None, normalized=False, common_x_range=False):
    """Create scientific plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Store handles and labels for legend
    handles = []
    labels = []
    
    for name, spec in spectra_dict.items():
        data = spec['data']
        x = data['x'].values
        y = data['y'].values
        color = spec['color']
        
        # Remove .txt extension from name for display
        display_name = name.replace('.txt', '')
        
        # Apply x range cropping
        if x_range is not None:
            x, y = crop_to_ranges(x, y, x_range)
        
        if len(x) == 0:
            continue
        
        # Apply offset
        y_plot = y + offset if offset != 0 else y
        
        # Plot
        if fill and normalized:
            fill_handle = ax.fill_between(x, 0, y_plot, alpha=0.3, color=color, label=display_name)
            line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5)
            handles.append(line_handle[0])
            labels.append(display_name)
        else:
            line_handle = ax.plot(x, y_plot, color=color, linewidth=1.5, label=display_name)
            handles.append(line_handle[0])
            labels.append(display_name)
    
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Create legend with proper colors
    if handles:
        legend = ax.legend(handles, labels, loc='best', fontsize=10, 
                          frameon=True, edgecolor='black')
        # Make legend text colors match line colors
        for text, handle in zip(legend.get_texts(), handles):
            text.set_color(handle.get_color())
    
    ax.tick_params(direction='out', length=4, width=0.8)
    
    plt.tight_layout()
    return fig

# Function for peak analysis
def analyze_peaks(spectra_dict, x_range=None, peak_width=20):
    """Analyze peaks in spectra"""
    results = []
    
    for name, spec in spectra_dict.items():
        data = spec['data']
        x = data['x'].values
        y = data['y'].values
        
        if x_range is not None:
            x, y = crop_to_ranges(x, y, x_range)
        
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
                'Spectrum': name,
                'Peak position': peak_x,
                'Intensity': peak_y,
                'Area': area
            })
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# Main app
def main():
    st.title("📊 Advanced Spectra Analysis Tool")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_files = st.file_uploader(
            "Upload spectra files (.txt)",
            type=['txt'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.success(f"Loaded {len(uploaded_files)} files")
            
            # Load data
            spectra_data = {}
            for file in uploaded_files:
                data = load_spectrum(file)
                if data is not None:
                    spectra_data[file.name] = {
                        'data': data,
                        'color': None  # Will be assigned later
                    }
            
            if spectra_data:
                st.markdown("---")
                st.header("📋 Select Spectra Order")
                
                # Select and order spectra
                selected_spectra = st.multiselect(
                    "Choose spectra to display",
                    options=list(spectra_data.keys()),
                    default=list(spectra_data.keys())
                )
                
                if selected_spectra:
                    # Order spectra
                    ordered_spectra = []
                    for i, name in enumerate(selected_spectra):
                        ordered_spectra.append(name)
                    
                    # Assign colors with default distinct colors
                    colors = {}
                    st.markdown("---")
                    st.header("🎨 Color Assignment")
                    
                    # Define default color palette
                    default_colors = [
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
                    ]
                    
                    for i, name in enumerate(ordered_spectra):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{name}**")
                        with col2:
                            # Use default color from palette, cycling if more spectra than colors
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
                    st.header("⚙️ Processing Options")
                    
                    # Common x range option
                    common_x_range = st.checkbox("Align all spectra to common x range", value=False)
                    
                    # X-axis ranges
                    st.subheader("X-axis Ranges")
                    x_range_option = st.radio(
                        "Select range mode",
                        ["Full range", "Custom ranges"]
                    )
                    
                    x_ranges = None
                    if x_range_option == "Custom ranges":
                        range_input = st.text_area(
                            "Enter ranges (e.g., 100-200, 300-400)",
                            placeholder="100-200, 300-400"
                        )
                        if range_input:
                            x_ranges = parse_x_ranges(range_input)
                    
                    # Axis labels
                    st.subheader("Axis Labels")
                    x_label = st.text_input("X-axis label", value="Raman shift (cm⁻¹)")
                    y_label = st.text_input("Y-axis label", value="Intensity (a.u.)")
                    
                    # Normalization options
                    st.subheader("Normalization")
                    norm_method = st.selectbox(
                        "Normalization method",
                        ["None", "Maximum intensity", "Area", "Peak intensity (range)"]
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
                    st.subheader("Offset")
                    offset_value = st.slider(
                        "Offset value",
                        min_value=0.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.1
                    )
                    
                    fill_area = st.checkbox("Fill area under normalized spectra", value=False)
                    
                    # Peak analysis options
                    st.markdown("---")
                    st.header("📈 Peak Analysis")
                    analyze_peaks_flag = st.checkbox("Enable peak analysis", value=False)
                    
                    if analyze_peaks_flag:
                        peak_analysis_range = None
                        if x_range_option == "Custom ranges" and range_input:
                            peak_analysis_range = x_ranges
                        
                        peak_width = st.slider(
                            "Peak width for area calculation",
                            min_value=5,
                            max_value=100,
                            value=20,
                            step=5
                        )
                    
                    # Parameter correlation
                    st.markdown("---")
                    st.header("📊 Parameter Correlation")
                    param_correlation = st.checkbox("Enable parameter correlation", value=False)
                    
                    if param_correlation:
                        st.info("For correlation analysis, assign numeric values to each spectrum")
                        param_values = {}
                        for name in ordered_spectra:
                            param_values[name] = st.number_input(
                                f"Value for {name}",
                                value=float(len(param_values)),
                                key=f"param_{name}"
                            )
                        
                        param_label = st.text_input("Parameter label", value="Sample number")
    
    # Main content area
    if uploaded_files and 'spectra_data' in locals() and spectra_data:
        # Apply common x range if selected
        current_spectra = spectra_data
        if common_x_range:
            current_spectra = align_x_ranges(current_spectra)
        
        # Prepare normalized spectra
        normalized_spectra = {}
        for name, spec in current_spectra.items():
            if name in ordered_spectra:
                data = spec['data']
                y_norm = normalize_spectrum(
                    data['x'].values, 
                    data['y'].values, 
                    norm_method, 
                    norm_range
                )
                normalized_spectra[name] = {
                    'data': pd.DataFrame({'x': data['x'], 'y': y_norm}),
                    'color': spec['color']
                }
        
        # Filter spectra based on selection
        filtered_spectra = {name: current_spectra[name] for name in ordered_spectra if name in current_spectra}
        filtered_norm_spectra = {name: normalized_spectra[name] for name in ordered_spectra if name in normalized_spectra}
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Raw Spectra", 
            "📊 Normalized Spectra", 
            "📉 Raw + Offset", 
            "📐 Normalized + Offset",
            "🔍 Peak Analysis",
            "📈 Parameter Correlation"
        ])
        
        with tab1:
            st.subheader("Raw Spectra")
            if filtered_spectra:
                fig = create_plot(
                    filtered_spectra, x_label, y_label, 
                    "Raw Spectra", 
                    x_range=x_ranges
                )
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No spectra selected")
        
        with tab2:
            st.subheader("Normalized Spectra")
            if filtered_norm_spectra:
                fig = create_plot(
                    filtered_norm_spectra, x_label, 
                    f"Normalized Intensity ({norm_method})", 
                    "Normalized Spectra",
                    x_range=x_ranges,
                    normalized=True
                )
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No spectra selected")
        
        with tab3:
            st.subheader(f"Raw Spectra with Offset (offset = {offset_value})")
            if filtered_spectra:
                fig = create_plot(
                    filtered_spectra, x_label, y_label, 
                    f"Raw Spectra (offset = {offset_value})",
                    offset=offset_value,
                    x_range=x_ranges
                )
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No spectra selected")
        
        with tab4:
            st.subheader(f"Normalized Spectra with Offset (offset = {offset_value})")
            if filtered_norm_spectra:
                fig = create_plot(
                    filtered_norm_spectra, x_label, 
                    f"Normalized Intensity ({norm_method})", 
                    f"Normalized Spectra (offset = {offset_value})",
                    offset=offset_value,
                    fill=fill_area,
                    x_range=x_ranges,
                    normalized=True
                )
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No spectra selected")
        
        with tab5:
            st.subheader("Peak Analysis")
            if analyze_peaks_flag and filtered_spectra:
                peaks_df = analyze_peaks(filtered_spectra, x_ranges, peak_width)
                if not peaks_df.empty:
                    st.dataframe(peaks_df, use_container_width=True)
                    
                    # Download button for peak analysis
                    csv = peaks_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download peak analysis as CSV",
                        data=csv,
                        file_name="peak_analysis.csv",
                        mime="text/csv"
                    )
                    
                    # Visualize peaks
                    st.subheader("Peak Visualization")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    for name, spec in filtered_spectra.items():
                        data = spec['data']
                        x = data['x'].values
                        y = data['y'].values
                        
                        if x_ranges is not None:
                            x, y = crop_to_ranges(x, y, x_ranges)
                        
                        ax.plot(x, y, color=spec['color'], linewidth=1.5, label=name, alpha=0.7)
                        
                        # Mark peaks
                        spec_peaks = peaks_df[peaks_df['Spectrum'] == name]
                        for _, peak in spec_peaks.iterrows():
                            ax.axvline(peak['Peak position'], color=spec['color'], 
                                      linestyle='--', alpha=0.5, linewidth=1)
                            ax.text(peak['Peak position'], peak['Intensity']*0.8, 
                                   f"{peak['Peak position']:.1f}", 
                                   fontsize=8, ha='center')
                    
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title("Detected Peaks")
                    ax.legend(loc='best')
                    ax.tick_params(direction='out')
                    
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No peaks detected in the selected range")
            else:
                st.info("Enable peak analysis in sidebar to analyze peaks")
        
        with tab6:
            st.subheader("Parameter Correlation Analysis")
            if param_correlation and filtered_spectra and param_values:
                # Prepare data for correlation
                param_list = []
                intensity_list = []
                area_list = []
                position_list = []
                
                # Get parameter values
                params = [param_values[name] for name in ordered_spectra if name in param_values]
                
                # Get peak analysis results if available
                if analyze_peaks_flag and 'peaks_df' in locals() and not peaks_df.empty:
                    for name in ordered_spectra:
                        if name in param_values:
                            spec_peaks = peaks_df[peaks_df['Spectrum'] == name]
                            if not spec_peaks.empty:
                                # Take the most intense peak
                                main_peak = spec_peaks.loc[spec_peaks['Intensity'].idxmax()]
                                param_list.append(param_values[name])
                                intensity_list.append(main_peak['Intensity'])
                                area_list.append(main_peak['Area'])
                                position_list.append(main_peak['Peak position'])
                
                if param_list:
                    # Create correlation plots
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].scatter(param_list, intensity_list, c='blue', alpha=0.6, s=50)
                    axes[0].set_xlabel(param_label)
                    axes[0].set_ylabel("Peak Intensity (a.u.)")
                    axes[0].set_title("Intensity vs Parameter")
                    axes[0].grid(True, alpha=0.3)
                    
                    axes[1].scatter(param_list, area_list, c='green', alpha=0.6, s=50)
                    axes[1].set_xlabel(param_label)
                    axes[1].set_ylabel("Peak Area")
                    axes[1].set_title("Area vs Parameter")
                    axes[1].grid(True, alpha=0.3)
                    
                    axes[2].scatter(param_list, position_list, c='red', alpha=0.6, s=50)
                    axes[2].set_xlabel(param_label)
                    axes[2].set_ylabel("Peak Position (cm⁻¹)")
                    axes[2].set_title("Position vs Parameter")
                    axes[2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show correlation table
                    corr_data = pd.DataFrame({
                        'Spectrum': [name for name in ordered_spectra if name in param_values],
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
                        file_name="correlation_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Enable peak analysis and select at least one peak to perform correlation analysis")
            else:
                st.info("Enable parameter correlation in sidebar and assign numeric values to spectra")
        
        # Export options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export raw data
            if filtered_spectra:
                export_data = pd.DataFrame()
                for name, spec in filtered_spectra.items():
                    data = spec['data']
                    export_data[f"{name}_x"] = data['x']
                    export_data[f"{name}_y"] = data['y']
                
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Export Raw Data (CSV)",
                    data=csv,
                    file_name="raw_spectra_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Export normalized data
            if filtered_norm_spectra:
                export_norm = pd.DataFrame()
                for name, spec in filtered_norm_spectra.items():
                    data = spec['data']
                    export_norm[f"{name}_x"] = data['x']
                    export_norm[f"{name}_y_norm"] = data['y']
                
                csv_norm = export_norm.to_csv(index=False)
                st.download_button(
                    label="📥 Export Normalized Data (CSV)",
                    data=csv_norm,
                    file_name="normalized_spectra_data.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Save current figure
            st.download_button(
                label="💾 Instructions",
                data="Spectra Analysis Tool v1.0\n\nFeatures:\n- Multiple spectra visualization\n- Normalization (max, area, custom peak)\n- Offset and fill options\n- Peak analysis\n- Parameter correlation\n- Data export",
                file_name="instructions.txt",
                mime="text/plain"
            )
    
    else:
        st.info("👈 Please upload spectra files using the sidebar to begin analysis")
        
        # Show example
        st.markdown("""
        ### 📖 How to use:
        1. **Upload files** - Select one or more .txt files with two columns (x y)
        2. **Select spectra** - Choose which spectra to display and their order
        3. **Customize colors** - Assign different colors to each spectrum
        4. **Set options** - Configure normalization, offset, x-axis ranges
        5. **Analyze** - Explore different visualization tabs
        6. **Export** - Download processed data and analysis results
        
        ### 📊 Features:
        - Raw and normalized spectra visualization
        - Offset with adjustable value
        - Fill under normalized spectra
        - Custom x-axis ranges
        - Peak detection and analysis
        - Parameter correlation (assign numeric values to spectra)
        - High-quality scientific plots
        - Data export in CSV format
        """)

if __name__ == "__main__":
    main()
