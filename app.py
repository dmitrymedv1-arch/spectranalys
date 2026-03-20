"""
🔬 Scientific Data Analyzer for Streamlit
Interactive statistical analysis and visualization tool for scientific research
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import base64
from datetime import datetime
import zipfile
import os

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Min/Max Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply scientific plotting style
plt.style.use('default')
plt.rcParams.update({
    # Font sizes and weights
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    
    # Axes appearance
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    
    # Tick parameters
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
    
    # Legend
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    
    # Figure
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    
    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})

class ScientificDataAnalyzer:
    def __init__(self):
        self.default_colors = self._generate_color_palette()
        
    def _generate_color_palette(self):
        """Generate 20 distinct colors for visualization"""
        return [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
            '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
            '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2',
            '#EF476F', '#7209B7', '#3A86FF', '#FB5607', '#8338EC'
        ]
    
    def parse_data(self, text):
        """Parse data with various separators"""
        if not text or not text.strip():
            return np.array([])
        
        # Replace commas with dots and split
        text = text.strip().replace(',', '.')
        # Split by spaces, newlines, tabs
        lines = text.replace('\n', ' ').replace('\t', ' ').split()
        
        data = []
        for item in lines:
            if item:
                try:
                    num = float(item)
                    if not np.isnan(num):
                        data.append(num)
                except ValueError:
                    continue
        
        return np.array(data) if data else np.array([])
    
    def calculate_statistics(self, data):
        """Calculate comprehensive statistics"""
        if len(data) == 0:
            return {}
        
        stats_dict = {
            'n': len(data),
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'variance': np.var(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'skewness': stats.skew(data) if len(data) > 2 else 0,
            'kurtosis': stats.kurtosis(data) if len(data) > 3 else 0,
            'range': np.max(data) - np.min(data),
            'cv': (np.std(data) / np.mean(data)) * 100 if np.mean(data) != 0 else 0,
            'mad': np.mean(np.abs(data - np.mean(data))),
            'sem': stats.sem(data) if len(data) > 1 else 0,
            'rms': np.sqrt(np.mean(np.square(data))) if len(data) > 0 else 0,
        }
        
        # Mode calculation
        try:
            modes, counts = np.unique(np.round(data, 5), return_counts=True)
            mode_idx = np.argmax(counts)
            stats_dict['mode'] = modes[mode_idx]
            stats_dict['mode_freq'] = counts[mode_idx]
        except:
            stats_dict['mode'] = np.nan
            stats_dict['mode_freq'] = 0
            
        return stats_dict

    def create_histogram_comparison(self, data_sets, set_names, set_colors):
        """Create comparative histogram"""
        fig, ax = plt.subplots(figsize=(8, 6))  # Уменьшен размер для научной публикации
        
        valid_sets = [(name, data) for name, data in data_sets.items() if len(data) > 0]
        
        if not valid_sets:
            ax.text(0.5, 0.5, 'No data available for plotting', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            return fig
        
        bins = 30
        
        for idx, (name, data) in enumerate(valid_sets):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            ax.hist(data, bins=bins, alpha=0.6, label=set_names.get(name, name), 
                   color=color, edgecolor='black', linewidth=0.8)  # Увеличена толщина границ
        
        ax.set_xlabel('Values', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Comparative Histogram', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)  # Изменен стиль сетки
        
        # Добавлены minor ticks
        ax.minorticks_on()
        ax.tick_params(which='both', direction='out', length=4, width=0.8)
        ax.tick_params(which='minor', length=2)
        
        return fig

    def create_normalized_histogram(self, data_sets, set_names, set_colors):
        """Create normalized histograms (PDF)"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        valid_sets = [(name, data) for name, data in data_sets.items() if len(data) > 0]
        
        if not valid_sets:
            ax.text(0.5, 0.5, 'No data available for plotting', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            return fig
        
        for idx, (name, data) in enumerate(valid_sets):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            
            # Kernel Density Estimation
            if len(data) > 1:
                try:
                    kde = stats.gaussian_kde(data)
                    xmin, xmax = data.min(), data.max()
                    x_range = xmax - xmin
                    x = np.linspace(xmin - 0.1*x_range, xmax + 0.1*x_range, 400)
                    ax.plot(x, kde(x), color=color, linewidth=2.0,  # Уменьшена толщина
                           label=f'{set_names.get(name, name)}')
                except:
                    pass
            
            # Normalized histogram
            ax.hist(data, bins=30, density=True, alpha=0.3, 
                   color=color, edgecolor='black', linewidth=0.8)
        
        ax.set_xlabel('Values', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax.set_title('Normalized Histograms', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=9, frameon=True, edgecolor='black')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        
        # Minor ticks
        ax.minorticks_on()
        ax.tick_params(which='both', direction='out', length=4, width=0.8)
        ax.tick_params(which='minor', length=2)
        
        return fig

    def create_box_plot(self, data_sets, set_names, set_colors):
        """Create box plots"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        valid_sets = [(name, data) for name, data in data_sets.items() if len(data) > 0]
        
        if not valid_sets:
            ax.text(0.5, 0.5, 'No data available for plotting', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            return fig
        
        data_to_plot = [data for _, data in valid_sets]
        labels = [set_names.get(name, name) for name, _ in valid_sets]
        
        box = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                        medianprops={'color': 'black', 'linewidth': 1.5},  # Уменьшена толщина
                        whiskerprops={'linewidth': 1.0, 'color': 'black'},
                        capprops={'linewidth': 1.0, 'color': 'black'},
                        boxprops={'linewidth': 1.0},
                        flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'gray'})
        
        # Different colors for each box
        for idx, patch in enumerate(box['boxes']):
            name, _ = valid_sets[idx]
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
        
        ax.set_ylabel('Values', fontsize=11, fontweight='bold')
        ax.set_title('Box Plot Comparison', fontsize=12, fontweight='bold', pad=10)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='y')
        
        # Minor ticks
        ax.minorticks_on()
        ax.tick_params(which='both', direction='out', length=4, width=0.8)
        ax.tick_params(which='minor', length=2)
        
        return fig
    
    def create_violin_plot(self, data_sets, set_names, set_colors):
        """Create violin plot"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        valid_sets = [(name, data) for name, data in data_sets.items() if len(data) > 0]
        
        if not valid_sets:
            ax.text(0.5, 0.5, 'No data available for plotting', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        data_to_plot = [data for _, data in valid_sets]
        labels = [set_names.get(name, name) for name, _ in valid_sets]
        
        violin = ax.violinplot(data_to_plot, showmeans=True, showmedians=True,
                              showextrema=True)
        
        # Colors for violin plot
        for idx, pc in enumerate(violin['bodies']):
            name, _ = valid_sets[idx]
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Customize mean and median lines
        violin['cmeans'].set_color('red')
        violin['cmeans'].set_linewidth(2)
        violin['cmedians'].set_color('blue')
        violin['cmedians'].set_linewidth(2)
        
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Values', fontsize=14)
        ax.set_title('Violin Plot of Dataset Distributions', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5, axis='y')
        
        return fig
    
    def create_4_parameter_analysis(self, data_sets, stats_data, set_names, set_colors):
        """Create comprehensive 4-parameter analysis (Min, Median, Mean, Max)"""
        fig = plt.figure(figsize=(14, 10))
        
        valid_stats = [(name, stats) for name, stats in stats_data.items() if stats]
        
        if not valid_stats:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No statistical data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Create 2x2 subplot grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Bar chart comparison
        ax1 = fig.add_subplot(gs[0, 0])
        x_pos = np.arange(len(valid_stats))
        
        for idx, (name, stats) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            ax1.bar(idx, stats['mean'], width=0.8, color=color, alpha=0.7, 
                   label=set_names.get(name, name))
            # Add error bar for standard deviation
            ax1.errorbar(idx, stats['mean'], yerr=stats['std'], 
                        fmt='none', ecolor='black', capsize=5, linewidth=1.5)
        
        ax1.set_xlabel('Datasets', fontsize=12)
        ax1.set_ylabel('Mean ± SD', fontsize=12)
        ax1.set_title('Mean Values with Standard Deviation', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([set_names.get(name, name) for name, _ in valid_stats], 
                           rotation=45, ha='right')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.1, axis='y')
        
        # 2. Radar chart for 4 parameters
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        
        parameters = ['Min', 'Median', 'Mean', 'Max']
        param_keys = ['min', 'median', 'mean', 'max']
        N = len(parameters)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        
        for idx, (name, stats) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            values = []
            for key in param_keys:
                val = stats.get(key, 0)
                # Normalize by max value across datasets
                max_val = max([s.get(key, 1) for _, s in valid_stats])
                if max_val != 0:
                    values.append(val / max_val)
                else:
                    values.append(0)
            
            values += values[:1]  # Close the radar
            current_angles = angles + angles[:1]
            
            ax2.plot(current_angles, values, 'o-', linewidth=2, 
                   label=set_names.get(name, name), color=color)
            ax2.fill(current_angles, values, alpha=0.1, color=color)
        
        ax2.set_xticks(angles)
        ax2.set_xticklabels(parameters)
        ax2.set_ylim(0, 1)
        ax2.set_title('Normalized 4-Parameter Radar Chart', fontsize=14)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        
        # 3. Parallel coordinates plot
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Prepare data for parallel coordinates
        parallel_data = []
        for name, stats in valid_stats:
            row = [
                stats.get('min', 0),
                stats.get('median', 0),
                stats.get('mean', 0),
                stats.get('max', 0)
            ]
            parallel_data.append(row)
        
        parallel_data = np.array(parallel_data)
        
        # Normalize for better visualization
        for i in range(parallel_data.shape[1]):
            col_min = parallel_data[:, i].min()
            col_max = parallel_data[:, i].max()
            if col_max > col_min:
                parallel_data[:, i] = (parallel_data[:, i] - col_min) / (col_max - col_min)
        
        x_parallel = np.arange(4)
        for idx, (name, _) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            ax3.plot(x_parallel, parallel_data[idx], 'o-', 
                    linewidth=2.5, markersize=8,
                    label=set_names.get(name, name), color=color)
        
        ax3.set_xticks(x_parallel)
        ax3.set_xticklabels(parameters)
        ax3.set_ylabel('Normalized Value', fontsize=12)
        ax3.set_title('Parallel Coordinates: 4-Parameter Comparison', fontsize=14)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.1)
        
        # 4. Bubble chart with 4 parameters
        ax4 = fig.add_subplot(gs[1, 1])
        
        sizes = []
        means = []
        medians = []
        ranges = []
        names_display = []
        
        for name, stats in valid_stats:
            sizes.append(stats['n'])
            means.append(stats['mean'])
            medians.append(stats['median'])
            ranges.append(stats['max'] - stats['min'])
            names_display.append(set_names.get(name, name))
        
        # Normalize bubble sizes
        if sizes:
            sizes_norm = np.array(sizes) / max(sizes) * 1500
        else:
            sizes_norm = []
        
        scatter = ax4.scatter(means, medians, s=sizes_norm, alpha=0.6,
                            c=[set_colors.get(name, self.default_colors[idx % len(self.default_colors)]) 
                               for idx, (name, _) in enumerate(valid_stats)],
                            edgecolors='black', linewidth=1)
        
        # Add dataset names
        for i, name in enumerate(names_display):
            ax4.annotate(name, (means[i], medians[i]), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        ax4.set_xlabel('Mean Value', fontsize=12)
        ax4.set_ylabel('Median Value', fontsize=12)
        ax4.set_title('Bubble Chart: Mean vs Median (size = sample size)', fontsize=14)
        ax4.grid(True, alpha=0.1)
        
        # Add colorbar for ranges
        if ranges:
            norm = plt.Normalize(min(ranges), max(ranges))
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax4, shrink=0.8)
            cbar.set_label('Range (Max-Min)', fontsize=10)
        
        fig.suptitle('Comprehensive 4-Parameter Analysis: Min, Median, Mean, Max', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        return fig
    
    def create_quadrant_analysis(self, stats_data, set_names, set_colors):
        """Create quadrant analysis based on 4 parameters"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        valid_stats = [(name, stats) for name, stats in stats_data.items() if stats]
        
        if not valid_stats:
            for ax in axes:
                ax.text(0.5, 0.5, 'No statistical data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig
        
        # Prepare data
        names = [set_names.get(name, name) for name, _ in valid_stats]
        mins = [stats['min'] for _, stats in valid_stats]
        medians = [stats['median'] for _, stats in valid_stats]
        means = [stats['mean'] for _, stats in valid_stats]
        maxs = [stats['max'] for _, stats in valid_stats]
        
        # 1. Min vs Max scatter
        for idx, (name, _) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            axes[0].scatter(mins[idx], maxs[idx], s=200, alpha=0.7, 
                          color=color, edgecolor='black', linewidth=1.5, 
                          label=set_names.get(name, name))
            axes[0].plot([mins[idx], mins[idx]], [mins[idx], maxs[idx]], 
                       color=color, alpha=0.3, linestyle='--')
        
        axes[0].plot([min(mins), max(mins)], [min(mins), max(mins)], 
                   'k--', alpha=0.5, label='y=x')
        axes[0].set_xlabel('Minimum Value', fontsize=12)
        axes[0].set_ylabel('Maximum Value', fontsize=12)
        axes[0].set_title('Range Analysis: Min vs Max', fontsize=14)
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.1)
        
        # 2. Median vs Mean with error bars
        for idx, (name, stats) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            axes[1].errorbar(means[idx], medians[idx], 
                           xerr=stats['std'], yerr=stats['iqr']/2,
                           fmt='o', color=color, alpha=0.7,
                           ecolor=color, elinewidth=1, capsize=4,
                           label=set_names.get(name, name), markersize=8)
        
        # Add y=x line for reference
        all_values = means + medians
        min_val, max_val = min(all_values), max(all_values)
        axes[1].plot([min_val, max_val], [min_val, max_val], 
                   'k--', alpha=0.5, label='Mean = Median')
        axes[1].set_xlabel('Mean Value', fontsize=12)
        axes[1].set_ylabel('Median Value', fontsize=12)
        axes[1].set_title('Central Tendency: Mean vs Median', fontsize=14)
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(True, alpha=0.1)
        
        # 3. Distribution width analysis
        ranges = [maxs[i] - mins[i] for i in range(len(mins))]
        iqrs = [stats['iqr'] for _, stats in valid_stats]
        
        for idx, (name, _) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            axes[2].bar(idx, ranges[idx], alpha=0.5, color=color, 
                       label=set_names.get(name, name))
            axes[2].bar(idx, iqrs[idx], alpha=0.8, color=color, 
                       edgecolor='black', linewidth=1)
        
        axes[2].set_xlabel('Dataset', fontsize=12)
        axes[2].set_ylabel('Value', fontsize=12)
        axes[2].set_title('Distribution Width: Total Range (light) vs IQR (dark)', fontsize=14)
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(names, rotation=45, ha='right')
        axes[2].legend(loc='best', fontsize=9)
        axes[2].grid(True, alpha=0.1, axis='y')
        
        # 4. 4-Parameter summary plot
        x_pos = np.arange(len(valid_stats))
        width = 0.15
        
        for idx, (name, stats) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            # Plot min, median, mean, max as separate bars
            axes[3].bar(x_pos[idx] - 1.5*width, stats['min'], width, 
                       color=color, alpha=0.3, label='Min' if idx == 0 else "")
            axes[3].bar(x_pos[idx] - 0.5*width, stats['median'], width, 
                       color=color, alpha=0.5, label='Median' if idx == 0 else "")
            axes[3].bar(x_pos[idx] + 0.5*width, stats['mean'], width, 
                       color=color, alpha=0.7, label='Mean' if idx == 0 else "")
            axes[3].bar(x_pos[idx] + 1.5*width, stats['max'], width, 
                       color=color, alpha=0.9, label='Max' if idx == 0 else "")
        
        axes[3].set_xlabel('Dataset', fontsize=12)
        axes[3].set_ylabel('Value', fontsize=12)
        axes[3].set_title('4-Parameter Summary for Each Dataset', fontsize=14)
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(names, rotation=45, ha='right')
        axes[3].legend(loc='best', fontsize=9)
        axes[3].grid(True, alpha=0.1, axis='y')
        
        fig.suptitle('Quadrant Analysis of 4 Key Parameters', fontsize=16, y=0.98)
        plt.tight_layout()
        
        return fig
    
    def create_statistical_summary_matrix(self, stats_data, set_names, set_colors):
        """Create a matrix plot of statistical summaries"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        valid_stats = [(name, stats) for name, stats in stats_data.items() if stats]
        
        if not valid_stats:
            for ax in axes.flatten():
                ax.text(0.5, 0.5, 'No statistical data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig
        
        # Prepare data arrays
        n_sets = len(valid_stats)
        names = [set_names.get(name, name) for name, _ in valid_stats]
        
        # 1. Heatmap of all statistics
        all_stats_list = []
        stat_names = []
        for name, stats in valid_stats:
            row = [
                stats.get('min', np.nan),
                stats.get('q1', np.nan),
                stats.get('median', np.nan),
                stats.get('mean', np.nan),
                stats.get('q3', np.nan),
                stats.get('max', np.nan),
                stats.get('std', np.nan),
                stats.get('iqr', np.nan),
                stats.get('skewness', np.nan),
                stats.get('kurtosis', np.nan)
            ]
            all_stats_list.append(row)
        
        stat_names = ['Min', 'Q1', 'Median', 'Mean', 'Q3', 'Max', 'Std', 'IQR', 'Skew', 'Kurt']
        
        im1 = axes[0, 0].imshow(all_stats_list, aspect='auto', cmap='RdYlBu_r')
        axes[0, 0].set_xticks(range(len(stat_names)))
        axes[0, 0].set_xticklabels(stat_names, rotation=45, ha='right')
        axes[0, 0].set_yticks(range(len(names)))
        axes[0, 0].set_yticklabels(names)
        axes[0, 0].set_title('Statistical Summary Heatmap', fontsize=14)
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. Correlation matrix of statistics between datasets
        if n_sets > 1:
            # Create correlation matrix
            corr_matrix = np.corrcoef(all_stats_list)
            im2 = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0, 1].set_xticks(range(n_sets))
            axes[0, 1].set_yticks(range(n_sets))
            axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
            axes[0, 1].set_yticklabels(names)
            axes[0, 1].set_title('Inter-Dataset Correlation Matrix', fontsize=14)
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # Add correlation values
            for i in range(n_sets):
                for j in range(n_sets):
                    text = axes[0, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                         ha="center", va="center",
                                         color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
        else:
            axes[0, 1].text(0.5, 0.5, 'Need at least 2 datasets\nfor correlation matrix',
                          ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. Parallel coordinates for key statistics
        ax3 = axes[1, 0]
        key_stats = ['min', 'median', 'mean', 'max', 'std', 'iqr']
        key_labels = ['Min', 'Median', 'Mean', 'Max', 'Std', 'IQR']
        
        # Normalize each statistic
        normalized_data = []
        for name, stats in valid_stats:
            row = [stats.get(key, 0) for key in key_stats]
            normalized_data.append(row)
        
        normalized_data = np.array(normalized_data)
        for i in range(len(key_stats)):
            col = normalized_data[:, i]
            if np.max(col) > np.min(col):
                normalized_data[:, i] = (col - np.min(col)) / (np.max(col) - np.min(col))
        
        for idx, (name, _) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            ax3.plot(range(len(key_stats)), normalized_data[idx], 'o-',
                    linewidth=2, markersize=6, color=color, 
                    label=set_names.get(name, name))
        
        ax3.set_xticks(range(len(key_stats)))
        ax3.set_xticklabels(key_labels, rotation=45, ha='right')
        ax3.set_ylabel('Normalized Value', fontsize=12)
        ax3.set_title('Parallel Coordinates: Key Statistics', fontsize=14)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.1)
        
        # 4. Bubble chart showing relationships
        ax4 = axes[1, 1]
        
        # Prepare bubble parameters
        bubble_sizes = [stats['n'] for _, stats in valid_stats]
        x_vals = [stats['mean'] for _, stats in valid_stats]
        y_vals = [stats['median'] for _, stats in valid_stats]
        colors = [set_colors.get(name, self.default_colors[idx % len(self.default_colors)]) 
                 for idx, (name, _) in enumerate(valid_stats)]
        
        # Normalize bubble sizes
        if bubble_sizes:
            max_size = max(bubble_sizes)
            sizes_norm = [size/max_size * 1000 for size in bubble_sizes]
        else:
            sizes_norm = []
        
        scatter = ax4.scatter(x_vals, y_vals, s=sizes_norm, alpha=0.6,
                            c=colors, edgecolors='black', linewidth=1)
        
        # Add dataset labels
        for i, name in enumerate(names):
            ax4.annotate(name, (x_vals[i], y_vals[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # Add y=x reference line
        if x_vals and y_vals:
            all_vals = x_vals + y_vals
            min_val, max_val = min(all_vals), max(all_vals)
            ax4.plot([min_val, max_val], [min_val, max_val], 
                    'k--', alpha=0.3, label='Mean = Median')
        
        ax4.set_xlabel('Mean Value', fontsize=12)
        ax4.set_ylabel('Median Value', fontsize=12)
        ax4.set_title('Mean vs Median (bubble size = sample size)', fontsize=14)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.1)
        
        fig.suptitle('Statistical Summary Matrix Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        return fig
    
    def create_log_comparison(self, data_sets, set_names, set_colors):
        """Create logarithmic scale comparison plots"""
        fig = plt.figure(figsize=(14, 10))
        
        valid_sets = [(name, data) for name, data in data_sets.items() if len(data) > 0]
        
        if not valid_sets:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data available for plotting', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Create 2x3 grid for logarithmic plots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Log histogram
        ax1 = fig.add_subplot(gs[0, 0])
        for idx, (name, data) in enumerate(valid_sets):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            positive_data = data[data > 0]
            if len(positive_data) > 0:
                ax1.hist(positive_data, bins=30, alpha=0.6, 
                        color=color, label=set_names.get(name, name), edgecolor='black')
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Values (log scale)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Logarithmic Histogram', fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.1, which='both')
        
        # 2. Log box plot
        ax2 = fig.add_subplot(gs[0, 1])
        log_data = []
        log_labels = []
        log_indices = []
        for idx, (name, data) in enumerate(valid_sets):
            positive_data = data[data > 0]
            if len(positive_data) > 1:
                log_data.append(np.log10(positive_data))
                log_labels.append(set_names.get(name, name))
                log_indices.append(idx)
        
        if log_data:
            box = ax2.boxplot(log_data, labels=log_labels, patch_artist=True)
            for box_idx, patch in enumerate(box['boxes']):
                orig_idx = log_indices[box_idx]
                name, _ = valid_sets[orig_idx]
                color = set_colors.get(name, self.default_colors[orig_idx % len(self.default_colors)])
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_ylabel('log10(Values)', fontsize=12)
            ax2.set_title('Box Plot of Log-Transformed Data', fontsize=14)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax2.grid(True, alpha=0.1, axis='y')
        
        # 3. Q-Q plot on log scale
        ax3 = fig.add_subplot(gs[0, 2])
        for idx, (name, data) in enumerate(valid_sets):
            if len(data) > 10:
                color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
                stats.probplot(data, dist="norm", plot=ax3)
                ax3.get_lines()[0].set_color(color)
                ax3.get_lines()[0].set_alpha(0.6)
                ax3.get_lines()[1].set_color('red')
        
        ax3.set_title('Q-Q Plot (vs Normal Distribution)', fontsize=14)
        ax3.grid(True, alpha=0.1)
        
        # 4. Cumulative distribution function
        ax4 = fig.add_subplot(gs[1, 0])
        for idx, (name, data) in enumerate(valid_sets):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            sorted_data = np.sort(data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax4.plot(sorted_data, y, '-', color=color, linewidth=2, 
                    label=set_names.get(name, name))
        
        ax4.set_xlabel('Values', fontsize=12)
        ax4.set_ylabel('Cumulative Probability', fontsize=12)
        ax4.set_title('Empirical Cumulative Distribution', fontsize=14)
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Log-log plot
        ax5 = fig.add_subplot(gs[1, 1])
        for idx, (name, data) in enumerate(valid_sets):
            if len(data) > 10:
                color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
                sorted_data = np.sort(data)
                rank = np.arange(1, len(sorted_data) + 1)
                ax5.loglog(sorted_data, rank, 'o-', markersize=3, 
                          linewidth=1, color=color, alpha=0.7, 
                          label=set_names.get(name, name))
        
        ax5.set_xlabel('Value (log scale)', fontsize=12)
        ax5.set_ylabel('Rank (log scale)', fontsize=12)
        ax5.set_title('Log-Log Rank Plot', fontsize=14)
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.1, which='both')
        
        # 6. Comparative density on log scale
        ax6 = fig.add_subplot(gs[1, 2])
        for idx, (name, data) in enumerate(valid_sets):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            positive_data = data[data > 0]
            if len(positive_data) > 1:
                # KDE on log-transformed data
                log_data = np.log10(positive_data)
                kde = stats.gaussian_kde(log_data)
                x_min, x_max = log_data.min(), log_data.max()
                x = np.linspace(x_min, x_max, 200)
                ax6.plot(10**x, kde(x), color=color, linewidth=2, 
                        label=set_names.get(name, name))
        
        ax6.set_xscale('log')
        ax6.set_xlabel('Values (log scale)', fontsize=12)
        ax6.set_ylabel('Density', fontsize=12)
        ax6.set_title('Density on Logarithmic Scale', fontsize=14)
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.1, which='both')
        
        fig.suptitle('Logarithmic Scale Analysis and Comparisons', fontsize=16, y=0.98)
        plt.tight_layout()
        
        return fig
    
    def create_bubble_chart_statistics(self, stats_data, set_names, set_colors):
        """Create bubble chart for statistical comparison"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        valid_stats = [(name, stats) for name, stats in stats_data.items() if stats]
        
        if len(valid_stats) < 2:
            ax.text(0.5, 0.5, 'At least 2 datasets required for comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Prepare data
        names = [set_names.get(name, name) for name, _ in valid_stats]
        means = [stats['mean'] for _, stats in valid_stats]
        stds = [stats['std'] for _, stats in valid_stats]
        medians = [stats['median'] for _, stats in valid_stats]
        iqrs = [stats['iqr'] for _, stats in valid_stats]
        counts = [stats['n'] for _, stats in valid_stats]
        
        # Normalize bubble sizes
        sizes = np.array(counts) / max(counts) * 2000
        
        # Create scatter plot
        scatter = ax.scatter(means, medians, s=sizes, alpha=0.7,
                           c=[set_colors.get(name, self.default_colors[idx % len(self.default_colors)]) 
                              for idx, (name, _) in enumerate(valid_stats)],
                           edgecolors='black', linewidth=1.5)
        
        # Add error bars for std and iqr
        for idx, (name, stats) in enumerate(valid_stats):
            color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
            # Horizontal error bar (std)
            ax.errorbar(means[idx], medians[idx], 
                       xerr=stds[idx], fmt='none',
                       ecolor=color, alpha=0.5, linewidth=1)
            # Vertical error bar (IQR/2)
            ax.errorbar(means[idx], medians[idx],
                       yerr=iqrs[idx]/2, fmt='none',
                       ecolor=color, alpha=0.5, linewidth=1)
            
            # Add dataset name
            ax.annotate(set_names.get(name, name), (means[idx], medians[idx]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor=color, alpha=0.2))
        
        # Add reference lines
        all_values = means + medians
        min_val, max_val = min(all_values), max(all_values)
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', alpha=0.3, linewidth=1, label='Mean = Median')
        
        ax.set_xlabel('Mean Value', fontsize=14)
        ax.set_ylabel('Median Value', fontsize=14)
        ax.set_title('Statistical Comparison Bubble Chart\n' +
                    'Bubble size = Sample size | Error bars = Std (horizontal) & IQR/2 (vertical)', 
                    fontsize=16, pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.1)
        ax.set_axisbelow(True)
        
        # Add colorbar for IQR
        if iqrs:
            norm = plt.Normalize(min(iqrs), max(iqrs))
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.01)
            cbar.set_label('Interquartile Range (IQR)', fontsize=12)
        
        return fig
    
    def create_interactive_plot(self, data_sets, set_names, set_colors):
        """Create an interactive plot using plotly"""
        try:
            valid_sets = [(name, data) for name, data in data_sets.items() if len(data) > 0]
            
            if not valid_sets:
                fig = go.Figure()
                fig.add_annotation(text="No data available for plotting",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Create interactive box plot
            fig = go.Figure()
            
            for idx, (name, data) in enumerate(valid_sets):
                color = set_colors.get(name, self.default_colors[idx % len(self.default_colors)])
                fig.add_trace(go.Box(
                    y=data,
                    name=set_names.get(name, name),
                    boxpoints='outliers',
                    marker_color=color,
                    line_color='black',
                    showlegend=True
                ))
            
            fig.update_layout(
                title="Interactive Box Plot Comparison",
                yaxis_title="Values",
                boxmode='group',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="serif", size=12),
                height=500
            )
            
            return fig
            
        except Exception as e:
            # Fallback to matplotlib if plotly fails
            st.warning(f"Plotly error: {e}, falling back to matplotlib")
            return self.create_box_plot(data_sets, set_names, set_colors)

def create_download_link(figures, prefix="figure"):
    """Create download link for all figures"""
    import zipfile
    from io import BytesIO
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{prefix}_{timestamp}.zip"
    
    # Create in-memory zip file
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        for i, fig in enumerate(figures):
            try:
                # Check if it's a plotly figure
                if hasattr(fig, 'write_image'):
                    # Try to save plotly figure as PNG
                    try:
                        import kaleido  # Import to ensure it's available
                        img_bytes = fig.to_image(format="png", width=1200, height=800)
                        zip_file.writestr(f"{prefix}_{i+1:02d}.png", img_bytes)
                    except Exception as e:
                        # If kaleido fails, save as HTML
                        st.warning(f"Kaleido export failed: {e}. Saving plot as HTML instead.")
                        html_content = fig.to_html(include_plotlyjs='cdn')
                        zip_file.writestr(f"{prefix}_{i+1:02d}.html", html_content)
                    
                else:
                    # Save matplotlib figure as PNG
                    fig_buffer = BytesIO()
                    fig.savefig(fig_buffer, format='png', bbox_inches='tight', dpi=300)
                    fig_buffer.seek(0)
                    zip_file.writestr(f"{prefix}_{i+1:02d}.png", fig_buffer.read())
                    
                    # Also save as PDF
                    fig_buffer_pdf = BytesIO()
                    fig.savefig(fig_buffer_pdf, format='pdf', bbox_inches='tight')
                    fig_buffer_pdf.seek(0)
                    zip_file.writestr(f"{prefix}_{i+1:02d}.pdf", fig_buffer_pdf.read())
                    
            except Exception as e:
                # If saving fails, create a placeholder
                error_msg = f"Error saving figure {i+1}: {str(e)}"
                zip_file.writestr(f"{prefix}_{i+1:02d}_ERROR.txt", error_msg)
    
    buffer.seek(0)
    
    # Check if zip file is empty
    if buffer.getbuffer().nbytes == 0:
        st.error("No figures could be saved for download. Please check if any plots were generated.")
        return None
    
    # Create download button
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">📥 Download All Figures as ZIP</a>'
    return href

def main():
    # Initialize analyzer
    analyzer = ScientificDataAnalyzer()
    
    # Sidebar configuration
    st.sidebar.title("🔬 Min/max Analyzer")
    st.sidebar.markdown("---")
    
    # Plot selection
    st.sidebar.subheader("📈 Plot Selection")
    plot_options = {
        'Comparative Histogram': st.sidebar.checkbox("Comparative Histogram", value=True),
        'Normalized Histograms': st.sidebar.checkbox("Normalized Histograms", value=True),
        'Box Plots': st.sidebar.checkbox("Box Plots", value=True),
        'Violin Plots': st.sidebar.checkbox("Violin Plots", value=True),
        '4-Parameter Analysis': st.sidebar.checkbox("4-Parameter Analysis", value=True),
        'Quadrant Analysis': st.sidebar.checkbox("Quadrant Analysis", value=True),
        'Statistical Matrix': st.sidebar.checkbox("Statistical Matrix", value=True),
        'Logarithmic Analysis': st.sidebar.checkbox("Logarithmic Analysis", value=True),
        'Bubble Chart': st.sidebar.checkbox("Bubble Chart", value=True),
        'Interactive Plot': st.sidebar.checkbox("Interactive Plot", value=True)
    }
    
    st.sidebar.markdown("---")
    
    # Color scheme
    st.sidebar.subheader("🎨 Color Settings")
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        ['Default', 'Viridis', 'Plasma', 'Set2', 'Set3', 'Tab20', 'Accent', 'Dark2']
    )
    
    # Update color scheme if needed
    if color_scheme != 'Default':
        try:
            if color_scheme in ['Viridis', 'Plasma']:
                cmap = plt.cm.get_cmap(color_scheme.lower())
                new_colors = [cmap(i) for i in np.linspace(0, 1, 20)]
                new_colors_hex = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' 
                                for r, g, b, _ in new_colors]
                analyzer.default_colors = new_colors_hex
        except:
            pass
    
    st.sidebar.markdown("---")
    
    # Main content
    st.title("🔬 Min/max Analyzer")
    st.markdown("Interactive statistical analysis and visualization tool for scientific research")
    
    # Instructions
    with st.expander("📖 Instructions & Features", expanded=False):
        st.markdown("""
        ### 📝 Data Input Format
        - Enter numbers separated by spaces, commas, or newlines
        - Decimal separator can be . or ,
        - Example: `1.23 2.34 3.45` or `1,23\\n2,34\\n3,45`
        
        ### 🎨 Customization Options
        - Give each dataset a meaningful name
        - Choose custom colors for each dataset
        - Select which plots to generate
        - Apply different color schemes
        
        ### 📊 Available Plots
        - **Comparative Histogram:** Side-by-side frequency comparison
        - **Normalized Histograms:** Probability density functions
        - **4-Parameter Analysis:** Comprehensive min/median/mean/max visualization
        - **Quadrant Analysis:** Multi-faceted comparison of key statistics
        - **Logarithmic Analysis:** Log-scale plots for wide-ranging data
        - **Statistical Matrix:** Heatmaps and correlation analysis
        """)
    
    st.markdown("---")
    
    # Data input section
    st.subheader("📥 Data Input")
    
    # Create tabs for datasets
    tabs = st.tabs([f"Dataset {i+1}" for i in range(10)])
    
    # Initialize session state for datasets
    if 'data_sets' not in st.session_state:
        st.session_state.data_sets = {}
    if 'set_names' not in st.session_state:
        st.session_state.set_names = {}
    if 'set_colors' not in st.session_state:
        st.session_state.set_colors = {}
    if 'stats_data' not in st.session_state:
        st.session_state.stats_data = {}
    
    # Load example data
    example_data_1 = """10.66791879
6.143209248
6.1
5.502455375
4.765145385
4.63
4.555425043
4.164980605
4.126343637
3.54
3.517386491
3.09
2.589827043
2.244136246
2.06
1.999933447
1.99896
1.47784
1.30915
1.1868
1.1868
0.816072865
0.74
0.72
0.72
0.703496997
0.628162903
0.61
0.604960343
0.596784758
0.55
0.51135
0.50806
0.501519916
0.456359991
0.4318
0.430257439
0.407245175
0.40495
0.38193
0.378964677
0.357
0.315324337
0.30626
0.304660485
0.29821595
0.294815954
0.293834644
0.291047114
0.28968
0.288
0.279
0.26
0.25182
0.22637
0.224932551
0.211723411
0.19731
0.193429261
0.187541109
0.186577749
0.17512204
0.1718769
0.159353711
0.143
0.139026145
0.12
0.108986138"""
    
    example_data_2 = """0.05246
0.06512
0.07358
0.106963079
0.115349256
0.13216
0.14666
0.15604
0.176262619
0.18076
0.20464
0.252
0.26
0.284
0.312
0.32
0.344
0.38
0.413245033
0.433566434
0.47132
0.522220329
0.546
0.613397635
1
1.091290368
1.2
1.20285195
1.318403662
1.7
1.72
2.12
3.31"""
    
    # Create dataset inputs
    for i, tab in enumerate(tabs):
        with tab:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Dataset name
                default_name = f"Dataset {i+1}"
                if i == 0:
                    default_name = "Dataset 1 (High Values)"
                elif i == 1:
                    default_name = "Dataset 2 (Low Values)"
                
                name = st.text_input("Dataset Name", 
                                   value=default_name,
                                   key=f"name_{i}")
                
                # Dataset color
                default_color = analyzer.default_colors[i % len(analyzer.default_colors)]
                color = st.color_picker("Dataset Color", 
                                       value=default_color,
                                       key=f"color_{i}")
                
                # Load example data button for first two datasets
                if i < 2:
                    if st.button(f"Load Example Data {i+1}", key=f"example_{i}"):
                        if i == 0:
                            st.session_state[f"data_{i}"] = example_data_1
                        else:
                            st.session_state[f"data_{i}"] = example_data_2
            
            with col2:
                # Data input
                data_key = f"data_{i}"
                if data_key not in st.session_state:
                    st.session_state[data_key] = ""
                
                data = st.text_area("Enter Data (one value per line or space-separated)", 
                                  value=st.session_state[data_key],
                                  height=200,
                                  key=f"textarea_{i}",
                                  placeholder="Enter numbers separated by spaces, commas, or newlines\nExample: 1.23 2.34 3.45")
                
                # Update session state
                st.session_state[f"data_{i}"] = data
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze Data", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    if analyze_button:
        # Clear previous data
        st.session_state.data_sets = {}
        st.session_state.set_names = {}
        st.session_state.set_colors = {}
        st.session_state.stats_data = {}
        
        # Collect data from all tabs
        valid_datasets = 0
        for i in range(10):
            data_key = f"data_{i}"
            name_key = f"name_{i}"
            color_key = f"color_{i}"
            
            if data_key in st.session_state and st.session_state[data_key]:
                data_text = st.session_state[data_key]
                name = st.session_state[name_key] if name_key in st.session_state else f"Dataset {i+1}"
                color = st.session_state[color_key] if color_key in st.session_state else analyzer.default_colors[i % len(analyzer.default_colors)]
                
                data = analyzer.parse_data(data_text)
                if len(data) > 0:
                    dataset_id = f"dataset_{i}"
                    st.session_state.data_sets[dataset_id] = data
                    st.session_state.set_names[dataset_id] = name
                    st.session_state.set_colors[dataset_id] = color
                    st.session_state.stats_data[dataset_id] = analyzer.calculate_statistics(data)
                    valid_datasets += 1
        
        if valid_datasets == 0:
            st.error("⚠️ No valid data found. Please enter data in at least one dataset.")
            return
        
        # Display dataset overview
        st.success(f"✅ Loaded {valid_datasets} dataset(s) for analysis")
        
        # Create overview table
        st.subheader("📊 Dataset Overview")
        
        overview_data = []
        for dataset_id, stats in st.session_state.stats_data.items():
            if stats:
                name = st.session_state.set_names.get(dataset_id, dataset_id)
                color = st.session_state.set_colors.get(dataset_id, "#cccccc")
                overview_data.append({
                    'Dataset': name,
                    'N': stats['n'],
                    'Min': f"{stats['min']:.4f}",
                    'Mean': f"{stats['mean']:.4f}",
                    'Median': f"{stats['median']:.4f}",
                    'Max': f"{stats['max']:.4f}",
                    'Std': f"{stats['std']:.4f}",
                    'IQR': f"{stats['iqr']:.4f}"
                })
        
        if overview_data:
            df_overview = pd.DataFrame(overview_data)
            st.dataframe(df_overview, use_container_width=True)
        
        # Create plots based on selection
        st.subheader("📈 Visualization Results")
        
        # Define plot functions mapping
        plot_functions = {
            'Comparative Histogram': lambda: analyzer.create_histogram_comparison(
                st.session_state.data_sets, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            'Normalized Histograms': lambda: analyzer.create_normalized_histogram(
                st.session_state.data_sets, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            'Box Plots': lambda: analyzer.create_box_plot(
                st.session_state.data_sets, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            'Violin Plots': lambda: analyzer.create_violin_plot(
                st.session_state.data_sets, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            '4-Parameter Analysis': lambda: analyzer.create_4_parameter_analysis(
                st.session_state.data_sets, 
                st.session_state.stats_data, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            'Quadrant Analysis': lambda: analyzer.create_quadrant_analysis(
                st.session_state.stats_data, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            'Statistical Matrix': lambda: analyzer.create_statistical_summary_matrix(
                st.session_state.stats_data, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            'Logarithmic Analysis': lambda: analyzer.create_log_comparison(
                st.session_state.data_sets, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            'Bubble Chart': lambda: analyzer.create_bubble_chart_statistics(
                st.session_state.stats_data, 
                st.session_state.set_names, 
                st.session_state.set_colors
            ),
            'Interactive Plot': lambda: analyzer.create_interactive_plot(
                st.session_state.data_sets, 
                st.session_state.set_names, 
                st.session_state.set_colors
            )
        }
        
        # Generate and display selected plots
        figures = []
        for plot_name, plot_func in plot_functions.items():
            if plot_options.get(plot_name, False):
                with st.spinner(f"Creating {plot_name}..."):
                    try:
                        fig = plot_func()
                        figures.append(fig)
                        
                        # Display plot
                        if plot_name == 'Interactive Plot':
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.pyplot(fig)
                        
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error creating {plot_name}: {str(e)}")
        
        # Display detailed statistics
        st.subheader("📋 Detailed Statistics")
        
        for dataset_id, stats in st.session_state.stats_data.items():
            if stats:
                with st.expander(f"{st.session_state.set_names.get(dataset_id, dataset_id)} - Detailed Statistics"):
                    stats_df = pd.DataFrame([stats]).T
                    stats_df.columns = ['Value']
                    st.dataframe(stats_df, use_container_width=True)
        
        # Download section
        if figures:
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Download All Figures")
                download_link = create_download_link(figures, "scientific_analysis")
                if download_link:
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.warning("Could not create download link. No valid figures to save.")
                st.markdown(download_link, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Export Statistics")
                if st.button("📊 Export Statistics to CSV"):
                    # Combine all statistics
                    all_stats = {}
                    for dataset_id, stats in st.session_state.stats_data.items():
                        if stats:
                            name = st.session_state.set_names.get(dataset_id, dataset_id)
                            all_stats[name] = stats
                    
                    if all_stats:
                        stats_df = pd.DataFrame(all_stats).T
                        csv = stats_df.to_csv().encode('utf-8')
                        
                        st.download_button(
                            label="📥 Download Statistics CSV",
                            data=csv,
                            file_name="statistical_analysis.csv",
                            mime="text/csv"
                        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🔬 Min/max Analyzer | Designed for scientific research publications  | developed by @daM, https://chimicatechnoacta.ru </p>
        <p>All plots are optimized for scientific papers with white background and black borders</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()




