"""
Visualization utilities for simulation results comparison.

This module provides plotting functions with standardized styling for
comparing different methods across various metrics.
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# Styling Configuration
# ==============================================================================

# Method colors (consistent across all plots)
METHOD_COLORS = {
    'rashomon': 'dodgerblue',
    'blasso': 'seagreen',
    'ssl': 'purple',
    'ppmx': 'darkorange',
    'bootstrap': 'orangered',
    'lasso': 'indianred',
}

# Method display names
METHOD_NAMES = {
    'rashomon': 'Rashomon Set',
    'blasso': 'Bayesian Lasso',
    'ssl': 'Spike-Slab Lasso',
    'ppmx': 'PPMx',
    'bootstrap': 'Bootstrap Lasso',
    'lasso': 'Lasso',
}

# Markers for line plots
METHOD_MARKERS = {
    'rashomon': 'd',   # diamond
    'blasso': 'o',     # circle
    'ssl': 's',        # square
    'ppmx': '*',       # star
    'bootstrap': '^',  # triangle
}

# Marker sizes
MARKER_SIZES = {
    'rashomon': 8,
    'blasso': 8,
    'ssl': 8,
    'ppmx': 10,  # Larger for star
    'bootstrap': 8,
}


def apply_clean_style(ax):
    """Apply clean styling to axis (remove top and right spines)."""
    ax.spines[['right', 'top']].set_visible(False)


# ==============================================================================
# Epsilon Curve Plotting
# ==============================================================================

def plot_epsilon_comparison(epsilon_curves, lasso_point=None, figsize=(10, 6),
                            save_path=None):
    """
    Plot epsilon curves comparing recovery rates across methods.

    Parameters
    ----------
    epsilon_curves : dict
        Dictionary mapping method names to epsilon curve dataframes.
        Each dataframe should have columns: ['eps_levels', 'profile_rate_eps']
    lasso_point : tuple, optional
        (x, y) coordinates for Lasso point estimate. If provided, plotted as scatter.
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save figure. If None, figure is not saved.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    apply_clean_style(ax)

    # Find max epsilon for consistent x-axis
    max_eps = max(np.max(df['eps_levels']) for df in epsilon_curves.values())

    # Reference line for perfect recovery
    len_x = 10
    x_horizontal = np.linspace(0, max_eps, num=len_x)
    y_best = np.array([1] * len_x)
    ax.plot(x_horizontal, y_best, color='black', ls='--', linewidth=1,
            alpha=0.5, label='Perfect recovery')

    # Plot epsilon curves for each method
    for method, df in epsilon_curves.items():
        color = METHOD_COLORS.get(method, 'gray')
        label = METHOD_NAMES.get(method, method)

        # Plot line
        ax.plot(df['eps_levels'], df['profile_rate_eps'],
                color=color, linewidth=2.5, zorder=3, clip_on=False, label=label)

        # Add scatter point at start for emphasis
        ax.scatter(df['eps_levels'].iloc[0], df['profile_rate_eps'].iloc[0],
                   color=color, edgecolor='black', s=60, zorder=3.2, clip_on=False)

    # Add Lasso point if provided
    if lasso_point is not None:
        ax.scatter([lasso_point[0]], [lasso_point[1]],
                   color=METHOD_COLORS['lasso'], edgecolor='black',
                   label=METHOD_NAMES['lasso'], s=60,
                   zorder=3.1, clip_on=False)

    # Axis settings
    ax.set_xlim(0, max_eps)
    ax.set_xlabel('Epsilon (%)', fontsize=13)
    ax.set_xscale('linear')

    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Recovery Rate of True Best Profile', fontsize=13)

    ax.set_title('Best Profile Recovery Rate', fontsize=14, pad=15)

    # Legend and grid
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# ==============================================================================
# Bar Chart Comparison
# ==============================================================================

def plot_hpd_bar_comparison(comparison_df, metrics_to_plot=None,
                            figsize=(20, 12), save_path=None):
    """
    Plot bar chart comparison of methods across multiple metrics.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: 'Method' and metric columns
    metrics_to_plot : list of dict, optional
        List of metric specifications. Each dict should have:
        - 'column': Column name in comparison_df
        - 'title': Plot title
        - 'ylabel': Y-axis label
        If None, plots default metrics.
    figsize : tuple, default=(20, 12)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes) matplotlib objects
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            {'column': 'HPD_Frequency', 'title': 'HPD/Top-95%\nFrequency',
             'ylabel': 'Inclusion Rate'},
            {'column': 'HPD_Presence', 'title': 'HPD/Top-95%\nBinary Presence',
             'ylabel': 'Coverage Rate'},
            {'column': 'Full_Frequency', 'title': 'Full Posterior\nFrequency',
             'ylabel': 'Inclusion Rate'},
        ]

    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Get colors based on method names in dataframe
    colors = [METHOD_COLORS.get(m.lower(), 'gray')
              for m in comparison_df['Method']]
    x_pos = np.arange(len(comparison_df))

    for idx, metric_spec in enumerate(metrics_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        column = metric_spec['column']
        if column not in comparison_df.columns:
            continue

        # Plot bars
        bars = ax.bar(x_pos, comparison_df[column], color=colors,
                      alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparison_df['Method'], rotation=30, fontsize=11)
        ax.set_ylabel(metric_spec['ylabel'], fontsize=13)
        ax.set_title(metric_spec['title'], fontsize=12)
        ax.set_ylim(0, 1.05)

        apply_clean_style(ax)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


# ==============================================================================
# Credible Interval Sweep
# ==============================================================================

def plot_credible_interval_sweep(sweep_results, methods=None,
                                 figsize=(16, 12), save_path=None):
    """
    Plot how metrics change with credible interval size.

    Creates a 2x2 grid showing:
    - Coverage vs credible level (full range and magnified)
    - Sample size vs credible level (full range and magnified)

    Parameters
    ----------
    sweep_results : pd.DataFrame
        DataFrame with columns:
        - 'credible_level': Credible interval sizes (0-1)
        - '{method}_coverage': Coverage for each method
        - '{method}_mean_size': Mean sample size for each method
    methods : list of str, optional
        List of methods to plot. If None, uses ['rashomon', 'blasso', 'ssl', 'ppmx']
    figsize : tuple, default=(16, 12)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, axes) matplotlib objects
    """
    if methods is None:
        methods = ['rashomon', 'blasso', 'ssl', 'ppmx']

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Convert credible_level to percentage
    x_full = sweep_results['credible_level'] * 100

    # Plot 1: Coverage vs Credible Level (full range)
    ax1 = axes[0, 0]
    for method in methods:
        col = f'{method}_coverage'
        if col not in sweep_results.columns:
            continue
        ax1.plot(x_full, sweep_results[col],
                 marker=METHOD_MARKERS.get(method, 'o'),
                 markersize=MARKER_SIZES.get(method, 8),
                 linewidth=2.5,
                 label=METHOD_NAMES.get(method, method),
                 color=METHOD_COLORS.get(method, 'gray'))

    ax1.set_xlabel('HPD Region Cutoff (%)', fontsize=13)
    ax1.set_ylabel('Recovery Rate', fontsize=13)
    ax1.set_title('Best Profile Recovery Rate in HPD Region', fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 105)
    apply_clean_style(ax1)

    # Plot 2: Mean sample size (full range)
    ax2 = axes[0, 1]
    for method in methods:
        col = f'{method}_mean_size'
        if col not in sweep_results.columns:
            continue
        ax2.plot(x_full, sweep_results[col],
                 marker=METHOD_MARKERS.get(method, 'o'),
                 markersize=MARKER_SIZES.get(method, 8),
                 linewidth=2.5,
                 label=METHOD_NAMES.get(method, method),
                 color=METHOD_COLORS.get(method, 'gray'))

    ax2.set_yscale('log')
    ax2.set_xlim(0, 105)
    ax2.set_xlabel('HPD Region Cutoff (%)', fontsize=13)
    ax2.set_ylabel('Number of Models', fontsize=13)
    ax2.set_title('Number of Models in HPD Region', fontsize=13)
    ax2.legend(loc='upper left', fontsize=11)
    apply_clean_style(ax2)

    # Plot 3: Coverage (magnified near 100%)
    ax3 = axes[1, 0]
    for method in methods:
        col = f'{method}_coverage'
        if col not in sweep_results.columns:
            continue
        ax3.plot(x_full, sweep_results[col],
                 marker=METHOD_MARKERS.get(method, 'o'),
                 markersize=MARKER_SIZES.get(method, 8),
                 linewidth=2.5,
                 label=METHOD_NAMES.get(method, method),
                 color=METHOD_COLORS.get(method, 'gray'))

    ax3.set_xlabel('HPD Region Cutoff (%)', fontsize=13)
    ax3.set_ylabel('Recovery Rate', fontsize=13)
    ax3.set_title('Best Profile Recovery Rate in HPD Region (Magnified)', fontsize=13)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlim(99, 100)
    ax3.set_xscale('log')
    apply_clean_style(ax3)

    # Plot 4: Sample size (magnified)
    ax4 = axes[1, 1]
    for method in methods:
        col = f'{method}_mean_size'
        if col not in sweep_results.columns:
            continue
        ax4.plot(x_full, sweep_results[col],
                 marker=METHOD_MARKERS.get(method, 'o'),
                 markersize=MARKER_SIZES.get(method, 8),
                 linewidth=2.5,
                 label=METHOD_NAMES.get(method, method),
                 color=METHOD_COLORS.get(method, 'gray'))

    ax4.set_yscale('log')
    ax4.set_xlim(99, 100)
    ax4.set_xscale('log')
    ax4.set_xlabel('HPD Region Cutoff (%)', fontsize=13)
    ax4.set_ylabel('Number of Models', fontsize=13)
    ax4.set_title('Number of Models in HPD Region (Magnified)', fontsize=13)
    apply_clean_style(ax4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
