"""
Visualization utilities for simulation results comparison.

This module provides plotting functions with standardized styling for
comparing different methods across various metrics.
"""

import numpy as np
import pandas as pd
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
    'tva': 'mediumpurple',
}

# Method display names
METHOD_NAMES = {
    'rashomon': 'Rashomon Set',
    'blasso': 'Bayesian Lasso',
    'ssl': 'Spike-Slab Lasso',
    'ppmx': 'PPMx',
    'bootstrap': 'Bootstrap Lasso',
    'lasso': 'Lasso',
    'tva': 'TVA',
}

# Markers for line plots
METHOD_MARKERS = {
    'rashomon': 'd',   # diamond
    'blasso': 'o',     # circle
    'ssl': 's',        # square
    'ppmx': '*',       # star
    'bootstrap': '^',  # triangle
    'tva': 'p',        # pentagon
}

# Marker sizes
MARKER_SIZES = {
    'rashomon': 8,
    'blasso': 8,
    'ssl': 8,
    'ppmx': 10,  # Larger for star
    'bootstrap': 8,
    'tva': 8,
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

    # Reference line for perfect recovery (no legend)
    len_x = 10
    x_horizontal = np.linspace(0, max_eps, num=len_x)
    y_best = np.array([1] * len_x)
    ax.plot(x_horizontal, y_best, color='black', ls='--', linewidth=1,
            alpha=0.5)

    # Plot epsilon curves for each method
    for method, df in epsilon_curves.items():
        color = METHOD_COLORS.get(method, 'gray')
        label = METHOD_NAMES.get(method, method)

        # Plot line (no label for legend)
        ax.plot(df['eps_levels'], df['profile_rate_eps'],
                color=color, linewidth=2.5, zorder=3, clip_on=False)

        # Add scatter point at start with legend
        ax.scatter(df['eps_levels'].iloc[0], df['profile_rate_eps'].iloc[0],
                   color=color, edgecolor='black', s=60, zorder=3.2,
                   clip_on=False, label=label)

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
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# ==============================================================================
# Bar Chart Comparison - Individual and Multi-Panel
# ==============================================================================

def plot_single_metric_bar(comparison_df, metric_column, ax=None,
                           ylabel='Metric Value', title='',
                           figsize=(8, 6), show_values=True):
    """
    Plot a single metric as a bar chart.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: 'Method' and metric column
    metric_column : str
        Column name for the metric to plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure.
    ylabel : str, default='Metric Value'
        Y-axis label
    title : str, default=''
        Plot title
    figsize : tuple, default=(8, 6)
        Figure size (only used if ax is None)
    show_values : bool, default=True
        Show value labels on bars

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if metric_column not in comparison_df.columns:
        raise ValueError(f"Column '{metric_column}' not found in dataframe")

    # Map display names back to method keys for color lookup
    name_to_key = {v: k for k, v in METHOD_NAMES.items()}
    colors = []
    for m in comparison_df['Method']:
        # Try exact match with METHOD_NAMES values
        key = name_to_key.get(m)
        if key is None:
            # Try lowercase match as fallback
            key = m.lower().replace(' ', '')
        colors.append(METHOD_COLORS.get(key, 'gray'))

    x_pos = np.arange(len(comparison_df))

    # Extract values more robustly - handle various DataFrame structures
    values = []
    for idx in comparison_df.index:
        val = comparison_df.loc[idx, metric_column]
        # Handle Series, DataFrame, or scalar
        if hasattr(val, 'item'):
            val = val.item()
        elif hasattr(val, 'iloc'):
            val = val.iloc[0] if len(val) > 0 else 0.0
        try:
            values.append(float(val) if pd.notna(val) else 0.0)
        except (TypeError, ValueError):
            values.append(0.0)

    # Plot bars
    bars = ax.bar(x_pos, values, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_df['Method'], rotation=30, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.05)

    apply_clean_style(ax)

    # Add value labels on bars
    if show_values:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    return ax


def plot_hpd_bar_comparison(comparison_df, metrics_to_plot=None,
                            figsize=(20, 6), save_path=None):
    """
    Plot bar chart comparison of methods across multiple metrics.

    For individual metric plots, use plot_single_metric_bar() directly.

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

    for idx, metric_spec in enumerate(metrics_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        column = metric_spec['column']
        if column not in comparison_df.columns:
            continue

        # Use the helper function to plot on this axis
        plot_single_metric_bar(comparison_df, column, ax=ax,
                               ylabel=metric_spec['ylabel'],
                               title=metric_spec['title'],
                               figsize=(20/n_cols, 12/n_rows),
                               show_values=True)

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
# Credible Interval Sweep - Individual Plot Functions
# ==============================================================================

def plot_coverage_vs_credible(sweep_results, methods=None, ax=None,
                              xlim=(0, 105), use_log_x=False,
                              show_legend=False, title_suffix=''):
    """
    Plot coverage (recovery rate) vs credible level.

    Parameters
    ----------
    sweep_results : pd.DataFrame
        DataFrame with 'credible_level' and '{method}_coverage' columns
    methods : list of str, optional
        Methods to plot. If None, uses ['rashomon', 'blasso', 'ssl', 'ppmx']
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure.
    xlim : tuple, default=(0, 105)
        X-axis limits
    use_log_x : bool, default=False
        Use log scale for x-axis
    show_legend : bool, default=False
        Show legend
    title_suffix : str, default=''
        Suffix to add to title (e.g., ' (Magnified)')

    Returns
    -------
    matplotlib.axes.Axes
    """
    if methods is None:
        methods = ['rashomon', 'blasso', 'ssl', 'ppmx']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    x_full = sweep_results['credible_level'] * 100

    for method in methods:
        col = f'{method}_coverage'
        if col not in sweep_results.columns:
            continue
        ax.plot(x_full, sweep_results[col],
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=MARKER_SIZES.get(method, 8),
                linewidth=2.5,
                label=METHOD_NAMES.get(method, method),
                color=METHOD_COLORS.get(method, 'gray'))

    ax.set_xlabel('HPD Region Cutoff (%)', fontsize=13)
    ax.set_ylabel('Recovery Rate', fontsize=13)
    ax.set_title(f'Best Profile Recovery Rate in HPD Region{title_suffix}', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(xlim)
    if use_log_x:
        ax.set_xscale('log')
    if show_legend:
        ax.legend(loc='best', fontsize=11)
    apply_clean_style(ax)

    return ax


def plot_sample_size_vs_credible(sweep_results, methods=None, ax=None,
                                 xlim=(0, 105), use_log_x=False,
                                 show_legend=True, title_suffix=''):
    """
    Plot sample size vs credible level.

    Parameters
    ----------
    sweep_results : pd.DataFrame
        DataFrame with 'credible_level' and '{method}_mean_size' columns
    methods : list of str, optional
        Methods to plot. If None, uses ['rashomon', 'blasso', 'ssl', 'ppmx']
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure.
    xlim : tuple, default=(0, 105)
        X-axis limits
    use_log_x : bool, default=False
        Use log scale for x-axis
    show_legend : bool, default=True
        Show legend
    title_suffix : str, default=''
        Suffix to add to title (e.g., ' (Magnified)')

    Returns
    -------
    matplotlib.axes.Axes
    """
    if methods is None:
        methods = ['rashomon', 'blasso', 'ssl', 'ppmx']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    x_full = sweep_results['credible_level'] * 100

    for method in methods:
        col = f'{method}_mean_size'
        if col not in sweep_results.columns:
            continue
        ax.plot(x_full, sweep_results[col],
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=MARKER_SIZES.get(method, 8),
                linewidth=2.5,
                label=METHOD_NAMES.get(method, method),
                color=METHOD_COLORS.get(method, 'gray'))

    ax.set_yscale('log')
    ax.set_xlim(xlim)
    if use_log_x:
        ax.set_xscale('log')
    ax.set_xlabel('HPD Region Cutoff (%)', fontsize=13)
    ax.set_ylabel('Number of Models', fontsize=13)
    ax.set_title(f'Number of Models in HPD Region{title_suffix}', fontsize=13)
    if show_legend:
        ax.legend(loc='upper left', fontsize=11)
    apply_clean_style(ax)

    return ax


def plot_credible_interval_sweep(sweep_results, methods=None,
                                 figsize=(16, 12), save_path=None):
    """
    Plot how metrics change with credible interval size.

    Creates a 2x2 grid showing:
    - Coverage vs credible level (full range and magnified)
    - Sample size vs credible level (full range and magnified)

    For individual plots, use plot_coverage_vs_credible() or
    plot_sample_size_vs_credible() directly.

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

    # Plot 1: Coverage (full range)
    plot_coverage_vs_credible(sweep_results, methods, ax=axes[0, 0],
                              xlim=(0, 105), show_legend=False)

    # Plot 2: Sample size (full range)
    plot_sample_size_vs_credible(sweep_results, methods, ax=axes[0, 1],
                                 xlim=(0, 105), show_legend=True)

    # Plot 3: Coverage (magnified)
    plot_coverage_vs_credible(sweep_results, methods, ax=axes[1, 0],
                              xlim=(99, 100+1e-2), use_log_x=True,
                              show_legend=False, title_suffix=' (Magnified)')

    # Plot 4: Sample size (magnified)
    plot_sample_size_vs_credible(sweep_results, methods, ax=axes[1, 1],
                                 xlim=(99, 100+1e-2), use_log_x=True,
                                 show_legend=False, title_suffix=' (Magnified)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
