"""
Unified worst case simulation analysis script.

This script loads worst case simulation data for all methods, aggregates results,
and generates comprehensive visualizations including individual metric plots,
panel comparisons, and Rashomon set heatmaps.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from analysis.data import aggregate_worst_case_results
from analysis.visualization import plot_method_comparison, plot_method_panel_comparison
from analysis.heatmaps import plot_rashomon_heatmap_grid


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Directory paths
RESULTS_DIR = '../Results/worst_case'
FIGURES_DIR = '../Figures/worst_case_revised'
FILE_PREFIX = 'worst_case_'

# Methods to analyze
METHODS = ['rashomon', 'lasso', 'tva', 'blasso', 'bootstrap', 'ssl', 'ppmx']
METHODS_ORDER = ['rashomon', 'blasso', 'bootstrap', 'lasso', 'ppmx', 'ssl', 'tva']

# Heatmap parameters
HEATMAP_REG = 1e-2  # Regularization for loss computation
HEATMAP_N_VALUES = [10, 50, 100, 1000]  # Sample sizes to plot
HEATMAP_XLIM = (2, 18)  # Model size limits

# Create figures directory if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================================
# SECTION 0: Setup
# ==============================================================================

print("=" * 80)
print("WORST CASE SIMULATION ANALYSIS")
print("=" * 80)
print("\nConfiguration:")
print(f"  Results directory: {RESULTS_DIR}")
print(f"  Figures directory: {FIGURES_DIR}")
print(f"  Methods: {', '.join(METHODS)}")
print(f"  Plotting order: {', '.join(METHODS_ORDER)}")


# ==============================================================================
# SECTION 1: Load and Aggregate Data
# ==============================================================================

print("=" * 80)
print("SECTION 1: Loading and Aggregating Data")
print("=" * 80)

raw_data = {}
aggregated_data = {}

for method in METHODS:
    print(f"Loading {method}...")
    filename = os.path.join(RESULTS_DIR, f"{FILE_PREFIX}{method}.csv")
    df = pd.read_csv(filename)

    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    raw_data[method] = df
    print(f"  {len(df)} rows loaded")

    # Aggregate results
    print(f"Aggregating {method}...")
    aggregated_data[method] = aggregate_worst_case_results(df, method)
    print(f"  Aggregated to {len(aggregated_data[method])} rows")

print("\nAll data loaded and aggregated successfully!")

print("\nAll data loaded and aggregated successfully!")


# ==============================================================================
# SECTION 2: Individual Metric Plots
# ==============================================================================

print("=" * 80)
print("SECTION 2: Individual Metric Plots")
print("=" * 80)

# Plot 1: MSE
print("Generating MSE plot...")
plot_method_comparison(
    aggregated_data,
    x_col='n_per_pol',
    y_col='MSE',
    ylabel='MSE',
    ylim=(0.8, 8),
    methods_order=METHODS_ORDER,
    save_path=os.path.join(FIGURES_DIR, 'MSE.png'),
    figsize=(5, 5)
)
plt.close()
print(f"  Saved to: {FIGURES_DIR}/MSE.png")

# Plot 2: IOU (Best feature set coverage)
print("Generating IOU plot...")
plot_method_comparison(
    aggregated_data,
    x_col='n_per_pol',
    y_col='IOU',
    ylabel='Best feature set coverage',
    ylim=(0, 1.01),
    methods_order=METHODS_ORDER,
    save_path=os.path.join(FIGURES_DIR, 'IOU.png'),
    figsize=(5, 5)
)
plt.close()
print(f"  Saved to: {FIGURES_DIR}/IOU.png")

# Plot 3: Minimum dosage inclusion
print("Generating min_dosage plot...")
plot_method_comparison(
    aggregated_data,
    x_col='n_per_pol',
    y_col='min_dosage',
    ylabel='Minimum dosage inclusion',
    ylim=(0, 1.01),
    methods_order=METHODS_ORDER,
    save_path=os.path.join(FIGURES_DIR, 'min_dosage.png'),
    figsize=(5, 5)
)
plt.close()
print(f"  Saved to: {FIGURES_DIR}/min_dosage.png")

# Plot 4: Best policy MSE
print("Generating best_pol_MSE plot...")
plot_method_comparison(
    aggregated_data,
    x_col='n_per_pol',
    y_col='best_pol_MSE',
    ylabel='Best policy MSE',
    ylim=None,
    methods_order=METHODS_ORDER,
    save_path=os.path.join(FIGURES_DIR, 'best_pol_MSE.png'),
    figsize=(5, 5)
)
plt.close()
print(f"  Saved to: {FIGURES_DIR}/best_pol_MSE.png")


# ==============================================================================
# SECTION 3: Panel Comparison
# ==============================================================================

print("=" * 80)
print("SECTION 3: Panel Comparison")
print("=" * 80)

print("Generating 1x3 panel comparison...")

metrics_config = [
    {'y_col': 'MSE', 'ylabel': 'MSE', 'ylim': (0.8, 8)},
    {'y_col': 'IOU', 'ylabel': 'Best feature set coverage', 'ylim': (0, 1.01)},
    {'y_col': 'best_pol_MSE', 'ylabel': 'Best policy MSE', 'ylim': None},
]

plot_method_panel_comparison(
    aggregated_data,
    x_col='n_per_pol',
    metrics_config=metrics_config,
    methods_order=METHODS_ORDER,
    save_path=os.path.join(FIGURES_DIR, 'panel_comparison.png'),
    figsize=(18, 5)
)
plt.close()
print(f"  Saved to: {FIGURES_DIR}/panel_comparison.png")


# ==============================================================================
# SECTION 4: Rashomon Heatmap Grid
# ==============================================================================

print("=" * 80)
print("SECTION 4: Rashomon Heatmap Grid")
print("=" * 80)

print("Generating 2x2 heatmap grid...")
print(f"  Sample sizes: {HEATMAP_N_VALUES}")
print(f"  Regularization: {HEATMAP_REG}")
print(f"  Model size limits: {HEATMAP_XLIM}")

plot_rashomon_heatmap_grid(
    raw_data['rashomon'],
    n_per_pol_values=HEATMAP_N_VALUES,
    reg=HEATMAP_REG,
    xlim=HEATMAP_XLIM,
    figsize=(12, 10),
    save_path=os.path.join(FIGURES_DIR, 'rset_2d_hist.png')
)
plt.close()
print(f"  Saved to: {FIGURES_DIR}/rset_2d_hist.png")


# ==============================================================================
# SECTION 5: Summary
# ==============================================================================

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print(f"  1. {FIGURES_DIR}/MSE.png")
print(f"  2. {FIGURES_DIR}/IOU.png")
print(f"  3. {FIGURES_DIR}/min_dosage.png")
print(f"  4. {FIGURES_DIR}/best_pol_MSE.png")
print(f"  5. {FIGURES_DIR}/panel_comparison.png")
print(f"  6. {FIGURES_DIR}/rset_2d_hist.png")
print("\nAnalysis completed successfully!")
