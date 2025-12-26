"""
Generate all plots for worst case simulation results.

This script loads and aggregates worst case simulation data for all methods,
then generates individual metric plots, a 1x3 panel comparison, and a 2x2
heatmap for Rashomon sets.
"""

import pandas as pd
from pathlib import Path

from analysis.data import aggregate_worst_case_results
from analysis.visualization import plot_method_comparison, plot_method_panel_comparison
from analysis.heatmaps import plot_rashomon_heatmap_grid


def main():
    """Load data, aggregate, and generate all plots."""
    print("Loading worst case simulation results...")

    # Define paths
    results_dir = Path("../Results/worst_case")
    figures_dir = Path("../Figures/worst_case")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data for all methods
    methods = ['rashomon', 'lasso', 'tva', 'blasso', 'bootstrap', 'ssl', 'ppmx']
    raw_data = {}
    aggregated_data = {}

    for method in methods:
        print(f"  Loading {method}...")
        filename = results_dir / f"worst_case_{method}.csv"
        df = pd.read_csv(filename)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        raw_data[method] = df

        # Aggregate results
        print(f"  Aggregating {method}...")
        aggregated_data[method] = aggregate_worst_case_results(df, method)

    print("\nAggregated data shapes:")
    for method, df in aggregated_data.items():
        print(f"  {method}: {df.shape}")

    # Define plotting order (rashomon first, then alphabetical)
    methods_order = ['rashomon', 'blasso', 'bootstrap', 'lasso', 'ppmx', 'ssl', 'tva']

    # =========================================================================
    # Generate individual metric plots
    # =========================================================================
    print("\nGenerating individual metric plots...")

    # Plot 1: MSE
    print("  MSE...")
    plot_method_comparison(
        aggregated_data,
        x_col='n_per_pol',
        y_col='MSE',
        ylabel='MSE',
        ylim=(0.8, 8),
        methods_order=methods_order,
        save_path=figures_dir / 'MSE.png',
        figsize=(5, 5)
    )

    # Plot 2: IOU (Best feature set coverage)
    print("  IOU...")
    plot_method_comparison(
        aggregated_data,
        x_col='n_per_pol',
        y_col='IOU',
        ylabel='Best feature set coverage',
        ylim=(0, 1.01),
        methods_order=methods_order,
        save_path=figures_dir / 'IOU.png',
        figsize=(5, 5)
    )

    # Plot 3: Minimum dosage inclusion
    print("  min_dosage...")
    plot_method_comparison(
        aggregated_data,
        x_col='n_per_pol',
        y_col='min_dosage',
        ylabel='Minimum dosage inclusion',
        ylim=(0, 1.01),
        methods_order=methods_order,
        save_path=figures_dir / 'min_dosage.png',
        figsize=(5, 5)
    )

    # Plot 4: Best policy MSE
    print("  best_pol_MSE...")
    plot_method_comparison(
        aggregated_data,
        x_col='n_per_pol',
        y_col='best_pol_MSE',
        ylabel='Best policy MSE',
        ylim=None,
        methods_order=methods_order,
        save_path=figures_dir / 'best_pol_MSE.png',
        figsize=(5, 5)
    )

    # =========================================================================
    # Generate 1x3 panel comparison
    # =========================================================================
    print("\nGenerating 1x3 panel comparison...")

    metrics_config = [
        {'y_col': 'MSE', 'ylabel': 'MSE', 'ylim': (0.8, 8)},
        {'y_col': 'IOU', 'ylabel': 'Best feature set coverage', 'ylim': (0, 1.01)},
        {'y_col': 'best_pol_MSE', 'ylabel': 'Best policy MSE', 'ylim': None},
    ]

    plot_method_panel_comparison(
        aggregated_data,
        x_col='n_per_pol',
        metrics_config=metrics_config,
        methods_order=methods_order,
        save_path=figures_dir / 'panel_comparison.png',
        figsize=(18, 5)
    )

    # =========================================================================
    # Generate 2x2 Rashomon heatmap
    # =========================================================================
    print("\nGenerating Rashomon heatmap grid...")

    plot_rashomon_heatmap_grid(
        raw_data['rashomon'],
        n_per_pol_values=[10, 50, 100, 1000],
        reg=1e-2,
        xlim=(2, 18),
        figsize=(12, 10),
        save_path=figures_dir / 'rset_2d_hist.png'
    )

    print(f"\nAll plots saved to {figures_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
