"""
Heatmap visualizations for Rashomon set analysis.

This module provides 2D histogram plotting functions for visualizing
the relationship between model size and posterior probability.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_rashomon_heatmap_grid(rashomon_raw_df, n_per_pol_values=None,
                               reg=1e-2, xlim=(2, 18), figsize=(12, 10),
                               save_path=None):
    """Plot 2x2 grid of heatmaps showing model size vs posterior probability.

    Parameters
    ----------
    rashomon_raw_df : pd.DataFrame
        Raw Rashomon results with columns: MSE, num_pools, n_per_pol.
    n_per_pol_values : list of int, optional
        Four sample sizes to plot (default: [10, 50, 100, 1000]).
    reg : float, optional
        Regularization parameter for loss computation (default: 1e-2).
    xlim : tuple, optional
        X-axis limits for model size (default: (2, 18)).
    figsize : tuple, optional
        Figure size (width, height) (default: (12, 10)).
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig, axes : matplotlib figure and 2x2 axes array
    """
    if n_per_pol_values is None:
        n_per_pol_values = [10, 50, 100, 1000]

    # Prepare data
    heatmap_df = rashomon_raw_df.copy()
    heatmap_df["loss"] = heatmap_df["MSE"] + reg * heatmap_df["num_pools"]
    heatmap_df["posterior"] = np.exp(-heatmap_df["loss"])

    # Split by sample size and scale posteriors
    heatmap_dfs = []
    for n_per_pol in n_per_pol_values:
        df_subset = heatmap_df[heatmap_df["n_per_pol"] == n_per_pol].copy()
        # Custom scaling: (posterior - max) / max
        df_subset["posterior"] = (
            (df_subset["posterior"] - np.max(df_subset["posterior"]))
            / np.max(df_subset["posterior"])
        )
        heatmap_dfs.append(df_subset)

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot each heatmap
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            df = heatmap_dfs[idx]
            n_val = n_per_pol_values[idx]

            # Create 2D histogram with log-normalized color scale
            h = axes[i, j].hist2d(
                df["num_pools"], df["posterior"],
                norm=colors.LogNorm(),
                cmap="OrRd",
                weights=[1e-2] * len(df)
            )

            axes[i, j].set_title(f"Samples per feature = {n_val}", fontsize=12)
            fig.colorbar(h[3], norm=colors.NoNorm, ax=axes[i, j])

            # Set axis limits
            axes[i, j].set_ylim(np.min(df["posterior"]), np.max(df["posterior"]))
            axes[i, j].set_xlim(xlim)

    # Add common axis labels
    fig.supylabel("Relative posterior probability ratio", fontsize=14)
    fig.supxlabel("Model size", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
