import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import colors
from scipy import stats


def plot_size_histogram(fig_size, model_sizes, xlabel, ylabel, title,
                        fname=None,
                        label_fontsize=10, title_fontsize=12,
                        **kwargs):
    fig, ax = plt.subplots(figsize=fig_size)

    ax.spines[['right', 'top']].set_visible(False)

    ax.hist(model_sizes, **kwargs)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    ax.set_title(title, fontsize=title_fontsize)

    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()


def plot_rps_heatmap(fig_size, model_sizes, rel_post_prob_ratio,
                     xlabel, ylabel, label_fontsize=10,
                     ticks=[1, 10, 100, 250, 700],
                     hline_val=None,
                     hline_range=None,
                     fname=None,
                     **kwargs):

    fig, ax = plt.subplots(figsize=fig_size)

    h = ax.hist2d(model_sizes, rel_post_prob_ratio, norm=colors.LogNorm(),
                  cmap="OrRd")

    cb = fig.colorbar(h[3], norm=colors.NoNorm, ax=ax)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticks)

    if hline_val:
        ax.plot(hline_range, [hline_val, hline_val], color="black", linestyle="--", linewidth=2)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()


def create_rps_heterogeneity_heatmap(
    plots_matrices, titles, labels, bin_labels,
    gap_between_panels=2, figsize=(12, 4), cmap="OrRd", vmin=0, vmax=1,
    xlabels=None, xlabel_fontsize=10, ylabel=None, fname=None, **kwargs
):
    """
    Create a generalized mosaic heatmap.

    Parameters:
    -----------
    plots_matrices : list of lists
        Shape: [n_rows][n_panels_per_row], where each element is a matrix
    titles : list of lists
        Shape: [n_rows][n_panels_per_row], titles for each panel
    labels : list of lists
        Shape: [n_rows][n_panels_per_row][n_cols_in_panel], labels for columns
    bin_labels : list
        Y-axis labels (rows)
    gap_between_panels : int or list
        Number of empty columns between panels. Can be:
        - Single int: same gap between all panels
        - List of ints: specific gap after each panel (length = n_panels - 1)
    """

    n_rows = len(plots_matrices)
    n_panels_per_row = len(plots_matrices[0])

    # Handle gap_between_panels as either int or list
    if isinstance(gap_between_panels, int):
        gaps = [gap_between_panels] * (n_panels_per_row - 1)
    else:
        gaps = gap_between_panels
        if len(gaps) != n_panels_per_row - 1:
            raise ValueError(f"gap_between_panels list must have length {n_panels_per_row - 1}")

    # Calculate column positions for each panel
    panel_col_counts = [[matrix.shape[1] for matrix in row] for row in plots_matrices]

    # Total columns needed PER ROW (can be different for each row)
    total_cols_per_row = [sum(row_counts) + sum(gaps) for row_counts in panel_col_counts]
    max_total_cols = max(total_cols_per_row)

    # Calculate starting column for each panel in each row WITH CENTERING
    panel_starts = []
    for row in range(n_rows):
        # Calculate offset to center this row
        row_offset = (max_total_cols - total_cols_per_row[row]) // 2

        starts_in_row = []
        current_col = row_offset  # Start from offset position
        for panel in range(n_panels_per_row):
            starts_in_row.append(current_col)
            current_col += panel_col_counts[row][panel]
            if panel < n_panels_per_row - 1:  # Don't add gap after last panel
                current_col += gaps[panel]
        panel_starts.append(starts_in_row)

    # Create figure with max columns
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=n_rows, ncols=max_total_cols, **kwargs)

    # Add colorbar axis
    cbar_ax = fig.add_axes([.95, .25, .03, .5])

    # Plot each matrix
    for i in range(n_rows):
        for panel in range(n_panels_per_row):
            matrix = plots_matrices[i][panel]
            num_cols = matrix.shape[1]
            panel_start_col = panel_starts[i][panel]

            # Plot each column separately
            for col in range(num_cols):
                col_pos = panel_start_col + col
                ax_col = fig.add_subplot(gs[i, col_pos])

                # Extract single column
                col_data = matrix[:, col:col+1]

                # Determine if this should have colorbar (last column of last panel in last row)
                show_cbar = (col == num_cols - 1 and panel == n_panels_per_row - 1 and i == n_rows - 1)

                sns.heatmap(
                    col_data,
                    cmap=cmap,
                    linewidths=0.5,
                    linecolor="black",
                    square=True,
                    vmin=vmin,
                    vmax=vmax,
                    ax=ax_col,
                    cbar=show_cbar,
                    cbar_ax=cbar_ax if show_cbar else None
                )

                ax_col.axhline(y=len(bin_labels), color='black', linewidth=1.5)
                ax_col.axvline(x=1, color='black', linewidth=1.5)

                # Remove ticks
                ax_col.set_yticks([])
                ax_col.set_xticks([])

                # Add y-axis labels on first column of first panel
                if col == 0 and panel == 0:
                    ax_col.set_yticks(np.arange(0, len(bin_labels)) + 0.5,
                                      bin_labels, rotation=0)

                # Add x-axis labels on bottom row
                if labels[i][panel][col]:
                    ax_col.set_xticks([0.5], [labels[i][panel][col]], rotation=90)

                # Add title to middle column of each panel
                if num_cols % 2 == 0 and col == 0:
                    ax_col.set_title(titles[i][panel], fontsize=10, x=num_cols*0.67, ha='center')
                if num_cols % 2 == 1 and col == (num_cols - 1) // 2:
                    ax_col.set_title(titles[i][panel], fontsize=10)

                if xlabels and col == (num_cols) // 2:
                    ax_col.set_xlabel(xlabels[i][panel], fontsize=xlabel_fontsize, labelpad=10)

    if ylabel:
        fig.supylabel(ylabel)

    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    return fig


def plot_ridgeline(data_list, posterior_probabilities, labels, xlabel="Effect", title="",
                   overlap=0.7, figsize=(10, 8), color="steelblue",
                   alpha=0.7, fname=None):
    """
    Create a ridgeline plot from a list of data arrays.

    Parameters:
    -----------
    data_list : list of arrays
        List of data arrays to plot
    labels : list of str
        Labels for each distribution
    overlap : float
        Overlap between distributions (higher = more overlap)
    """
    fig, axes = plt.subplots(len(data_list), 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=-1)  # Negative value creates overlap
    if len(data_list) == 1:
        axes = [axes]

    # Calculate global x-range
    all_data = np.concatenate(data_list)
    x_min, x_max = np.min(all_data), np.max(all_data)
    # x_range = x_max - x_min
    # x_min -= 0.1 * x_range
    # x_max += 0.1 * x_range

    # Plot each distribution
    for idx, (ax, data, label) in enumerate(zip(axes, data_list, labels)):
        # Create KDE
        x = np.linspace(x_min, x_max, 1000)
        kde = stats.gaussian_kde(data, weights=posterior_probabilities[idx])
        y = kde(x)

        # Plot
        ax.fill_between(x, y, alpha=alpha)
        ax.plot(x, y, linewidth=2)

        # Plot zero line for reference
        ax.plot([0, 0], [0, ax.get_ylim()[1]], color='black', linestyle='--', linewidth=1)

        # Styling
        ax.set_ylabel(label, rotation=0, ha='right', va='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.set_yticks([])
        ax.set_ylim(0)
        ax.set_xlim(x_min, x_max)

        # # Only show x-axis for bottom plot
        # if idx < len(data_list) - 1:
        #     ax.spines['bottom'].set_visible(False)
        #     ax.set_xticks([])
        # else:
        # if idx < len(data_list) - 1:
        #     print("Hiding x-axis for idx:", idx)
        #     ax.set_xticks([])
        # else:
        #     print("Showing x-axis for idx:", idx)
        #     # ax.set_xticks(ax.get_xticks())

    axes[-1].set_xlabel(xlabel)
    axes[0].set_title(title, pad=20)

    # plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    return fig, axes
