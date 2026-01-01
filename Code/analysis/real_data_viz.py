import matplotlib.pyplot as plt

from matplotlib import colors


def plot_size_histogram(fig_size, model_sizes, xlabel, ylabel, title,
                        fname=None,
                        label_fontsize=14, title_fontsize=16,
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
                     fname=None,
                     **kwargs):

    fig, ax = plt.subplots(figsize=fig_size)

    h = ax.hist2d(model_sizes, rel_post_prob_ratio, norm=colors.LogNorm(),
                  cmap="OrRd")

    cb = fig.colorbar(h[3], norm=colors.NoNorm, ax=ax)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticks)

    if hline_val:
        ax.plot([-2, 80], [hline_val, hline_val], color="black", linestyle="--", linewidth=2)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()
