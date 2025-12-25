"""
Analysis-specific metrics for comparing methods across simulation results.

This module provides metrics for evaluating how well different methods
(Rashomon, Bayesian Lasso, Spike-Slab Lasso, PPMx, Bootstrap) recover
the true best profile across different model selection criteria:
- Epsilon-balls (distance from optimal)
- HPD regions (highest posterior density)
- Top-K models (by loss)
"""

import numpy as np
import pandas as pd


# ==============================================================================
# Helper Functions
# ==============================================================================

def _filter_to_hpd_region(df, credible_mass=0.95, posterior_weight_col='posterior_weight',
                          group_cols=None):
    """
    Filter dataframe to HPD (Highest Posterior Density) region.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with posterior weights
    credible_mass : float, default=0.95
        Fraction of posterior mass to include
    posterior_weight_col : str, default='posterior_weight'
        Column with posterior weights
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only samples in HPD region
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    df_copy = df.copy()

    # Sort by weight within each group (descending)
    df_copy = df_copy.sort_values(group_cols + [posterior_weight_col],
                                  ascending=[True] * len(group_cols) + [False])

    # Cumulative weight
    df_copy['cumsum_weight'] = df_copy.groupby(group_cols)[posterior_weight_col].cumsum()

    # Keep only samples in HPD region
    hpd_samples = df_copy[df_copy['cumsum_weight'] <= credible_mass].copy()

    return hpd_samples


def _filter_to_top_k(df, loss_col, k_fraction=0.95, group_cols=None):
    """
    Filter dataframe to top-K models by loss.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with loss values
    loss_col : str
        Column with loss values (lower is better)
    k_fraction : float, default=0.95
        Fraction of models to include
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only top-K models
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    df_copy = df.copy()

    # Sort by loss within each group (ascending - lower loss is better)
    df_copy = df_copy.sort_values(group_cols + [loss_col],
                                  ascending=[True] * len(group_cols) + [True])

    # Assign rank
    df_copy['rank'] = df_copy.groupby(group_cols).cumcount()

    # Determine K for each group
    total_per_group = df_copy.groupby(group_cols).size().reset_index(name='total')
    total_per_group['k_threshold'] = (total_per_group['total'] * k_fraction).astype(int)

    # Merge back
    df_copy = df_copy.merge(total_per_group[group_cols + ['k_threshold']],
                            on=group_cols)

    # Keep only top-K
    top_k_samples = df_copy[df_copy['rank'] < df_copy['k_threshold']].copy()

    return top_k_samples


def _compute_metric_on_region(df, profile_col, metric_type='frequency',
                              group_cols=None):
    """
    Compute metric (frequency or binary presence) on a filtered region.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe (e.g., HPD region or top-K)
    profile_col : str
        Column name for the profile to check
    metric_type : str, default='frequency'
        Type of metric: 'frequency' (mean) or 'binary' (at least one)
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    tuple
        (per_group_df, overall_mean)
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    df_copy = df.copy()
    df_copy['has_best_profile'] = (df_copy[profile_col] > 0).astype(int)

    if metric_type == 'frequency':
        # Fraction of samples with the profile
        metric_per_group = df_copy.groupby(group_cols)['has_best_profile'].mean()
    elif metric_type == 'binary':
        # At least one sample with the profile (0 or 1)
        metric_per_group = df_copy.groupby(group_cols)['has_best_profile'].max()
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    metric_per_group = metric_per_group.reset_index()
    metric_per_group.rename(columns={'has_best_profile': 'metric_value'}, inplace=True)

    overall_mean = metric_per_group['metric_value'].mean()

    return metric_per_group, overall_mean


# ==============================================================================
# Frequency-Based Metrics
# ==============================================================================

def compute_frequency_inclusion(df, profile_col, group_cols=None):
    """
    Compute frequency-based inclusion rate.

    Returns the fraction of samples in each group that have the profile.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples
    profile_col : str
        Column name for the profile to check
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    tuple
        (freq_per_group, overall_freq)
        - freq_per_group: DataFrame with frequency per group
        - overall_freq: Mean frequency across all groups
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    df_copy = df.copy()
    df_copy['has_best_profile'] = (df_copy[profile_col] > 0).astype(int)

    # Fraction per group
    freq_per_group = df_copy.groupby(group_cols)['has_best_profile'].mean().reset_index()
    freq_per_group.rename(columns={'has_best_profile': 'frequency_inclusion'}, inplace=True)

    # Overall mean
    overall_freq = freq_per_group['frequency_inclusion'].mean()

    return freq_per_group, overall_freq


def compute_hpd_frequency(df, profile_col, credible_mass=0.95,
                          posterior_weight_col='posterior_weight',
                          group_cols=None):
    """
    Compute frequency within Highest Posterior Density (HPD) region.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with posterior weights
    profile_col : str
        Column name for the profile to check
    credible_mass : float, default=0.95
        Fraction of posterior mass to include
    posterior_weight_col : str, default='posterior_weight'
        Column with posterior weights
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    tuple
        (freq_per_group, overall_freq, hpd_size_per_group)
        - freq_per_group: DataFrame with HPD frequency per group
        - overall_freq: Mean HPD frequency across groups
        - hpd_size_per_group: DataFrame with number of samples in HPD per group
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    # Filter to HPD region
    hpd_samples = _filter_to_hpd_region(df, credible_mass, posterior_weight_col, group_cols)

    # Get all unique groups from original data
    all_groups = df[group_cols].drop_duplicates()

    # Compute frequency within HPD
    if len(hpd_samples) > 0:
        hpd_samples['has_best_profile'] = (hpd_samples[profile_col] > 0).astype(int)
        freq_per_group = hpd_samples.groupby(group_cols)['has_best_profile'].mean().reset_index()
        freq_per_group.rename(columns={'has_best_profile': 'hpd_frequency'}, inplace=True)

        # Count samples in HPD
        hpd_size_per_group = hpd_samples.groupby(group_cols).size().reset_index(name='hpd_size')
    else:
        freq_per_group = pd.DataFrame(columns=group_cols + ['hpd_frequency'])
        hpd_size_per_group = pd.DataFrame(columns=group_cols + ['hpd_size'])

    # Merge to ensure all groups are represented (missing ones get 0)
    freq_all_groups = all_groups.merge(freq_per_group, on=group_cols, how='left')
    freq_all_groups['hpd_frequency'] = freq_all_groups['hpd_frequency'].fillna(0)

    hpd_size_all_groups = all_groups.merge(hpd_size_per_group, on=group_cols, how='left')
    hpd_size_all_groups['hpd_size'] = hpd_size_all_groups['hpd_size'].fillna(0)

    overall_freq = freq_all_groups['hpd_frequency'].mean()

    return freq_all_groups, overall_freq, hpd_size_all_groups


def compute_top_k_frequency(df, profile_col, loss_col, k_fraction=0.95,
                            group_cols=None):
    """
    Compute frequency in top-K models by loss.

    For Bootstrap and other non-Bayesian methods: use loss to define "top models".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with loss values
    profile_col : str
        Column name for the profile to check
    loss_col : str
        Column with loss values (lower is better)
    k_fraction : float, default=0.95
        Fraction of models to include
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    tuple
        (freq_per_group, overall_freq)
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    # Filter to top-K
    top_k_samples = _filter_to_top_k(df, loss_col, k_fraction, group_cols)

    # Compute frequency
    if len(top_k_samples) > 0:
        top_k_samples['has_best_profile'] = (top_k_samples[profile_col] > 0).astype(int)
        freq_per_group = top_k_samples.groupby(group_cols)['has_best_profile'].mean().reset_index()
        freq_per_group.rename(columns={'has_best_profile': 'topk_frequency'}, inplace=True)
    else:
        freq_per_group = pd.DataFrame(columns=group_cols + ['topk_frequency'])

    overall_freq = freq_per_group['topk_frequency'].mean() if len(freq_per_group) > 0 else 0.0

    return freq_per_group, overall_freq


# ==============================================================================
# Binary Presence Metrics
# ==============================================================================

def compute_binary_presence(df, profile_col, group_cols=None):
    """
    Compute binary presence: does the profile appear at least once?

    Returns the fraction of groups where the profile appears at least once (0 or 1 per group).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples
    profile_col : str
        Column name for the profile to check
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    tuple
        (presence_per_group, overall_presence)
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    df_copy = df.copy()
    df_copy['has_best_profile'] = (df_copy[profile_col] > 0).astype(int)

    # Check if ANY sample in each group has the profile
    presence_per_group = df_copy.groupby(group_cols)['has_best_profile'].max().reset_index()
    presence_per_group.rename(columns={'has_best_profile': 'binary_presence'}, inplace=True)

    # Overall mean (fraction of groups with at least one)
    overall_presence = presence_per_group['binary_presence'].mean()

    return presence_per_group, overall_presence


def compute_hpd_binary_presence(df, profile_col, credible_mass=0.95,
                                posterior_weight_col='posterior_weight',
                                group_cols=None):
    """
    Compute binary presence in HPD region.

    Returns the fraction of groups where the profile appears at least once in HPD.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with posterior weights
    profile_col : str
        Column name for the profile to check
    credible_mass : float, default=0.95
        Fraction of posterior mass to include
    posterior_weight_col : str, default='posterior_weight'
        Column with posterior weights
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    tuple
        (presence_per_group, overall_presence)
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    # Filter to HPD region
    hpd_samples = _filter_to_hpd_region(df, credible_mass, posterior_weight_col, group_cols)

    # Get all unique groups from original data
    all_groups = df[group_cols].drop_duplicates()

    # Binary presence within HPD region
    if len(hpd_samples) > 0:
        hpd_samples['has_best_profile'] = (hpd_samples[profile_col] > 0).astype(int)
        presence_per_group = hpd_samples.groupby(group_cols)['has_best_profile'].max().reset_index()
        presence_per_group.rename(columns={'has_best_profile': 'hpd_binary_presence'}, inplace=True)
    else:
        presence_per_group = pd.DataFrame(columns=group_cols + ['hpd_binary_presence'])

    # Merge to ensure all groups are represented (missing ones get 0)
    presence_all_groups = all_groups.merge(presence_per_group, on=group_cols, how='left')
    presence_all_groups['hpd_binary_presence'] = presence_all_groups['hpd_binary_presence'].fillna(0)

    overall_presence = presence_all_groups['hpd_binary_presence'].mean()

    return presence_all_groups, overall_presence


def compute_top_k_binary_presence(df, profile_col, loss_col, k_fraction=0.95,
                                  group_cols=None):
    """
    Compute binary presence in top-K models.

    Returns the fraction of groups where the profile appears at least once in top-K.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with loss values
    profile_col : str
        Column name for the profile to check
    loss_col : str
        Column with loss values (lower is better)
    k_fraction : float, default=0.95
        Fraction of models to include
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    tuple
        (presence_per_group, overall_presence)
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    # Filter to top-K
    top_k_samples = _filter_to_top_k(df, loss_col, k_fraction, group_cols)

    # Binary presence within top-K
    if len(top_k_samples) > 0:
        top_k_samples['has_best_profile'] = (top_k_samples[profile_col] > 0).astype(int)
        presence_per_group = top_k_samples.groupby(group_cols)['has_best_profile'].max().reset_index()
        presence_per_group.rename(columns={'has_best_profile': 'topk_binary_presence'}, inplace=True)
    else:
        presence_per_group = pd.DataFrame(columns=group_cols + ['topk_binary_presence'])

    overall_presence = presence_per_group['topk_binary_presence'].mean() if len(presence_per_group) > 0 else 0.0

    return presence_per_group, overall_presence


# ==============================================================================
# Epsilon-Ball Metrics
# ==============================================================================

def compute_epsilon_curve(df, loss_col, true_best_profile, n_bins=15,
                          eps_extend_max=25, eps_max_cutoff=None,
                          group_cols=None):
    """
    Compute epsilon curve for recovery rate analysis.

    Tracks how the recovery rate of the true best profile changes as the
    epsilon-ball (distance from optimal) expands.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples/models
    loss_col : str
        Column name for loss/posterior metric
    true_best_profile : str
        Column name for true best profile
    n_bins : int, default=15
        Number of bins for epsilon histogram
    eps_extend_max : float, default=25
        Maximum epsilon value for extending the curve
    eps_max_cutoff : float, optional
        Optional maximum epsilon value for binning. If provided,
        only epsilon values <= eps_max_cutoff will be used for creating bins
    group_cols : list of str, optional
        Columns to group by. Default is ['n_per_pol', 'sim_num']

    Returns
    -------
    tuple
        (plot_ext_df, eps_bins)
        - plot_ext_df: DataFrame ready for plotting with columns
          [n_per_pol, eps_levels, is_present_level, profile_rate_eps]
        - eps_bins: List of epsilon bin edges
    """
    if group_cols is None:
        group_cols = ['n_per_pol', 'sim_num']

    eps_df = df.copy()

    # Compute epsilon: distance from mode (best model) as percentage
    eps_df['eps'] = eps_df.groupby(group_cols)[loss_col].transform(
        lambda x: (x - x.min()) / x.min() * 100
    )

    # Create epsilon bins with optional cutoff
    if eps_max_cutoff is not None:
        # Filter data for binning only up to cutoff
        eps_for_binning = eps_df[eps_df['eps'] <= eps_max_cutoff]['eps']
        if len(eps_for_binning) == 0:
            # If no data within cutoff, use all data
            eps_for_binning = eps_df['eps']
        eps_hist = np.histogram(eps_for_binning, bins=n_bins)[1].tolist()
        # Ensure the max bin edge doesn't exceed the cutoff
        eps_hist = [e for e in eps_hist if e <= eps_max_cutoff]
        eps_levels = [-0.1] + eps_hist + [eps_max_cutoff]
    else:
        eps_hist = np.histogram(eps_df['eps'], bins=n_bins)[1].tolist()
        eps_levels = [-0.1] + eps_hist + [np.max(eps_df['eps']) + 0.1]

    eps_bins = eps_levels[1:]
    eps_df['eps_levels'] = pd.cut(eps_df['eps'], bins=eps_levels, labels=eps_bins)

    # Mark best profile presence
    eps_df['right_best_profile'] = eps_df[true_best_profile]
    eps_df['best_profile_present'] = eps_df['right_best_profile']

    # Create plot dataframe
    plot_df = eps_df.copy()
    plot_df['is_present_level'] = plot_df.groupby(
        group_cols + ['eps_levels'], observed=False
    )['best_profile_present'].transform('max')

    # Fill in missing eps_levels for some groups
    n_per_pol_index = list(np.unique(eps_df['n_per_pol']))
    sims_index = list(np.unique(eps_df['sim_num']))
    eps_index = list(eps_bins)

    new_index = pd.MultiIndex.from_product(
        [n_per_pol_index, sims_index, eps_index],
        names=['n_per_pol', 'sim_num', 'eps_levels']
    )
    plot_df = plot_df.drop_duplicates(['n_per_pol', 'sim_num', 'eps_levels'])
    plot_df = plot_df.set_index(['n_per_pol', 'sim_num', 'eps_levels']).reindex(new_index).reset_index()

    plot_df['is_present_level'] = plot_df['is_present_level'].fillna(0)

    # Cumulative sum: once found, stays found
    plot_df['is_present_level'] = plot_df.groupby(group_cols)['is_present_level'].cumsum()
    plot_df.loc[plot_df['is_present_level'] > 0, 'is_present_level'] = 1

    # Compute profile rate across groups
    plot_df['profile_rate_eps'] = plot_df.groupby(
        ['n_per_pol', 'eps_levels'], observed=False
    )['is_present_level'].transform('mean')

    # Convert eps_levels from categorical to numeric using the right edge of each bin
    plot_df['eps_levels'] = plot_df['eps_levels'].apply(
        lambda x: x.right if hasattr(x, 'right') else x
    )

    plot_df = plot_df.dropna(subset=['eps_levels'])
    plot_df = plot_df.drop_duplicates(['n_per_pol', 'eps_levels'])
    plot_df = plot_df.sort_values('eps_levels')

    # Extend to max epsilon for plotting
    new_rows = []
    max_eps_in_data = np.max(plot_df['eps_levels'])
    eps_extended_range = np.arange(max_eps_in_data, eps_extend_max, step=3)
    is_present_max = np.max(plot_df['is_present_level'])
    profile_eps_max = np.max(plot_df['profile_rate_eps'])

    for eps_extended_i in eps_extended_range:
        new_rows.append([plot_df['n_per_pol'].iloc[0], eps_extended_i,
                        is_present_max, profile_eps_max])

    new_rows_df = pd.DataFrame(new_rows,
                               columns=['n_per_pol', 'eps_levels',
                                        'is_present_level', 'profile_rate_eps'])
    plot_ext_df = pd.concat([
        plot_df[['n_per_pol', 'eps_levels', 'is_present_level', 'profile_rate_eps']],
        new_rows_df
    ])

    return plot_ext_df, eps_bins
