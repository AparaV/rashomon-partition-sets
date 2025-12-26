"""
Data loading and preprocessing utilities for simulation results.
"""

import pandas as pd
from rashomon import hasse


# Method-specific configuration
METHOD_CONFIG = {
    'rashomon': {
        'filename': 'rashomon',
        'loss_col': 'loss',
        'needs_loss_computation': True,
        'sign_flip': False,
        'columns_to_drop': ['num_pools', 'MSE', 'IOU', 'min_dosage', 'best_pol_diff'],
    },
    'lasso': {
        'filename': 'lasso',
        'loss_col': 'L1_loss',
        'needs_loss_computation': False,
        'sign_flip': False,
        'columns_to_drop': ['MSE', 'IOU', 'min_dosage', 'best_pol_diff'],
    },
    'blasso': {
        'filename': 'blasso',
        'loss_col': 'neg_log_posterior',
        'needs_loss_computation': False,
        'sign_flip': False,
        'columns_to_drop': [],
    },
    'ssl': {
        'filename': 'ssl',
        'loss_col': 'neg_log_posterior',
        'needs_loss_computation': False,
        'sign_flip': False,
        'columns_to_drop': ['MSE', 'IOU', 'min_dosage', 'best_pol_diff', 'converged',
                            'max_rhat', 'mean_inclusion_prob', 'n_selected_features'],
    },
    'ppmx': {
        'filename': 'ppmx',
        'loss_col': 'neg_log_posterior',
        'needs_loss_computation': False,
        'sign_flip': True,  # PPMx stores positive log posterior, needs negation
        'columns_to_drop': ['MSE', 'IOU', 'min_dosage', 'best_pol_diff', 'converged',
                            'max_rhat', 'IOU_coverage', 'min_dosage_coverage',
                            'n_clusters', 'acceptance_rate'],
    },
    'bootstrap': {
        'filename': 'bootstrap',
        'loss_col': 'penalized_loss',
        'needs_loss_computation': False,
        'sign_flip': False,
        'columns_to_drop': [],
    },
    'tva': {
        'filename': 'tva',
        'loss_col': 'neg_log_posterior',
        'needs_loss_computation': False,
        'sign_flip': False,
        'columns_to_drop': ['MSE', 'IOU', 'min_dosage', 'best_pol_diff', 'converged',
                            'max_rhat'],
    },
}


def setup_profile_config(n_arms=4):
    """
    Set up profile configuration for treatment arm analysis.

    Parameters
    ----------
    n_arms : int, default=4
        Number of treatment arms

    Returns
    -------
    dict
        Dictionary containing:
        - 'profiles': list of all possible profiles
        - 'profiles_map': dict mapping profiles to indices
        - 'profile_cols': list of profile column names (as strings)
        - 'best_profile': tuple representing the true best profile
        - 'best_profile_idx': index of the best profile
        - 'true_best_profile': string representation of best profile
    """
    profiles, profiles_map = hasse.enumerate_profiles(n_arms)
    profile_cols = [str(x) for x in profiles]
    best_profile = (1, 0, 1, 0)
    best_profile_idx = profiles_map[best_profile]
    true_best_profile = str(best_profile)

    return {
        'profiles': profiles,
        'profiles_map': profiles_map,
        'profile_cols': profile_cols,
        'best_profile': best_profile,
        'best_profile_idx': best_profile_idx,
        'true_best_profile': true_best_profile,
    }


def load_simulation_results(method, n_per_pol=30, n_sims=100,
                            results_dir='../Results/4arms',
                            file_prefix='4arms_'):
    """
    Load simulation results for a specific method.

    Parameters
    ----------
    method : str
        Method name. One of: 'rashomon', 'lasso', 'blasso', 'ssl', 'ppmx',
        'bootstrap', 'bootstrap_samples'
    n_per_pol : int, default=30
        Number of samples per policy
    n_sims : int, default=100
        Number of simulation runs
    results_dir : str, default='../Results/4arms'
        Directory containing results files
    file_prefix : str, default='4arms_'
        Prefix for result filenames

    Returns
    -------
    pd.DataFrame
        Loaded and cleaned dataframe

    Raises
    ------
    ValueError
        If method is not recognized
    """
    if method not in METHOD_CONFIG:
        raise ValueError(f"Unknown method: {method}. "
                         f"Available methods: {list(METHOD_CONFIG.keys())}")

    config = METHOD_CONFIG[method]
    filename = f"{results_dir}/{file_prefix}{config['filename']}_{n_per_pol}_{n_sims}.csv"

    df = pd.read_csv(filename)

    # Drop the unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    return df


def prepare_method_dataframe(df, method, true_best_profile,
                             lambda_reg=1e-1):
    """
    Prepare method-specific dataframe with computed features.

    This function:
    1. Computes loss if needed (for Rashomon with regularization)
    2. Flips sign if needed (for PPMx)
    3. Adds right/wrong best profile indicators
    4. Drops unnecessary columns
    5. Sorts by loss/posterior

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_simulation_results()
    method : str
        Method name for configuration lookup
    true_best_profile : str
        Column name of the true best profile
    lambda_reg : float, default=1e-1
        Regularization parameter for Rashomon loss computation

    Returns
    -------
    pd.DataFrame
        Processed dataframe ready for analysis
    """
    if method not in METHOD_CONFIG:
        raise ValueError(f"Unknown method: {method}")

    config = METHOD_CONFIG[method]
    df = df.copy()

    # Handle Rashomon loss computation
    if config['needs_loss_computation'] and method == 'rashomon':
        df['loss'] = df['MSE'] + lambda_reg * df['num_pools']
        df['neg_log_posterior'] = -df['loss']  # For consistency

    # Handle PPMx sign flip
    if config['sign_flip']:
        df['neg_log_posterior'] = -df['neg_log_posterior']

    # Add best profile indicators
    df['right_best_profile'] = df[true_best_profile]
    df['wrong_best_profile'] = 1 - df['right_best_profile']

    # Drop unnecessary columns
    cols_to_drop = [col for col in config['columns_to_drop'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)

    # Sort by loss/posterior if column exists
    loss_col = config['loss_col']
    if loss_col and loss_col in df.columns:
        df = df.sort_values(['n_per_pol', 'sim_num', loss_col], ascending=True)
    else:
        df = df.sort_values(['n_per_pol', 'sim_num'], ascending=True)

    # Add best_profile_present indicator
    df['best_profile_present'] = (df[true_best_profile] > 0).astype(int)

    return df


def aggregate_worst_case_results(raw_df, method, metrics=None):
    """
    Aggregate worst case simulation results.

    This function handles two aggregation patterns:
    1. Point estimates (lasso, tva): directly average metrics across simulations
    2. Sample-based (rashomon, blasso, bootstrap, ssl, ppmx): first average
       within each simulation, then average across simulations

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw results with columns: n_per_pol, sim_num, and metric columns.
    method : str
        Method name. One of: 'rashomon', 'lasso', 'tva', 'blasso',
        'bootstrap', 'ssl', 'ppmx'.
    metrics : list of str, optional
        Metrics to aggregate. Default depends on method:
        - rashomon: ['num_pools', 'MSE', 'IOU', 'min_dosage', 'best_pol_MSE']
        - others: ['MSE', 'IOU', 'min_dosage', 'best_pol_MSE']

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with columns: n_per_pol and metric columns.
        For sample-based methods, averages are computed at two levels.
        For point estimates, a single average is computed.

    Notes
    -----
    - Assumes 'best_pol_diff' column exists and computes 'best_pol_MSE' from it
    - Drops unnecessary columns based on method type
    - Returns one row per n_per_pol value
    """
    df = raw_df.copy()
    df['best_pol_MSE'] = df['best_pol_diff'] ** 2

    # Set default metrics
    if metrics is None:
        if method == 'rashomon':
            metrics = ['num_pools', 'MSE', 'IOU', 'min_dosage', 'best_pol_MSE']
        else:
            metrics = ['MSE', 'IOU', 'min_dosage', 'best_pol_MSE']

    # Point estimate methods: lasso and tva
    if method in ['lasso', 'tva']:
        # Directly average across simulations (grouped by n_per_pol)
        for metric in metrics:
            df[metric] = df.groupby('n_per_pol')[metric].transform('mean')

        # Drop duplicates and unnecessary columns
        df = df.drop_duplicates('n_per_pol')
        cols_to_drop = ['best_pol_diff', 'sim_num']
        if method == 'lasso':
            cols_to_drop.append('L1_loss')
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    # Sample-based methods: rashomon, blasso, bootstrap, ssl, ppmx
    else:
        # Step 1: Average within each simulation
        group_by_cols = ['n_per_pol', 'sim_num']
        for metric in metrics:
            df[metric] = df.groupby(group_by_cols)[metric].transform('mean')

        # Drop duplicates within simulations
        df = df.drop_duplicates(group_by_cols)

        # Drop method-specific columns
        cols_to_drop = ['best_pol_diff']
        if method == 'rashomon':
            pass  # Rashomon doesn't have extra columns to drop
        elif method == 'blasso':
            cols_to_drop.extend(['sample_idx', 'neg_log_posterior'])
        elif method == 'bootstrap':
            cols_to_drop.extend(['sample_idx', 'penalized_loss'])
        elif method == 'ssl':
            cols_to_drop.extend(['sample_idx', 'neg_log_posterior'])
        elif method == 'ppmx':
            cols_to_drop.extend(['sample_idx', 'neg_log_posterior'])

        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

        # Step 2: Average across simulations
        for metric in metrics:
            df[metric] = df.groupby('n_per_pol')[metric].transform('mean')

        # Drop duplicates and sim_num column
        df = df.drop_duplicates('n_per_pol')
        df = df.drop('sim_num', axis=1)

    return df
