"""
Unified simulation results analysis script.

This script consolidates analysis from both notebooks:
- analyze_reff_simulations.ipynb (epsilon-ball analysis)
- analyze_posterior_comparison.ipynb (HPD-region analysis)

It demonstrates both perspectives on the same simulation data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rashomon.metrics import compute_posterior_weights
from analysis import (
    # Data loading
    load_simulation_results,
    setup_profile_config,
    prepare_method_dataframe,
    # Metrics
    compute_epsilon_curve,
    compute_frequency_inclusion,
    compute_hpd_frequency,
    compute_top_k_frequency,
    compute_binary_presence,
    compute_hpd_binary_presence,
    compute_top_k_binary_presence,
    # Visualization
    plot_epsilon_comparison,
    plot_hpd_bar_comparison,
    plot_coverage_vs_credible,
    plot_sample_size_vs_credible,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Simulation parameters
N_PER_POL = 30
N_SIMS = 100
RESULTS_DIR = '../Results/4arms'
FILE_PREFIX = '4arms_'

# Analysis parameters
LAMBDA_RASHOMON = 1e-1  # Regularization for Rashomon loss
CREDIBLE_MASS = 0.95    # HPD region size (95%)

# Epsilon curve parameters
N_BINS = 15
EPS_EXTEND_MAX = 15
EPS_MAX_CUTOFF = 15  # For Bayesian methods, cap epsilon at 25%

# Methods to analyze
METHODS_BAYESIAN = ['rashomon', 'blasso', 'ssl', 'ppmx']  # Methods with posterior
METHODS_FREQUENTIST = ['bootstrap_samples']  # Methods without posterior
METHOD_LASSO = 'lasso'  # Point estimate baseline

# Figures directory
FIGURES_DIR = '../Figures/reff_revisions/'

# Create figures directory if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================================
# SECTION 0: Setup
# ==============================================================================

print("=" * 80)
print("UNIFIED SIMULATION ANALYSIS")
print("=" * 80)
print("\nConfiguration:")
print(f"  Samples per policy: {N_PER_POL}")
print(f"  Number of simulations: {N_SIMS}")
print(f"  Credible mass: {CREDIBLE_MASS * 100}%")
print(f"  Lambda (Rashomon): {LAMBDA_RASHOMON}")

# Setup profile configuration
profile_config = setup_profile_config(n_arms=4)
true_best_profile = profile_config['true_best_profile']
best_profile_idx = profile_config['best_profile_idx']

print("Profile configuration:")
print(f"  True best profile: {true_best_profile}")
print(f"  Profile index: {best_profile_idx}")
print(f"  Total profiles: {len(profile_config['profile_cols'])}")


# ==============================================================================
# SECTION 1: Load and Prepare Data
# ==============================================================================

print("=" * 80)
print("SECTION 1: Loading Data")
print("=" * 80)

# Dictionary to store all loaded dataframes
data = {}

# Load Bayesian methods
for method in METHODS_BAYESIAN:
    print(f"Loading {method}...")
    df = load_simulation_results(method, N_PER_POL, N_SIMS, RESULTS_DIR, FILE_PREFIX)
    df = prepare_method_dataframe(df, method, true_best_profile, LAMBDA_RASHOMON)
    data[method] = df
    print(f"  {len(df)} samples loaded")

# Load Bootstrap
print("Loading bootstrap_samples...")
df = load_simulation_results('bootstrap_samples', N_PER_POL, N_SIMS, RESULTS_DIR, FILE_PREFIX)
df = prepare_method_dataframe(df, 'bootstrap_samples', true_best_profile)
data['bootstrap'] = df
print(f"  {len(df)} samples loaded")

# Load Lasso (for point estimate comparison)
print("Loading lasso...")
df = load_simulation_results(METHOD_LASSO, N_PER_POL, N_SIMS, RESULTS_DIR, FILE_PREFIX)
df = prepare_method_dataframe(df, METHOD_LASSO, true_best_profile)
data['lasso'] = df
print(f"  {len(df)} samples loaded")

print("\nAll data loaded successfully!")


# ==============================================================================
# SECTION 2: Compute Posterior Weights (for Bayesian methods)
# ==============================================================================

print("=" * 80)
print("SECTION 2: Computing Posterior Weights")
print("=" * 80)

for method in METHODS_BAYESIAN:
    print(f"Computing posterior weights for {method}...")
    data[method] = compute_posterior_weights(data[method])

# For bootstrap, use penalized loss to compute weights
print("Computing posterior weights for bootstrap...")
data['bootstrap'] = compute_posterior_weights(
    data['bootstrap'],
    neg_log_posterior_col='penalized_loss'
)

print("\nPosterior weights computed!")


# ==============================================================================
# SECTION 3: Epsilon-Ball Analysis (Recovery Rate vs Distance from Optimal)
# ==============================================================================

print("=" * 80)
print("SECTION 3: Epsilon-Ball Analysis")
print("=" * 80)

epsilon_curves = {}

# Compute epsilon curves for all methods
print("Computing epsilon curves...")
for method in METHODS_BAYESIAN:
    print(f"  {method}...", end=' ')
    curve, _ = compute_epsilon_curve(
        data[method],
        loss_col='loss' if method == 'rashomon' else 'neg_log_posterior',
        true_best_profile=true_best_profile,
        n_bins=N_BINS if method == 'rashomon' else N_BINS,
        eps_extend_max=EPS_EXTEND_MAX,
        eps_max_cutoff=None if method == 'rashomon' else EPS_MAX_CUTOFF
    )
    epsilon_curves[method] = curve
    print(f"{len(curve)} points")

# Bootstrap
print("  bootstrap...", end=' ')
curve, _ = compute_epsilon_curve(
    data['bootstrap'],
    loss_col='penalized_loss',
    true_best_profile=true_best_profile,
    n_bins=N_BINS,
    eps_extend_max=EPS_EXTEND_MAX,
    eps_max_cutoff=EPS_MAX_CUTOFF
)
epsilon_curves['bootstrap'] = curve
print(f"{len(curve)} points")

# Compute Lasso point (for comparison)
lasso_profile_means = data['lasso'].groupby('n_per_pol')[true_best_profile].mean()
lasso_point = (0, lasso_profile_means.iloc[0])

print("\nEpsilon curves computed!")

# Plot epsilon comparison
print("\nGenerating epsilon comparison plot...")
fig, ax = plot_epsilon_comparison(
    epsilon_curves,
    lasso_point=lasso_point,
    save_path=f'{FIGURES_DIR}/epsilon_comparison.png'
)
plt.close()
print(f"  Saved to: {FIGURES_DIR}/epsilon_comparison.png")
print()


# ==============================================================================
# SECTION 4: HPD-Region Analysis (High Posterior Density)
# ==============================================================================

print("=" * 80)
print("SECTION 4: HPD-Region Analysis")
print("=" * 80)

# Compute metrics for all methods
metrics_results = {
    'Method': [],
    'HPD_Frequency': [],
    'HPD_Presence': [],
    'Full_Frequency': [],
    'Full_Presence': [],
}

# Rashomon
print("Computing metrics for rashomon...")
_, hpd_freq, _ = compute_hpd_frequency(data['rashomon'], true_best_profile, CREDIBLE_MASS)
_, hpd_pres = compute_hpd_binary_presence(data['rashomon'], true_best_profile, CREDIBLE_MASS)
_, full_freq = compute_frequency_inclusion(data['rashomon'], true_best_profile)
_, full_pres = compute_binary_presence(data['rashomon'], true_best_profile)
metrics_results['Method'].append('Rashomon')
metrics_results['HPD_Frequency'].append(hpd_freq)
metrics_results['HPD_Presence'].append(hpd_pres)
metrics_results['Full_Frequency'].append(full_freq)
metrics_results['Full_Presence'].append(full_pres)

# Bayesian Lasso
print("Computing metrics for blasso...")
_, hpd_freq, _ = compute_hpd_frequency(data['blasso'], true_best_profile, CREDIBLE_MASS)
_, hpd_pres = compute_hpd_binary_presence(data['blasso'], true_best_profile, CREDIBLE_MASS)
_, full_freq = compute_frequency_inclusion(data['blasso'], true_best_profile)
_, full_pres = compute_binary_presence(data['blasso'], true_best_profile)
metrics_results['Method'].append('Bayesian Lasso')
metrics_results['HPD_Frequency'].append(hpd_freq)
metrics_results['HPD_Presence'].append(hpd_pres)
metrics_results['Full_Frequency'].append(full_freq)
metrics_results['Full_Presence'].append(full_pres)

# SSL
print("Computing metrics for ssl...")
_, hpd_freq, _ = compute_hpd_frequency(data['ssl'], true_best_profile, CREDIBLE_MASS)
_, hpd_pres = compute_hpd_binary_presence(data['ssl'], true_best_profile, CREDIBLE_MASS)
_, full_freq = compute_frequency_inclusion(data['ssl'], true_best_profile)
_, full_pres = compute_binary_presence(data['ssl'], true_best_profile)
metrics_results['Method'].append('SSL')
metrics_results['HPD_Frequency'].append(hpd_freq)
metrics_results['HPD_Presence'].append(hpd_pres)
metrics_results['Full_Frequency'].append(full_freq)
metrics_results['Full_Presence'].append(full_pres)

# PPMx
print("Computing metrics for ppmx...")
_, hpd_freq, _ = compute_hpd_frequency(data['ppmx'], true_best_profile, CREDIBLE_MASS)
_, hpd_pres = compute_hpd_binary_presence(data['ppmx'], true_best_profile, CREDIBLE_MASS)
_, full_freq = compute_frequency_inclusion(data['ppmx'], true_best_profile)
_, full_pres = compute_binary_presence(data['ppmx'], true_best_profile)
metrics_results['Method'].append('PPMx')
metrics_results['HPD_Frequency'].append(hpd_freq)
metrics_results['HPD_Presence'].append(hpd_pres)
metrics_results['Full_Frequency'].append(full_freq)
metrics_results['Full_Presence'].append(full_pres)

# Bootstrap (using top-K instead of HPD)
print("Computing metrics for bootstrap...")
_, topk_freq = compute_top_k_frequency(data['bootstrap'], true_best_profile,
                                       'penalized_loss', CREDIBLE_MASS)
_, topk_pres = compute_top_k_binary_presence(data['bootstrap'], true_best_profile,
                                             'penalized_loss', CREDIBLE_MASS)
_, full_freq = compute_frequency_inclusion(data['bootstrap'], true_best_profile)
_, full_pres = compute_binary_presence(data['bootstrap'], true_best_profile)
metrics_results['Method'].append('Bootstrap')
metrics_results['HPD_Frequency'].append(topk_freq)
metrics_results['HPD_Presence'].append(topk_pres)
metrics_results['Full_Frequency'].append(full_freq)
metrics_results['Full_Presence'].append(full_pres)

# Create comparison dataframe
comparison_df = pd.DataFrame(metrics_results)

# Plot bar comparison
print("Generating HPD bar comparison plot...")
fig, axes = plot_hpd_bar_comparison(
    comparison_df,
    save_path=f'{FIGURES_DIR}/hpd_bar_comparison.png'
)
plt.close()
print(f"  Saved to: {FIGURES_DIR}/hpd_bar_comparison.png")
print()


# ==============================================================================
# SECTION 5: Credible Interval Sweep Analysis
# ==============================================================================

print("=" * 80)
print("SECTION 5: Credible Interval Sweep Analysis")
print("=" * 80)

# Define credible levels to sweep
credible_levels = list(np.arange(0.05, 1, 0.05)) + list(np.geomspace(0.99, 1.0, num=10))

print(f"Sweeping {len(credible_levels)} credible levels from 5% to 100%...")

# Pre-compute cumulative weights for efficiency
print("Pre-computing cumulative weights...")
results_dfs = {}
for method in METHODS_BAYESIAN:
    df_copy = data[method].copy()
    df_copy = df_copy.sort_values(['n_per_pol', 'sim_num', 'posterior_weight'],
                                  ascending=[True, True, False])
    df_copy['cumsum_weight'] = df_copy.groupby(['n_per_pol', 'sim_num'])['posterior_weight'].cumsum()
    df_copy['has_best_profile'] = (df_copy[true_best_profile] > 0).astype(int)
    results_dfs[method] = df_copy

# Get all unique simulations (for normalization)
all_sims = data['rashomon'][['n_per_pol', 'sim_num']].drop_duplicates()
n_sims_total = len(all_sims)

# Sweep through credible levels
sweep_results = {
    'credible_level': [],
}
for method in METHODS_BAYESIAN:
    sweep_results[f'{method}_coverage'] = []
    sweep_results[f'{method}_mean_size'] = []

print("Computing metrics for each credible level...")
for level in credible_levels:
    sweep_results['credible_level'].append(level)

    for method in METHODS_BAYESIAN:
        df = results_dfs[method]

        # Filter to HPD region
        hpd_samples = df[df['cumsum_weight'] <= level].copy()

        if len(hpd_samples) > 0:
            # Binary presence
            presence = hpd_samples.groupby(['n_per_pol', 'sim_num'])['has_best_profile'].max()
            presence_all = all_sims.merge(presence.reset_index(),
                                          on=['n_per_pol', 'sim_num'], how='left')
            coverage = presence_all['has_best_profile'].fillna(0).mean()

            # Mean size
            hpd_sizes = hpd_samples.groupby(['n_per_pol', 'sim_num']).size()
            sizes_all = all_sims.merge(hpd_sizes.reset_index(name='hpd_size'),
                                       on=['n_per_pol', 'sim_num'], how='left')
            mean_size = sizes_all['hpd_size'].fillna(0).mean()
        else:
            coverage = 0.0
            mean_size = 0.0

        sweep_results[f'{method}_coverage'].append(coverage)
        sweep_results[f'{method}_mean_size'].append(mean_size)

sweep_df = pd.DataFrame(sweep_results)
print("Credible interval sweep completed!")

# Generate individual credible interval plots
print("\nGenerating credible interval sweep plots...")

# Plot 1: Coverage (full range) - with legend
fig1, ax1 = plt.subplots(figsize=(8, 6))
plot_coverage_vs_credible(sweep_df, methods=METHODS_BAYESIAN, ax=ax1,
                          xlim=(0, 105), show_legend=True)
plt.savefig(f'{FIGURES_DIR}/credible_interval_sweep_coverage.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved to: {FIGURES_DIR}/credible_interval_sweep_coverage.png")

# Plot 2: Sample size (full range) - with legend
fig2, ax2 = plt.subplots(figsize=(8, 6))
plot_sample_size_vs_credible(sweep_df, methods=METHODS_BAYESIAN, ax=ax2,
                             xlim=(0, 105), show_legend=True)
plt.savefig(f'{FIGURES_DIR}/credible_interval_sweep_sample_size.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved to: {FIGURES_DIR}/credible_interval_sweep_sample_size.png")

# Plot 3: Coverage (magnified) - no legend
fig3, ax3 = plt.subplots(figsize=(8, 6))
plot_coverage_vs_credible(sweep_df, methods=METHODS_BAYESIAN, ax=ax3,
                          xlim=(99, 100+1e-2), use_log_x=True,
                          show_legend=False, title_suffix=' (Magnified)')
plt.savefig(f'{FIGURES_DIR}/credible_interval_sweep_coverage_magnified.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved to: {FIGURES_DIR}/credible_interval_sweep_coverage_magnified.png")

# Plot 4: Sample size (magnified) - no legend
fig4, ax4 = plt.subplots(figsize=(8, 6))
plot_sample_size_vs_credible(sweep_df, methods=METHODS_BAYESIAN, ax=ax4,
                             xlim=(99, 100+1e-2), use_log_x=True,
                             show_legend=False, title_suffix=' (Magnified)')
plt.savefig(f'{FIGURES_DIR}/credible_interval_sweep_sample_size_magnified.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved to: {FIGURES_DIR}/credible_interval_sweep_sample_size_magnified.png")


# ==============================================================================
# SECTION 6: Summary
# ==============================================================================

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print(f"  1. {FIGURES_DIR}/epsilon_comparison.png")
print(f"  2. {FIGURES_DIR}/hpd_bar_comparison.png")
print(f"  3. {FIGURES_DIR}/credible_interval_sweep_coverage.png")
print(f"  4. {FIGURES_DIR}/credible_interval_sweep_sample_size.png")
print(f"  5. {FIGURES_DIR}/credible_interval_sweep_coverage_magnified.png")
print(f"  6. {FIGURES_DIR}/credible_interval_sweep_sample_size_magnified.png")
print("Analysis completed successfully!")
