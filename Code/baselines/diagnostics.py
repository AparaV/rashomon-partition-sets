"""
Convergence diagnostics and visualization utilities for MCMC chains.

Includes:
- Gelman-Rubin convergence diagnostic (R-hat statistic)
- Trace plots for visual inspection
- Autocorrelation plots
- Posterior distribution histograms
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import warnings


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """
    Compute Gelman-Rubin convergence diagnostic (R-hat) for MCMC chains.
    
    The Gelman-Rubin statistic compares between-chain and within-chain variance
    to assess convergence. Values close to 1 (typically < 1.1) indicate convergence.
    
    Reference: Gelman & Rubin (1992) "Inference from Iterative Simulation Using
    Multiple Sequences" Statistical Science, 7(4), 457-472.
    
    Parameters
    ----------
    chains : ndarray of shape (n_chains, n_samples, n_params)
        MCMC samples from multiple chains
        
    Returns
    -------
    rhat : ndarray of shape (n_params,)
        R-hat statistic for each parameter
        
    Notes
    -----
    The computation follows:
    - B = between-chain variance
    - W = within-chain variance
    - V_hat = weighted average of W and B
    - R_hat = sqrt(V_hat / W)
    """
    n_chains, n_samples, n_params = chains.shape
    
    if n_chains < 2:
        raise ValueError("Need at least 2 chains to compute Gelman-Rubin statistic")
    
    # Compute chain means: shape (n_chains, n_params)
    chain_means = np.mean(chains, axis=1)
    
    # Overall mean across all chains: shape (n_params,)
    overall_mean = np.mean(chain_means, axis=0)
    
    # Between-chain variance: B = (n / (m-1)) * sum((chain_mean - overall_mean)^2)
    B = n_samples / (n_chains - 1) * np.sum((chain_means - overall_mean) ** 2, axis=0)
    
    # Within-chain variance: W = (1/m) * sum(s_j^2) where s_j^2 is variance of chain j
    chain_variances = np.var(chains, axis=1, ddof=1)  # shape (n_chains, n_params)
    W = np.mean(chain_variances, axis=0)
    
    # Estimated variance: V_hat = ((n-1)/n)*W + ((m+1)/m)*B
    V_hat = ((n_samples - 1) / n_samples) * W + ((n_chains + 1) / n_chains) * B
    
    # R-hat: sqrt(V_hat / W)
    # Add small constant to avoid division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rhat = np.sqrt(V_hat / (W + 1e-10))
    
    # Handle edge cases
    rhat = np.where(np.isnan(rhat) | np.isinf(rhat), 1.0, rhat)
    
    return rhat


def check_convergence(
    chains: np.ndarray,
    threshold: float = 1.1
) -> bool:
    """
    Check if all parameters have converged based on Gelman-Rubin statistic.
    
    Parameters
    ----------
    chains : ndarray of shape (n_chains, n_samples, n_params)
        MCMC samples from multiple chains
    threshold : float, default=1.1
        Maximum acceptable R-hat value for convergence
        
    Returns
    -------
    converged : bool
        True if all parameters have R-hat < threshold
    """
    rhat = gelman_rubin(chains)
    return np.all(rhat < threshold)


def plot_traces(
    chains: np.ndarray,
    param_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    max_params: int = 10
) -> plt.Figure:
    """
    Plot MCMC traces for visual convergence inspection.
    
    Parameters
    ----------
    chains : ndarray of shape (n_chains, n_samples, n_params)
        MCMC samples from multiple chains
    param_names : list of str, optional
        Names for each parameter (defaults to β₀, β₁, ...)
    figsize : tuple of int, default=(12, 8)
        Figure size in inches
    save_path : str, optional
        Path to save the figure (if None, displays instead)
    max_params : int, default=10
        Maximum number of parameters to plot (to avoid overcrowding)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    n_chains, n_samples, n_params = chains.shape
    
    # Limit number of parameters to plot
    if n_params > max_params:
        warnings.warn(
            f"Too many parameters ({n_params}). Plotting only first {max_params}."
        )
        chains = chains[:, :, :max_params]
        n_params = max_params
    
    # Default parameter names
    if param_names is None:
        param_names = [f'β_{i}' for i in range(n_params)]
    
    # Compute R-hat for titles
    rhat = gelman_rubin(chains)
    
    # Create subplots
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Plot each parameter
    for param_idx in range(n_params):
        ax = axes[param_idx]
        
        # Plot each chain
        for chain_idx in range(n_chains):
            ax.plot(
                chains[chain_idx, :, param_idx],
                alpha=0.7,
                linewidth=0.5,
                label=f'Chain {chain_idx + 1}'
            )
        
        ax.set_title(f'{param_names[param_idx]} (R̂ = {rhat[param_idx]:.3f})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        
        if param_idx == 0:
            ax.legend(loc='best', fontsize=8)
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_autocorr(
    chain: np.ndarray,
    param_idx: int = 0,
    max_lag: int = 50,
    param_name: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot autocorrelation function for a single parameter in a chain.
    
    Parameters
    ----------
    chain : ndarray of shape (n_samples, n_params) or (n_samples,)
        MCMC samples from a single chain
    param_idx : int, default=0
        Index of parameter to plot (if chain is 2D)
    max_lag : int, default=50
        Maximum lag to compute autocorrelation
    param_name : str, optional
        Name of the parameter (defaults to β_{param_idx})
    figsize : tuple of int, default=(8, 4)
        Figure size in inches
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Extract parameter if 2D
    if chain.ndim == 2:
        samples = chain[:, param_idx]
    else:
        samples = chain
    
    if param_name is None:
        param_name = f'β_{param_idx}'
    
    # Compute autocorrelation
    n_samples = len(samples)
    max_lag = min(max_lag, n_samples - 1)
    
    # Demean
    samples_centered = samples - np.mean(samples)
    
    # Compute autocorrelation for each lag
    autocorr = np.zeros(max_lag + 1)
    variance = np.var(samples)
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            autocorr[lag] = np.mean(
                samples_centered[:-lag] * samples_centered[lag:]
            ) / (variance + 1e-10)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(max_lag + 1), autocorr, width=0.8, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±0.05')
    ax.axhline(y=-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Autocorrelation Function: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_posterior_hist(
    chains: np.ndarray,
    param_idx: int = 0,
    param_name: Optional[str] = None,
    bins: int = 50,
    true_value: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of posterior distribution for a parameter.
    
    Parameters
    ----------
    chains : ndarray of shape (n_chains, n_samples, n_params)
        MCMC samples from multiple chains
    param_idx : int, default=0
        Index of parameter to plot
    param_name : str, optional
        Name of the parameter
    bins : int, default=50
        Number of histogram bins
    true_value : float, optional
        True parameter value to mark on plot (if known)
    figsize : tuple of int, default=(8, 5)
        Figure size in inches
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    if param_name is None:
        param_name = f'β_{param_idx}'
    
    # Flatten all chains for this parameter
    samples = chains[:, :, param_idx].flatten()
    
    # Compute statistics
    mean_val = np.mean(samples)
    median_val = np.median(samples)
    ci_lower = np.percentile(samples, 2.5)
    ci_upper = np.percentile(samples, 97.5)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(samples, bins=bins, density=True, alpha=0.7, edgecolor='black')
    
    # Mark statistics
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    ax.axvline(ci_lower, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(ci_upper, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, 
               label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    
    # Mark true value if provided
    if true_value is not None:
        ax.axvline(true_value, color='green', linestyle='-', linewidth=2, 
                   label=f'True: {true_value:.3f}')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Posterior Distribution: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig
