"""
Baseline methods for comparison with Rashomon Partition Sets.

This module provides alternative regression methods including:
- BayesianLasso: Bayesian Lasso with Laplace priors via Gibbs sampling
- BootstrapLasso: Bootstrap Lasso for empirical uncertainty quantification
"""

from .bayesian_lasso import BayesianLasso
from .bootstrap_lasso import BootstrapLasso
from .diagnostics import (
    gelman_rubin,
    check_convergence,
    plot_traces,
    plot_autocorr,
    plot_posterior_hist
)

__all__ = [
    'BayesianLasso',
    'BootstrapLasso',
    'gelman_rubin',
    'check_convergence',
    'plot_traces',
    'plot_autocorr',
    'plot_posterior_hist'
]
