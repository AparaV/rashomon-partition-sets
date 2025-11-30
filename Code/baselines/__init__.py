"""
Baseline methods for comparison with Rashomon Partition Sets.

This module provides alternative regression methods including:
- BayesianLasso: Bayesian Lasso with Laplace priors via Gibbs sampling
- BootstrapLasso: Bootstrap Lasso for empirical uncertainty quantification
- TVA: Treatment Variance Analysis using Puffer transform
- PPMx: Product Partition Model with Covariates
"""

from .bayesian_lasso import BayesianLasso
from .bootstrap_lasso import BootstrapLasso
from .tva import TVA
from .ppmx import PPMx
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
    'TVA',
    'PPMx',
    'gelman_rubin',
    'check_convergence',
    'plot_traces',
    'plot_autocorr',
    'plot_posterior_hist'
]
