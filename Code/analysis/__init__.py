"""
Analysis utilities for Rashomon simulation results.

This module provides tools for loading, processing, and analyzing simulation results
from various methods (Rashomon sets, Bayesian Lasso, Spike-Slab Lasso, PPMx, Bootstrap).
"""

from .data import (
    load_simulation_results,
    setup_profile_config,
    prepare_method_dataframe,
    METHOD_CONFIG,
)

from .metrics import (
    compute_frequency_inclusion,
    compute_hpd_frequency,
    compute_top_k_frequency,
    compute_binary_presence,
    compute_hpd_binary_presence,
    compute_top_k_binary_presence,
    compute_epsilon_curve,
)

__all__ = [
    # Data loading
    'load_simulation_results',
    'setup_profile_config',
    'prepare_method_dataframe',
    'METHOD_CONFIG',
    # Metrics
    'compute_frequency_inclusion',
    'compute_hpd_frequency',
    'compute_top_k_frequency',
    'compute_binary_presence',
    'compute_hpd_binary_presence',
    'compute_top_k_binary_presence',
    'compute_epsilon_curve',
]
