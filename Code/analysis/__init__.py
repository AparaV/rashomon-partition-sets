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

__all__ = [
    'load_simulation_results',
    'setup_profile_config',
    'prepare_method_dataframe',
    'METHOD_CONFIG',
]
