"""
Configuration constants for analysis module.

This module centralizes styling and display configuration for all plotting
functions to ensure consistency across all visualizations.
"""

# ==============================================================================
# Method Styling Configuration
# ==============================================================================

# Method colors (consistent across all plots)
METHOD_COLORS = {
    'rashomon': 'dodgerblue',
    'blasso': 'seagreen',
    'ssl': 'purple',
    'ppmx': 'darkorange',
    'bootstrap': 'orangered',
    'lasso': 'indianred',
    'tva': 'mediumpurple',
}

# Method display names
METHOD_NAMES = {
    'rashomon': 'Rashomon Set',
    'blasso': 'Bayesian Lasso',
    'ssl': 'Spike-Slab Lasso',
    'ppmx': 'PPMx',
    'bootstrap': 'Bootstrap Lasso',
    'lasso': 'Lasso',
    'tva': 'TVA',
}

# Markers for line plots
METHOD_MARKERS = {
    'rashomon': 'd',   # diamond
    'blasso': 'o',     # circle
    'ssl': 's',        # square
    'ppmx': '*',       # star
    'bootstrap': '^',  # triangle
    'tva': 'p',        # pentagon
}

# Marker sizes
MARKER_SIZES = {
    'rashomon': 8,
    'blasso': 8,
    'ssl': 8,
    'ppmx': 10,  # Larger for star
    'bootstrap': 8,
    'tva': 8,
}
