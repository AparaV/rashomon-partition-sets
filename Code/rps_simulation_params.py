import numpy as np
from rashomon import hasse, counter


def create_simulation_params(M, R_vals):
    """
    Create simulation parameters for given M and R values
    Args:
        M: Number of features
        R_vals: Array of factor levels per feature (e.g., [4, 4, 4, 4])
    """
    
    # Generate all profiles
    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)
    
    # Initialize parameters for all profiles
    sigma = []
    mu = []
    var = []
    
    # Profile 0: (0, 0, 0, ..., 0) - baseline
    sigma_0 = None
    mu_0 = np.array([0])
    var_0 = np.array([1])
    
    sigma.append(sigma_0)
    mu.append(mu_0)
    var.append(var_0)
    
    h = 1  # Start with 1 pool for baseline
    
    # Configure all non-baseline profiles with all features active
    target_profile_idx = num_profiles - 1  # Profile (1, 1, 1, ..., 1)
    
    for i, profile in enumerate(profiles):
        if i == 0:  # Skip baseline
            continue
            
        if i == target_profile_idx:  # All features active profile
            m = np.sum(profile)
            sigma_i = np.ones(shape=(m, R_vals[0]-2))  # All splits active
            
            # Create heterogeneous effects
            num_pools = counter.num_pools(sigma_i)
            mu_i = np.random.uniform(2, 6, size=num_pools)  # Random effects
            var_i = np.ones(num_pools) * 1.0  # Fixed variance
            
        else:
            # Other profiles get simpler structure
            m = np.sum(profile)
            if m > 0:
                sigma_i = np.ones(shape=(m, R_vals[0]-2))
                mu_i = np.array([np.random.uniform(1, 3)])
                var_i = np.array([1.0])
            else:
                sigma_i = None
                mu_i = np.array([0])
                var_i = np.array([1])
        
        if sigma_i is not None:
            h += counter.num_pools(sigma_i)
        
        sigma.append(sigma_i)
        mu.append(mu_i)
        var.append(var_i)
    
    # Set H slightly above total pools needed
    H = h + 5
    
    return {
        'M': M,
        'R': R_vals,
        'sigma': sigma,
        'mu': mu,
        'var': var,
        'H': H,
        'profiles': profiles,
        'profile_map': profile_map,
        'target_profile_idx': target_profile_idx
    }


def get_profile_params(params, profile_idx):
    """
    Extract parameters for a specific profile
    """
    return {
        'sigma': params['sigma'][profile_idx],
        'mu': params['mu'][profile_idx],
        'var': params['var'][profile_idx],
        'profile': params['profiles'][profile_idx]
    }
