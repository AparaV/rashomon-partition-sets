"""Quick test to verify Bayesian Lasso integration with simulation data."""

import numpy as np
import sys
sys.path.insert(0, '/Users/apara/Documents/Research/2022_Bayesian_TVA/Rashomon/Code')

from rashomon import hasse
import reff_4 as params

# Set up like in simulations.py
M = params.M
R = params.R
num_profiles = 2**M
profiles, profile_map = hasse.enumerate_profiles(M)
all_policies = hasse.enumerate_policies(M, R)
num_policies = len(all_policies)

print(f"M (arms): {M}")
print(f"R (levels per arm): {R}")
print(f"Number of policies: {num_policies}")
print(f"Number of profiles: {num_profiles}")

# Test with small data
n_per_pol = 10
num_data = num_policies * n_per_pol
print(f"\nData dimensions:")
print(f"  n_samples: {num_data}")
print(f"  n_features (policies): {num_policies}")

# Create dummy matrix
G = hasse.alpha_matrix(all_policies)
print(f"  Alpha matrix G shape: {G.shape}")

# This is a challenging problem: n=810, p=81
# With default params, this will be slow
# Let's test with reduced iterations

from baselines import BayesianLasso

print("\nTesting Bayesian Lasso with reduced iterations...")
X_test = np.random.randn(num_data, num_policies)
y_test = np.random.randn(num_data)

# Test with very small number of iterations
model = BayesianLasso(
    n_iter=100,  # Very small for testing
    burnin=20,
    thin=2,
    lambda_prior=1.0,
    random_state=42,
    verbose=True
)

import time
start = time.time()
model.fit(X_test, y_test, n_chains=2)  # Only 2 chains for speed
elapsed = time.time() - start

print(f"\nTime elapsed: {elapsed:.2f} seconds")
print(f"Converged: {model.converged_}")
print(f"Max R-hat: {np.max(model.rhat_):.4f}")
print(f"\nEstimate for full run (n_iter=2000, n_chains=4):")
print(f"  ~{elapsed * 20 * 2:.1f} seconds per iteration")
