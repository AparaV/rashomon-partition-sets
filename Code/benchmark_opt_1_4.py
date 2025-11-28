"""
Benchmark optimizations 1-4 for Bayesian Lasso.

Optimizations:
1. Vectorized lambda sampling (chi-squared relationship)
2. Avoid residual recomputation (pass from beta to tau2)
3. Reuse workspace matrix (pre-allocate, use np.copyto)
4. Remove unused X parameter (code cleanup)
"""

import numpy as np
import time
from baselines import BayesianLasso

# Set random seed for reproducibility
np.random.seed(42)

# Problem dimensions matching simulation
n = 1280  # Half of 2560 for faster testing
p = 256

print(f"Problem size: n={n}, p={p}")
print("="*60)

# Generate synthetic data
X = np.random.randn(n, p)
true_beta = np.random.randn(p) * 0.1
y = X @ true_beta + np.random.randn(n) * 0.5

# Test parameters
n_iter = 200  # Short run for timing
burnin = 100
thin = 2
n_chains = 2  # Minimum for Gelman-Rubin

print(f"MCMC parameters: {n_iter} iterations, {burnin} burnin, thin={thin}, {n_chains} chains")
print()

# Benchmark optimized version
print("Benchmarking OPTIMIZED version (with optimizations 1-4)...")
model = BayesianLasso(
    n_iter=n_iter,
    burnin=burnin,
    thin=thin,
    lambda_prior=0.05,
    random_state=42,
    verbose=False
)

start_time = time.time()
model.fit(X, y, n_chains=n_chains)
elapsed = time.time() - start_time

print(f"Time: {elapsed:.2f} seconds")
print(f"Time per 100 iterations: {elapsed / (n_iter / 100):.2f} seconds")
print(f"Coefficient norm: {np.linalg.norm(model.coef_):.4f}")
print(f"Max R-hat: {np.max(model.rhat_):.4f}")
print(f"Converged: {model.converged_}")
print()

# Estimate time for full simulation
full_iter = 2000
full_chains = 3
estimated_time = elapsed * (full_iter / n_iter) * (full_chains / n_chains)
print(f"Estimated time for full simulation ({full_iter} iters, {full_chains} chains):")
print(f"  {estimated_time:.1f} seconds = {estimated_time/60:.2f} minutes")
print()

# Quick correctness check
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
print(f"Training MSE: {mse:.4f}")
print()

print("="*60)
print("Summary of optimizations implemented:")
print("1. ✓ Vectorized lambda sampling (chi-squared method)")
print("2. ✓ Avoid residual recomputation (compute once per iteration)")
print("3. ✓ Reuse workspace matrix (pre-allocated A_workspace)")
print("4. ✓ Remove unused X parameter from _sample_beta")
print()
print(f"Expected combined speedup: 20-30% on top of previous 2.11x")
print(f"Previous: ~2.84s per 100 iterations")
print(f"Target: ~2.3s per 100 iterations (15% reduction)")
