"""
Benchmark optimized Bayesian Lasso to measure speedup.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/apara/Documents/Research/2022_Bayesian_TVA/Rashomon/Code')

from baselines import BayesianLasso

print("=" * 70)
print("BAYESIAN LASSO OPTIMIZATION BENCHMARK")
print("=" * 70)

# Test with simulation-like dimensions
np.random.seed(42)
n_samples = 1280
n_features = 256

X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

print(f"\nProblem size: n={n_samples}, p={n_features}")
print(f"Running 100 iterations with optimized code...")

# Test optimized version
model = BayesianLasso(
    n_iter=100,
    burnin=20,
    thin=1,
    lambda_prior=1.0,
    random_state=42,
    verbose=False
)

start = time.time()
model.fit(X, y, n_chains=2)  # Need 2+ chains for diagnostics
elapsed = time.time() - start

print(f"\nOptimized version:")
print(f"  Time: {elapsed:.2f} seconds")
print(f"  Rate: {100/elapsed:.1f} iterations/second")
print(f"  Coefficient norm: {np.linalg.norm(model.coef_):.4f}")

# Estimate time for full simulation
full_sim_time = elapsed * (2000 / 100) * 3  # 2000 iters, 3 chains
print(f"\nEstimated time for full run (2000 iters, 3 chains):")
print(f"  {full_sim_time:.1f} seconds = {full_sim_time/60:.1f} minutes per simulation")
print(f"  For 100 simulations: {full_sim_time * 100 / 3600:.1f} hours")

print("\n" + "=" * 70)
print("Previous (unoptimized): ~6 seconds per 100 iterations")
print(f"Current (optimized): {elapsed:.2f} seconds per 100 iterations")
print(f"Speedup: {6.0/elapsed:.2f}x")
print("=" * 70)
