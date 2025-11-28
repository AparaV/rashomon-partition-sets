"""
Compare performance before and after optimizations 1-4.
"""

import numpy as np
import time
from baselines import BayesianLasso

np.random.seed(42)

# Problem dimensions
n = 1280
p = 256

# Generate data
X = np.random.randn(n, p)
true_beta = np.random.randn(p) * 0.1
y = X @ true_beta + np.random.randn(n) * 0.5

# Parameters
n_iter = 200
burnin = 100
thin = 2
n_chains = 2

print("="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"Problem: n={n}, p={p}")
print(f"MCMC: {n_iter} iterations, {burnin} burnin, {n_chains} chains")
print()

# Run multiple trials
n_trials = 3
times = []

for trial in range(n_trials):
    model = BayesianLasso(
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        lambda_prior=0.05,
        random_state=42 + trial,
        verbose=False
    )
    
    start = time.time()
    model.fit(X, y, n_chains=n_chains)
    elapsed = time.time() - start
    times.append(elapsed)
    
    print(f"Trial {trial+1}: {elapsed:.2f}s ({elapsed / (n_iter/100):.2f}s per 100 iters)")

avg_time = np.mean(times)
std_time = np.std(times)
time_per_100 = avg_time / (n_iter / 100)

print()
print(f"Average: {avg_time:.2f} ± {std_time:.2f}s")
print(f"Per 100 iterations: {time_per_100:.2f}s")
print()

# Compare with previous benchmarks
print("="*70)
print("SPEEDUP ANALYSIS")
print("="*70)
print("Before optimization 1 (original):          ~6.00s per 100 iters")
print("After optimization 1&2 (previous):         ~2.84s per 100 iters (2.11x)")
print(f"After optimizations 1-4 (current):         ~{time_per_100:.2f}s per 100 iters", end="")

speedup_from_previous = 2.84 / time_per_100
total_speedup = 6.00 / time_per_100
improvement = (2.84 - time_per_100) / 2.84 * 100

print(f" ({speedup_from_previous:.2f}x from previous, {total_speedup:.2f}x total)")
print(f"Additional improvement: {improvement:.1f}%")
print()

# Estimate for full simulation
full_time = time_per_100 * (2000 / 100) * (3 / n_chains)
print(f"Estimated time per simulation (2000 iters, 3 chains): {full_time/60:.2f} minutes")
print(f"Estimated time for 100 simulations: {full_time*100/3600:.2f} hours")
