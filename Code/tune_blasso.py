"""
Tune Bayesian Lasso hyperparameters to achieve convergence.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from baselines import BayesianLasso
from rashomon import hasse
from rashomon.extract_pools import extract_pools

np.random.seed(42)

# Generate data
sigma = np.array([[1, 1, 1], [0, 0, 0]], dtype='float64')
sigma_profile = (1, 1)
M, _ = sigma.shape
R = np.array([5, 5])

profiles, profile_map = hasse.enumerate_profiles(M)
all_policies = hasse.enumerate_policies(M, R)
policies = [x for x in all_policies if hasse.policy_to_profile(x) == sigma_profile]
pi_pools, pi_policies = extract_pools(policies, sigma)

mu_pools = np.array([0, 1.5, 3, 4.5])
mu = np.zeros((len(policies), 1))
for idx, policy in enumerate(policies):
    pool_i = pi_policies[idx]
    mu[idx, 0] = mu_pools[pool_i]
var = 1

n_per_pol = 50
num_data = len(policies) * n_per_pol

X = np.ndarray(shape=(num_data, M))
D = np.ndarray(shape=(num_data, 1), dtype='int_')
y = np.ndarray(shape=(num_data, 1))

for idx, policy in enumerate(policies):
    y_i = np.random.normal(mu[idx], var, size=(n_per_pol, 1))
    start_idx = idx * n_per_pol
    end_idx = (idx + 1) * n_per_pol
    X[start_idx:end_idx, ] = policy
    D[start_idx:end_idx, ] = idx
    y[start_idx:end_idx, ] = y_i

G = hasse.alpha_matrix(policies)
D_matrix = hasse.get_dummy_matrix(D, G, len(policies))

print(f"Data: n={num_data}, p={D_matrix.shape[1]}, y_range=[{y.min():.2f}, {y.max():.2f}]")
print()

# Test configurations
configs = [
    # (n_iter, burnin, thin, n_chains, lambda_prior, tau2_a, tau2_b)
    (2000, 500, 2, 3, 5.0, 1e-1, 1e-1),  # Current best lambda, more iterations
    (3000, 1000, 2, 4, 5.0, 1e-1, 1e-1),  # Even more iterations
    (2000, 500, 2, 3, 10.0, 1e-1, 1e-1),  # Stronger regularization
    (2000, 500, 2, 3, 5.0, 1.0, 1.0),    # Different tau2 prior
    (2000, 500, 2, 3, 5.0, 1e-2, 1e-2),  # Tighter tau2 prior
    (3000, 1000, 3, 4, 10.0, 1.0, 1.0),  # Conservative combo
]

print("Testing configurations:")
print("-" * 100)
print(f"{'n_iter':>7} {'burnin':>7} {'thin':>5} {'chains':>7} {'lambda':>7} {'tau2_a':>7} {'tau2_b':>7} | {'MSE':>8} {'converged':>10} {'max_rhat':>9} {'time(s)':>8}")
print("-" * 100)

import time

best_mse = float('inf')
best_config = None
best_model = None

for config in configs:
    n_iter, burnin, thin, n_chains, lambda_p, tau2_a, tau2_b = config
    
    start = time.time()
    blasso = BayesianLasso(
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        lambda_prior=lambda_p,
        tau2_a=tau2_a,
        tau2_b=tau2_b,
        fit_intercept=False,
        random_state=42,
        verbose=False
    )
    
    blasso.fit(D_matrix, y, n_chains=n_chains)
    elapsed = time.time() - start
    
    y_pred = blasso.predict(D_matrix)
    mse = mean_squared_error(y, y_pred)
    
    print(f"{n_iter:7d} {burnin:7d} {thin:5d} {n_chains:7d} {lambda_p:7.1f} {tau2_a:7.2e} {tau2_b:7.2e} | "
          f"{mse:8.4f} {str(blasso.converged_):>10} {np.max(blasso.rhat_):9.3f} {elapsed:8.1f}")
    
    if blasso.converged_ and mse < best_mse:
        best_mse = mse
        best_config = config
        best_model = blasso

print("-" * 100)
print()

if best_model is not None:
    print("BEST CONFIGURATION (converged):")
    n_iter, burnin, thin, n_chains, lambda_p, tau2_a, tau2_b = best_config
    print(f"  n_iter: {n_iter}")
    print(f"  burnin: {burnin}")
    print(f"  thin: {thin}")
    print(f"  n_chains: {n_chains}")
    print(f"  lambda_prior: {lambda_p}")
    print(f"  tau2_a: {tau2_a}")
    print(f"  tau2_b: {tau2_b}")
    print(f"  MSE: {best_mse:.4f}")
    print(f"  Max R-hat: {np.max(best_model.rhat_):.3f}")
    print(f"  Coef range: [{np.min(best_model.coef_):.3f}, {np.max(best_model.coef_):.3f}]")
else:
    print("WARNING: No configuration converged!")
    print("Try:")
    print("  - Even more iterations (5000+)")
    print("  - More chains (5+)")
    print("  - Different priors")
    print("  - Check if problem is well-conditioned")
