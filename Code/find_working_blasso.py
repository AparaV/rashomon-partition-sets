"""
Systematically search for Bayesian Lasso hyperparameters that work.
Test with different sample sizes and hyperparameter combinations.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from baselines import BayesianLasso
from rashomon import hasse
from rashomon.extract_pools import extract_pools
import time

def generate_data(n_per_pol, seed=42):
    """Generate data with specified samples per policy."""
    np.random.seed(seed)
    
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
    
    return D_matrix, y

print("=" * 100)
print("TESTING BAYESIAN LASSO WITH DIFFERENT SAMPLE SIZES")
print("=" * 100)
print()

# Test different sample sizes
sample_sizes = [10, 25, 50, 100, 200]

print("First, let's see if sample size affects convergence with fixed hyperparameters:")
print("-" * 100)
print(f"{'n_per_pol':>10} {'n_total':>10} | {'MSE':>8} {'converged':>10} {'max_rhat':>9} {'mean_rhat':>10} {'coef_mean':>10} {'time(s)':>8}")
print("-" * 100)

for n_per_pol in sample_sizes:
    D_matrix, y = generate_data(n_per_pol)
    
    # Get baseline Lasso MSE
    lasso = Lasso(alpha=1e-3, fit_intercept=False)
    lasso.fit(D_matrix, y.ravel())
    lasso_mse = mean_squared_error(y, lasso.predict(D_matrix))
    
    # Test Bayesian Lasso
    start = time.time()
    blasso = BayesianLasso(
        n_iter=2000,
        burnin=500,
        thin=2,
        lambda_prior=5.0,
        tau2_a=1.0,
        tau2_b=1.0,
        fit_intercept=False,
        random_state=42,
        verbose=False
    )
    blasso.fit(D_matrix, y, n_chains=3)
    elapsed = time.time() - start
    
    y_pred = blasso.predict(D_matrix)
    blasso_mse = mean_squared_error(y, y_pred)
    
    print(f"{n_per_pol:10d} {len(y):10d} | {blasso_mse:8.4f} {str(blasso.converged_):>10} "
          f"{np.max(blasso.rhat_):9.3f} {np.mean(blasso.rhat_):10.3f} "
          f"{np.mean(blasso.coef_):10.4f} {elapsed:8.1f}   (Lasso MSE: {lasso_mse:.4f})")

print()
print()

# Now let's do a grid search with n_per_pol=100
print("=" * 100)
print("GRID SEARCH WITH n_per_pol=100 (n_total=1600)")
print("=" * 100)
print()

D_matrix, y = generate_data(100)
lasso = Lasso(alpha=1e-3, fit_intercept=False)
lasso.fit(D_matrix, y.ravel())
lasso_mse = mean_squared_error(y, lasso.predict(D_matrix))
print(f"Baseline Lasso MSE: {lasso_mse:.4f}")
print()

# Grid search parameters
configs = []

# Try different lambda values
for lam in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    configs.append((3000, 1000, 2, 4, lam, 1.0, 1.0))

# Try different tau2 priors
for tau2 in [0.01, 0.1, 1.0, 10.0]:
    configs.append((3000, 1000, 2, 4, 5.0, tau2, tau2))

# Try more iterations
for n_iter in [5000, 10000]:
    configs.append((n_iter, n_iter//4, 2, 4, 5.0, 1.0, 1.0))

# Try more chains
for n_chains in [5, 6]:
    configs.append((3000, 1000, 2, n_chains, 5.0, 1.0, 1.0))

print("Testing configurations:")
print("-" * 100)
print(f"{'n_iter':>7} {'burnin':>7} {'thin':>5} {'chains':>7} {'lambda':>7} {'tau2':>7} | "
      f"{'MSE':>8} {'conv':>5} {'max_rhat':>9} {'mean_rhat':>10} {'time(s)':>8}")
print("-" * 100)

best_config = None
best_mse = float('inf')
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
    
    try:
        blasso.fit(D_matrix, y, n_chains=n_chains)
        elapsed = time.time() - start
        
        y_pred = blasso.predict(D_matrix)
        mse = mean_squared_error(y, y_pred)
        
        print(f"{n_iter:7d} {burnin:7d} {thin:5d} {n_chains:7d} {lambda_p:7.1f} {tau2_a:7.2f} | "
              f"{mse:8.4f} {str(blasso.converged_)[0]:>5} {np.max(blasso.rhat_):9.3f} "
              f"{np.mean(blasso.rhat_):10.3f} {elapsed:8.1f}")
        
        if blasso.converged_ and mse < best_mse:
            best_mse = mse
            best_config = config
            best_model = blasso
    except Exception as e:
        print(f"{n_iter:7d} {burnin:7d} {thin:5d} {n_chains:7d} {lambda_p:7.1f} {tau2_a:7.2f} | ERROR: {str(e)[:40]}")

print("-" * 100)
print()

if best_model is not None:
    print("=" * 100)
    print("BEST CONFIGURATION FOUND (CONVERGED)")
    print("=" * 100)
    n_iter, burnin, thin, n_chains, lambda_p, tau2_a, tau2_b = best_config
    print(f"  n_iter: {n_iter}")
    print(f"  burnin: {burnin}")
    print(f"  thin: {thin}")
    print(f"  n_chains: {n_chains}")
    print(f"  lambda_prior: {lambda_p}")
    print(f"  tau2_a: {tau2_a}")
    print(f"  tau2_b: {tau2_b}")
    print()
    print(f"Performance:")
    print(f"  MSE: {best_mse:.4f} (vs Lasso: {lasso_mse:.4f})")
    print(f"  Ratio: {best_mse/lasso_mse:.2f}x")
    print(f"  Max R-hat: {np.max(best_model.rhat_):.3f}")
    print(f"  Mean R-hat: {np.mean(best_model.rhat_):.3f}")
    print(f"  Coef range: [{np.min(best_model.coef_):.3f}, {np.max(best_model.coef_):.3f}]")
    print(f"  Coef mean: {np.mean(best_model.coef_):.3f} ± {np.std(best_model.coef_):.3f}")
    print()
    print("✓ SUCCESS! Use these hyperparameters in worst_case_sims.py")
else:
    print("=" * 100)
    print("NO CONVERGED CONFIGURATION FOUND")
    print("=" * 100)
    print()
    
    # Find best non-converged
    print("Best non-converged results:")
    print()
    
    # Rerun a subset to find best non-converged
    test_configs = [
        (5000, 1000, 2, 5, 10.0, 1.0, 1.0),
        (10000, 2000, 2, 5, 10.0, 1.0, 1.0),
        (5000, 1000, 2, 5, 50.0, 1.0, 1.0),
    ]
    
    print("Testing more aggressive configurations:")
    print("-" * 100)
    for config in test_configs:
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
        
        print(f"Config: n_iter={n_iter}, burnin={burnin}, chains={n_chains}, lambda={lambda_p}")
        print(f"  MSE: {mse:.4f}, Converged: {blasso.converged_}, max R-hat: {np.max(blasso.rhat_):.3f}")
        print(f"  Time: {elapsed:.1f}s")
        print()
    
    print()
    print("CONCLUSION: Bayesian Lasso may not be suitable for this problem structure.")
    print("Consider using Bootstrap Lasso instead for uncertainty quantification.")
