"""
Focus on n_per_pol=200 where we saw promising results.
Fine-tune hyperparameters to achieve full convergence.
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
print("FINE-TUNING BAYESIAN LASSO WITH n_per_pol=200 (n=3200)")
print("=" * 100)
print()

D_matrix, y = generate_data(200)

lasso = Lasso(alpha=1e-3, fit_intercept=False)
lasso.fit(D_matrix, y.ravel())
lasso_mse = mean_squared_error(y, lasso.predict(D_matrix))
print(f"Baseline Lasso MSE: {lasso_mse:.4f}")
print(f"Data: n={len(y)}, p={D_matrix.shape[1]}")
print()

# Promising configurations based on initial results
configs = [
    # (n_iter, burnin, thin, n_chains, lambda_prior, tau2_a, tau2_b, description)
    (2000, 500, 2, 3, 5.0, 1.0, 1.0, "baseline"),
    (3000, 1000, 2, 4, 5.0, 1.0, 1.0, "more iterations + chains"),
    (5000, 1500, 2, 4, 5.0, 1.0, 1.0, "even more iterations"),
    (10000, 3000, 2, 5, 5.0, 1.0, 1.0, "aggressive iterations + chains"),
    (5000, 1500, 3, 4, 5.0, 1.0, 1.0, "more thinning"),
    (3000, 1000, 2, 4, 10.0, 1.0, 1.0, "stronger regularization"),
    (3000, 1000, 2, 4, 1.0, 1.0, 1.0, "weaker regularization"),
    (3000, 1000, 2, 4, 5.0, 0.1, 0.1, "tighter tau2 prior"),
    (3000, 1000, 2, 4, 5.0, 10.0, 10.0, "looser tau2 prior"),
]

print("Testing configurations:")
print("-" * 100)
print(f"{'Description':>30} | {'n_iter':>7} {'burnin':>7} {'chains':>7} | "
      f"{'MSE':>8} {'conv':>5} {'max_rhat':>9} {'time(s)':>8}")
print("-" * 100)

results = []

for config in configs:
    n_iter, burnin, thin, n_chains, lambda_p, tau2_a, tau2_b, desc = config
    
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
    
    results.append({
        'config': config,
        'mse': mse,
        'converged': blasso.converged_,
        'max_rhat': np.max(blasso.rhat_),
        'mean_rhat': np.mean(blasso.rhat_),
        'coef_mean': np.mean(blasso.coef_),
        'coef_std': np.std(blasso.coef_),
        'time': elapsed,
        'model': blasso
    })
    
    print(f"{desc:>30} | {n_iter:7d} {burnin:7d} {n_chains:7d} | "
          f"{mse:8.4f} {str(blasso.converged_)[0]:>5} {np.max(blasso.rhat_):9.3f} {elapsed:8.1f}")

print("-" * 100)
print()

# Find best converged
converged_results = [r for r in results if r['converged']]

if converged_results:
    best = min(converged_results, key=lambda x: x['mse'])
    n_iter, burnin, thin, n_chains, lambda_p, tau2_a, tau2_b, desc = best['config']
    
    print("=" * 100)
    print("✓ SUCCESS! FOUND CONVERGED CONFIGURATION")
    print("=" * 100)
    print()
    print(f"Description: {desc}")
    print(f"Hyperparameters:")
    print(f"  n_iter: {n_iter}")
    print(f"  burnin: {burnin}")
    print(f"  thin: {thin}")
    print(f"  n_chains: {n_chains}")
    print(f"  lambda_prior: {lambda_p}")
    print(f"  tau2_a: {tau2_a}")
    print(f"  tau2_b: {tau2_b}")
    print()
    print(f"Performance:")
    print(f"  MSE: {best['mse']:.4f}")
    print(f"  Lasso MSE: {lasso_mse:.4f}")
    print(f"  Ratio: {best['mse']/lasso_mse:.2f}x")
    print(f"  Max R-hat: {best['max_rhat']:.3f}")
    print(f"  Mean R-hat: {best['mean_rhat']:.3f}")
    print(f"  Coef: {best['coef_mean']:.3f} ± {best['coef_std']:.3f}")
    print(f"  Runtime: {best['time']:.1f}s")
    print()
    print("=" * 100)
    print("RECOMMENDATION FOR worst_case_sims.py")
    print("=" * 100)
    print()
    print(f"With n_per_pol >= 200:")
    print(f"  bayesian_n_iter = {n_iter}")
    print(f"  bayesian_burnin = {burnin}")
    print(f"  bayesian_thin = {thin}")
    print(f"  bayesian_n_chains = {n_chains}")
    print(f"  bayesian_lambda_prior = {lambda_p}")
    print(f"  bayesian_tau2_a = {tau2_a}")
    print(f"  bayesian_tau2_b = {tau2_b}")
    print()
else:
    print("=" * 100)
    print("NO FULLY CONVERGED CONFIGURATION")
    print("=" * 100)
    print()
    
    # Find best by MSE
    best = min(results, key=lambda x: x['mse'])
    n_iter, burnin, thin, n_chains, lambda_p, tau2_a, tau2_b, desc = best['config']
    
    print(f"Best MSE (not converged): {desc}")
    print(f"  MSE: {best['mse']:.4f} (Lasso: {lasso_mse:.4f})")
    print(f"  Max R-hat: {best['max_rhat']:.3f}")
    print(f"  Mean R-hat: {best['mean_rhat']:.3f}")
    print()
    
    # Find best by convergence
    best_conv = min(results, key=lambda x: x['max_rhat'])
    n_iter, burnin, thin, n_chains, lambda_p, tau2_a, tau2_b, desc = best_conv['config']
    
    print(f"Best convergence (not converged): {desc}")
    print(f"  MSE: {best_conv['mse']:.4f}")
    print(f"  Max R-hat: {best_conv['max_rhat']:.3f}")
    print(f"  Mean R-hat: {best_conv['mean_rhat']:.3f}")
    print()
    
    # Test with even larger sample size
    print("=" * 100)
    print("TESTING WITH EVEN LARGER SAMPLE: n_per_pol=300 (n=4800)")
    print("=" * 100)
    print()
    
    D_matrix_large, y_large = generate_data(300)
    lasso_large = Lasso(alpha=1e-3, fit_intercept=False)
    lasso_large.fit(D_matrix_large, y_large.ravel())
    lasso_mse_large = mean_squared_error(y_large, lasso_large.predict(D_matrix_large))
    
    print(f"Lasso MSE: {lasso_mse_large:.4f}")
    print()
    
    # Test best config from before
    n_iter, burnin, thin, n_chains, lambda_p, tau2_a, tau2_b, desc = best['config']
    
    start = time.time()
    blasso_large = BayesianLasso(
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
    blasso_large.fit(D_matrix_large, y_large, n_chains=n_chains)
    elapsed = time.time() - start
    
    y_pred_large = blasso_large.predict(D_matrix_large)
    mse_large = mean_squared_error(y_large, y_pred_large)
    
    print(f"Bayesian Lasso with n=4800:")
    print(f"  MSE: {mse_large:.4f}")
    print(f"  Converged: {blasso_large.converged_}")
    print(f"  Max R-hat: {np.max(blasso_large.rhat_):.3f}")
    print(f"  Mean R-hat: {np.mean(blasso_large.rhat_):.3f}")
    print(f"  Time: {elapsed:.1f}s")
    
    if blasso_large.converged_:
        print()
        print("✓ CONVERGED WITH LARGER SAMPLE!")
        print(f"  Use n_per_pol >= 300 in worst_case_sims.py")
