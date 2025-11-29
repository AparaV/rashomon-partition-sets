"""
Diagnostic script to understand why Bayesian Lasso has high MSE.
Compares coefficients, predictions, and explores hyperparameter sensitivity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from rashomon import hasse, loss, metrics
from rashomon.extract_pools import extract_pools
from baselines import BayesianLasso, BootstrapLasso

# Set random seed
np.random.seed(42)

#
# Generate same data as worst_case_sims.py
#
sigma = np.array([[1, 1, 1],
                  [0, 0, 0]], dtype='float64')
sigma_profile = (1, 1)
M, _ = sigma.shape
R = np.array([5, 5])

# Enumerate policies and find pools
num_policies = np.prod(R-1)
profiles, profile_map = hasse.enumerate_profiles(M)
all_policies = hasse.enumerate_policies(M, R)
policies = [x for x in all_policies if hasse.policy_to_profile(x) == sigma_profile]
pi_pools, pi_policies = extract_pools(policies, sigma)

# Ground truth
mu_pools = np.array([0, 1.5, 3, 4.5])
mu = np.zeros((num_policies, 1))
for idx, policy in enumerate(policies):
    pool_i = pi_policies[idx]
    mu[idx, 0] = mu_pools[pool_i]
var = 1

print(f"Problem dimensions:")
print(f"  Number of policies: {num_policies}")
print(f"  M (features): {M}")
print(f"  True mu range: [{np.min(mu):.2f}, {np.max(mu):.2f}]")
print()

# Generate data with different sample sizes
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

# Create design matrix
G = hasse.alpha_matrix(policies)
D_matrix = hasse.get_dummy_matrix(D, G, num_policies)

print(f"Data dimensions:")
print(f"  n_samples: {num_data}")
print(f"  X shape: {X.shape}")
print(f"  D_matrix shape: {D_matrix.shape}")
print(f"  y range: [{np.min(y):.2f}, {np.max(y):.2f}]")
print(f"  y mean: {np.mean(y):.2f}, std: {np.std(y):.2f}")
print()

#
# Compare different methods
#

print("="*70)
print("COMPARING METHODS")
print("="*70)
print()

# 1. Regular Lasso
print("1. Regular Lasso (reg=1e-3)")
lasso = linear_model.Lasso(1e-3, fit_intercept=False)
lasso.fit(D_matrix, y)
y_lasso = lasso.predict(D_matrix)
mse_lasso = mean_squared_error(y, y_lasso)
print(f"   MSE: {mse_lasso:.4f}")
print(f"   Prediction range: [{np.min(y_lasso):.2f}, {np.max(y_lasso):.2f}]")
print(f"   Coef range: [{np.min(lasso.coef_):.4f}, {np.max(lasso.coef_):.4f}]")
print(f"   Coef norm: {np.linalg.norm(lasso.coef_):.4f}")
print(f"   Non-zero coefs: {np.sum(np.abs(lasso.coef_) > 1e-6)}/{len(lasso.coef_)}")
print()

# 2. Bayesian Lasso with different lambda values
lambda_values = [1e-3, 1e-2, 1e-1, 5e-1, 1.0, 2.0, 5.0]
print("2. Bayesian Lasso with different lambda values:")

blasso_results = []
for lam in lambda_values:
    blasso = BayesianLasso(
        n_iter=500,
        burnin=100,
        thin=2,
        lambda_prior=lam,
        tau2_a=1e-1,
        tau2_b=1e-1,
        fit_intercept=False,
        random_state=42,
        verbose=False
    )
    blasso.fit(D_matrix, y, n_chains=2)
    y_blasso = blasso.predict(D_matrix)
    mse_blasso = mean_squared_error(y, y_blasso)
    
    print(f"   lambda={lam:6.3f}: MSE={mse_blasso:.4f}, "
          f"pred_range=[{np.min(y_blasso):6.2f}, {np.max(y_blasso):6.2f}], "
          f"coef_norm={np.linalg.norm(blasso.coef_):6.4f}, "
          f"converged={blasso.converged_}, max_rhat={np.max(blasso.rhat_):.3f}")
    
    blasso_results.append({
        'lambda': lam,
        'mse': mse_blasso,
        'pred_min': np.min(y_blasso),
        'pred_max': np.max(y_blasso),
        'coef_norm': np.linalg.norm(blasso.coef_),
        'converged': blasso.converged_,
        'max_rhat': np.max(blasso.rhat_),
        'coef': blasso.coef_.copy()
    })

print()

# Find best lambda
best_idx = np.argmin([r['mse'] for r in blasso_results])
best_lambda = blasso_results[best_idx]['lambda']
best_mse = blasso_results[best_idx]['mse']
print(f"Best lambda: {best_lambda} with MSE: {best_mse:.4f}")
print(f"Regular Lasso MSE: {mse_lasso:.4f}")
print(f"Ratio (blasso/lasso): {best_mse/mse_lasso:.2f}x")
print()

# 3. Bootstrap Lasso with different alpha values
print("3. Bootstrap Lasso with different alpha values:")
alpha_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

bootstrap_results = []
for alpha in alpha_values:
    bootstrap = BootstrapLasso(
        n_bootstrap=500,
        alpha=alpha,
        confidence_level=0.95,
        fit_intercept=False,
        random_state=42,
        verbose=False
    )
    bootstrap.fit(D_matrix, y)
    y_bootstrap = bootstrap.predict(D_matrix)
    mse_bootstrap = mean_squared_error(y, y_bootstrap)
    
    print(f"   alpha={alpha:6.4f}: MSE={mse_bootstrap:.4f}, "
          f"pred_range=[{np.min(y_bootstrap):6.2f}, {np.max(y_bootstrap):6.2f}], "
          f"coef_norm={np.linalg.norm(bootstrap.coef_):6.4f}")
    
    bootstrap_results.append({
        'alpha': alpha,
        'mse': mse_bootstrap,
        'pred_min': np.min(y_bootstrap),
        'pred_max': np.max(y_bootstrap),
        'coef_norm': np.linalg.norm(bootstrap.coef_),
        'coef': bootstrap.coef_.copy()
    })

print()

# Find best alpha
best_boot_idx = np.argmin([r['mse'] for r in bootstrap_results])
best_alpha = bootstrap_results[best_boot_idx]['alpha']
best_boot_mse = bootstrap_results[best_boot_idx]['mse']
print(f"Best alpha: {best_alpha} with MSE: {best_boot_mse:.4f}")
print(f"Ratio (bootstrap/lasso): {best_boot_mse/mse_lasso:.2f}x")
print()

#
# Detailed comparison of best models
#
print("="*70)
print("DETAILED COMPARISON")
print("="*70)
print()

# Refit best models
blasso_best = BayesianLasso(
    n_iter=1000,
    burnin=200,
    thin=2,
    lambda_prior=best_lambda,
    tau2_a=1e-1,
    tau2_b=1e-1,
    fit_intercept=False,
    random_state=42,
    verbose=False
)
blasso_best.fit(D_matrix, y, n_chains=3)
y_blasso_best = blasso_best.predict(D_matrix)

bootstrap_best = BootstrapLasso(
    n_bootstrap=1000,
    alpha=best_alpha,
    confidence_level=0.95,
    fit_intercept=False,
    random_state=42,
    verbose=False
)
bootstrap_best.fit(D_matrix, y)
y_bootstrap_best = bootstrap_best.predict(D_matrix)

print(f"Lasso MSE: {mse_lasso:.4f}")
print(f"Bayesian Lasso MSE (lambda={best_lambda}): {mean_squared_error(y, y_blasso_best):.4f}")
print(f"Bootstrap Lasso MSE (alpha={best_alpha}): {mean_squared_error(y, y_bootstrap_best):.4f}")
print()

# Compare coefficients
print("Coefficient comparison:")
print(f"  Lasso - mean: {np.mean(lasso.coef_):.4f}, std: {np.std(lasso.coef_):.4f}")
print(f"  BLasso - mean: {np.mean(blasso_best.coef_):.4f}, std: {np.std(blasso_best.coef_):.4f}")
print(f"  Bootstrap - mean: {np.mean(bootstrap_best.coef_):.4f}, std: {np.std(bootstrap_best.coef_):.4f}")
print()

# Check convergence
print(f"Bayesian Lasso diagnostics:")
print(f"  Converged: {blasso_best.converged_}")
print(f"  Max R-hat: {np.max(blasso_best.rhat_):.3f}")
print(f"  Mean R-hat: {np.mean(blasso_best.rhat_):.3f}")
print(f"  R-hat > 1.1: {np.sum(blasso_best.rhat_ > 1.1)} / {len(blasso_best.rhat_)}")
print()

#
# Visualizations
#
print("Creating diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: MSE vs lambda for Bayesian Lasso
ax = axes[0, 0]
lambdas = [r['lambda'] for r in blasso_results]
mses = [r['mse'] for r in blasso_results]
ax.semilogx(lambdas, mses, 'o-', color='blue', label='Bayesian Lasso')
ax.axhline(mse_lasso, color='red', linestyle='--', label='Regular Lasso')
ax.set_xlabel('Lambda (prior regularization)')
ax.set_ylabel('MSE')
ax.set_title('Bayesian Lasso: MSE vs Lambda')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: MSE vs alpha for Bootstrap
ax = axes[0, 1]
alphas = [r['alpha'] for r in bootstrap_results]
boot_mses = [r['mse'] for r in bootstrap_results]
ax.semilogx(alphas, boot_mses, 'o-', color='green', label='Bootstrap Lasso')
ax.axhline(mse_lasso, color='red', linestyle='--', label='Regular Lasso')
ax.set_xlabel('Alpha (Lasso regularization)')
ax.set_ylabel('MSE')
ax.set_title('Bootstrap Lasso: MSE vs Alpha')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Coefficient comparison
ax = axes[0, 2]
coef_indices = np.arange(len(lasso.coef_))
ax.scatter(coef_indices, lasso.coef_, alpha=0.6, label='Lasso', s=20)
ax.scatter(coef_indices, blasso_best.coef_, alpha=0.6, label='Bayesian Lasso', s=20)
ax.scatter(coef_indices, bootstrap_best.coef_, alpha=0.6, label='Bootstrap', s=20)
ax.set_xlabel('Coefficient index')
ax.set_ylabel('Coefficient value')
ax.set_title('Coefficient Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Predictions vs True
ax = axes[1, 0]
ax.scatter(y, y_lasso, alpha=0.3, s=10, label='Lasso')
ax.scatter(y, y_blasso_best, alpha=0.3, s=10, label='Bayesian Lasso')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('True y')
ax.set_ylabel('Predicted y')
ax.set_title('Predictions vs True Values')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Residuals
ax = axes[1, 1]
residuals_lasso = y.ravel() - y_lasso.ravel()
residuals_blasso = y.ravel() - y_blasso_best.ravel()
ax.hist(residuals_lasso, bins=30, alpha=0.5, label=f'Lasso (std={np.std(residuals_lasso):.2f})')
ax.hist(residuals_blasso, bins=30, alpha=0.5, label=f'BLasso (std={np.std(residuals_blasso):.2f})')
ax.set_xlabel('Residual')
ax.set_ylabel('Count')
ax.set_title('Residual Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: R-hat distribution for Bayesian Lasso
ax = axes[1, 2]
ax.hist(blasso_best.rhat_, bins=30, edgecolor='black')
ax.axvline(1.1, color='red', linestyle='--', label='Threshold (1.1)')
ax.set_xlabel('R-hat')
ax.set_ylabel('Count')
ax.set_title(f'Bayesian Lasso R-hat Distribution\n(Converged: {blasso_best.converged_})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../Figures/worst_case/blasso_diagnostics.png', dpi=300, bbox_inches='tight')
print("Saved diagnostic plots to ../Figures/worst_case/blasso_diagnostics.png")
plt.show()

#
# Recommendations
#
print()
print("="*70)
print("RECOMMENDATIONS")
print("="*70)
print()
print(f"Based on this analysis:")
print(f"1. For Bayesian Lasso, use lambda={best_lambda} (current: 0.5)")
print(f"   This gives MSE={best_mse:.4f} vs current MSE with lambda=0.5")
print(f"2. For Bootstrap Lasso, use alpha={best_alpha} (current: 0.001)")
print(f"   This gives MSE={best_boot_mse:.4f}")
print(f"3. Regular Lasso MSE: {mse_lasso:.4f}")
print()
print(f"MSE Ratios:")
print(f"  Best Bayesian Lasso / Regular Lasso: {best_mse/mse_lasso:.2f}x")
print(f"  Best Bootstrap Lasso / Regular Lasso: {best_boot_mse/mse_lasso:.2f}x")
print()

if not blasso_best.converged_:
    print("WARNING: Bayesian Lasso did not converge!")
    print("Consider:")
    print("  - Increasing n_iter (current: 1000)")
    print("  - Increasing burnin (current: 200)")
    print("  - Using more chains (current: 3)")
    print()
