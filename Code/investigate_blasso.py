"""
Investigate why Bayesian Lasso struggles with this problem.
"""

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
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

print("=" * 70)
print("PROBLEM DIAGNOSTICS")
print("=" * 70)
print()

print(f"Data dimensions:")
print(f"  n_samples: {num_data}")
print(f"  n_features (policies): {D_matrix.shape[1]}")
print(f"  Samples per policy: {n_per_pol}")
print()

# Check design matrix properties
print(f"Design matrix D properties:")
print(f"  Shape: {D_matrix.shape}")
print(f"  Rank: {np.linalg.matrix_rank(D_matrix)}")
print(f"  Condition number: {np.linalg.cond(D_matrix):.2e}")
print(f"  Is one-hot encoded: {np.allclose(D_matrix.sum(axis=1), 1.0)}")
print()

# Check XtX properties
XtX = D_matrix.T @ D_matrix
print(f"D'D matrix:")
print(f"  Shape: {XtX.shape}")
print(f"  Diagonal (first 5): {np.diag(XtX)[:5]}")
print(f"  Is diagonal: {np.allclose(XtX, np.diag(np.diag(XtX)))}")
print(f"  Off-diagonal max: {np.max(np.abs(XtX - np.diag(np.diag(XtX)))):.2e}")
print()

# Since D is one-hot, D'D should be diagonal with counts
print(f"D'D should be diagonal with n_per_pol={n_per_pol} on diagonal:")
print(f"  Expected: {n_per_pol}")
print(f"  Actual diagonal (unique): {np.unique(np.diag(XtX))}")
print()

# Check if problem is sparse
print(f"Outcome distribution by policy:")
policy_means = []
policy_stds = []
for idx in range(len(policies)):
    mask = (D == idx).ravel()
    policy_y = y[mask]
    policy_means.append(np.mean(policy_y))
    policy_stds.append(np.std(policy_y))

print(f"  Policy mean outcomes: min={np.min(policy_means):.2f}, max={np.max(policy_means):.2f}, range={np.ptp(policy_means):.2f}")
print(f"  Policy std outcomes: min={np.min(policy_stds):.2f}, max={np.max(policy_stds):.2f}")
print()

# Fit OLS to see what coefficients should look like
from sklearn.linear_model import LinearRegression
ols = LinearRegression(fit_intercept=False)
ols.fit(D_matrix, y)
print(f"OLS coefficients (should match policy means):")
print(f"  Range: [{np.min(ols.coef_):.3f}, {np.max(ols.coef_):.3f}]")
print(f"  Mean: {np.mean(ols.coef_):.3f}, Std: {np.std(ols.coef_):.3f}")
print(f"  Non-zero: {np.sum(np.abs(ols.coef_) > 1e-6)}/{len(ols.coef_)}")
print(f"  MSE: {mean_squared_error(y, ols.predict(D_matrix)):.4f}")
print()

# Compare with Lasso
lasso = linear_model.Lasso(1e-3, fit_intercept=False)
lasso.fit(D_matrix, y)
print(f"Lasso coefficients (alpha=1e-3):")
print(f"  Range: [{np.min(lasso.coef_):.3f}, {np.max(lasso.coef_):.3f}]")
print(f"  Mean: {np.mean(lasso.coef_):.3f}, Std: {np.std(lasso.coef_):.3f}")
print(f"  Non-zero: {np.sum(np.abs(lasso.coef_) > 1e-6)}/{len(lasso.coef_)}")
print(f"  MSE: {mean_squared_error(y, lasso.predict(D_matrix)):.4f}")
print()

# The issue: With one-hot encoding, the problem becomes:
# Each coefficient is just the mean of its group
# Bayesian Lasso puts strong priors that shrink toward zero
# This is inappropriate when we know each group should have distinct means

print("=" * 70)
print("WHY BAYESIAN LASSO FAILS")
print("=" * 70)
print()
print("1. Problem structure:")
print("   - D is one-hot encoded (each sample belongs to exactly one policy)")
print("   - D'D is diagonal (policies are orthogonal)")
print("   - Each coefficient β_j is simply the mean outcome for policy j")
print()
print("2. Bayesian Lasso prior:")
print("   - Puts Laplace prior on each β_j, centered at 0")
print("   - Shrinks coefficients toward zero")
print("   - Appropriate for sparse regression, NOT for group means")
print()
print("3. Why it doesn't converge:")
print("   - Data strongly suggests β_j ≈ group_mean_j (OLS solution)")
print("   - Prior strongly suggests β_j ≈ 0")
print("   - These are incompatible when group means are far from 0")
print("   - MCMC struggles between these two modes")
print()
print("4. Solution:")
print("   - Use a different prior (e.g., normal prior centered at data mean)")
print("   - Or use a hierarchical model that pools toward grand mean, not zero")
print("   - Or adjust lambda_prior to be very weak (large values)")
print()

# Test with very weak prior
print("Testing Bayesian Lasso with very weak prior (lambda=100)...")
from baselines import BayesianLasso

blasso_weak = BayesianLasso(
    n_iter=2000,
    burnin=500,
    thin=2,
    lambda_prior=100.0,  # Very weak regularization
    tau2_a=1.0,
    tau2_b=1.0,
    fit_intercept=False,
    random_state=42,
    verbose=False
)
blasso_weak.fit(D_matrix, y, n_chains=3)
y_pred_weak = blasso_weak.predict(D_matrix)
mse_weak = mean_squared_error(y, y_pred_weak)

print(f"  MSE: {mse_weak:.4f}")
print(f"  Converged: {blasso_weak.converged_}")
print(f"  Max R-hat: {np.max(blasso_weak.rhat_):.3f}")
print(f"  Coef range: [{np.min(blasso_weak.coef_):.3f}, {np.max(blasso_weak.coef_):.3f}]")
print()

if mse_weak < 2.0:  # Much better than before
    print("✓ Weak prior helps! With lambda=100, we get reasonable results.")
else:
    print("✗ Even weak prior doesn't help. The prior structure is fundamentally wrong.")
