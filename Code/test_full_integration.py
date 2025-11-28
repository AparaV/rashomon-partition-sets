"""
End-to-end test of Bayesian Lasso integration with simulation pipeline.
Uses minimal iterations to verify the full workflow.
"""

import numpy as np
import sys
import os
sys.path.insert(0, '/Users/apara/Documents/Research/2022_Bayesian_TVA/Rashomon/Code')

from copy import deepcopy
from sklearn import linear_model
from rashomon import hasse, loss, metrics, extract_pools
from rashomon.aggregate import RAggregate
from baselines import BayesianLasso
import reff_4 as params

print("=" * 70)
print("BAYESIAN LASSO INTEGRATION TEST")
print("=" * 70)

# Setup (from simulations.py)
M = params.M
R = params.R
sigma = params.sigma
mu = params.mu
var = params.var
blasso_lambda = params.blasso_lambda

num_profiles = 2**M
profiles, profile_map = hasse.enumerate_profiles(M)
all_policies = hasse.enumerate_policies(M, R)
num_policies = len(all_policies)

print(f"\nProblem dimensions:")
print(f"  M (arms): {M}")
print(f"  Policies: {num_policies}")
print(f"  Profiles: {num_profiles}")

# Identify pools (from simulations.py)
policies_profiles = {}
policies_profiles_masked = {}
policies_ids_profiles = {}
pi_policies = {}
pi_pools = {}

for k, profile in enumerate(profiles):
    policies_temp = [(i, x) for i, x in enumerate(all_policies) if hasse.policy_to_profile(x) == profile]
    unzipped_temp = list(zip(*policies_temp))
    policies_ids_k = list(unzipped_temp[0])
    policies_k = list(unzipped_temp[1])
    policies_profiles[k] = deepcopy(policies_k)
    policies_ids_profiles[k] = policies_ids_k
    
    profile_mask = list(map(bool, profile))
    for idx, pol in enumerate(policies_k):
        policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
    policies_profiles_masked[k] = policies_k
    
    if np.sum(profile) > 0:
        pi_pools_k, pi_policies_k = extract_pools.extract_pools(policies_k, sigma[k])
        pi_policies[k] = pi_policies_k
        pi_pools[k] = {}
        for x, y in pi_pools_k.items():
            y_full = [policies_profiles[k][i] for i in y]
            y_agg = [all_policies.index(i) for i in y_full]
            pi_pools[k][x] = y_agg
    else:
        pi_policies[k] = {0: 0}
        pi_pools[k] = {0: [0]}

# Ground truth
best_per_profile = [np.max(mu_k) for mu_k in mu]
true_best_profile = np.argmax(best_per_profile)
true_best_profile_idx = int(true_best_profile)
true_best_effect = np.max(mu[true_best_profile])
true_best = pi_pools[true_best_profile][np.argmax(mu[true_best_profile])]
min_dosage_best_policy = metrics.find_min_dosage(true_best, all_policies)

print(f"  True best profile: {true_best_profile} ({profiles[true_best_profile]})")
print(f"  True best effect: {true_best_effect:.2f}")

# Generate data for ONE simulation
n_per_pol = 5
np.random.seed(42)

def generate_data(mu, var, n_per_pol, all_policies, pi_policies, M):
    num_data = num_policies * n_per_pol
    X = np.zeros(shape=(num_data, M))
    D = np.zeros(shape=(num_data, 1), dtype='int_')
    y = np.zeros(shape=(num_data, 1))
    
    idx_ctr = 0
    for k, profile in enumerate(profiles):
        policies_k = policies_profiles[k]
        for idx, policy in enumerate(policies_k):
            policy_idx = [i for i, x in enumerate(all_policies) if x == policy]
            pool_id = pi_policies[k][idx]
            mu_i = mu[k][pool_id]
            var_i = var[k][pool_id]
            y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))
            
            start_idx = idx_ctr * n_per_pol
            end_idx = (idx_ctr + 1) * n_per_pol
            X[start_idx:end_idx, ] = policy
            D[start_idx:end_idx, ] = policy_idx[0]
            y[start_idx:end_idx, ] = y_i
            idx_ctr += 1
    
    return X, D, y

print(f"\nGenerating data with n_per_pol={n_per_pol}...")
X, D, y = generate_data(mu, var, n_per_pol, all_policies, pi_policies, M)
G = hasse.alpha_matrix(all_policies)
D_matrix = hasse.get_dummy_matrix(D, G, num_policies)

print(f"  Data shape: X={X.shape}, D={D.shape}, y={y.shape}")
print(f"  Design matrix shape: {D_matrix.shape}")

# Run Bayesian Lasso with minimal iterations
print(f"\nRunning Bayesian Lasso (minimal iterations for testing)...")
import time
start = time.time()

blasso = BayesianLasso(
    n_iter=200,  # Minimal for testing
    burnin=50,
    thin=2,
    lambda_prior=blasso_lambda,
    random_state=42,
    verbose=True
)
blasso.fit(D_matrix, y, n_chains=2)
elapsed = time.time() - start

print(f"\n  Time elapsed: {elapsed:.1f} seconds")
print(f"  Converged: {blasso.converged_}")
print(f"  Max R-hat: {np.max(blasso.rhat_):.4f}")

# Make predictions
y_blasso = blasso.predict(D_matrix)

# Compute metrics
print(f"\nComputing metrics...")
blasso_results = metrics.compute_all_metrics(
    y, y_blasso, D, true_best, all_policies, profile_map,
    min_dosage_best_policy, true_best_effect
)

print(f"  MSE: {blasso_results['sqrd_err']:.4f}")
print(f"  IOU: {blasso_results['iou']:.4f}")
print(f"  Best policy diff: {blasso_results['best_pol_diff']:.4f}")
print(f"  Min dosage included: {blasso_results['min_dos_inc']}")
print(f"  Best profile identified: {blasso_results['best_prof'][true_best_profile_idx]}")

print("\n" + "=" * 70)
print("INTEGRATION TEST SUCCESSFUL!")
print("=" * 70)
print(f"\nEstimated time for full simulation (n_per_pol=30, iters=100):")
print(f"  ~{elapsed * 100 * 6:.0f} seconds = {elapsed * 100 * 6 / 60:.1f} minutes")
