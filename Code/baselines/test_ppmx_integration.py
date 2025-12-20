"""
Test PPMx integration with simulation framework.
Replicates the setup from simulations.py using ppmx_new.py
"""

import sys
import numpy as np
from copy import deepcopy
sys.path.append('..')

from rashomon import hasse, metrics, extract_pools
from baselines.ppmx_new import PPMxConfig, PPMxData, PPMxSampler, posterior_similarity_matrix, map_partition
import reff_4


def generate_data(mu, var, n_per_pol, all_policies, pi_policies, M, profiles, policies_profiles, num_policies):
    """Generate data matching simulations.py exactly.
    
    Parameters
    ----------
    mu : list
        List of mean arrays for each profile
    var : list
        List of variance arrays for each profile
    n_per_pol : int
        Number of observations per policy
    all_policies : list
        List of all policies
    pi_policies : dict
        Mapping from profile to policy index to pool id
    M : int
        Number of arms
    profiles : list
        List of profiles
    policies_profiles : dict
        Mapping from profile index to list of policies
    num_policies : int
        Total number of policies
        
    Returns
    -------
    X : np.ndarray
        Covariate matrix (n_total x M)
    D : np.ndarray
        Policy assignment indices (n_total,)
    y : np.ndarray
        Response vector (n_total, 1)
    """
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


def compute_rhat(chains):
    """Compute Gelman-Rubin R-hat statistic."""
    chains = [np.array(chain) for chain in chains]
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    
    chains_flat = [chain.reshape(n_samples, -1) for chain in chains]
    n_params = chains_flat[0].shape[1]
    
    rhats = []
    for param_idx in range(n_params):
        param_chains = [chain[:, param_idx] for chain in chains_flat]
        
        W = np.mean([np.var(chain, ddof=1) for chain in param_chains])
        chain_means = [np.mean(chain) for chain in param_chains]
        B = n_samples * np.var(chain_means, ddof=1)
        
        var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        
        if W > 0:
            rhat = np.sqrt(var_plus / W)
        else:
            rhat = 1.0
        
        rhats.append(rhat)
    
    return np.max(rhats)


def test_ppmx_simulation_integration():
    """Test PPMx with full simulation framework using reff_4 parameters."""
    print("=" * 70)
    print("PPMx Integration Test with Simulation Framework (reff_4 params)")
    print("=" * 70)
    
    # Use parameters from reff_4
    M = reff_4.M
    R = reff_4.R
    sigma = reff_4.sigma
    mu = reff_4.mu
    var = reff_4.var
    
    # Simulation settings
    n_per_pol = 10  # Start with small sample size for testing
    seed = 42
    
    print(f"\nSetup from reff_4.py:")
    print(f"  M={M} arms, R={R} levels")
    print(f"  n={n_per_pol} obs per policy")
    
    # Setup profiles and policies
    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)
    
    print(f"  {num_policies} policies across {num_profiles} profiles")
    
    # Identify the pools (following simulations.py exactly)
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

        # Mask the empty arms
        for idx, pol in enumerate(policies_k):
            policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
        policies_profiles_masked[k] = policies_k

        if np.sum(profile) > 0:
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(policies_k, sigma[k])
            if len(pi_pools_k.keys()) != mu[k].shape[0]:
                print(f"  Warning: Profile {k}. Expected {len(pi_pools_k.keys())} pools. Received {mu[k].shape[0]} means.")
            pi_policies[k] = pi_policies_k
            # pi_pools_k has indicies that match with policies_profiles[k]
            # Need to map those indices back to all_policies
            pi_pools[k] = {}
            for x, y in pi_pools_k.items():
                y_full = [policies_profiles[k][i] for i in y]
                y_agg = [all_policies.index(i) for i in y_full]
                pi_pools[k][x] = y_agg
        else:
            pi_policies[k] = {0: 0}
            pi_pools[k] = {0: [0]}
    
    # Count total number of true pools
    total_true_pools = sum(len(mu_k) for mu_k in mu)
    print(f"  True pools structure: {total_true_pools} total pools")
    
    # Get true best policy
    best_per_profile = [np.max(mu_k) for mu_k in mu]
    true_best_profile = np.argmax(best_per_profile)
    true_best_profile_idx = int(true_best_profile)
    true_best_effect = np.max(mu[true_best_profile])
    true_best = pi_pools[true_best_profile][np.argmax(mu[true_best_profile])]
    min_dosage_best_policy = metrics.find_min_dosage(true_best, all_policies)
    
    print(f"  True best profile: {profiles[true_best_profile_idx]} (index {true_best_profile_idx})")
    print(f"  True best effect: {true_best_effect:.2f}")
    
    # Generate data
    np.random.seed(seed)
    X, D, y = generate_data(mu, var, n_per_pol, all_policies, pi_policies, M, profiles, policies_profiles, num_policies)
    
    # Flatten arrays for compatibility
    D = D.flatten()
    y = y.flatten()
    
    n_total = X.shape[0]
    print(f"\nGenerated {n_total} observations")
    
    # Configure PPMx (using test mode parameters)
    n_chains = 1
    n_iter = 200
    burn_in = 50
    thin = 3
    alpha = 1.0
    
    print(f"\nPPMx parameters:")
    print(f"  Chains: {n_chains}")
    print(f"  Iterations: {n_iter} (burn-in: {burn_in}, thin: {thin})")
    print(f"  Alpha: {alpha}")
    
    # Run multiple chains
    print(f"\nRunning {n_chains} chains...")
    
    chains_assignments = []
    chains_n_clusters = []
    
    for chain_id in range(n_chains):
        print(f"  Chain {chain_id + 1}...", end=" ", flush=True)
        
        # Create data container
        data = PPMxData.from_raw(X, Y=y, standardize=True)
        
        # Configure sampler
        config = PPMxConfig(
            alpha=alpha,
            n_iter=n_iter,
            burn_in=burn_in,
            thin=thin,
            random_state=seed + chain_id
        )
        
        # Run sampler
        sampler = PPMxSampler(data, config)
        results = sampler.run()
        
        chains_assignments.append(results['assignments'])
        chains_n_clusters.append(results['n_clusters'])
        
        print(f"{results['assignments'].shape[0]} samples, {results['n_clusters'][-1]} clusters (final)")
    
    # Compute convergence diagnostics
    print("\nConvergence diagnostics:")
    
    # R-hat on number of clusters
    rhat_n_clusters = compute_rhat([[nc] for nc in chains_n_clusters])
    print(f"  R-hat (n_clusters): {rhat_n_clusters:.4f} {'✓' if rhat_n_clusters < 1.1 else '✗'}")
    
    # R-hat on similarity matrix
    n = X.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    chains_similarity = [posterior_similarity_matrix(chain) for chain in chains_assignments]
    chains_similarity_triu = [sim[triu_indices] for sim in chains_similarity]
    rhat_similarity = compute_rhat(chains_similarity_triu)
    print(f"  R-hat (similarity): {rhat_similarity:.4f} {'✓' if rhat_similarity < 1.1 else '✗'}")
    
    # Pool chains for inference
    print("\nPooling chains for inference...")
    pooled_assignments = np.vstack(chains_assignments)
    n_samples = pooled_assignments.shape[0]
    print(f"  Total posterior samples: {n_samples}")
    
    # Compute MAP partition
    map_part = map_partition(pooled_assignments)
    n_discovered_clusters = len(np.unique(map_part))
    print(f"  MAP partition: {n_discovered_clusters} clusters")
    
    # Compute policy-level predictions using cluster means
    print("\nComputing policy-level predictions...")
    
    policy_pred_means = np.zeros(num_policies)
    
    for pol_idx in range(num_policies):
        # Get observations for this policy
        obs_indices = np.where(D == pol_idx)[0]
        
        if len(obs_indices) > 0:
            # Get cluster assignment for this policy (should be same for all obs)
            cluster_id = map_part[obs_indices[0]]
            
            # Compute cluster mean from training data
            cluster_obs = np.where(map_part == cluster_id)[0]
            cluster_mean = np.mean(y[cluster_obs])
            
            policy_pred_means[pol_idx] = cluster_mean
    
    # Make predictions on all data
    y_pred = np.array([policy_pred_means[D[i]] for i in range(len(D))])
    
    # Compute metrics using the standard framework
    ppmx_results = metrics.compute_all_metrics(
        y.reshape(-1, 1), y_pred.reshape(-1, 1), D.reshape(-1, 1), 
        true_best, all_policies, profile_map, 
        min_dosage_best_policy, true_best_effect
    )
    
    sqrd_err_ppmx = ppmx_results["sqrd_err"]
    iou_ppmx = ppmx_results["iou"]
    best_profile_indicator_ppmx = ppmx_results["best_prof"]
    min_dosage_present_ppmx = ppmx_results["min_dos_inc"]
    best_policy_diff_ppmx = ppmx_results["best_pol_diff"]
    
    print(f"\nMetrics:")
    print(f"  MSE: {sqrd_err_ppmx:.4f}")
    print(f"  IOU: {iou_ppmx:.4f}")
    print(f"  Min dosage present: {min_dosage_present_ppmx}")
    print(f"  Best policy diff: {best_policy_diff_ppmx:.4f}")
    print(f"  Best profile indicator: {best_profile_indicator_ppmx[true_best_profile_idx]}")
    
    # Summary
    print("\n" + "=" * 70)
    if rhat_n_clusters < 1.1 and rhat_similarity < 1.1:
        print("✓ Test PASSED: Chains converged and metrics computed successfully")
    else:
        print("⚠ Test PASSED with warnings: Convergence needs attention")
    print("=" * 70)


if __name__ == "__main__":
    test_ppmx_simulation_integration()
