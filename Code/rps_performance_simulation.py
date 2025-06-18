import numpy as np
import pandas as pd
import time
import itertools
from typing import Dict

from rashomon.aggregate import RAggregate_profile
from rashomon import hasse, loss, extract_pools
from rps_simulation_params import create_simulation_params


def compute_pool_matching_error(rps_pools: np.ndarray, gt_pools: np.ndarray,
                                pi_pools_rps: Dict, pi_pools_gt: Dict) -> float:
    """
    Compute error between pool means by matching pools based on policy membership
    """
    if len(rps_pools) != len(gt_pools):
        return float('inf')

    if len(rps_pools) == 1:
        # Single pool case - direct comparison
        return abs(rps_pools[0] - gt_pools[0])    # Multi-pool case: find best matching by checking all permutations
    # For small number of pools, this is feasible
    
    # Get pool IDs
    rps_pool_ids = list(pi_pools_rps.keys())
    gt_pool_ids = list(pi_pools_gt.keys())

    min_error = float('inf')

    # Try all possible assignments
    for gt_perm in itertools.permutations(gt_pool_ids):
        total_error = 0.0
        valid_assignment = True

        for i, rps_pool_id in enumerate(rps_pool_ids):
            gt_pool_id = gt_perm[i]

            # Check if pools contain same policies
            rps_policy_set = set(pi_pools_rps[rps_pool_id])
            gt_policy_set = set(pi_pools_gt[gt_pool_id])

            if rps_policy_set == gt_policy_set:
                # Pools match - add error
                total_error += abs(rps_pools[rps_pool_id] - gt_pools[gt_pool_id])
            else:
                # Pools don't match - invalid assignment
                valid_assignment = False
                break

        if valid_assignment:
            avg_error = total_error / len(rps_pools)
            min_error = min(min_error, avg_error)

    return min_error


def run_single_simulation(params: Dict, H_val: int, epsilon: float,
                          data_gen_seed: int, n_per_policy: int = 50) -> Dict:
    """
    Run a single simulation with given parameters
    """
    np.random.seed(data_gen_seed)

    M = params['M']
    R = params['R']
    profiles = params['profiles']
    target_profile_idx = params['target_profile_idx']
    target_profile = profiles[target_profile_idx]

    # Generate data for target profile only
    all_policies = hasse.enumerate_policies(M, R)
    target_policies = [x for x in all_policies if hasse.policy_to_profile(x) == target_profile]

    # Generate synthetic data
    num_policies = len(target_policies)
    num_data = num_policies * n_per_policy

    X = np.zeros(shape=(num_data, M))
    D = np.zeros(shape=(num_data, 1), dtype=int)
    y = np.zeros(shape=(num_data, 1))

    # Use the ground truth parameters for target profile
    sigma_true = params['sigma'][target_profile_idx]
    mu_true = params['mu'][target_profile_idx]
    var_true = params['var'][target_profile_idx]

    pi_pools_true, pi_policies_true = extract_pools.extract_pools(target_policies, sigma_true)

    idx_ctr = 0
    for pol_idx, policy in enumerate(target_policies):
        pool_id = pi_policies_true[pol_idx]
        mu_pool = mu_true[pool_id]
        var_pool = var_true[pool_id]

        y_pol = np.random.normal(mu_pool, np.sqrt(var_pool), size=(n_per_policy, 1))

        start_idx = idx_ctr * n_per_policy
        end_idx = (idx_ctr + 1) * n_per_policy

        X[start_idx:end_idx, :] = policy
        D[start_idx:end_idx, 0] = pol_idx
        y[start_idx:end_idx, 0] = y_pol.flatten()

        idx_ctr += 1

    # Time RPS algorithm (adaptive/clever version)
    theta = epsilon  # Use epsilon as Rashomon threshold
    reg = 0.1
    start_time = time.time()
    rashomon_set = RAggregate_profile(
        M=np.sum(target_profile),
        R=int(R[0]),  # Assuming uniform R
        H=H_val,
        D=D,
        y=y,
        theta=theta,
        profile=target_profile,
        reg=reg,
        policies=target_policies
    )
    rps_time = time.time() - start_time

    # Compute absolute errors in pool means compared to ground truth
    errors = []
    policy_means = loss.compute_policy_means(D, y, len(target_policies))

    for rps_idx in range(len(rashomon_set)):
        rps_sigma = rashomon_set.sigma[rps_idx]

        # Get RPS pool means
        pi_pools_rps, _ = extract_pools.extract_pools(target_policies, rps_sigma)
        pool_means_rps = loss.compute_pool_means(policy_means, pi_pools_rps)

        # Compare against the true partition that generated the data
        if np.array_equal(rps_sigma, sigma_true):
            # Found the exact true partition - compare pool means directly
            error = compute_pool_matching_error(
                pool_means_rps, mu_true, pi_pools_rps, pi_pools_true
            )
        else:
            # Different partition structure - compute loss difference
            # Compare losses rather than means
            true_loss = loss.compute_loss(D, y, len(target_policies), pi_pools_true, mu_true, reg)
            rps_loss = loss.compute_loss(D, y, len(target_policies), pi_pools_rps, pool_means_rps, reg)
            error = abs(rps_loss - true_loss)

        errors.append(error)

    return {
        'M': M,
        'R_val': R[0],
        'H': H_val,
        'epsilon': epsilon,
        'seed': data_gen_seed,
        'n_per_policy': n_per_policy,
        'rps_time': rps_time,
        'num_rps_partitions': len(rashomon_set),
        'found_true_partition': int(any(np.array_equal(sigma, sigma_true) for sigma in rashomon_set.sigma)),
        'mean_error_vs_truth': np.mean(errors) if errors else 0.0,
        'max_error_vs_truth': np.max(errors) if errors else 0.0,
        'min_error_vs_truth': np.min(errors) if errors else 0.0
    }


def run_parameter_sweep():
    """
    Main function to run the parameter sweep simulation
    """
    # Parameter ranges
    M_values = [3, 4, 5]  # Number of features
    R_values = [3, 4, 5]  # Factor levels (uniform across features)
    H_multipliers = [1.0, 1.5, 2.0]  # Multipliers for H relative to minimum needed
    epsilon_values = [0.5, 1.0, 2.0, 4.0]  # Rashomon thresholds

    # Simulation settings
    n_data_generations = 50  # Number of random data generations (10-100 as requested)
    n_per_policy = 30  # Samples per policy

    results = []

    for M in M_values:
        for R_val in R_values:
            print(f"Running simulations for M={M}, R={R_val}")

            # Create parameters
            R_array = np.array([R_val] * M)
            params = create_simulation_params(M, R_array)

            # Calculate base H needed
            base_H = params['H']

            for H_mult in H_multipliers:
                H_val = int(base_H * H_mult)

                for epsilon in epsilon_values:
                    print(f"  H={H_val}, epsilon={epsilon}")

                    # Run multiple data generations
                    for seed in range(n_data_generations):
                        # try:
                        result = run_single_simulation(
                            params, H_val, epsilon, seed, n_per_policy
                        )
                        results.append(result)

                        if (seed + 1) % 10 == 0:
                            print(f"    Completed {seed + 1}/{n_data_generations} generations")

                        # except Exception as e:
                        #     print(f"    Error in seed {seed}: {e}")
                        #     continue

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("../Results/timed_sims/rps_performance_results.csv", index=False)
    print("Simulation complete. Results saved to ../Results/timed_sims/rps_performance_results.csv")

    return df


if __name__ == "__main__":
    run_parameter_sweep()
