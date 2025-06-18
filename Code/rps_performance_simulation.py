import numpy as np
import pandas as pd
import time
from typing import Dict
import os

from rashomon.aggregate import RAggregate_profile
from rashomon import hasse, loss, extract_pools
from rps_simulation_params import create_simulation_params


def generate_ground_truth_data(params: Dict, data_gen_seed: int, n_per_policy: int = 50) -> Dict:
    """
    Generate ground truth data for a given M, R, and seed.
    This is independent of H and epsilon values.
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

    # Map true beta (pool means) to each policy/feature
    true_beta = np.zeros(len(target_policies))
    for policy_idx, policy in enumerate(target_policies):
        pool_id = pi_policies_true[policy_idx]
        true_beta[policy_idx] = mu_true[pool_id]

    return {
        'M': M,
        'R': R,
        'target_policies': target_policies,
        'target_profile': target_profile,
        'sigma_true': sigma_true,
        'mu_true': mu_true,
        'var_true': var_true,
        'pi_pools_true': pi_pools_true,
        'pi_policies_true': pi_policies_true,
        'true_beta': true_beta,
        'X': X,
        'D': D,
        'y': y,
        'seed': data_gen_seed,
        'n_per_policy': n_per_policy
    }


def evaluate_rps_performance(ground_truth_data: Dict, H_val: int, epsilon: float) -> Dict:
    """
    Evaluate RPS algorithm performance on pre-generated ground truth data
    """
    # Extract ground truth data
    M = ground_truth_data['M']
    R = ground_truth_data['R']
    target_policies = ground_truth_data['target_policies']
    target_profile = ground_truth_data['target_profile']
    sigma_true = ground_truth_data['sigma_true']
    true_beta = ground_truth_data['true_beta']
    D = ground_truth_data['D']
    y = ground_truth_data['y']

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

    # Compute proper Bayesian posterior approximation error
    policy_means = loss.compute_policy_means(D, y, len(target_policies))

    # Compute Q values for all partitions in RPS
    rps_q_values = []
    rps_betas = []

    for rps_idx in range(len(rashomon_set)):
        rps_sigma = rashomon_set.sigma[rps_idx]

        # Get RPS pool means and map to policies
        pi_pools_rps, _ = extract_pools.extract_pools(target_policies, rps_sigma)
        pool_means_rps = loss.compute_pool_means(policy_means, pi_pools_rps)

        # Map pool means to each policy
        rps_beta = np.zeros(len(target_policies))
        for policy_idx, policy in enumerate(target_policies):
            # Find which pool this policy belongs to
            for pool_id, pool_policies in pi_pools_rps.items():
                if policy_idx in pool_policies:
                    rps_beta[policy_idx] = pool_means_rps[pool_id]
                    break

        rps_betas.append(rps_beta)

        # Compute Q value for this partition
        q_value = loss.compute_Q(D, y, rps_sigma, target_policies, policy_means, reg)
        rps_q_values.append(q_value)

    # Step 2: Compute RPS posterior approximation using softmax weights
    rps_q_values = np.array(rps_q_values)
    # Use negative Q values for softmax (lower Q = higher probability)
    weights = np.exp(-rps_q_values)
    weights = weights / np.sum(weights)  # Normalize to get probabilities

    # Step 3: Compute posterior mean beta as weighted average
    posterior_beta = np.zeros(len(target_policies))
    for i, beta in enumerate(rps_betas):
        posterior_beta += weights[i] * beta

    # Step 4: Compute L2 norm of difference
    error = np.linalg.norm(true_beta - posterior_beta, ord=2)

    return {
        'M': M,
        'R_val': R[0],
        'H': H_val,
        'epsilon': epsilon,
        'seed': ground_truth_data['seed'],
        'n_per_policy': ground_truth_data['n_per_policy'],
        'rps_time': rps_time,
        'num_rps_partitions': len(rashomon_set),
        'found_true_partition': int(any(np.array_equal(sigma, sigma_true) for sigma in rashomon_set.sigma)),
        'posterior_beta_error': error,
        'num_unique_betas': len(np.unique([tuple(beta) for beta in rps_betas])),
        'posterior_entropy': -np.sum(weights * np.log(weights + 1e-10))
    }


def run_parameter_sweep():
    """
    Main function to run the parameter sweep simulation using efficient approach:
    Generate ground truth data once per (M, R, seed) and reuse for all H and epsilon
    """
    # Parameter ranges
    M_values = [3, 4]  # , 5]  # Number of features
    R_values = [3, 4]  # , 5]  # Factor levels (uniform across features)
    H_multipliers = [1.0, 1.5]  # , 2.0]  # Multipliers for H relative to minimum needed
    epsilon_values = [10]  # , 1.0, 1.5]  # Rashomon thresholds

    # Simulation settings
    n_data_generations = 2  # Number of random data generations (10-100 as requested)
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

            # Generate ground truth data once per (M, R, seed)
            for seed in range(n_data_generations):
                print(f"  Generating ground truth data for seed {seed}")
                
                # Generate ground truth data once for this (M, R, seed)
                ground_truth_data = generate_ground_truth_data(params, seed, n_per_policy)
                
                # Now evaluate RPS for all H and epsilon combinations using this ground truth
                for H_mult in H_multipliers:
                    H_val = int(base_H * H_mult)

                    for epsilon in epsilon_values:
                        print(f"    Evaluating H={H_val}, epsilon={epsilon}")

                        try:
                            result = evaluate_rps_performance(ground_truth_data, H_val, epsilon)
                            results.append(result)
                        except Exception as e:
                            print(f"      Error in H={H_val}, epsilon={epsilon}: {e}")
                            continue

                if (seed + 1) % 10 == 0:
                    print(f"  Completed {seed + 1}/{n_data_generations} data generations")

    # Save results
    df = pd.DataFrame(results)
    dir = "../Results/timed_sims/"
    os.makedirs(dir, exist_ok=True)
    df.to_csv(f"{dir}rps_performance_results.csv", index=False)
    print(f"Simulation complete. Results saved to {dir}rps_performance_results.csv")

    return df


if __name__ == "__main__":
    run_parameter_sweep()
