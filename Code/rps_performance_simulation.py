import numpy as np
import pandas as pd
import time
from typing import Dict
import os

from rashomon.aggregate import RAggregate_profile, _brute_RAggregate_profile
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

    reg = 0.1

    # Step 1: Find Q for ALL possible partitions using brute force with theta = infinity
    print("    Computing Q values for all partitions...")
    start_time = time.time()
    all_partitions_set = _brute_RAggregate_profile(
        M=np.sum(target_profile),
        R=int(R[0]),  # Assuming uniform R
        H=np.inf,
        D=D,
        y=y,
        theta=np.inf,  # Set to infinity to get ALL partitions
        profile=target_profile,
        reg=reg,
        policies=target_policies
    )
    all_partitions_time = time.time() - start_time

    # Step 2: Compute Q values and posterior probabilities for all partitions
    policy_means = loss.compute_policy_means(D, y, len(target_policies))

    all_q_values = []
    all_betas = []

    for partition_idx in range(len(all_partitions_set)):
        partition_sigma = all_partitions_set.sigma[partition_idx]

        # Get partition pool means and map to policies
        pi_pools_partition, _ = extract_pools.extract_pools(target_policies, partition_sigma)
        pool_means_partition = loss.compute_pool_means(policy_means, pi_pools_partition)

        # Map pool means to each policy
        partition_beta = np.zeros(len(target_policies))
        for policy_idx, policy in enumerate(target_policies):
            for pool_id, pool_policies in pi_pools_partition.items():
                if policy_idx in pool_policies:
                    partition_beta[policy_idx] = pool_means_partition[pool_id]
                    break

        all_betas.append(partition_beta)

        # Compute Q value for this partition
        q_value = loss.compute_Q(D, y, partition_sigma, target_policies, policy_means, reg)
        all_q_values.append(q_value)

    all_q_values = np.array(all_q_values)

    # Step 3: Compute posterior probabilities using e^Q_i / sum(e^Q_j)
    # Note: We use negative Q values since lower Q = better fit = higher probability
    posterior_weights = np.exp(-all_q_values)
    norm_constant = np.sum(posterior_weights)
    posterior_weights = posterior_weights / np.sum(posterior_weights)

    # Step 4: Find MAP partition (highest posterior probability)
    map_idx = np.argmax(posterior_weights)
    map_q_value = all_q_values[map_idx]
    map_posterior_prob = posterior_weights[map_idx]

    # Step 5: Redefine theta as q_0 * (1 + epsilon) where q_0 is MAP's Q value
    theta = map_q_value * (1 + epsilon)

    print(f"    MAP Q value: {map_q_value:.4f}, MAP posterior prob: {map_posterior_prob:.4f}")
    print(f"    Using theta = {theta:.4f} (= {map_q_value:.4f} * (1 + {epsilon}))")

    # Step 6: Run RPS algorithm with the new theta
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

    # Step 7: Compute RPS-specific posterior approximation error
    # Get Q values and betas for partitions in the RPS (subset of all partitions)
    rps_q_values = []
    rps_betas = []

    for rps_idx in range(len(rashomon_set)):
        rps_sigma = rashomon_set.sigma[rps_idx]

        # Find this partition in our all_partitions list to get its Q value
        for all_idx, all_sigma in enumerate(all_partitions_set.sigma):
            if np.array_equal(rps_sigma, all_sigma):
                rps_q_values.append(all_q_values[all_idx])
                rps_betas.append(all_betas[all_idx])
                break

    # Compute RPS posterior weights (subset of all posterior weights)
    rps_q_values = np.array(rps_q_values)
    rps_weights = np.exp(-rps_q_values)
    rps_weights = rps_weights / np.sum(rps_weights)  # Renormalize within RPS

    # Compute RPS posterior mean beta as weighted average
    rps_posterior_beta = np.zeros(len(target_policies))
    for i, beta in enumerate(rps_betas):
        rps_posterior_beta += rps_weights[i] * beta

    # Compute full posterior mean beta using all partitions
    full_posterior_beta = np.zeros(len(target_policies))
    for i, beta in enumerate(all_betas):
        full_posterior_beta += posterior_weights[i] * beta

    # Compute errors
    rps_error = np.linalg.norm(true_beta - rps_posterior_beta, ord=2)
    full_error = np.linalg.norm(true_beta - full_posterior_beta, ord=2)

    return {
        'M': M,
        'R_val': R[0],
        'H': H_val,
        'epsilon': epsilon,
        'seed': ground_truth_data['seed'],
        'n_per_policy': ground_truth_data['n_per_policy'],
        'all_partitions_time': all_partitions_time,
        'rps_time': rps_time,
        'total_partitions': len(all_partitions_set),
        'num_rps_partitions': len(rashomon_set),
        'map_q_value': map_q_value,
        'map_posterior_prob': map_posterior_prob,
        'norm_constant': norm_constant,
        'theta_used': theta,
        'found_true_partition': int(any(np.array_equal(sigma, sigma_true) for sigma in rashomon_set.sigma)),
        'rps_posterior_beta_error': rps_error,  # L2 norm error using RPS posterior
        'full_posterior_beta_error': full_error,  # L2 norm error using full posterior
        # 'rps_posterior_entropy': -np.sum(rps_weights * np.log(rps_weights + 1e-10)),
        # 'full_posterior_entropy': -np.sum(posterior_weights * np.log(posterior_weights + 1e-10)),
        # 'num_unique_rps_betas': len(np.unique([tuple(beta) for beta in rps_betas])),
        # 'num_unique_all_betas': len(np.unique([tuple(beta) for beta in all_betas]))
    }


def run_parameter_sweep():
    """
    Main function to run the parameter sweep simulation using efficient approach:
    Generate ground truth data once per (M, R, seed) and reuse for all H and epsilon
    """
    # Parameter ranges
    M_values = [3]  # , 5]  # Number of features
    R_values = [4]  # , 5]  # Factor levels (uniform across features)
    H_multipliers = [1.0]  # , 2.0]  # Multipliers for H relative to minimum needed
    epsilon_values = [0.1]  # Multipliers for theta = q_0 * (1 + epsilon)

    # Simulation settings
    n_data_generations = 1  # Number of random data generations (10-100 as requested)
    n_per_policy = 30  # Samples per policy

    results = []

    for i, M in enumerate(M_values):
        for j, R_val in enumerate(R_values):
            print(f"Running simulations for M={M}, R={R_val}")

            # Create parameters
            seed = i * len(R_values) + j  # Unique seed for each (M, R) combination
            R_array = np.array([R_val] * M)
            params = create_simulation_params(M, R_array, seed)

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
