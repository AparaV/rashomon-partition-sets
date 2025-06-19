import numpy as np
import pandas as pd
import time
from typing import Dict
import os
import pickle

from rashomon.aggregate import RAggregate_profile, _brute_RAggregate_profile
from rashomon import hasse, loss, extract_pools
from rps_simulation_params import create_simulation_params

from sklearn.metrics import mean_squared_error

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
    true_beta = np.zeros(len(target_policies))
    for pol_idx, policy in enumerate(target_policies):
        pool_id = pi_policies_true[pol_idx]
        mu_pool = mu_true[pool_id]
        var_pool = var_true[pool_id]

        true_beta[pol_idx] = mu_true[pool_id]
        y_pol = np.random.normal(mu_pool, np.sqrt(var_pool), size=(n_per_policy, 1))

        start_idx = idx_ctr * n_per_policy
        end_idx = (idx_ctr + 1) * n_per_policy

        X[start_idx:end_idx, :] = policy
        D[start_idx:end_idx, 0] = pol_idx
        y[start_idx:end_idx, 0] = y_pol.flatten()

        idx_ctr += 1

    # # Map true beta (pool means) to each policy/feature
    # # TODO: This can be made more efficient by directly using pi_pools_true
    # for policy_idx, policy in enumerate(target_policies):
    #     pool_id = pi_policies_true[policy_idx]

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


def compute_all_partitions_and_map(ground_truth_data: Dict, reg: float = 0.1, full_partition: bool = False) -> Dict:
    """
    Compute all partitions, Q values, posterior probabilities, and find MAP.
    This handles steps 1-4 of the RPS performance evaluation.

    Args:
        ground_truth_data: Dictionary containing ground truth data
        H_val: H parameter for RPS (used for debugging)
        reg: Regularization parameter

    Returns:
        Dictionary containing all partitions, Q values, betas, posterior weights, and MAP info
    """
    # Extract ground truth data
    M = ground_truth_data['M']
    R = ground_truth_data['R']
    sigma_true = ground_truth_data['sigma_true']
    target_policies = ground_truth_data['target_policies']
    target_profile = ground_truth_data['target_profile']
    D = ground_truth_data['D']
    y = ground_truth_data['y']

    # # Step 1: Find Q for ALL possible partitions using brute force with theta = infinity
    if full_partition:
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
        print(f"    Found {len(all_partitions_set)} partitions in {all_partitions_time:.2f} seconds")
    else:
        all_partitions_set = None
        all_partitions_time = None

    # Step 2: Compute Q values and posterior probabilities for all partitions
    policy_means = loss.compute_policy_means(D, y, len(target_policies))

    # policy_data = np.array([[idx for idx in enumerate(target_policies)]])
    policy_data = np.zeros(shape=(len(target_policies), 1), dtype=int)
    for pol_idx, policy in enumerate(target_policies):
        policy_data[pol_idx, 0] = pol_idx
    # hasse_edges = extract_pools.lattice_edges(target_policies)
    # hasse_edges = None

    all_q_values = []
    all_betas = []
    all_sigmas = []
    if full_partition:
        for partition_idx in range(len(all_partitions_set)):
            partition_sigma = all_partitions_set.sigma[partition_idx]
            q_value = all_partitions_set.Q[partition_idx]
            pools_idx = all_partitions_set.pools[partition_idx]
            # pi_pools_idx = pools_idx['pi_pools']
            pi_policies_idx = pools_idx['pi_policies']
            mu_pools_idx = pools_idx['mu_pools']
            D_pool = [pi_policies_idx[pol_id] for pol_id in policy_data[:, 0]]
            partition_beta = mu_pools_idx[D_pool]

            # partition_beta = loss.predict(policy_data, partition_sigma, target_policies, policy_means, hasse_edges)

            all_betas.append(partition_beta)
            all_sigmas.append(partition_sigma)
            all_q_values.append(q_value)

        all_q_values = np.array(all_q_values)
        all_betas = np.array(all_betas)  # Convert to array for potential vectorization

    # Step 3: Compute posterior probabilities using e^Q_i / sum(e^Q_j)
    # Note: We use negative Q values since lower Q = better fit = higher probability
    if full_partition:
        posterior_weights = np.exp(-all_q_values)
        norm_constant = np.sum(posterior_weights)
        posterior_weights = posterior_weights / norm_constant
    else:
        norm_constant = None

    # Step 4: Find MAP partition (highest posterior probability)
    if full_partition:
        map_idx = np.argmax(posterior_weights)
        map_q_value = all_q_values[map_idx]
        map_posterior_prob = posterior_weights[map_idx]
    else:
        map_idx = None
        map_q_value = loss.compute_Q(D, y, sigma_true, target_policies, policy_means, reg)
        map_posterior_prob = None

    # Compute full posterior mean beta using all partitions
    if full_partition:
        full_posterior_beta = np.zeros(len(target_policies))
        for i, beta in enumerate(all_betas):
            full_posterior_beta += posterior_weights[i] * beta
    else:
        full_posterior_beta = None

    return {
        'all_partitions_set': all_partitions_set,
        'all_q_values': all_q_values,
        'all_betas': all_betas,
        'all_sigmas': all_sigmas,
        'norm_constant': norm_constant,
        'map_idx': map_idx,
        'map_q_value': map_q_value,
        'map_posterior_prob': map_posterior_prob,
        'full_posterior_beta': full_posterior_beta,
        'policy_means': policy_means,
        'all_partitions_time': all_partitions_time,
        'reg': reg
    }


def evaluate_rps_performance(
    ground_truth_data: Dict, epsilon: float, all_partitions_results: Dict = None
) -> Dict:
    """
    Evaluate RPS algorithm performance on pre-generated ground truth data

    Args:
        ground_truth_data: Dictionary containing ground truth data
        H_val: H parameter for RPS
        epsilon: Epsilon parameter for setting theta
        all_partitions_results: Optional precomputed results from compute_all_partitions_and_map
                               (if not provided, will be computed)

    Returns:
        Dictionary containing performance metrics
    """
    # Compute all partitions and MAP if not provided
    if all_partitions_results is None:
        all_partitions_results = compute_all_partitions_and_map(ground_truth_data)

    all_partitions_set = all_partitions_results['all_partitions_set']
    # all_q_values = all_partitions_results['all_q_values']
    # all_betas = all_partitions_results['all_betas']
    # all_sigmas = all_partitions_results['all_sigmas']
    norm_constant = all_partitions_results['norm_constant']
    full_posterior_beta = all_partitions_results['full_posterior_beta']
    reg = all_partitions_results['reg']
    policy_means = all_partitions_results['policy_means']

    # Extract ground truth data
    M = ground_truth_data['M']
    R = ground_truth_data['R']
    target_policies = ground_truth_data['target_policies']
    target_profile = ground_truth_data['target_profile']
    sigma_true = ground_truth_data['sigma_true']
    true_beta = ground_truth_data['true_beta']
    D = ground_truth_data['D']
    y = ground_truth_data['y']

    # Step 5: Redefine theta as q_0 * (1 + epsilon) where q_0 is MAP's Q value
    theta = all_partitions_results['map_q_value'] * (1 + epsilon)

    # Print MAP information
    map_q = all_partitions_results['map_q_value']
    map_prob = all_partitions_results['map_posterior_prob']
    if map_prob is not None:
        print(f"    MAP Q value: {map_q:.4f}, MAP posterior prob: {map_prob:.4f}")
    else:
        print(f"    MAP Q value: {map_q:.4f}, MAP posterior prob: {map_prob} (not computed)")
    print(f"    Using theta = {theta:.4f} (= {map_q:.4f} * (1 + {epsilon}))")

    # Step 6: Run RPS algorithm with the new theta
    start_time = time.time()
    rashomon_set = RAggregate_profile(
        M=np.sum(target_profile),
        R=int(R[0]),  # Assuming uniform R
        H=np.inf,
        D=D,
        y=y,
        theta=theta,
        profile=target_profile,
        reg=reg,
        policies=target_policies
    )
    rps_time = time.time() - start_time
    print(f"    RPS found {rashomon_set.size} partitions in {rps_time:.2f} seconds")

    # Step 7: Compute RPS-specific posterior approximation error
    # Get Q values and betas for partitions in the RPS (subset of all partitions)
    rps_q_values = []
    rps_betas = []

    policy_data = np.zeros(shape=(len(target_policies), 1), dtype=int)
    for pol_idx, policy in enumerate(target_policies):
        policy_data[pol_idx, 0] = pol_idx
    # hasse_edges = extract_pools.lattice_edges(target_policies)

    for rps_idx in range(len(rashomon_set)):
        # rps_sigma = rashomon_set.sigma[rps_idx]
        q_value = rashomon_set.Q[rps_idx]

        pools_idx = rashomon_set.pools[rps_idx]
        # pi_pools_idx = pools_idx['pi_pools']
        pi_policies_idx = pools_idx['pi_policies']
        mu_pools_idx = pools_idx['mu_pools']
        D_pool = [pi_policies_idx[pol_id] for pol_id in policy_data[:, 0]]
        rps_beta = mu_pools_idx[D_pool]

        # rps_beta, h = loss.predict(policy_data, rps_sigma, target_policies, policy_means, hasse_edges, return_num_pools=True)
        # mse = mean_squared_error(true_beta, rps_beta)
        # q_value = mse + reg * h

        # rps_beta = loss.predict(policy_data, rps_sigma, target_policies, policy_means, hasse_edges)
        rps_betas.append(rps_beta)
        # q_value = loss.compute_Q(D, y, rps_sigma, target_policies, policy_means, reg)
        rps_q_values.append(q_value)

    # Compute RPS posterior weights (subset of all posterior weights)
    rps_q_values = np.array(rps_q_values)
    rps_weights = np.exp(-rps_q_values)
    rps_norm_constant = np.sum(rps_weights)
    rps_weights = rps_weights / rps_norm_constant  # Renormalize within RPS

    # Compute RPS posterior mean beta as weighted average
    rps_posterior_beta = np.zeros(len(target_policies))
    # print(len(rps_betas), len(rps_weights))
    # print(rps_betas[0].shape, rps_posterior_beta.shape, rps_weights[0])
    for i, beta in enumerate(rps_betas):
        rps_posterior_beta += rps_weights[i] * beta

    # Compute errors
    rps_error = np.linalg.norm(true_beta - rps_posterior_beta, ord=2)
    if full_posterior_beta is None:
        full_error = None
    else:
        full_error = np.linalg.norm(true_beta - full_posterior_beta, ord=2)

    return {
        'M': M,
        'R_val': R[0],
        # 'H': H_val,
        'epsilon': epsilon,
        'seed': ground_truth_data['seed'],
        'n_per_policy': ground_truth_data['n_per_policy'],
        'all_partitions_time': all_partitions_results['all_partitions_time'],
        'rps_time': rps_time,
        'total_partitions': 2 ** (M * (R[0]-1)),
        'num_rps_partitions': len(rashomon_set),
        'map_q_value': all_partitions_results['map_q_value'],
        'map_posterior_prob': all_partitions_results['map_posterior_prob'],
        'norm_constant': norm_constant,
        'rps_norm_constant': rps_norm_constant,
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

    # Setup 1: Fix M, R. Vary epsilon
    setup_id = 1
    M_R_values = [(3, 4), (4, 4)]
    epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
    full_partition = True

    # # Setup 2: Fix M, epsilon. Vary R
    # setup_id = 2
    # M_R_values = [(3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]
    # epsilon_values = [0.01]
    # full_partition = False

    # # Setup 3: Fix R, epsilon. Vary M
    # setup_id = 3
    # M_R_values = [(3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4)]
    # epsilon_values = [0.01]
    # full_partition = False

    # Simulation settings
    n_data_generations = 10  # Number of random data generations
    n_per_policy = 30  # Samples per policy

    dir = f"../Results/timed_sims/setup_{setup_id}/"
    os.makedirs(dir, exist_ok=True)

    df = None
    for i, M_R_set in enumerate(M_R_values):

        M = M_R_set[0]  # Extract M from the first tuple
        R_val = M_R_set[1]  # Extract R from the second tuple

        fname = f"rps_performance_results_{M}_{R_val}_setup_{setup_id}.csv"
        path = f"{dir}{fname}"

        # Check if results already exist
        if os.path.exists(path):
            print(f"Results for M={M_R_set[0]}, R={M_R_set[1]} already exist. Skipping.")
            continue

        results = []

        print(f"Running simulations for M={M}, R={R_val}")

        # Create parameters
        seed_MR = i  # Unique seed for each (M, R) combination
        R_array = np.array([R_val] * M)
        params = create_simulation_params(M, R_array, seed_MR)

        # Calculate base H needed
        base_H = params['H']

        # Generate ground truth data once per (M, R, seed)
        for seed in range(n_data_generations):
            print(f"  Generating ground truth data for seed {seed}")

            # Generate ground truth data once for this (M, R, seed)
            ground_truth_data = generate_ground_truth_data(params, seed, n_per_policy)

            all_partitions_results = compute_all_partitions_and_map(ground_truth_data, full_partition=full_partition)
            if full_partition:
                pkl_file = f"all_partitions_results_M{M}_R{R_val}_seed{seed}.pkl"
                pkl_path = os.path.join(dir, pkl_file)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(all_partitions_results, f)

            for epsilon in epsilon_values:
                print(f"    Evaluating epsilon={epsilon}")

                result = evaluate_rps_performance(
                    ground_truth_data, epsilon, all_partitions_results
                )
                results.append(result)

            if (seed + 1) % 10 == 0:
                print(f"  Completed {seed + 1}/{n_data_generations} data generations")

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
        print(f"Simulation complete. Results saved to {path}.csv")

    return df


if __name__ == "__main__":
    run_parameter_sweep()
