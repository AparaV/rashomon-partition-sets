import os
import argparse
import importlib
import pickle
import numpy as np

from copy import deepcopy

from rashomon import hasse
from rashomon import extract_pools
from rashomon.aggregate import RAggregate_slopes


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument("--params", type=str,
                        help=".py file where parameters are stored")
    parser.add_argument("--sample_size", type=int,
                        help="Number of samples per feature combination")
    parser.add_argument("--iters", type=int,
                        help="Number of iterations")
    parser.add_argument("--output_prefix", type=str,
                        help="Prefix for output file name")
    args = parser.parse_args()
    return args


def generate_data(beta, var, n_per_pol, all_policies, pi_policies, M, policies_profiles):
    num_data = num_policies * n_per_pol
    X = np.zeros(shape=(num_data, M))
    D = np.zeros(shape=(num_data, 1), dtype='int_')
    y = np.zeros(shape=(num_data, 1))
    ones = np.zeros(shape=(n_per_pol, 1)) + 1

    idx_ctr = 0
    for k, profile in enumerate(profiles):
        policies_k = policies_profiles[k]

        for idx, policy in enumerate(policies_k):
            policy_idx = [i for i, x in enumerate(all_policies) if x == policy]

            start_idx = idx_ctr * n_per_pol
            end_idx = (idx_ctr + 1) * n_per_pol

            X[start_idx:end_idx, ] = policy
            X_subset = X[start_idx:end_idx, ]
            X_ones = np.concatenate((ones, X_subset), axis=1)
            D[start_idx:end_idx, ] = policy_idx[0]

            pool_id = pi_policies[k][idx]
            beta_i = beta[k][pool_id].reshape((-1, 1))
            var_i = var[k][pool_id]
            y_i = np.matmul(X_ones, beta_i) + np.random.normal(0, var_i, size=(n_per_pol, 1))

            y[start_idx:end_idx, ] = y_i

            idx_ctr += 1

    return X, D, y


if __name__ == "__main__":

    args = parse_arguments()

    # from sim_4_params import M, R, sigma, mu, var
    params_module_name = args.params
    params = importlib.import_module(params_module_name, package=None)
    M = params.M
    R = params.R
    sigma = params.sigma
    beta = params.beta
    var = params.var
    H = params.H
    theta = params.theta
    reg = params.reg

    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)

    # Simulation parameters and variables
    samples_per_pol = [args.sample_size]
    num_sims = args.iters

    # Output file names
    start_sim = 0
    output_dir = "../Results/"
    output_suffix = f"_{args.sample_size}_{args.iters}.csv"
    rashomon_fname = args.output_prefix + "_rashomon" + ".pkl"

    # Identify the pools
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
            if len(pi_pools_k.keys()) != beta[k].shape[0]:
                print(f"Profile {k}. Expected {len(pi_pools_k.keys())} pools. Received {beta[k].shape[0]} means.")
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

    # Simulation results data structure
    rashomon_list = []

    np.random.seed(3)

    #
    # Simulations
    #
    for n_per_pol in samples_per_pol:

        print(f"Number of samples: {n_per_pol}")

        for sim_i in range(start_sim, start_sim + num_sims):
            print(sim_i)
            # np.random.seed(sim_i)

            if (sim_i + 1) % 20 == 0:
                print(f"\tSimulation {sim_i+1}")

            # Generate data
            X, D, y = generate_data(beta, var, n_per_pol, all_policies, pi_policies, M, policies_profiles)

            #
            # Run Rashomon
            #
            R_set, rashomon_profiles = RAggregate_slopes(M, R, H, D, X, y, theta, reg=reg, verbose=True)
            print(f"\t\t{theta},{len(R_set)}")

            result = {
                "R_set": R_set,
                "R_profiles": rashomon_profiles,
                "D": D,
                "X": X,
                "y": y
            }
            rashomon_list.append(result)

    with open(os.path.join(output_dir, rashomon_fname), "wb") as f:
        pickle.dump(rashomon_list, f, pickle.HIGHEST_PROTOCOL)
