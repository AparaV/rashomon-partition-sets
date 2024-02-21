import os
import argparse
import importlib
import pickle
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn import linear_model, metrics

from rashomon import tva
# from rashomon import metrics
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
    # parser.add_argument("--output_dir", type=str,
    #                     help="Where should output be saved")
    parser.add_argument("--output_prefix", type=str,
                        help="Prefix for output file name")
    # parser.add_argument("--theta", type=float,
    #                     help="Rashomon threshold")
    parser.add_argument("--method", type=str,
                        help="One of {r, lasso, ct}")
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
    lasso_reg = params.lasso_reg

    num_profiles = 2**M
    profiles, profile_map = tva.enumerate_profiles(M)
    all_policies = tva.enumerate_policies(M, R)
    num_policies = len(all_policies)

    # Simulation parameters and variables
    samples_per_pol = [args.sample_size]
    num_sims = args.iters

    # Output file names
    start_sim = 0
    output_dir = "../Results/"
    # output_suffix = f"_{args.sample_size}_{args.iters}_{start_sim}.csv"
    output_suffix = f"_{args.sample_size}_{args.iters}.csv"
    rashomon_fname = args.output_prefix + "_rashomon" + ".pkl"
    lasso_fname = args.output_prefix + "_lasso" + output_suffix
    ct_fname = args.output_prefix + "_ct" + output_suffix

    # Identify the pools
    policies_profiles = {}
    policies_profiles_masked = {}
    policies_ids_profiles = {}
    pi_policies = {}
    pi_pools = {}
    for k, profile in enumerate(profiles):

        policies_temp = [(i, x) for i, x in enumerate(all_policies) if tva.policy_to_profile(x) == profile]
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

    # The transformation matrix for Lasso
    G = tva.alpha_matrix(all_policies)

    # Simulation results data structure
    method = args.method
    if method not in ["r", "lasso", "ct"]:
        print(f"method should be one of [r, lasso, ct]. Received {method}. Defaulting to r")
        method = "r"
    rashomon_list = []
    lasso_list = []
    ct_list = []

    np.random.seed(3)

    #
    # Simulations
    #
    for n_per_pol in samples_per_pol:

        print(f"Number of samples: {n_per_pol}")

        for sim_i in range(start_sim, start_sim + num_sims):
            print(sim_i)
            np.random.seed(sim_i)

            if (sim_i + 1) % 20 == 0:
                print(f"\tSimulation {sim_i+1}")

            # Generate data
            X, D, y = generate_data(beta, var, n_per_pol, all_policies, pi_policies, M, policies_profiles)
            # policy_means = loss.compute_policy_means(D, y, num_policies)
            # The dummy matrix for Lasso
            D_matrix = tva.get_dummy_matrix(D, G, num_policies)

            #
            # Run Rashomon
            #
            if method == "r":

                # Adaptive expand R set threshold until we find a model that
                # identifies the true best profile
                # found_best_profile = False
                # counter = 0
                # this_theta = theta

                # current_results = []
                R_set, rashomon_profiles = RAggregate_slopes(M, R, np.inf, D, X, y, theta, reg=1, verbose=False)
                print(f"\t\t{theta},{len(R_set)}")

                result = {
                    "R_set": R_set,
                    "R_profiles": rashomon_profiles,
                    "D": D,
                    "X": X,
                    "y": y
                }
                rashomon_list.append(result)

                # best_loss = np.inf

                # for idx, r_set in enumerate(R_set):

                #     # MSE
                #     pi_policies_profiles_r = {}
                #     for k, profile in enumerate(profiles):
                #         _, pi_policies_r_k = extract_pools.extract_pools(
                #             policies_profiles_masked[k],
                #             rashomon_profiles[k].sigma[r_set[k]]
                #         )
                #         pi_policies_profiles_r[k] = pi_policies_r_k

                #     pi_pools_r, pi_policies_r = extract_pools.aggregate_pools(
                #         pi_policies_profiles_r, policies_ids_profiles)
                #     # pool_means_r = loss.compute_pool_means(policy_means, pi_pools_r)
                #     y_r_est = metrics.make_predictions(D, pi_policies_r, pool_means_r)

                #     r_set_results = metrics.compute_all_metrics(
                #         y, y_r_est, D, true_best, all_policies, profile_map,
                #         min_dosage_best_policy, true_best_effect)
                #     sqrd_err = r_set_results["sqrd_err"]
                #     iou_r = r_set_results["iou"]
                #     best_profile_indicator = r_set_results["best_prof"]
                #     min_dosage_present = r_set_results["min_dos_inc"]
                #     best_pol_diff = r_set_results["best_pol_diff"]
                #     this_loss = sqrd_err + reg * len(pi_pools_r)

                #     this_list = [
                #         n_per_pol, sim_i, len(pi_pools_r), sqrd_err, iou_r, min_dosage_present, best_pol_diff
                #         ]
                #     this_list += best_profile_indicator
                #     current_results.append(this_list)

                #     if this_loss < best_loss:
                #         best_loss = this_loss

                #     if best_profile_indicator[true_best_profile_idx] == 1:
                #         found_best_profile = True
                #             print("Found", this_loss)

                #     if found_best_profile:
                #         print("\tFound best profile")

                #     eps = 0.2
                #     eps_factor = 1 + eps
                #     if np.isinf(best_loss):
                #         best_loss = this_theta
                #     if found_best_profile or this_theta >= (eps_factor * best_loss):
                #         found_best_profile = True
                #     else:
                #         # this_theta = eps_factor * best_loss
                #         this_theta += 0.1
                #         found_best_profile = False
                #     counter += 1

                #     if this_theta >= 6.5:
                #         found_best_profile = True
                #     if len(R_set) > 1e5:
                #     if len(R_set) >= 8377:
                #         found_best_profile = True
                #     this_theta += 0.5
                #     break

                # rashomon_list += current_results

            #
            # Run Lasso
            #
            if method == "lasso":
                lasso = linear_model.Lasso(lasso_reg, fit_intercept=False)
                lasso.fit(D_matrix, y)
                alpha_est = lasso.coef_
                y_tva = lasso.predict(D_matrix)

                sqrd_err = metrics.mean_squared_error(y_tva, y)
                L1_loss = sqrd_err + reg * np.linalg.norm(alpha_est, ord=1)

                this_list = [n_per_pol, sim_i, sqrd_err, L1_loss]
                lasso_list.append(this_list)

    profiles_str = [str(prof) for prof in profiles]

    if method == "r":
        with open(os.path.join(output_dir, rashomon_fname), "wb") as f:
            pickle.dump(rashomon_list, f, pickle.HIGHEST_PROTOCOL)

    if method == "lasso":
        lasso_cols = ["n_per_pol", "sim_num", "MSE", "L1_loss"]
        lasso_cols += profiles_str
        lasso_df = pd.DataFrame(lasso_list, columns=lasso_cols)
        lasso_df.to_csv(os.path.join(output_dir, lasso_fname))
