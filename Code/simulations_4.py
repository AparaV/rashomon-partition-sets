# import warnings
import numpy as np
import pandas as pd

from copy import deepcopy
# from sklearn import linear_model

from rashomon import tva
from rashomon import loss
from rashomon import metrics
# from rashomon import causal_trees
from rashomon import extract_pools
from rashomon.aggregate import RAggregate


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

            # profile_id = tva.policy_to_profile(policy)
            pool_id = pi_policies[k][idx]
            # pool_i = pi_policies[idx]
            mu_i = mu[k][pool_id]
            # var_i = var[policy_idx[0]]
            var_i = var[k][pool_id]
            y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

            start_idx = idx_ctr * n_per_pol
            end_idx = (idx_ctr + 1) * n_per_pol

            X[start_idx:end_idx, ] = policy
            # D[start_idx:end_idx, ] = idx
            D[start_idx:end_idx, ] = policy_idx[0]
            y[start_idx:end_idx, ] = y_i

            idx_ctr += 1

    return X, D, y


if __name__ == "__main__":

    from sim_4_params import M, R, sigma, mu, var

    num_profiles = 2**M
    profiles, profile_map = tva.enumerate_profiles(M)
    all_policies = tva.enumerate_policies(M, R)
    num_policies = len(all_policies)

    np.random.seed(3)

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
            if len(pi_pools_k.keys()) != mu[k].shape[0]:
                print(f"Profile {k}. Expected {len(pi_pools_k.keys())} pools. Received {mu[k].shape[0]} means.")
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

    best_per_profile = [np.max(mu_k) for mu_k in mu]
    true_best_profile = np.argmax(best_per_profile)
    true_best_profile_idx = 5
    true_best_effect = np.max(mu[true_best_profile])
    true_best = pi_pools[true_best_profile][np.argmax(mu[true_best_profile])]
    min_dosage_best_policy = metrics.find_min_dosage(true_best, all_policies)

    # The transformation matrix for Lasso
    G = tva.alpha_matrix(all_policies)

    # Simulation parameters and variables
    # samples_per_pol = [10, 100, 500, 1000, 5000]
    samples_per_pol = [5, 10, 25, 50, 100, 250, 500, 1000]
    # samples_per_pol = [5]
    num_sims = 100

    H = 15
    theta = 3.9
    reg = 1e-1

    # Simulation results data structure
    rashomon_list = []
    lasso_list = []
    ct_list = []

    #
    # Simulations
    #
    for n_per_pol in samples_per_pol:

        print(f"Number of samples: {n_per_pol}")

        for sim_i in range(num_sims):
            print(sim_i)

            if (sim_i + 1) % 20 == 0:
                print(f"\tSimulation {sim_i+1}")

            # Generate data
            X, D, y = generate_data(mu, var, n_per_pol, all_policies, pi_policies, M)
            policy_means = loss.compute_policy_means(D, y, num_policies)
            # The dummy matrix for Lasso
            # D_matrix = tva.get_dummy_matrix(D, G, num_policies)

            #
            # Run Rashomon
            #
            if n_per_pol == 5:
                if sim_i == 0 or sim_i == 3:
                    theta = 6.5
                else:
                    theta = 6.5
            elif n_per_pol == 10:
                theta = 4.2
            elif n_per_pol <= 50:
                theta = 4.4
            elif n_per_pol <= 250:
                theta = 4.3
            else:
                theta = 4.2

            # Adaptive expand R set threshold until we find a model that
            # identifies the true best profile
            found_best_profile = False
            counter = 0
            this_theta = theta

            while not found_best_profile:
                if (counter + 1) % 10 == 0:
                    print(f"\tSimulation {sim_i}. Tried {counter} times. Theta = {this_theta}")

                current_results = []
                R_set, rashomon_profiles = RAggregate(M, R, H, D, y, this_theta, reg)
                print(f"\t\t{this_theta},{len(R_set)}")

                for r_set in R_set:

                    # MSE
                    pi_policies_profiles_r = {}
                    for k, profile in enumerate(profiles):
                        _, pi_policies_r_k = extract_pools.extract_pools(
                            policies_profiles_masked[k],
                            rashomon_profiles[k].sigma[r_set[k]]
                        )
                        pi_policies_profiles_r[k] = pi_policies_r_k

                    pi_pools_r, pi_policies_r = extract_pools.aggregate_pools(
                        pi_policies_profiles_r, policies_ids_profiles)
                    pool_means_r = loss.compute_pool_means(policy_means, pi_pools_r)
                    y_r_est = metrics.make_predictions(D, pi_policies_r, pool_means_r)

                    r_set_results = metrics.compute_all_metrics(
                        y, y_r_est, D, true_best, all_policies, profile_map, min_dosage_best_policy, true_best_effect)
                    sqrd_err = r_set_results["sqrd_err"]
                    iou_r = r_set_results["iou"]
                    best_profile_indicator = r_set_results["best_prof"]
                    min_dosage_present = r_set_results["min_dos_inc"]
                    best_pol_diff = r_set_results["best_pol_diff"]

                    this_list = [n_per_pol, sim_i, len(pi_pools_r), sqrd_err, iou_r, min_dosage_present, best_pol_diff]
                    this_list += best_profile_indicator
                    current_results.append(this_list)

                    if best_profile_indicator[true_best_profile_idx] == 1:
                        found_best_profile = True
                if this_theta >= 6.5:
                    found_best_profile = True
                if len(R_set) > 1e5:
                    found_best_profile = True
                this_theta += 0.5
                counter += 1

            rashomon_list += current_results

    profiles_str = [str(prof) for prof in profiles]

    rashomon_cols = ["n_per_pol", "sim_num", "num_pools", "MSE", "IOU", "min_dosage", "best_pol_diff"]
    rashomon_cols += profiles_str
    rashomon_df = pd.DataFrame(rashomon_list, columns=rashomon_cols)

    rashomon_df.to_csv("../Results/simulation_4_rashomon.csv")
