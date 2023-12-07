import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from rashomon import tva
from rashomon import loss
from rashomon import metrics
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

    np.random.seed(3)

    #
    # Fix ground truth
    #
    M = 2
    R = np.array([5, 5])

    num_profiles = 2**M
    profiles, profile_map = tva.enumerate_profiles(M)
    all_policies = tva.enumerate_policies(M, R)
    num_policies = len(all_policies)

    # Fix the partitions
    # Profile 0: (0, 0)
    sigma_0 = None
    mu_0 = np.array([0])

    # Profile 1: (0, 1)
    sigma_1 = np.array([[1, 1, 0]])
    mu_1 = np.array([0, 3.5])

    # Profile 2: (1, 0)
    sigma_2 = np.array([[0, 0, 0]])
    mu_2 = np.array([3.2, 3.4, 3.3, 3.4])

    # Profile 3: (1, 1)
    sigma_3 = np.array([[1, 1, 1],
                        [1, 1, 1]])
    mu_3 = np.array([2])

    # Set data parameters
    sigma = [sigma_0, sigma_1, sigma_2, sigma_3]
    mu = [mu_0, mu_1, mu_2, mu_3]
    var = [[1], [1, 1], [2, 2, 2, 2], [1, 1, 1, 1]]

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
    true_best_effect = np.max(mu[true_best_profile])
    true_best = pi_pools[true_best_profile][np.argmax(mu[true_best_profile])]
    min_dosage_best_policy = metrics.find_min_dosage(true_best, all_policies)

    # The transformation matrix for Lasso
    G = tva.alpha_matrix(all_policies)

    # Simulation parameters and variables
    samples_per_pol = [10, 100, 1000, 5000]
    num_sims = 100

    H = 15
    theta = 2.5
    reg = 1e-1

    # Simulation results data structure
    rashomon_list = []
    lasso_list = []

    #
    # Simulations
    #
    for n_per_pol in samples_per_pol:

        print(f"Number of samples: {n_per_pol}")

        for sim_i in range(num_sims):

            if (sim_i + 1) % 20 == 0:
                print(f"\tSimulation {sim_i+1}")

            # Generate data
            X, D, y = generate_data(mu, var, n_per_pol, all_policies, pi_policies, M)
            policy_means = loss.compute_policy_means(D, y, num_policies)
            # The dummy matrix for Lasso
            D_matrix = tva.get_dummy_matrix(D, G, num_policies)

            #
            # Run Rashomon
            #
            R_set, rashomon_profiles = RAggregate(M, R, H, D, y, theta, reg)

            for r_set in R_set:

                # MSE
                pi_policies_profiles_r = {}
                for k, profile in enumerate(profiles):
                    _, pi_policies_r_k = extract_pools.extract_pools(
                        policies_profiles_masked[k],
                        rashomon_profiles[k].sigma[r_set[k]]
                    )
                    pi_policies_profiles_r[k] = pi_policies_r_k

                pi_pools_r, pi_policies_r = extract_pools.aggregate_pools(pi_policies_profiles_r, policies_ids_profiles)
                pool_means_r = loss.compute_pool_means(policy_means, pi_pools_r)
                y_pred = metrics.make_predictions(D, pi_policies_r, pool_means_r)

                sqrd_err = mean_squared_error(y, y_pred)

                # IOU
                rash_max = metrics.find_best_policies(D, y_pred)
                iou = metrics.intersect_over_union(set(true_best), set(rash_max))

                # Profiles
                best_profile_indicator = metrics.find_profiles(rash_max, all_policies, profile_map)

                # Min dosage membership
                min_dosage_present = metrics.check_membership(min_dosage_best_policy, rash_max)

                # Best policy difference
                best_pol_diff = true_best_effect - np.max(pool_means_r)

                this_list = [n_per_pol, sim_i, len(pi_pools_r), sqrd_err, iou, min_dosage_present, best_pol_diff]
                this_list += best_profile_indicator
                rashomon_list.append(this_list)

            #
            # Run Lasso
            #
            lasso = linear_model.Lasso(reg, fit_intercept=False)
            lasso.fit(D_matrix, y)
            alpha_est = lasso.coef_
            y_tva = lasso.predict(D_matrix)

            # MSE
            sqrd_err = mean_squared_error(y_tva, y)
            L1_loss = sqrd_err + reg * np.linalg.norm(alpha_est, ord=1)

            # IOU
            tva_best = metrics.find_best_policies(D, y_tva)
            iou_tva = metrics.intersect_over_union(set(true_best), set(tva_best))

            # Profiles
            best_profile_indicator_tva = metrics.find_profiles(tva_best, all_policies, profile_map)

            # Min dosage inclusion
            min_dosage_present_tva = metrics.check_membership(min_dosage_best_policy, tva_best)

            # Best policy MSE
            best_policy_error_tva = true_best_effect - np.max(y_tva)

            this_list = [n_per_pol, sim_i, sqrd_err, L1_loss, iou_tva, min_dosage_present_tva, best_policy_error_tva]
            this_list += best_profile_indicator_tva
            lasso_list.append(this_list)

    rashomon_cols = ["n_per_pol", "sim_num", "num_pools", "MSE", "IOU", "min_dosage", "best_pol_diff"]
    rashomon_cols += [str(prof) for prof in profiles]
    rashomon_df = pd.DataFrame(rashomon_list, columns=rashomon_cols)

    lasso_cols = ["n_per_pol", "sim_num", "MSE", "L1_loss", "IOU", "min_dosage", "best_pol_diff"]
    lasso_cols += [str(prof) for prof in profiles]
    lasso_df = pd.DataFrame(lasso_list, columns=lasso_cols)

    rashomon_df.to_csv("../Results/simulation_3_rashomon.csv")
    lasso_df.to_csv("../Results/simulation_3_lasso.csv")
