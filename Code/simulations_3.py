import warnings
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn import linear_model

from rashomon import tva
from rashomon import loss
from rashomon import metrics
from rashomon import causal_trees
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
    M = 3
    R = np.array([4, 4, 4])

    num_profiles = 2**M
    profiles, profile_map = tva.enumerate_profiles(M)
    all_policies = tva.enumerate_policies(M, R)
    num_policies = len(all_policies)

    # Fix the partitions
    # Profile 0: (0, 0, 0)
    sigma_0 = None
    mu_0 = np.array([0])
    var_0 = np.array([1])

    # Profile 1: (0, 0, 1)
    sigma_1 = np.array([[1, 1]])
    mu_1 = np.array([1])
    var_1 = np.array([1])

    # Profile 2: (0, 1, 0)
    sigma_2 = np.array([[1, 0]])
    mu_2 = np.array([0, 3.8])
    var_2 = np.array([1, 1])

    # Profile 3: (0, 1, 1)
    sigma_3 = np.array([[1, 1],
                        [1, 1]])
    mu_3 = np.array([1.5])
    var_3 = np.array([1])

    # Profile 4: (1, 0, 0)
    sigma_4 = np.array([[1, 1]])
    mu_4 = np.array([1])
    var_4 = np.array([1])

    # Profile 5: (1, 0, 1)
    sigma_5 = np.array([[0, 1],
                        [1, 0]])
    mu_5 = np.array([3.6, 3.3, 3.5, 3.4])
    var_5 = np.array([2, 2, 1, 2])

    # Profile 6: (1, 1, 0)
    sigma_6 = np.array([[1, 1],
                        [1, 1]])
    mu_6 = np.array([2])
    var_6 = np.array([1])

    # Profile 1: (1, 1, 1)
    sigma_7 = np.array([[1, 1],
                        [1, 1],
                        [1, 1]])
    mu_7 = np.array([3])
    var_7 = np.array([1])

    sigma = [sigma_0, sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7]
    mu = [mu_0, mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7]
    var = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7]

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
    theta = 2.3
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
                rashomon_list.append(this_list)

            #
            # Run Lasso
            #
            lasso = linear_model.Lasso(reg, fit_intercept=False)
            lasso.fit(D_matrix, y)
            alpha_est = lasso.coef_
            y_tva = lasso.predict(D_matrix)

            tva_results = metrics.compute_all_metrics(
                y, y_tva, D, true_best, all_policies, profile_map, min_dosage_best_policy, true_best_effect)
            sqrd_err = tva_results["sqrd_err"]
            iou_tva = tva_results["iou"]
            best_profile_indicator_tva = tva_results["best_prof"]
            min_dosage_present_tva = tva_results["min_dos_inc"]
            best_policy_diff_tva = tva_results["best_pol_diff"]
            L1_loss = sqrd_err + reg * np.linalg.norm(alpha_est, ord=1)

            this_list = [n_per_pol, sim_i, sqrd_err, L1_loss, iou_tva, min_dosage_present_tva, best_policy_diff_tva]
            this_list += best_profile_indicator_tva
            lasso_list.append(this_list)

            #
            # Causal trees
            #
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice.")
                warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
                ct_res = causal_trees.ctl(M, R, D, y, D_matrix)
            y_ct = ct_res[3]

            ct_results = metrics.compute_all_metrics(
                y, y_ct, D, true_best, all_policies, profile_map, min_dosage_best_policy, true_best_effect)
            sqrd_err = ct_results["sqrd_err"]
            iou_ct = ct_results["iou"]
            best_profile_indicator_ct = ct_results["best_prof"]
            min_dosage_present_ct = ct_results["min_dos_inc"]
            best_policy_diff_ct = ct_results["best_pol_diff"]

            this_list = [n_per_pol, sim_i, sqrd_err, iou_ct, min_dosage_present_ct, best_policy_diff_ct]
            this_list += best_profile_indicator_ct
            ct_list.append(this_list)

    profiles_str = [str(prof) for prof in profiles]

    rashomon_cols = ["n_per_pol", "sim_num", "num_pools", "MSE", "IOU", "min_dosage", "best_pol_diff"]
    rashomon_cols += profiles_str
    rashomon_df = pd.DataFrame(rashomon_list, columns=rashomon_cols)

    lasso_cols = ["n_per_pol", "sim_num", "MSE", "L1_loss", "IOU", "min_dosage", "best_pol_diff"]
    lasso_cols += profiles_str
    lasso_df = pd.DataFrame(lasso_list, columns=lasso_cols)

    ct_cols = ["n_per_pol", "sim_num", "MSE", "IOU", "min_dosage", "best_pol_diff"]
    ct_cols += profiles_str
    ct_df = pd.DataFrame(ct_list, columns=ct_cols)

    rashomon_df.to_csv("../Results/simulation_3_rashomon.csv")
    lasso_df.to_csv("../Results/simulation_3_lasso.csv")
    ct_df.to_csv("../Results/simulation_3_causal_trees.csv")
