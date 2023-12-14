import warnings
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from rashomon import loss
from rashomon import tva
from rashomon import metrics
from rashomon import causal_trees
from rashomon.aggregate import RAggregate_profile
from rashomon.extract_pools import extract_pools


def generate_data(mu, var, n_per_pol, policies, pi_policies, M):
    num_data = len(policies) * n_per_pol
    X = np.ndarray(shape=(num_data, M))
    D = np.ndarray(shape=(num_data, 1), dtype='int_')
    y = np.ndarray(shape=(num_data, 1))

    for idx, policy in enumerate(policies):
        pool_i = pi_policies[idx]
        mu_i = mu[pool_i]
        var_i = var[pool_i]
        y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

        start_idx = idx * n_per_pol
        end_idx = (idx + 1) * n_per_pol

        X[start_idx:end_idx, ] = policy
        D[start_idx:end_idx, ] = idx
        y[start_idx:end_idx, ] = y_i

    return X, D, y


if __name__ == "__main__":

    np.random.seed(3)

    #
    # Fix ground truth
    #
    sigma = np.array([[1, 1, 0],
                      [0, 1, 0]], dtype='float64')
    sigma_profile = (1, 1)

    M, _ = sigma.shape
    R = np.array([5, 5])

    # Enumerate policies and find pools
    num_policies = np.prod(R-1)
    profiles, profile_map = tva.enumerate_profiles(M)
    all_policies = tva.enumerate_policies(M, R)
    policies = [x for x in all_policies if tva.policy_to_profile(x) == sigma_profile]
    pi_pools, pi_policies = extract_pools(policies, sigma)
    num_pools = len(pi_pools)

    # The transformation matrix for Lasso
    G = tva.alpha_matrix(policies)

    # Set data parameters
    mu = np.array([0, 1.5, 3, 3, 6, 4.5])
    se = 1
    var = se * np.ones_like(mu)

    true_best = pi_pools[np.argmax(mu)]
    true_best_effect = np.max(mu)
    min_dosage_best_policy = metrics.find_min_dosage(true_best, policies)

    # Simulation parameters and variables
    samples_per_pol = [10, 100, 1000, 5000]
    num_sims = 100

    H = 10
    theta = 2
    reg = 0.1

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
            X, D, y = generate_data(mu, var, n_per_pol, policies, pi_policies, M)
            # The dummy matrix for Lasso
            D_matrix = tva.get_dummy_matrix(D, G, num_policies)
            pol_means = loss.compute_policy_means(D, y, num_policies)

            #
            # Run Rashomon
            #
            P_set = RAggregate_profile(M, R, H, D, y, theta, sigma_profile, reg)
            if not P_set.seen(sigma):
                print("P_set missing true sigma")

            for s_i in P_set:
                pi_pools_i, pi_policies_i = extract_pools(policies, s_i)
                pool_means_i = loss.compute_pool_means(pol_means, pi_pools_i)

                Q = loss.compute_Q(D, y, s_i, policies, pol_means, reg=0.1)
                y_pred = metrics.make_predictions(D, pi_policies_i, pool_means_i)
                sqrd_err = mean_squared_error(y, y_pred)

                # IOU
                pol_max = metrics.find_best_policies(D, y_pred)
                iou = metrics.intersect_over_union(set(true_best), set(pol_max))

                # Min dosage membership
                min_dosage_present = metrics.check_membership(min_dosage_best_policy, pol_max)

                # Best policy difference
                best_pol_diff = np.max(mu) - np.max(pool_means_i)

                this_list = [n_per_pol, sim_i, len(pi_pools_i), sqrd_err, iou, min_dosage_present, best_pol_diff]
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

            # Min dosage inclusion
            min_dosage_present_tva = metrics.check_membership(min_dosage_best_policy, tva_best)

            # Best policy MSE
            best_policy_error_tva = np.max(mu) - np.max(y_tva)

            this_list = [n_per_pol, sim_i, sqrd_err, L1_loss, iou_tva, min_dosage_present_tva, best_policy_error_tva]
            lasso_list.append(this_list)

            #
            # Causal trees
            #
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice.")
                warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
                ct_res = causal_trees.ctl_single_profile(D, y, D_matrix)
            y_ct = ct_res[3]

            ct_results = metrics.compute_all_metrics(
                y, y_ct, D, true_best, all_policies, profile_map, min_dosage_best_policy, true_best_effect)
            sqrd_err = ct_results["sqrd_err"]
            iou_ct = ct_results["iou"]
            min_dosage_present_ct = ct_results["min_dos_inc"]
            best_policy_diff_ct = ct_results["best_pol_diff"]

            this_list = [n_per_pol, sim_i, sqrd_err, iou_ct, min_dosage_present_ct, best_policy_diff_ct]
            ct_list.append(this_list)

    rashomon_cols = ["n_per_pol", "sim_num", "num_pools", "MSE", "IOU", "min_dosage", "best_pol_diff"]
    rashomon_df = pd.DataFrame(rashomon_list, columns=rashomon_cols)

    lasso_cols = ["n_per_pol", "sim_num", "MSE", "L1_loss", "IOU", "min_dosage", "best_pol_diff"]
    lasso_df = pd.DataFrame(lasso_list, columns=lasso_cols)

    ct_cols = ["n_per_pol", "sim_num", "MSE", "IOU", "min_dosage", "best_pol_diff"]
    ct_df = pd.DataFrame(ct_list, columns=ct_cols)

    rashomon_df.to_csv("../Results/worst_case_rashomon.csv")
    lasso_df.to_csv("../Results/worst_case_lasso.csv")
    ct_df.to_csv("../Results/worst_case_causal_trees.csv")
