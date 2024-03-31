import os
import pickle
import argparse
import importlib
# import warnings
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from econml.grf import CausalForest

from rashomon import tva
from rashomon import loss
from rashomon import metrics
# from rashomon import causal_trees
from rashomon import extract_pools
from rashomon.aggregate import RAggregate, find_te_het_partitions


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
                        help="One of {r, lasso, cf}")
    args = parser.parse_args()
    return args


def generate_data(mu, var, n_per_pol, all_policies, pi_policies, profiles, policies_profiles, M):
    num_policies = len(all_policies)
    num_data = num_policies * n_per_pol
    X = np.zeros(shape=(num_data, M))
    D = np.zeros(shape=(num_data, 1), dtype='int_')
    y = np.zeros(shape=(num_data, 1)) + np.inf
    mu_true = np.zeros(shape=(num_data, 1))

    idx_ctr = 0
    for k, profile in enumerate(profiles):
        policies_k = policies_profiles[k]

        for idx, policy in enumerate(policies_k):
            policy_idx = [i for i, x in enumerate(all_policies) if x == policy]

            if pi_policies[k] is None and np.isnan(mu[k]):
                continue

            pool_id = pi_policies[k][idx]
            mu_i = mu[k][pool_id]
            var_i = var[k][pool_id]
            y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

            start_idx = idx_ctr * n_per_pol
            end_idx = (idx_ctr + 1) * n_per_pol

            X[start_idx:end_idx, ] = policy
            D[start_idx:end_idx, ] = policy_idx[0]
            y[start_idx:end_idx, ] = y_i
            mu_true[start_idx:end_idx, ] = mu_i

            idx_ctr += 1

    absent_idx = np.where(np.isinf(y))[0]
    X = np.delete(X, absent_idx, 0)
    y = np.delete(y, absent_idx, 0)
    D = np.delete(D, absent_idx, 0)
    mu_true = np.delete(mu_true, absent_idx, 0)

    return X, D, y, mu_true


if __name__ == "__main__":

    args = parse_arguments()

    # from sim_4_params import M, R, sigma, mu, var
    params_module_name = args.params
    params = importlib.import_module(params_module_name, package=None)
    M = params.M
    R = params.R
    sigma_tmp = params.sigma_tmp
    mu_tmp = params.mu_tmp
    var_tmp = params.var_tmp
    interested_profiles = params.interested_profiles
    H = params.H
    theta = params.theta
    reg = params.reg
    lasso_reg = params.lasso_reg

    num_profiles = 2**M
    profiles, profile_map = tva.enumerate_profiles(M)
    all_policies = tva.enumerate_policies(M, R)
    num_policies = len(all_policies)

    interested_profile_idx = []
    sigma = []
    mu = []
    var = []
    for k, profile in enumerate(profiles):
        sigma_k = None
        mu_k = np.nan
        var_k = np.nan
        for i, p in enumerate(interested_profiles):
            if p == profile:
                sigma_k = sigma_tmp[i]
                mu_k = mu_tmp[i]
                var_k = var_tmp[i]
                break
        sigma.append(sigma_k)
        mu.append(mu_k)
        var.append(var_k)

    # Simulation parameters and variables
    samples_per_pol = [args.sample_size]
    num_sims = args.iters

    # Output file names
    start_sim = 0
    output_dir = "../Results/TE/"
    # output_suffix = f"_{args.sample_size}_{args.iters}_{start_sim}.csv"
    output_suffix = f"_{args.sample_size}_{args.iters}"
    rashomon_fname = args.output_prefix + "_rashomon" + output_suffix + ".csv"
    lasso_fname = args.output_prefix + "_lasso" + output_suffix + ".csv"
    cf_fname = args.output_prefix + "_cf" + output_suffix + ".csv"
    rashomon_conf_matrix_fname = args.output_prefix + "_rashomon" + output_suffix + "_conf_mat.pkl"
    cf_conf_matrix_fname = args.output_prefix + "_cf" + output_suffix + "_conf_mat.pkl"
    lasso_conf_matrix_fname = args.output_prefix + "_lasso" + output_suffix + "_conf_mat.pkl"

    # Simulation results data structure
    method = args.method
    if method not in ["r", "lasso", "cf"]:
        print(f"method should be one of [r, lasso, cf]. Received {method}. Defaulting to r")
        method = "r"
    rashomon_list = []
    lasso_list = []
    cf_list = []
    conf_matrices = []

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

        if sigma[k] is None:
            pi_policies[k] = None
            pi_pools[k] = None
            continue

        if np.sum(profile) > 0:
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(policies_k, sigma[k])
            if len(pi_pools_k.keys()) != mu[k].shape[0]:
                print(pi_pools_k)
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
    true_best_profile = np.nanargmax(best_per_profile)
    true_best_profile_idx = int(true_best_profile)
    true_best_effect = np.max(mu[true_best_profile])
    true_best = pi_pools[true_best_profile][np.argmax(mu[true_best_profile])]
    min_dosage_best_policy = metrics.find_min_dosage(true_best, all_policies)

    # The transformation matrix for Lasso
    G = tva.alpha_matrix(all_policies)

    # True TE
    # trt_arm_idx = 0

    # ctl_profile_idx = 7
    # trt_profile_idx = 15

    # ctl_profile = (0, 1, 1, 1)
    # trt_profile = (1, 1, 1, 1)

    trt_arm_idx = params.trt_arm_idx

    ctl_profile_idx = params.ctl_profile_idx
    trt_profile_idx = params.trt_profile_idx

    ctl_profile = params.ctl_profile
    trt_profile = params.trt_profile

    # Subset data for interested profiles
    trt_policies_ids = policies_ids_profiles[trt_profile_idx]
    ctl_policies_ids = policies_ids_profiles[ctl_profile_idx]
    tc_policies_ids = trt_policies_ids + ctl_policies_ids

    trt_policies = policies_profiles_masked[trt_profile_idx]
    ctl_policies = policies_profiles_masked[ctl_profile_idx]

    trt_pools, trt_pools_policies = extract_pools.extract_pools(trt_policies, sigma[trt_profile_idx])
    ctl_pools, ctl_pools_policies = extract_pools.extract_pools(ctl_policies, sigma[ctl_profile_idx])

    D_trt = np.array(list(trt_pools_policies.keys()))
    D_ctl = np.array(list(ctl_pools_policies.keys()))

    D_trt_pooled = [trt_pools_policies[pol_id] for pol_id in D_trt]
    D_ctl_pooled = [ctl_pools_policies[pol_id] for pol_id in D_ctl]
    y_trt = mu[trt_profile_idx][D_trt_pooled]
    y_ctl = mu[ctl_profile_idx][D_ctl_pooled]

    X_trt = np.array(policies_profiles[trt_profile_idx])[:, 1:]

    te_true = y_trt - y_ctl
    max_te = np.max(te_true)
    max_te_policies_p = D_trt[np.where(te_true == max_te)]
    max_te_policies = [policies_ids_profiles[trt_profile_idx][x] for x in max_te_policies_p]
    min_dosage_best_te = metrics.find_min_dosage(max_te_policies, all_policies)

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
            X, D, y, mu_true = generate_data(mu, var, n_per_pol, all_policies,
                                             pi_policies, profiles, policies_profiles, M)
            policy_means = loss.compute_policy_means(D, y, num_policies)
            # The dummy matrix for Lasso
            D_matrix = tva.get_dummy_matrix(D, G, num_policies)

            # trt_idx = 0
            feature_idx = list(np.arange(0, trt_arm_idx)) + list(np.arange(trt_arm_idx+1, M))

            T = np.zeros(shape=y.shape)
            T[X[:, trt_arm_idx] > 0] = 1

            y_0d = y.reshape((-1,))
            X_cf = X[:, feature_idx]

            X_trt_subset = X[X[:, trt_arm_idx] > 0, :]
            X_trt_subset = X_trt_subset[:, feature_idx]
            y_trt_subset = y[X[:, trt_arm_idx] > 0]

            D_trt_subset = D[X[:, trt_arm_idx] > 0]
            D_matrix_trt_subset = D_matrix[X[:, trt_arm_idx] > 0, :]

            D_trt_univ = np.array([policies_ids_profiles[trt_profile_idx][x] for x in D_trt]).reshape((-1, 1))
            D_ctl_univ = np.array([policies_ids_profiles[ctl_profile_idx][x] for x in D_ctl]).reshape((-1, 1))
            D_matrix_trt = tva.get_dummy_matrix(D_trt_univ, G, num_policies)
            D_matrix_ctl = tva.get_dummy_matrix(D_ctl_univ, G, num_policies)

            mask = np.isin(D, tc_policies_ids)
            D_tc = D[mask].reshape((-1, 1))
            y_tc = y[mask].reshape((-1, 1))

            #
            # Run Rashomon
            #
            if method == "r":
                R_set, rashomon_profiles = RAggregate(M, R, H, D, y, theta, reg, verbose=True)
                print(len(R_set))

                te_partitions = []
                num_models = 0
                for R_set_idx, R_set_i in enumerate(R_set):

                    print(f"Looking at combination {R_set_idx+1} out of {len(R_set)} combinations")

                    # Get treatment and control partitions
                    sigma_trt_R_set_idx = R_set_i[trt_profile_idx]
                    sigma_trt_i = rashomon_profiles[trt_profile_idx].sigma[sigma_trt_R_set_idx]
                    sigma_ctl_R_set_idx = R_set_i[ctl_profile_idx]
                    sigma_ctl_i = rashomon_profiles[ctl_profile_idx].sigma[sigma_ctl_R_set_idx]

                    P_qe = find_te_het_partitions(
                        sigma_trt_i, sigma_ctl_i, trt_profile_idx, ctl_profile_idx, trt_policies, ctl_policies,
                        trt_arm_idx, all_policies, policies_ids_profiles,
                        D_tc, y_tc, policy_means, theta, reg
                    )
                    num_models += P_qe.size

                    te_partitions.append(P_qe)

                print(f"Found {num_models} after TE partitioning")

                conf_matrix_sim_i = []
                for idx, r_set in enumerate(R_set):
                    conf_matrix_list_idx = []

                    sigma_trt_R_set_idx = r_set[trt_profile_idx]
                    sigma_trt_i = rashomon_profiles[trt_profile_idx].sigma[sigma_trt_R_set_idx]
                    sigma_ctl_R_set_idx = r_set[ctl_profile_idx]
                    sigma_ctl_i = rashomon_profiles[ctl_profile_idx].sigma[sigma_ctl_R_set_idx]

                    trt_pools_0, _ = extract_pools.extract_pools(trt_policies, sigma_trt_i)
                    ctl_pools_0, _ = extract_pools.extract_pools(ctl_policies, sigma_ctl_i)

                    for (ti, ci) in zip(trt_policies, ctl_policies):
                        ti = tuple(ti[:trt_arm_idx] + ti[(trt_arm_idx+1):])
                        if ti != ci:
                            raise RuntimeError("Treatment and control pairs do not match!")

                    trt_pools = tva.profile_ids_to_univ_ids(trt_pools_0, trt_policies_ids)
                    ctl_pools = tva.profile_ids_to_univ_ids(ctl_pools_0, ctl_policies_ids)

                    P_qe_idx = te_partitions[idx]

                    for te_pool_id, sigma_int in enumerate(P_qe_idx.sigma):
                        sigma_pools, sigma_policies = extract_pools.get_trt_ctl_pooled_partition(
                            trt_pools, ctl_pools, sigma_int
                        )
                        mu_pools = loss.compute_pool_means(policy_means, sigma_pools)
                        D_tc_pool = [sigma_policies[pol_id] for pol_id in D_tc[:, 0]]
                        mu_D = mu_pools[D_tc_pool]

                        # Find TE
                        D_trt_pooled_i = [sigma_policies[pol_id] for pol_id in trt_policies_ids]
                        D_ctl_pooled_i = [sigma_policies[pol_id] for pol_id in ctl_policies_ids]
                        y_trt_i = mu_pools[D_trt_pooled_i]
                        y_ctl_i = mu_pools[D_ctl_pooled_i]

                        te_i = y_trt_i - y_ctl_i

                        metrics_results_i = metrics.compute_te_het_metrics(
                            te_true, te_i,
                            max_te, max_te_policies,
                            D_trt, policies_ids_profiles[trt_profile_idx]
                        )
                        mse_te_i = metrics_results_i["mse_te"]
                        max_te_err_i = metrics_results_i["max_te_err"]
                        iou_i = metrics_results_i["iou"]
                        conf_mat_i = metrics_results_i["conf_matrix"]

                        # Compute overall MSE
                        mse_i = mean_squared_error(y_tc[:, 0], mu_D)

                        # Count number of pools
                        num_pools_i = len(sigma_pools.keys())

                        results_i = [
                            n_per_pol, sim_i, idx, te_pool_id,
                            mse_te_i, max_te_err_i, iou_i,
                            mse_i, num_pools_i
                        ]
                        rashomon_list.append(results_i)

                        conf_matrix_list_idx.append(conf_mat_i)

                    conf_matrix_sim_i.append(conf_matrix_list_idx)

                conf_matrices.append(conf_matrix_sim_i)

            #
            # Run Lasso
            #
            if method == "lasso":
                lasso = linear_model.Lasso(lasso_reg, fit_intercept=False)
                lasso.fit(D_matrix, y)
                alpha_est = lasso.coef_
                y_trt_lasso = lasso.predict(D_matrix_trt)
                y_ctl_lasso = lasso.predict(D_matrix_ctl)

                te_lasso = y_trt_lasso - y_ctl_lasso

                tva_results = metrics.compute_te_het_metrics(
                    te_true, te_lasso,
                    max_te, max_te_policies,
                    D_trt, policies_ids_profiles[trt_profile_idx]
                    )

                mse_te_i = tva_results["mse_te"]
                max_te_err_i = tva_results["max_te_err"]
                iou_i = tva_results["iou"]
                conf_mat_i = tva_results["conf_matrix"]

                this_list = [n_per_pol, sim_i, mse_te_i, max_te_err_i, iou_i]
                lasso_list.append(this_list)

                conf_matrices.append(conf_mat_i)

            #
            # Causal forest
            #
            if method == "cf":
                cf_est = CausalForest(
                    criterion="mse", n_estimators=100,
                    min_samples_leaf=1,
                    # max_depth=None,
                    # min_samples_split=2,
                    random_state=3,
                    # fit_intercept=False
                    )

                cf_est.fit(X_cf, T, y_0d)
                te_cf = cf_est.predict(X_trt)

                cf_res = metrics.compute_te_het_metrics(
                    te_true, te_cf, max_te, max_te_policies,
                    D_trt, policies_ids_profiles[trt_profile_idx]
                    )
                mse_te_i = cf_res["mse_te"]
                max_te_err_i = cf_res["max_te_err"]
                iou_i = cf_res["iou"]
                conf_mat_i = cf_res["conf_matrix"]

                results_i = [
                    n_per_pol, sim_i,
                    mse_te_i, max_te_err_i, iou_i,
                ]
                cf_list.append(results_i)

                conf_matrices.append(conf_mat_i)

    if method == "r":
        rashomon_cols = [
            "n_per_pol", "sim_num", "idx", "te_idx",
            "MSE_TE", "max_te_diff", "IOU", "MSE", "num_pools"
        ]
        rashomon_df = pd.DataFrame(rashomon_list, columns=rashomon_cols)
        rashomon_df.to_csv(os.path.join(output_dir, rashomon_fname))

        with open(os.path.join(output_dir, rashomon_conf_matrix_fname), "wb") as f:
            pickle.dump(conf_matrices, f, pickle.HIGHEST_PROTOCOL)

    if method == "lasso":
        lasso_cols = ["n_per_pol", "sim_num", "MSE_TE", "max_te_diff", "IOU"]
        lasso_df = pd.DataFrame(lasso_list, columns=lasso_cols)
        lasso_df.to_csv(os.path.join(output_dir, lasso_fname))

        with open(os.path.join(output_dir, lasso_conf_matrix_fname), "wb") as f:
            pickle.dump(conf_matrices, f, pickle.HIGHEST_PROTOCOL)

    if method == "cf":
        cf_cols = ["n_per_pol", "sim_num", "MSE_TE", "max_te_diff", "IOU"]
        cf_df = pd.DataFrame(cf_list, columns=cf_cols)
        cf_df.to_csv(os.path.join(output_dir, cf_fname))

        with open(os.path.join(output_dir, cf_conf_matrix_fname), "wb") as f:
            pickle.dump(conf_matrices, f, pickle.HIGHEST_PROTOCOL)
