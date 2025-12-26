import os
import numpy as np
import pandas as pd
import importlib

from rashomon import hasse
from copy import deepcopy
from rashomon import extract_pools
from rashomon import metrics

from reff_simulations import generate_data


def read_ppmx_sim_data(file_dir, sim_num):
    """
    Contents of file_dir:
        - fitted_vals_{sim_num}.csv
        - nclusters_{sim_num}.csv
        - posteriors_{sim_num}.csv
    Where sim_num is from 0 to num_sims-1
    """

    fitted_vals = pd.read_csv(file_dir + f"/fitted_vals_{sim_num}.csv", sep=";")
    fitted_vals = fitted_vals.drop(columns=["Unnamed: 0"])
    # Values are of format str'0,2456' -> float 0.2456
    fitted_vals = fitted_vals.map(lambda x: float(str(x).replace(',', '.'))).to_numpy()
    nclusters = pd.read_csv(file_dir + f"/nclusters_{sim_num}.csv", sep=";")
    nclusters = nclusters.drop(columns=["Unnamed: 0"])
    nclusters = nclusters.to_numpy().squeeze()
    posteriors = pd.read_csv(file_dir + f"/posteriors_{sim_num}.csv", sep=";")
    posteriors = posteriors.drop(columns=["Unnamed: 0"])
    posteriors = posteriors.map(lambda x: float(str(x).replace(',', '.'))).to_numpy()
    return fitted_vals, nclusters, posteriors


def read_ground_truth(file_dir, sim_num):
    """
    Contents of file_dir:
        - sim_data_30_{sim_num}.csv
    Where sim_num is from 0 to num_sims-1
    """

    ground_truth = pd.read_csv(file_dir + f"/sim_data_30_{sim_num}.csv", sep=",")
    X = ground_truth.drop(columns=["y"]).to_numpy()
    y = ground_truth["y"].to_numpy()
    return X, y


def setup_params(params_module_name="reff_4"):

    params = importlib.import_module(params_module_name, package=None)
    M = params.M
    R = params.R
    n_per_pol = 30
    sigma = params.sigma
    mu = params.mu
    var = params.var

    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)

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
    # print(true_best_profile)
    true_best_profile_idx = int(true_best_profile)
    true_best_effect = np.max(mu[true_best_profile])
    true_best = pi_pools[true_best_profile][np.argmax(mu[true_best_profile])]
    min_dosage_best_policy = metrics.find_min_dosage(true_best, all_policies)

    # The transformation matrix for Lasso
    G = hasse.alpha_matrix(all_policies)

    _, D, _ = generate_data(mu, var, n_per_pol, profiles, num_policies, all_policies, policies_profiles, pi_policies, M)
    D_matrix = hasse.get_dummy_matrix(D, G, num_policies)

    results = {
        "M": M,
        "R": R,
        "n_per_pol": n_per_pol,
        "all_policies": all_policies,
        "num_profiles": num_profiles,
        "profiles": profiles,
        "profile_map": profile_map,
        "D": D,
        "D_matrix": D_matrix,
        "true_best": true_best,
        "true_best_effect": true_best_effect,
        "true_best_profile_idx": true_best_profile_idx,
        "min_dosage_best_policy": min_dosage_best_policy,
    }

    return results


if __name__ == "__main__":

    raw_file_dir = "../Data/reff_sims"
    results_file_dir = "../Results/reff/ppmx"
    output_dir = "../Results/4arms/"
    verbose = True

    params = setup_params("reff_4")
    M = params["M"]
    R = params["R"]
    n_per_pol = params["n_per_pol"]
    all_policies = params["all_policies"]
    num_profiles = params["num_profiles"]
    profiles = params["profiles"]
    profile_map = params["profile_map"]
    D = params["D"]
    D_matrix = params["D_matrix"]
    true_best = params["true_best"]
    true_best_effect = params["true_best_effect"]
    true_best_profile_idx = params["true_best_profile_idx"]
    min_dosage_best_policy = params["min_dosage_best_policy"]

    profiles_str = [str(prof) for prof in profiles]

    ppmx_list = []
    num_sims = 100

    output_prefix = "4arms"
    output_suffix = f"_{n_per_pol}_{num_sims}"
    ppmx_fname = output_prefix + "_ppmx" + output_suffix + ".csv"

    for sim_i in range(num_sims):

        if verbose:
            print(f"Processing simulation {sim_i}...")

        X, y = read_ground_truth(raw_file_dir, sim_num=sim_i)
        fitted_vals, nclusters, posteriors = read_ppmx_sim_data(results_file_dir, sim_num=sim_i)

        niters = fitted_vals.shape[0]
        niters = 1000

        iou_coverage = metrics.compute_iou_coverage(None, D_matrix, D, true_best, y_pred=fitted_vals, n_samples=niters)
        min_dosage_coverage = metrics.compute_min_dosage_coverage(
            None, D_matrix, D, min_dosage_best_policy, y_pred=fitted_vals, n_samples=niters)

        profile_indicators_sum = np.zeros(len(profiles))

        for j in range(niters):
            y_fitted = fitted_vals[j, :]
            ncluster_j = nclusters[j]
            posterior_j = posteriors[j, 0]
            converged = None
            max_rhat = None
            acceptance_rate = None

            # Compute metrics for this sample
            sample_results = metrics.compute_all_metrics(
                y, y_fitted, D, true_best, all_policies, profile_map,
                min_dosage_best_policy, true_best_effect)

            sqrd_err_sample = sample_results["sqrd_err"]
            iou_sample = sample_results["iou"]
            profile_indicator_sample = sample_results["best_prof"]
            min_dosage_sample = sample_results["min_dos_inc"]
            best_pol_diff_sample = sample_results["best_pol_diff"]

            # Accumulate for average
            profile_indicators_sum += np.array(profile_indicator_sample)

            # Store individual sample results with summary metrics
            sample_list = [
                n_per_pol, sim_i, j,
                posterior_j,  # loss (negative log posterior)
                sqrd_err_sample,  # MSE component of loss
                iou_sample,
                min_dosage_sample,
                best_pol_diff_sample,
                converged,
                max_rhat,
                iou_coverage,
                min_dosage_coverage,
                ncluster_j,  # number of clusters for this sample
                acceptance_rate
            ]
            sample_list += profile_indicator_sample
            ppmx_list.append(sample_list)

    ppmx_cols = [
        "n_per_pol", "sim_num", "sample_idx",
        "neg_log_posterior", "MSE", "IOU", "min_dosage", "best_pol_diff",
        "converged", "max_rhat", "IOU_coverage", "min_dosage_coverage",
        "n_clusters", "acceptance_rate"
    ]
    ppmx_cols += profiles_str
    ppmx_df = pd.DataFrame(ppmx_list, columns=ppmx_cols)
    ppmx_df.to_csv(os.path.join(output_dir, ppmx_fname))
    if verbose:
        print(f"\nSaved PPMx results to {ppmx_fname}")
