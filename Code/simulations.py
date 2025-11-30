import os
import argparse
import importlib
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn import linear_model

from rashomon import hasse
from rashomon import loss
from rashomon import metrics
from rashomon import extract_pools
from rashomon.aggregate import RAggregate
from baselines import BayesianLasso
from baselines import BootstrapLasso
from baselines import PPMx


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
    parser.add_argument("--method", type=str,
                        help="One of {r, lasso, blasso, bootstrap, ppmx}")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with reduced iterations (5 iterations, reduced MCMC/bootstrap samples)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress information (default: True)"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
        help="Disable progress printing"
    )
    args = parser.parse_args()
    return args


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

            pool_id = pi_policies[k][idx]
            mu_i = mu[k][pool_id]
            var_i = var[k][pool_id]
            y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

            start_idx = idx_ctr * n_per_pol
            end_idx = (idx_ctr + 1) * n_per_pol

            X[start_idx:end_idx, ] = policy
            D[start_idx:end_idx, ] = policy_idx[0]
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
    mu = params.mu
    var = params.var
    H = params.H
    theta = params.theta
    reg = params.reg
    lasso_reg = params.lasso_reg

    # Bayesian Lasso parameters (with defaults)
    if args.test:
        blasso_n_iter = 1000
        blasso_burnin = 200
        blasso_thin = 2
        blasso_n_chains = 3
        blasso_lambda = getattr(params, 'blasso_lambda', 1.0)
        blasso_tau2_a = getattr(params, 'blasso_tau2_a', 1.0)
        blasso_tau2_b = getattr(params, 'blasso_tau2_b', 1.0)
    else:
        blasso_n_iter = getattr(params, 'blasso_n_iter', 2000)
        blasso_burnin = getattr(params, 'blasso_burnin', 500)
        blasso_thin = getattr(params, 'blasso_thin', 2)
        blasso_n_chains = getattr(params, 'blasso_n_chains', 4)
        blasso_lambda = getattr(params, 'blasso_lambda', 1.0)
        blasso_tau2_a = getattr(params, 'blasso_tau2_a', 1.0)
        blasso_tau2_b = getattr(params, 'blasso_tau2_b', 1.0)

    # Bootstrap Lasso parameters (with defaults)
    if args.test:
        bootstrap_n_iter = 100
        bootstrap_alpha = getattr(params, 'bootstrap_alpha', 1.0)
        bootstrap_confidence_level = getattr(params, 'bootstrap_confidence_level', 0.95)
        bootstrap_random_state = getattr(params, 'bootstrap_random_state', None)
    else:
        bootstrap_n_iter = getattr(params, 'bootstrap_n_iter', 1000)
        bootstrap_alpha = getattr(params, 'bootstrap_alpha', 1.0)
        bootstrap_confidence_level = getattr(params, 'bootstrap_confidence_level', 0.95)
        bootstrap_random_state = getattr(params, 'bootstrap_random_state', None)

    # PPMx parameters (with defaults)
    if args.test:
        ppmx_n_iter = 1000
        ppmx_burnin = 200
        ppmx_thin = 2
        ppmx_alpha = getattr(params, 'ppmx_alpha', 1.0)
        ppmx_cohesion = getattr(params, 'ppmx_cohesion', 'gaussian')
        ppmx_similarity_weight = getattr(params, 'ppmx_similarity_weight', 0.5)
        ppmx_similarity_bandwidth = getattr(params, 'ppmx_similarity_bandwidth', 1.0)
    else:
        ppmx_n_iter = getattr(params, 'ppmx_n_iter', 5000)
        ppmx_burnin = getattr(params, 'ppmx_burnin', 1000)
        ppmx_thin = getattr(params, 'ppmx_thin', 2)
        ppmx_alpha = getattr(params, 'ppmx_alpha', 1.0)
        ppmx_cohesion = getattr(params, 'ppmx_cohesion', 'gaussian')
        ppmx_similarity_weight = getattr(params, 'ppmx_similarity_weight', 0.5)
        ppmx_similarity_bandwidth = getattr(params, 'ppmx_similarity_bandwidth', 1.0)

    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)

    # Simulation parameters and variables
    if args.test:
        # Override for test mode
        samples_per_pol = [args.sample_size] if args.sample_size else [10]
        num_sims = 5
        verbose = args.verbose
        if verbose:
            print("Running in TEST mode: 5 iterations, reduced MCMC/bootstrap samples")
    else:
        samples_per_pol = [args.sample_size]
        num_sims = args.iters
        verbose = args.verbose

    # Output file names
    start_sim = 0
    output_dir = "../Results/4arms/"
    # output_suffix = f"_{args.sample_size}_{args.iters}_{start_sim}.csv"
    output_suffix = f"_{samples_per_pol[0]}_{num_sims}"
    if args.test:
        output_suffix += "_test"
    output_suffix += ".csv"
    rashomon_fname = args.output_prefix + "_rashomon" + output_suffix
    lasso_fname = args.output_prefix + "_lasso" + output_suffix
    blasso_fname = args.output_prefix + "_blasso" + output_suffix
    bootstrap_fname = args.output_prefix + "_bootstrap" + output_suffix
    ppmx_fname = args.output_prefix + "_ppmx" + output_suffix

    if verbose:
        print(f"Method to run: {args.method}")
        print(f"Sample size per policy: {samples_per_pol[0]}")
        print(f"Number of iterations: {num_sims}")

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

    # Simulation results data structure
    method = args.method
    if method not in ["r", "lasso", "blasso", "bootstrap", "ppmx"]:
        print(f"method should be one of [r, lasso, blasso, bootstrap, ppmx]. Received {method}. Defaulting to r")
        method = "r"
    rashomon_list = []
    lasso_list = []
    blasso_list = []
    bootstrap_list = []
    ppmx_list = []

    np.random.seed(3)

    #
    # Simulations
    #
    for n_per_pol in samples_per_pol:

        if verbose:
            print(f"\nNumber of samples per policy: {n_per_pol}")

        for sim_i in range(start_sim, start_sim + num_sims):
            if verbose:
                print(f"Simulation {sim_i}")
            np.random.seed(sim_i)

            if verbose and (sim_i + 1) % 20 == 0:
                print(f"\tSimulation {sim_i+1}")

            # Generate data
            X, D, y = generate_data(mu, var, n_per_pol, all_policies, pi_policies, M)
            policy_means = loss.compute_policy_means(D, y, num_policies)
            # The dummy matrix for Lasso
            D_matrix = hasse.get_dummy_matrix(D, G, num_policies)

            #
            # Run Rashomon
            #
            if method == "r":
                # if n_per_pol == 5:
                #     if sim_i == 0 or sim_i == 3:
                #         theta = 6.5
                #     else:
                #         theta = 6.5
                # elif n_per_pol == 10:
                #     theta = 4.2
                # elif n_per_pol <= 50:
                #     theta = 4.4
                # elif n_per_pol <= 250:
                #     theta = 4.3
                # else:
                #     theta = 4.2

                # Adaptive expand R set threshold until we find a model that
                # identifies the true best profile
                found_best_profile = False
                counter = 0
                this_theta = theta

                while not found_best_profile:
                    if verbose and (counter + 1) % 10 == 0:
                        print(f"\tSimulation {sim_i}. Tried {counter} times. Theta = {this_theta}")

                    current_results = []
                    R_set, rashomon_profiles = RAggregate(M, R, H, D, y, this_theta, reg)
                    if verbose:
                        print(f"\t\t{this_theta},{len(R_set)}")

                    best_loss = np.inf

                    for idx, r_set in enumerate(R_set):
                        # print(idx)

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
                            y, y_r_est, D, true_best, all_policies, profile_map,
                            min_dosage_best_policy, true_best_effect)
                        sqrd_err = r_set_results["sqrd_err"]
                        iou_r = r_set_results["iou"]
                        best_profile_indicator = r_set_results["best_prof"]
                        min_dosage_present = r_set_results["min_dos_inc"]
                        best_pol_diff = r_set_results["best_pol_diff"]
                        this_loss = sqrd_err + reg * len(pi_pools_r)

                        this_list = [
                            n_per_pol, sim_i, len(pi_pools_r), sqrd_err, iou_r, min_dosage_present, best_pol_diff
                            ]
                        this_list += best_profile_indicator
                        current_results.append(this_list)

                        if this_loss < best_loss:
                            best_loss = this_loss

                        if best_profile_indicator[true_best_profile_idx] == 1:
                            found_best_profile = True
                            # print("Found", this_loss)

                    if found_best_profile and verbose:
                        print("\tFound best profile")

                    eps = 0.2
                    eps_factor = 1 + eps
                    if np.isinf(best_loss):
                        best_loss = this_theta
                    if found_best_profile or this_theta >= (eps_factor * best_loss):
                        found_best_profile = True
                    else:
                        # this_theta = eps_factor * best_loss
                        this_theta += 0.1
                        found_best_profile = False
                    counter += 1

                    # if this_theta >= 6.5:
                    #     found_best_profile = True
                    # if len(R_set) > 1e5:
                    # if len(R_set) >= 8377:
                    #     found_best_profile = True
                    # this_theta += 0.5
                    # break

                rashomon_list += current_results

            #
            # Run Lasso
            #
            if method == "lasso":
                lasso = linear_model.Lasso(lasso_reg, fit_intercept=False)
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
            # Run Bayesian Lasso
            #
            if method == "blasso":
                blasso = BayesianLasso(
                    n_iter=blasso_n_iter,
                    burnin=blasso_burnin,
                    thin=blasso_thin,
                    lambda_prior=blasso_lambda,
                    tau2_a=blasso_tau2_a,
                    tau2_b=blasso_tau2_b,
                    random_state=sim_i,
                    verbose=False
                )
                blasso.fit(D_matrix, y, n_chains=blasso_n_chains)

                # Predictions using posterior mean
                y_blasso = blasso.predict(D_matrix)
                # Predictions using MAP estimate
                y_blasso_map = blasso.predict_map(D_matrix)

                # Compute metrics for posterior mean
                blasso_results = metrics.compute_all_metrics(
                    y, y_blasso, D, true_best, all_policies, profile_map,
                    min_dosage_best_policy, true_best_effect)
                sqrd_err_blasso = blasso_results["sqrd_err"]
                iou_blasso = blasso_results["iou"]
                best_profile_indicator_blasso = blasso_results["best_prof"]
                min_dosage_present_blasso = blasso_results["min_dos_inc"]
                best_policy_diff_blasso = blasso_results["best_pol_diff"]
                
                # Compute metrics for MAP estimate
                blasso_map_results = metrics.compute_all_metrics(
                    y, y_blasso_map, D, true_best, all_policies, profile_map,
                    min_dosage_best_policy, true_best_effect)
                sqrd_err_blasso_map = blasso_map_results["sqrd_err"]
                iou_blasso_map = blasso_map_results["iou"]
                best_profile_indicator_blasso_map = blasso_map_results["best_prof"]
                min_dosage_present_blasso_map = blasso_map_results["min_dos_inc"]
                best_policy_diff_blasso_map = blasso_map_results["best_pol_diff"]

                # Store convergence information
                converged = blasso.converged_
                max_rhat = np.max(blasso.rhat_)

                # Extract posterior samples and compute coverage metrics
                n_chains, n_samples_mcmc, n_features = blasso.chains_.shape
                coef_samples = blasso.chains_.reshape(n_chains * n_samples_mcmc, n_features)
                iou_coverage = metrics.compute_iou_coverage(coef_samples, D_matrix, D, true_best)
                min_dosage_coverage = metrics.compute_min_dosage_coverage(
                    coef_samples, D_matrix, D, min_dosage_best_policy)

                # Compute average profile indicators across MCMC samples
                profile_indicators_sum = np.zeros(len(profiles))
                n_posterior_samples = coef_samples.shape[0]
                for sample_idx in range(n_posterior_samples):
                    y_sample = np.dot(D_matrix, coef_samples[sample_idx])
                    best_policies_sample = metrics.find_best_policies(D, y_sample)
                    profile_indicator_sample = metrics.find_profiles(best_policies_sample, all_policies, profile_map)
                    profile_indicators_sum += np.array(profile_indicator_sample)
                avg_profile_indicators = (profile_indicators_sum / n_posterior_samples).tolist()

                this_list = [
                    n_per_pol, sim_i, 
                    sqrd_err_blasso, iou_blasso, min_dosage_present_blasso, best_policy_diff_blasso,
                    sqrd_err_blasso_map, iou_blasso_map, min_dosage_present_blasso_map, best_policy_diff_blasso_map,
                    converged, max_rhat, iou_coverage, min_dosage_coverage
                ]
                # Add profile indicators: mean, then MAP, then average across all samples
                this_list += best_profile_indicator_blasso
                this_list += best_profile_indicator_blasso_map
                this_list += avg_profile_indicators
                blasso_list.append(this_list)

            #
            # Run Bootstrap Lasso
            #
            if method == "bootstrap":
                bootstrap = BootstrapLasso(
                    n_bootstrap=bootstrap_n_iter,
                    alpha=bootstrap_alpha,
                    confidence_level=bootstrap_confidence_level,
                    fit_intercept=False,
                    random_state=sim_i if bootstrap_random_state is None else bootstrap_random_state,
                    verbose=False
                )
                bootstrap.fit(D_matrix, y)

                y_bootstrap = bootstrap.predict(D_matrix)

                bootstrap_results = metrics.compute_all_metrics(
                    y, y_bootstrap, D, true_best, all_policies, profile_map,
                    min_dosage_best_policy, true_best_effect)
                sqrd_err_bootstrap = bootstrap_results["sqrd_err"]
                iou_bootstrap = bootstrap_results["iou"]
                best_profile_indicator_bootstrap = bootstrap_results["best_prof"]
                min_dosage_present_bootstrap = bootstrap_results["min_dos_inc"]
                best_policy_diff_bootstrap = bootstrap_results["best_pol_diff"]

                # Store bootstrap-specific diagnostics
                coverage = bootstrap.coverage_
                mean_ci_width = np.mean(bootstrap.coef_ci_[:, 1] - bootstrap.coef_ci_[:, 0])
                feature_importance = bootstrap.get_feature_importance()
                n_stable_features = np.sum(feature_importance > 0.5)  # Features selected in >50% of iterations

                # Extract bootstrap samples and compute coverage metrics
                coef_samples = bootstrap.bootstrap_coefs_
                iou_coverage = metrics.compute_iou_coverage(coef_samples, D_matrix, D, true_best)
                min_dosage_coverage = metrics.compute_min_dosage_coverage(
                    coef_samples, D_matrix, D, min_dosage_best_policy)

                # Compute average profile indicators across bootstrap samples
                profile_indicators_sum = np.zeros(len(profiles))
                n_bootstrap_samples = coef_samples.shape[0]
                for sample_idx in range(n_bootstrap_samples):
                    y_sample = np.dot(D_matrix, coef_samples[sample_idx])
                    best_policies_sample = metrics.find_best_policies(D, y_sample)
                    profile_indicator_sample = metrics.find_profiles(best_policies_sample, all_policies, profile_map)
                    profile_indicators_sum += np.array(profile_indicator_sample)
                avg_profile_indicators = (profile_indicators_sum / n_bootstrap_samples).tolist()

                this_list = [
                    n_per_pol, sim_i, sqrd_err_bootstrap, iou_bootstrap,
                    min_dosage_present_bootstrap, best_policy_diff_bootstrap,
                    coverage, mean_ci_width, n_stable_features,
                    iou_coverage, min_dosage_coverage
                ]
                this_list += avg_profile_indicators
                bootstrap_list.append(this_list)

            #
            # Run PPMx
            #
            if method == "ppmx":
                ppmx = PPMx(
                    n_iter=ppmx_n_iter,
                    burnin=ppmx_burnin,
                    thin=ppmx_thin,
                    alpha=ppmx_alpha,
                    cohesion=ppmx_cohesion,
                    similarity_weight=ppmx_similarity_weight,
                    similarity_bandwidth=ppmx_similarity_bandwidth,
                    random_state=sim_i,
                    verbose=False
                )
                ppmx.fit(X, y, D)

                y_ppmx = ppmx.predict(X)

                ppmx_results = metrics.compute_all_metrics(
                    y, y_ppmx, D, true_best, all_policies, profile_map,
                    min_dosage_best_policy, true_best_effect)
                sqrd_err_ppmx = ppmx_results["sqrd_err"]
                iou_ppmx = ppmx_results["iou"]
                best_profile_indicator_ppmx = ppmx_results["best_prof"]
                min_dosage_present_ppmx = ppmx_results["min_dos_inc"]
                best_policy_diff_ppmx = ppmx_results["best_pol_diff"]

                # Compute mean number of clusters
                mean_n_clusters = np.mean(ppmx.n_clusters_samples_)

                # Extract posterior samples and compute coverage metrics
                n_posterior_samples = len(ppmx.partition_samples_)
                coef_samples = np.zeros((n_posterior_samples, num_policies))

                for sample_idx in range(n_posterior_samples):
                    partition = ppmx.partition_samples_[sample_idx]
                    cluster_means = ppmx.cluster_means_samples_[sample_idx]

                    # Convert partition to coefficient vector
                    for policy_id in range(num_policies):
                        cluster_id = partition[policy_id]
                        coef_samples[sample_idx, policy_id] = cluster_means[cluster_id]

                # Compute coverage metrics
                iou_coverage = metrics.compute_iou_coverage(coef_samples, D_matrix, D, true_best)
                min_dosage_coverage = metrics.compute_min_dosage_coverage(
                    coef_samples, D_matrix, D, min_dosage_best_policy)

                # Compute average profile indicators across posterior samples
                profile_indicators_sum = np.zeros(len(profiles))
                for sample_idx in range(n_posterior_samples):
                    y_sample = np.dot(D_matrix, coef_samples[sample_idx])
                    best_policies_sample = metrics.find_best_policies(D, y_sample)
                    profile_indicator_sample = metrics.find_profiles(best_policies_sample, all_policies, profile_map)
                    profile_indicators_sum += np.array(profile_indicator_sample)
                avg_profile_indicators = (profile_indicators_sum / n_posterior_samples).tolist()

                this_list = [
                    n_per_pol, sim_i, sqrd_err_ppmx, iou_ppmx,
                    min_dosage_present_ppmx, best_policy_diff_ppmx,
                    mean_n_clusters, ppmx.acceptance_rate_,
                    iou_coverage, min_dosage_coverage
                ]
                this_list += avg_profile_indicators
                ppmx_list.append(this_list)

    profiles_str = [str(prof) for prof in profiles]

    if method == "r":
        rashomon_cols = ["n_per_pol", "sim_num", "num_pools", "MSE", "IOU", "min_dosage", "best_pol_diff"]
        rashomon_cols += profiles_str
        rashomon_df = pd.DataFrame(rashomon_list, columns=rashomon_cols)
        rashomon_df.to_csv(os.path.join(output_dir, rashomon_fname))
        if verbose:
            print(f"\nSaved Rashomon results to {rashomon_fname}")

    if method == "lasso":
        lasso_cols = ["n_per_pol", "sim_num", "MSE", "L1_loss", "IOU", "min_dosage", "best_pol_diff"]
        lasso_cols += profiles_str
        lasso_df = pd.DataFrame(lasso_list, columns=lasso_cols)
        lasso_df.to_csv(os.path.join(output_dir, lasso_fname))
        if verbose:
            print(f"\nSaved Lasso results to {lasso_fname}")

    if method == "blasso":
        blasso_cols = [
            "n_per_pol", "sim_num", 
            "MSE_mean", "IOU_mean", "min_dosage_mean", "best_pol_diff_mean",
            "MSE_map", "IOU_map", "min_dosage_map", "best_pol_diff_map",
            "converged", "max_rhat", "IOU_coverage", "min_dosage_coverage"
        ]
        # Add profile columns for posterior mean, MAP, and average
        blasso_cols += [f"{prof}_mean" for prof in profiles_str]
        blasso_cols += [f"{prof}_map" for prof in profiles_str]
        blasso_cols += [f"{prof}_avg" for prof in profiles_str]
        blasso_df = pd.DataFrame(blasso_list, columns=blasso_cols)
        blasso_df.to_csv(os.path.join(output_dir, blasso_fname))
        if verbose:
            print(f"\nSaved Bayesian Lasso results to {blasso_fname}")

    if method == "bootstrap":
        bootstrap_cols = ["n_per_pol", "sim_num", "MSE", "IOU", "min_dosage", "best_pol_diff",
                          "coverage", "mean_ci_width", "n_stable_features",
                          "IOU_coverage", "min_dosage_coverage"]
        bootstrap_cols += profiles_str
        bootstrap_df = pd.DataFrame(bootstrap_list, columns=bootstrap_cols)
        bootstrap_df.to_csv(os.path.join(output_dir, bootstrap_fname))
        if verbose:
            print(f"\nSaved Bootstrap Lasso results to {bootstrap_fname}")

    if method == "ppmx":
        ppmx_cols = ["n_per_pol", "sim_num", "MSE", "IOU", "min_dosage", "best_pol_diff",
                     "mean_n_clusters", "acceptance_rate",
                     "IOU_coverage", "min_dosage_coverage"]
        ppmx_cols += profiles_str
        ppmx_df = pd.DataFrame(ppmx_list, columns=ppmx_cols)
        ppmx_df.to_csv(os.path.join(output_dir, ppmx_fname))
        if verbose:
            print(f"\nSaved PPMx results to {ppmx_fname}")

    if verbose:
        print("\nSimulations complete!")
