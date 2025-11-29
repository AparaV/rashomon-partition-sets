import argparse
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from rashomon import loss
from rashomon import hasse
from rashomon import metrics
from rashomon.aggregate import RAggregate_profile
from rashomon.extract_pools import extract_pools
from baselines import BayesianLasso, BootstrapLasso

from typing import Dict


def parse_arguments():
    """Parse command line arguments for simulation control."""
    parser = argparse.ArgumentParser(
        description="Run worst case simulations with selected baseline methods"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["rashomon", "lasso", "tva", "blasso", "bootstrap"],
        default=["rashomon", "lasso", "tva", "blasso", "bootstrap"],
        help="Methods to run (default: all methods)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="Sample sizes per policy (default: [10, 20, 50, 100, 500, 1000])"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of simulation iterations (default: 100)"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix to add to output filenames"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with reduced iterations (2 sample sizes, 5 iterations, 100 bootstrap/MCMC samples)"
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
    return parser.parse_args()


def puffer_transform(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Puffer transform the data to make it suitable for regression.

    Arguments:
    y (np.ndarray): Target variable
    X (np.ndarray): Feature matrix

    Returns:
    y_transformed (np.ndarray): Transformed target variable
    X_transformed (np.ndarray): Transformed feature matrix
    """

    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag(1 / S)
    S_mat = np.diag(S)

    # start_1 = time.time()
    F = np.matmul(S_inv, U.T)
    F = np.matmul(U, F)
    # end_1 = time.time()
    # print(f"Method 1: Time for F: {end_1 - start_1:.4f} seconds")

    F_inv = np.matmul(U, S_mat)
    F_inv = np.matmul(F_inv, U.T)

    # start_2 = time.time()
    # F_2 = np.matmul(U, S_inv)
    # F_2 = np.matmul(F_2, U.T)
    # end_2 = time.time()
    # print(f"Method 2: Time for F: {end_2 - start_2:.4f} seconds")

    # print(f"F is equal: {np.allclose(F, F_2)}")

    y_transformed = np.matmul(F, y)
    X_transformed = np.matmul(U, Vh)

    # print(y.shape, X.shape)
    # print(y_transformed.shape, X_transformed.shape)

    return y_transformed, X_transformed, F, F_inv


def run_lasso(y: np.ndarray, X: np.ndarray, reg: float, D: np.ndarray, true_best, min_dosage_best_policy,
              puff_details: Dict = None) -> dict:
    """ Run Lasso regression on the data."""

    lasso = linear_model.Lasso(reg, fit_intercept=False)
    lasso.fit(X, y)
    alpha_est = lasso.coef_
    y_lasso = lasso.predict(X)

    y_lasso_outcome = y_lasso.copy()
    sqrd_err = mean_squared_error(y_lasso, y)
    if puff_details is not None:
        # X_outcome = np.matmul(puffer_inv, X)
        # # alpha_outcome = np.matmul(puffer_inv, alpha_est)
        # y_outcome = np.matmul(puffer_inv, y)
        # # y_lasso_outcome = np.matmul(puffer_inv, y_lasso)
        # y_lasso_outcome = np.matmul(X_outcome, alpha_est)
        X_outcome = puff_details["X"]
        y_outcome = puff_details["y"]
        y_lasso_outcome = np.matmul(X_outcome, alpha_est)

        mse = mean_squared_error(y_lasso_outcome, y_outcome)
        # print(f"y_outcome_space shape: {y_lasso_outcome}")
        # print(f"y_lasso shape: {y_lasso}")
    else:
        # y_lasso_outcome = y_lasso
        # sqrd_err = mean_squared_error(y_lasso_outcome, y)
        mse = sqrd_err

    # MSE
    L1_loss = sqrd_err + reg * np.linalg.norm(alpha_est, ord=1)

    # IOU
    lasso_best = metrics.find_best_policies(D, y_lasso_outcome)
    iou_lasso = metrics.intersect_over_union(set(true_best), set(lasso_best))

    # Min dosage inclusion
    min_dosage_present_lasso = metrics.check_membership(min_dosage_best_policy, lasso_best)

    # Best policy MSE
    best_policy_error_lasso = np.max(mu) - np.max(y_lasso_outcome)

    result = {
     "sqrd_err": mse,
     "L1_loss": L1_loss,
     "iou_lasso": iou_lasso,
     "min_dosage_present_lasso": min_dosage_present_lasso,
     "best_policy_error_lasso": best_policy_error_lasso
    }

    return result


def run_bayesian_lasso(y: np.ndarray, X: np.ndarray, D: np.ndarray, true_best, min_dosage_best_policy,
                       blasso_params: Dict, sim_seed: int, verbose: bool = False) -> dict:
    """Run Bayesian Lasso regression on the data."""
    
    blasso = BayesianLasso(
        n_iter=blasso_params["n_iter"],
        burnin=blasso_params["burnin"],
        thin=blasso_params["thin"],
        lambda_prior=blasso_params["lambda_prior"],
        tau2_a=blasso_params["tau2_a"],
        tau2_b=blasso_params["tau2_b"],
        fit_intercept=False,
        random_state=sim_seed,
        verbose=verbose
    )
    
    blasso.fit(X, y, n_chains=blasso_params["n_chains"])
    y_blasso = blasso.predict(X)
    
    # Compute metrics
    mse = mean_squared_error(y, y_blasso)
    
    # IOU
    blasso_best = metrics.find_best_policies(D, y_blasso)
    iou_blasso = metrics.intersect_over_union(set(true_best), set(blasso_best))
    
    # Min dosage inclusion
    min_dosage_present_blasso = metrics.check_membership(min_dosage_best_policy, blasso_best)
    
    # Best policy error
    best_policy_error_blasso = np.max(mu) - np.max(y_blasso)
    
    # Convergence diagnostics
    converged = blasso.converged_
    max_rhat = np.max(blasso.rhat_)
    
    result = {
        "sqrd_err": mse,
        "iou_blasso": iou_blasso,
        "min_dosage_present_blasso": min_dosage_present_blasso,
        "best_policy_error_blasso": best_policy_error_blasso,
        "converged": converged,
        "max_rhat": max_rhat
    }
    
    return result


def run_bootstrap_lasso(y: np.ndarray, X: np.ndarray, D: np.ndarray, true_best, min_dosage_best_policy,
                        bootstrap_params: Dict, sim_seed: int, verbose: bool = False) -> dict:
    """Run Bootstrap Lasso regression on the data."""
    
    bootstrap = BootstrapLasso(
        n_bootstrap=bootstrap_params["n_iter"],
        alpha=bootstrap_params["alpha"],
        confidence_level=bootstrap_params["confidence_level"],
        fit_intercept=False,
        random_state=sim_seed,
        verbose=verbose
    )
    
    bootstrap.fit(X, y)
    y_bootstrap = bootstrap.predict(X)
    
    # Compute metrics
    mse = mean_squared_error(y, y_bootstrap)
    
    # IOU
    bootstrap_best = metrics.find_best_policies(D, y_bootstrap)
    iou_bootstrap = metrics.intersect_over_union(set(true_best), set(bootstrap_best))
    
    # Min dosage inclusion
    min_dosage_present_bootstrap = metrics.check_membership(min_dosage_best_policy, bootstrap_best)
    
    # Best policy error
    best_policy_error_bootstrap = np.max(mu) - np.max(y_bootstrap)
    
    # Bootstrap diagnostics
    coverage = bootstrap.coverage_
    mean_ci_width = np.mean(bootstrap.coef_ci_[:, 1] - bootstrap.coef_ci_[:, 0])
    feature_importance = bootstrap.get_feature_importance()
    n_stable_features = np.sum(feature_importance > 0.5)
    
    result = {
        "sqrd_err": mse,
        "iou_bootstrap": iou_bootstrap,
        "min_dosage_present_bootstrap": min_dosage_present_bootstrap,
        "best_policy_error_bootstrap": best_policy_error_bootstrap,
        "coverage": coverage,
        "mean_ci_width": mean_ci_width,
        "n_stable_features": n_stable_features
    }
    
    return result


def generate_data(mu, var, n_per_pol, policies, pi_policies, M):
    num_data = len(policies) * n_per_pol
    X = np.ndarray(shape=(num_data, M))
    D = np.ndarray(shape=(num_data, 1), dtype='int_')
    y = np.ndarray(shape=(num_data, 1))

    for idx, policy in enumerate(policies):
        # pool_i = pi_policies[idx]
        # mu_i = mu[pool_i]
        # var_i = var[pool_i]
        # y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

        y_i = np.random.normal(mu[idx], var, size=(n_per_pol, 1))

        start_idx = idx * n_per_pol
        end_idx = (idx + 1) * n_per_pol

        X[start_idx:end_idx, ] = policy
        D[start_idx:end_idx, ] = idx
        y[start_idx:end_idx, ] = y_i

    # y = mu + np.random.normal(0, var, size=(num_data, 1))

    return X, D, y


if __name__ == "__main__":

    args = parse_arguments()
    
    np.random.seed(3)

    #
    # Fix ground truth
    #
    sigma = np.array([[1, 1, 1],
                      [0, 0, 0]], dtype='float64')
    sigma_profile = (1, 1)

    M, _ = sigma.shape
    R = np.array([5, 5])

    # Enumerate policies and find pools
    num_policies = np.prod(R-1)
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    policies = [x for x in all_policies if hasse.policy_to_profile(x) == sigma_profile]
    pi_pools, pi_policies = extract_pools(policies, sigma)
    # num_pools = len(pi_pools)

    for pool, policy_pools in pi_pools.items():
        print(f"Pool {pool}: {policy_pools}")

    # The transformation matrix for Lasso
    G = hasse.alpha_matrix(policies)

    # # Anirudh's method
    # alpha = np.zeros((num_policies, 1))
    # for i, pol in enumerate(policies):
    #     if pol[0] <= 2 and pol[1] <= 2:
    #         alpha[i, 0] = 0
    #     else:
    #         arm_a = pol[0]
    #         arm_b = pol[1] - 1
    #         alpha[i, 0] = -1 + 0.1 * arm_b  #+ np.random.normal(0, 0.01)
    #     print(f"Policy {i}: {pol}, alpha: {alpha[i, 0]}")
    # mu = np.matmul(G, alpha)[::-1]
    # print(mu)

    # Set data parameters
    # # Original parameters
    # mu_pools = np.array([0, 1.5, 3, 3, 6, 4.5])

    # New parameters to break TVA
    mu_pools = np.array([0, 1.5, 3, 4.5])
    mu = np.zeros((num_policies, 1))
    for idx, policy in enumerate(policies):
        pool_i = pi_policies[idx]
        mu[idx, 0] = mu_pools[pool_i]
        # var_i = var[pool_i]
    # se = 1
    # var = se * np.ones_like(mu)
    var = 1

    # true_best = pi_pools[np.argmax(mu)]
    near_max_indices = np.where(np.abs(np.max(mu) - mu) <= 1e-6)[0]
    true_best = near_max_indices

    true_best_effect = np.max(mu)
    min_dosage_best_policy = metrics.find_min_dosage(true_best, policies)

    # Parse command line arguments
    methods_to_run = args.methods
    verbose = args.verbose
    
    # Simulation parameters and variables
    if args.test:
        samples_per_pol = [10, 50]
        num_sims = 5
        if verbose:
            print("Running in TEST mode: 2 sample sizes, 5 iterations, reduced MCMC/bootstrap samples")
    else:
        samples_per_pol = args.samples if args.samples is not None else [10, 20, 50, 100, 500, 1000]
        num_sims = args.iters if args.iters is not None else 100
    
    if verbose:
        print(f"Methods to run: {methods_to_run}")
        print(f"Sample sizes: {samples_per_pol}")
        print(f"Iterations: {num_sims}")

    H = np.inf
    theta = 1.1
    reg = 1e-3
    reg_rps = 1e-2
    reg_tva = 1e-3
    
    # Bayesian Lasso parameters
    if args.test:
        blasso_params = {
            "n_iter": 5000,
            "burnin": 2000,
            "thin": 2,
            "n_chains": 3,
            "lambda_prior": 5,
            "tau2_a": 1,
            "tau2_b": 1
        }
    else:
        blasso_params = {
            "n_iter": 2000,
            "burnin": 500,
            "thin": 2,
            "n_chains": 3,
            "lambda_prior": 5e-1,
            "tau2_a": 1e-1,
            "tau2_b": 1e-1
        }
    
    # Bootstrap Lasso parameters
    if args.test:
        bootstrap_params = {
            "n_iter": 100,
            "alpha": 1e-3,
            "confidence_level": 0.95
        }
    else:
        bootstrap_params = {
            "n_iter": 500,
            "alpha": 1e-3,
            "confidence_level": 0.95
        }

    # Simulation results data structure (initialize only for selected methods)
    rashomon_list = [] if "rashomon" in methods_to_run else None
    lasso_list = [] if "lasso" in methods_to_run else None
    tva_list = [] if "tva" in methods_to_run else None
    blasso_list = [] if "blasso" in methods_to_run else None
    bootstrap_list = [] if "bootstrap" in methods_to_run else None

    #
    # Simulations
    #
    for n_per_pol in samples_per_pol:

        if verbose:
            print(f"Number of samples: {n_per_pol}")

        for sim_i in range(num_sims):

            if verbose and (sim_i + 1) % 20 == 0:
                print(f"\tSimulation {sim_i+1}")

            # Generate data
            X, D, y = generate_data(mu, var, n_per_pol, policies, pi_policies, M)
            # The dummy matrix for Lasso
            D_matrix = hasse.get_dummy_matrix(D, G, num_policies)
            pol_means = loss.compute_policy_means(D, y, num_policies)

            y_puffer, D_puffer, F, F_inv = puffer_transform(y, D_matrix)
            # break

            #
            # Run Rashomon
            #
            if "rashomon" in methods_to_run:
                P_set = RAggregate_profile(M, R, H, D, y, theta, sigma_profile, reg_rps)
                # if not P_set.seen(sigma):
                #     print("P_set missing true sigma")

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

            # Run Lasso regression
            if "lasso" in methods_to_run:
                lasso_result = run_lasso(y, D_matrix, reg, D, true_best, min_dosage_best_policy, puff_details=None)
                lasso_list_i = [n_per_pol, sim_i, lasso_result["sqrd_err"], lasso_result["L1_loss"],
                                lasso_result["iou_lasso"], lasso_result["min_dosage_present_lasso"],
                                lasso_result["best_policy_error_lasso"]]
                lasso_list.append(lasso_list_i)

            # Run TVA regression
            if "tva" in methods_to_run:
                gamma = -1
                scaling = n_per_pol ** gamma
                puff_details = {"F": F, "X": D_matrix, "y": y}
                tva_result = run_lasso(y_puffer, D_puffer, reg_tva * scaling, D, true_best, min_dosage_best_policy,
                                       puff_details=puff_details)
                tva_list_i = [n_per_pol, sim_i, tva_result["sqrd_err"], tva_result["L1_loss"],
                              tva_result["iou_lasso"], tva_result["min_dosage_present_lasso"],
                              tva_result["best_policy_error_lasso"]]
                tva_list.append(tva_list_i)
            
            # Run Bayesian Lasso
            if "blasso" in methods_to_run:
                blasso_result = run_bayesian_lasso(y, D_matrix, D, true_best, min_dosage_best_policy,
                                                   blasso_params, sim_i, verbose=False)
                blasso_list_i = [n_per_pol, sim_i, blasso_result["sqrd_err"],
                                blasso_result["iou_blasso"], blasso_result["min_dosage_present_blasso"],
                                blasso_result["best_policy_error_blasso"], blasso_result["converged"],
                                blasso_result["max_rhat"]]
                blasso_list.append(blasso_list_i)
            
            # Run Bootstrap Lasso
            if "bootstrap" in methods_to_run:
                bootstrap_result = run_bootstrap_lasso(y, D_matrix, D, true_best, min_dosage_best_policy,
                                                       bootstrap_params, sim_i, verbose=False)
                bootstrap_list_i = [n_per_pol, sim_i, bootstrap_result["sqrd_err"],
                                   bootstrap_result["iou_bootstrap"], bootstrap_result["min_dosage_present_bootstrap"],
                                   bootstrap_result["best_policy_error_bootstrap"], bootstrap_result["coverage"],
                                   bootstrap_result["mean_ci_width"], bootstrap_result["n_stable_features"]]
                bootstrap_list.append(bootstrap_list_i)

    # Save results for methods that were run
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    if args.test:
        suffix += "_test"
    
    if "rashomon" in methods_to_run:
        rashomon_cols = ["n_per_pol", "sim_num", "num_pools", "MSE", "IOU", "min_dosage", "best_pol_diff"]
        rashomon_df = pd.DataFrame(rashomon_list, columns=rashomon_cols)
        rashomon_df.to_csv(f"../Results/worst_case/worst_case_rashomon{suffix}.csv")
        if verbose:
            print(f"Saved Rashomon results to worst_case_rashomon{suffix}.csv")

    if "lasso" in methods_to_run:
        lasso_cols = ["n_per_pol", "sim_num", "MSE", "L1_loss", "IOU", "min_dosage", "best_pol_diff"]
        lasso_df = pd.DataFrame(lasso_list, columns=lasso_cols)
        lasso_df.to_csv(f"../Results/worst_case/worst_case_lasso{suffix}.csv")
        if verbose:
            print(f"Saved Lasso results to worst_case_lasso{suffix}.csv")

    if "tva" in methods_to_run:
        tva_cols = ["n_per_pol", "sim_num", "MSE", "TVA_loss", "IOU", "min_dosage", "best_pol_diff"]
        tva_df = pd.DataFrame(tva_list, columns=tva_cols)
        tva_df.to_csv(f"../Results/worst_case/worst_case_tva{suffix}.csv")
        if verbose:
            print(f"Saved TVA results to worst_case_tva{suffix}.csv")
    
    if "blasso" in methods_to_run:
        blasso_cols = ["n_per_pol", "sim_num", "MSE", "IOU", "min_dosage", "best_pol_diff", "converged", "max_rhat"]
        blasso_df = pd.DataFrame(blasso_list, columns=blasso_cols)
        blasso_df.to_csv(f"../Results/worst_case/worst_case_blasso{suffix}.csv")
        if verbose:
            print(f"Saved Bayesian Lasso results to worst_case_blasso{suffix}.csv")
    
    if "bootstrap" in methods_to_run:
        bootstrap_cols = ["n_per_pol", "sim_num", "MSE", "IOU", "min_dosage", "best_pol_diff", 
                         "coverage", "mean_ci_width", "n_stable_features"]
        bootstrap_df = pd.DataFrame(bootstrap_list, columns=bootstrap_cols)
        bootstrap_df.to_csv(f"../Results/worst_case/worst_case_bootstrap{suffix}.csv")
        if verbose:
            print(f"Saved Bootstrap Lasso results to worst_case_bootstrap{suffix}.csv")
    
    if verbose:
        print("\nSimulations complete!")
