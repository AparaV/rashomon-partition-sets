import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from rashomon import loss
from rashomon import hasse
from rashomon import metrics
from rashomon.aggregate import RAggregate_profile
from rashomon.extract_pools import extract_pools


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
              puffer_inv: np.ndarray = None) -> dict:
    """ Run Lasso regression on the data."""

    lasso = linear_model.Lasso(reg, fit_intercept=False)
    lasso.fit(X, y)
    alpha_est = lasso.coef_
    y_lasso = lasso.predict(X)

    if puffer_inv is not None:
        y_outcome = np.matmul(puffer_inv, y)
        y_lasso_outcome = np.matmul(puffer_inv, y_lasso)

        mse = mean_squared_error(y_lasso_outcome, y_outcome)
        sqrd_err = mean_squared_error(y_lasso, y)
        # print(f"y_outcome_space shape: {y_lasso_outcome}")
        # print(f"y_lasso shape: {y_lasso}")
    else:
        y_lasso_outcome = y_lasso
        mse = mean_squared_error(y_lasso_outcome, y)
        sqrd_err = mse

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
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    policies = [x for x in all_policies if hasse.policy_to_profile(x) == sigma_profile]
    pi_pools, pi_policies = extract_pools(policies, sigma)
    num_pools = len(pi_pools)

    # The transformation matrix for Lasso
    G = hasse.alpha_matrix(policies)

    # Set data parameters
    mu = np.array([0, 1.5, 3, 3, 6, 4.5])
    se = 1
    var = se * np.ones_like(mu)

    true_best = pi_pools[np.argmax(mu)]
    true_best_effect = np.max(mu)
    min_dosage_best_policy = metrics.find_min_dosage(true_best, policies)

    # Simulation parameters and variables
    samples_per_pol = [10, 20, 50, 100, 500, 1000]
    # samples_per_pol = [10]
    num_sims = 100
    # num_sims = 2

    H = 10
    theta = 2
    reg = 0.1

    # Simulation results data structure
    rashomon_list = []
    lasso_list = []
    tva_list = []

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
            D_matrix = hasse.get_dummy_matrix(D, G, num_policies)
            pol_means = loss.compute_policy_means(D, y, num_policies)

            y_puffer, D_puffer, F, F_inv = puffer_transform(y, D_matrix)
            # break

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

            # Run Lasso regression
            lasso_result = run_lasso(y, D_matrix, reg, D, true_best, min_dosage_best_policy, puffer_inv=None)
            lasso_list_i = [n_per_pol, sim_i, lasso_result["sqrd_err"], lasso_result["L1_loss"],
                            lasso_result["iou_lasso"], lasso_result["min_dosage_present_lasso"],
                            lasso_result["best_policy_error_lasso"]]
            lasso_list.append(lasso_list_i)

            # Run TVA regression
            tva_result = run_lasso(y_puffer, D_puffer, 1e-3 / n_per_pol, D, true_best, min_dosage_best_policy,
                                   puffer_inv=F_inv)
            tva_list_i = [n_per_pol, sim_i, tva_result["sqrd_err"], tva_result["L1_loss"],
                          tva_result["iou_lasso"], tva_result["min_dosage_present_lasso"],
                          tva_result["best_policy_error_lasso"]]
            tva_list.append(tva_list_i)

    rashomon_cols = ["n_per_pol", "sim_num", "num_pools", "MSE", "IOU", "min_dosage", "best_pol_diff"]
    rashomon_df = pd.DataFrame(rashomon_list, columns=rashomon_cols)

    lasso_cols = ["n_per_pol", "sim_num", "MSE", "L1_loss", "IOU", "min_dosage", "best_pol_diff"]
    lasso_df = pd.DataFrame(lasso_list, columns=lasso_cols)

    tva_cols = ["n_per_pol", "sim_num", "MSE", "TVA_loss", "IOU", "min_dosage", "best_pol_diff"]
    tva_df = pd.DataFrame(tva_list, columns=tva_cols)

    rashomon_df.to_csv("../Results/worst_case/worst_case_rashomon.csv")
    lasso_df.to_csv("../Results/worst_case/worst_case_lasso.csv")
    tva_df.to_csv("../Results/worst_case/worst_case_tva.csv")
