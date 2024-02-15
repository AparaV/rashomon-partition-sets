import pickle
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy

from rashomon import tva
# from rashomon import loss
# from rashomon import counter
# from rashomon import metrics
# from rashomon import extract_pools
from rashomon.aggregate import RAggregate
# from rashomon.sets import RashomonSet, RashomonProblemCache, RashomonSubproblemCache


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument("--outcome_col", type=int,
                        help="Index of outcome column")
    parser.add_argument("--reg", type=float,
                        help="Regularization parameter")
    parser.add_argument("--q", type=float,
                        help="q threshold")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    outcome_col_id = args.outcome_col

    # With q = 4.6e-3, we got
    # Best model loss 0.004299737679159057 and epsilon 0.06977152970196171
    # Smallest model 64.0, largest model 67.0
    # And there were 13333 models
    # reg = 1e-4
    # q = 4.6e-3
    reg = args.reg
    q = args.q
    H = np.inf

    data_fname = "../Data/banerjee_miracle.csv"
    results_dir = "../Results/microfinance/"
    chosen_covariates_idx = [2, 3, 4, 5, 7, 8]

    df = pd.read_csv(data_fname)

    # outcome_cols = cols[10:]
    # covariate_cols_id = [2, 3, 4, 5, 6, 7, 8, 9]
    # covariate_cols = [cols[x] for x in covariate_cols_id]

    df["treatment"] = df["treatment"] + 1
    df["hh_edu"] = df["hh_edu"] + 1
    df["hh_gender"] = df["hh_gender"] + 1

    cols = df.columns
    outcome_col = cols[outcome_col_id]
    chosen_covariates = [cols[x] for x in chosen_covariates_idx]

    print(f"Covariates used are {chosen_covariates}")
    print(f"Outcome is {outcome_col}")

    df2 = df.copy()
    df2 = df2.dropna(subset=[outcome_col], axis=0)

    Z = df2.to_numpy()

    X = Z[:, chosen_covariates_idx]
    y = Z[:, outcome_col_id]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y = y.reshape((-1, 1))

    num_data = X.shape[0]
    print(f"There are {num_data} data points")

    #
    # Setup policy means
    #
    M = 6
    R = np.array([3, 3, 3, 4, 4, 4])

    num_profiles = 2**M
    profiles, profile_map = tva.enumerate_profiles(M)

    all_policies = tva.enumerate_policies(M, R)
    num_policies = len(all_policies)

    policies_profiles = {}
    policies_profiles_masked = {}
    policies_ids_profiles = {}
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

    D = np.zeros(shape=y.shape, dtype=np.int64)
    # profiles_in_data = []
    for i in range(num_data):
        policy_i = tuple([int(x) for x in X[i, :]])
        policy_idx = [idx for idx in range(num_policies) if all_policies[idx] == policy_i]
        # profiles_in_data.append(tva.policy_to_profile(policy_i))
        D[i, 0] = int(policy_idx[0])

    # policy_means = loss.compute_policy_means(D, y, num_policies)

    # nodata_idx = np.where(policy_means[:, 1] == 0)[0]
    # policy_means[nodata_idx, 0] = -np.inf
    # policy_means[nodata_idx, 1] = 1
    # mu_policies = policy_means[:, 0] / policy_means[:, 1]

    # true_best_eff = np.max(mu_policies)
    # print(true_best_eff)
    # np.where(mu_policies == true_best_eff)

    #
    # Find the Rashomon set
    #
    R_set, R_profiles = RAggregate(M, R, H, D, y, q, reg=reg, verbose=True)
    print(f"There are {len(R_set)} models in the Rashomon set")

    #
    # Compute properties of the Rashomon set
    #
    model_losses = []
    model_sizes = []

    for r_set in R_set:
        loss_r = 0
        size_r = 0
        for profile, model_prof in enumerate(r_set):
            loss_r_prof = R_profiles[profile].loss[model_prof]
            size_r_prof = R_profiles[profile].pools[model_prof]
            loss_r += loss_r_prof
            size_r += size_r_prof

        model_losses.append(loss_r)
        model_sizes.append(size_r)

    if len(R_set) > 0:
        q0 = np.min(model_losses)
        eps = (np.max(model_losses) - np.min(model_losses)) / q0

        print(f"Best model loss {q0} and epsilon {eps}")
        print(f"Smallest model {np.min(model_sizes)}, largest model {np.max(model_sizes)}")
    else:
        q0 = -np.inf
        eps = np.inf

    #
    # Pickle the results
    #
    res_dict = {
        "outcome": outcome_col,
        "reg": reg,
        "q": q,
        "q0": q0,
        "eps": eps,
        "H": H,
        "R_set": R_set,
        "R_profiles": R_profiles
    }

    pkl_fname = results_dir + outcome_col + ".pkl"

    with open(pkl_fname, "wb") as f:
        pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)

    print("Done pickling!")
