import pickle
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy

from rashomon import tva
from rashomon.aggregate import RAggregate


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

    trt_het = True
    edu_het = False
    gen_het = False

    output_fname_suffix = ""
    if trt_het:
        output_fname_suffix += "_trt"
    if edu_het:
        output_fname_suffix += "_edu"
    if gen_het:
        output_fname_suffix += "_gen"

    args = parse_arguments()

    outcome_col_id = args.outcome_col

    reg = args.reg
    q = args.q
    H = np.inf

    data_fname = "../Data/banerjee_miracle.csv"
    results_dir = "../Results/microfinance/"
    # chosen_covariates_idx = [2, 3, 4, 5, 7, 8]
    chosen_covariates_idx = [2, 3, 4, 6, 7, 9, 10]

    df = pd.read_csv(data_fname)

    if not trt_het:
        df["treatment"] = df["treatment"] + 1
    if not edu_het:
        df["hh_edu"] = df["hh_edu"] + 1
    if not gen_het:
        df["hh_gender"] = df["hh_gender"] + 1

    trt_max_dosage = int(np.max(df["treatment"]) + 1)
    edu_max_dosage = int(np.max(df["hh_edu"]) + 1)
    gen_max_dosage = int(np.max(df["hh_gender"]) + 1)

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
    M = 7
    R = np.array([trt_max_dosage, edu_max_dosage, gen_max_dosage, 4, 4, 4, 4])

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
    for i in range(num_data):
        policy_i = tuple([int(x) for x in X[i, :]])
        policy_idx = [idx for idx in range(num_policies) if all_policies[idx] == policy_i]
        D[i, 0] = int(policy_idx[0])

    #
    # Find the Rashomon set
    #
    # R_set, R_profiles = RAggregate(M, R, H, D, y, q, reg=reg, verbose=True)
    R_set, R_profiles = RAggregate(M, R, H, D, y, q, reg=reg, verbose=True, bruteforce=True)
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

    results_subdir = results_dir + outcome_col + "/"
    pkl_fname = results_subdir + outcome_col + output_fname_suffix + ".pkl"
    print(f"Pickling to {pkl_fname}")

    with open(pkl_fname, "wb") as f:
        pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)

    print("Done pickling!")
