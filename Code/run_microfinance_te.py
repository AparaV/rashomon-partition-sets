import pickle
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy

from rashomon import tva, loss
from rashomon.aggregate import find_te_het_partitions, find_feasible_sum_subsets
from rashomon.sets import RashomonSet


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument("--outcome_col", type=int,
                        help="Index of outcome column")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    df_original = pd.read_csv("../Data/banerjee_miracle.csv")
    results_dir = "../Results/microfinance/"
    cols = df_original.columns

    chosen_covariates_idx = [2, 3, 4, 6, 7, 9, 10]
    outcome_col_id = 14
    outcome_col = cols[outcome_col_id]
    chosen_covariates = [cols[x] for x in chosen_covariates_idx]

    print(f"Covariates used are {chosen_covariates}")
    print(f"Outcome is {outcome_col}")

    suffix_possibilities = [
        "",
        "_trt_edu_gen", "_trt_edu", "_trt_gen", "_trt",
        "_edu_gen", "_edu",
        "_gen"
    ]

    for suffix in suffix_possibilities:

        print(f"Working on {suffix}")

        results_subdir = results_dir + outcome_col
        outcome_fname = outcome_col + suffix

        pickle_pools_results = results_subdir + "/" + outcome_fname + ".pkl"
        pickle_te_pools_fname = results_subdir + "/" + outcome_fname + "_te.pkl"

        # Read pickle file
        with open(pickle_pools_results, "rb") as f:
            res_dict = pickle.load(f)

        reg = res_dict["reg"]
        q = res_dict["q"]
        eps = res_dict["eps"]
        R_set = res_dict["R_set"]
        R_profiles = res_dict["R_profiles"]

        # If empty, continue on
        if len(R_set) == 0:
            print("\tEmpty Rashomon set")
            # pruned_te_rashomon_profiles = []
            te_res_dict = {
                "reg": reg,
                "q": q,
                "eps": eps,
                "R_set": [],
                "R_profiles": []
            }
            with open(pickle_te_pools_fname, "wb") as f:
                pickle.dump(te_res_dict, f, pickle.HIGHEST_PROTOCOL)
            continue

        # Otherwise, find TE partitions
        df = df_original.copy()

        if "trt" not in suffix:
            df["treatment"] = df["treatment"] + 1
        if "edu" not in suffix:
            df["hh_edu"] = df["hh_edu"] + 1
        if "gen" not in suffix:
            df["hh_gender"] = df["hh_gender"] + 1

        trt_max_dosage = int(np.max(df["treatment"]) + 1)
        edu_max_dosage = int(np.max(df["hh_edu"]) + 1)
        gen_max_dosage = int(np.max(df["hh_gender"]) + 1)

        df2 = df.copy()
        df2 = df2.dropna(subset=[outcome_col], axis=0)

        Z = df2.to_numpy()

        X = Z[:, chosen_covariates_idx]
        y = Z[:, outcome_col_id]
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        y = y.reshape((-1, 1))

        num_data = X.shape[0]

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

        policy_means = loss.compute_policy_means(D, y, num_policies)

        # Compute the partitions
        trt_arm_idx = 0

        profiles, profiles_map = tva.enumerate_profiles(M)
        profiles_x, profiles_x_map = tva.enumerate_profiles(M-1)

        te_rashomon_profiles = [[]] * len(profiles_x)

        for x, profile_x in enumerate(profiles_x):
            print(f"\tWorking on feature profile {tuple(['x'] + list(profile_x))}")
            te_rashomon_x = []
            seen_pairs_bytes = []

            trt_profile = tuple([1] + list(profile_x))
            ctl_profile = tuple([0] + list(profile_x))

            trt_profile_idx = profiles_map[trt_profile]
            ctl_profile_idx = profiles_map[ctl_profile]

            trt_policies_ids = policies_ids_profiles[trt_profile_idx]
            ctl_policies_ids = policies_ids_profiles[ctl_profile_idx]
            tc_policies_ids = trt_policies_ids + ctl_policies_ids

            trt_policies = policies_profiles_masked[trt_profile_idx]
            ctl_policies = policies_profiles_masked[ctl_profile_idx]

            # Subset data
            mask = np.isin(D, tc_policies_ids)
            D_tc = D[mask].reshape((-1, 1))
            y_tc = y[mask].reshape((-1, 1))

            for R_est_idx, R_set_i in enumerate(R_set):
                # Get treatment and control partitions
                sigma_trt_R_set_idx = R_set_i[trt_profile_idx]
                sigma_trt_i = R_profiles[trt_profile_idx].sigma[sigma_trt_R_set_idx]
                sigma_ctl_R_set_idx = R_set_i[ctl_profile_idx]
                sigma_ctl_i = R_profiles[ctl_profile_idx].sigma[sigma_ctl_R_set_idx]

                if sigma_trt_i is None:
                    trt_bytes_rep = str.encode("None")
                else:
                    trt_bytes_rep = sigma_trt_i.tobytes()
                if sigma_ctl_i is None:
                    ctl_bytes_rep = str.encode("None")
                else:
                    ctl_bytes_rep = sigma_ctl_i.tobytes()
                bytes_rep = trt_bytes_rep + ctl_bytes_rep

                if bytes_rep in seen_pairs_bytes:
                    continue
                seen_pairs_bytes.append(bytes_rep)

                if sigma_trt_i is None and sigma_ctl_i is None:
                    te_rashomon_x_i = RashomonSet(shape=None)
                    te_rashomon_x_i.P_qe = [None]
                    Q_ctl = R_profiles[trt_profile_idx].Q[sigma_trt_R_set_idx]
                    Q_trt = R_profiles[ctl_profile_idx].Q[sigma_ctl_R_set_idx]
                    te_rashomon_x_i.Q = np.append(te_rashomon_x_i.Q, Q_ctl + Q_trt)

                else:
                    te_rashomon_x_i = find_te_het_partitions(
                        sigma_trt_i, sigma_ctl_i, trt_profile_idx, ctl_profile_idx, trt_policies, ctl_policies,
                        trt_arm_idx, all_policies, policies_ids_profiles,
                        D_tc, y_tc, policy_means,
                        theta=q, reg=reg, normalize=num_data
                    )

                for idx, sigma_idx in enumerate(te_rashomon_x_i.sigma):
                    q_idx = te_rashomon_x_i.Q[idx]
                    te_rashomon_x.append((sigma_idx, q_idx, sigma_trt_i, sigma_ctl_i))

            te_rashomon_profiles[x] = te_rashomon_x

        losses = []

        # Sort partitions by loss
        for x, profile_x in enumerate(profiles_x):
            te_rashomon_x = te_rashomon_profiles[x]
            losses_x = [loss for _, loss, _, _ in te_rashomon_x]
            losses_x = np.array(losses_x)
            argsort_indices = np.argsort(losses_x)

            sorted_te_rashomon_x = []
            for idx in argsort_indices:
                sorted_te_rashomon_x.append(te_rashomon_x[idx])

            te_rashomon_profiles[x] = sorted_te_rashomon_x
            losses.append(np.sort(losses_x))

        first_loss = [x[0] for x in losses]
        last_loss = [x[-1] for x in losses]
        print(f"\tMin = {np.sum(first_loss)}, Max = {np.sum(last_loss)}")
        eps = (np.sum(last_loss) - np.sum(first_loss)) / np.sum(first_loss)

        loss_combinations = find_feasible_sum_subsets(losses, q)

        # Remove partitions that don't appear in any feasible loss combination
        loss_comb_np = np.array(loss_combinations)
        max_id_per_profile = np.max(loss_comb_np, axis=0)

        pruned_te_rashomon_profiles = [[]] * len(profiles_x)

        for k, max_id in enumerate(max_id_per_profile):
            pruned_te_rashomon_profiles[k] = te_rashomon_profiles[k][:(max_id+1)]

        pickle_res_dict = {
            "reg": reg,
            "q": q,
            "eps": eps,
            "R_set": loss_combinations,
            "R_profiles": pruned_te_rashomon_profiles
        }

        with open(pickle_te_pools_fname, "wb") as f:
            pickle.dump(pickle_res_dict, f, pickle.HIGHEST_PROTOCOL)
