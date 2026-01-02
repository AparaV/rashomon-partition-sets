import pickle
import argparse
import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument("--outcome_col", type=int,
                        help="Index of outcome column")
    parser.add_argument("--reg", type=float,
                        help="Regularization parameter")
    parser.add_argument("--eps", type=float,
                        help="Desired common epsilon threshold")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    outcome_col_id = args.outcome_col

    reg = args.reg
    lambda_str = f"_{reg:.2e}"

    data_fname = "../Data/banerjee_miracle.csv"
    results_dir = "../Results/microfinance/"
    chosen_covariates_idx = [2, 3, 4, 6, 7, 9, 10]

    df = pd.read_csv(data_fname)

    cols = df.columns
    outcome_col = cols[outcome_col_id]
    chosen_covariates = [cols[x] for x in chosen_covariates_idx]

    print(f"Covariates used are {chosen_covariates}")
    print(f"Outcome is {outcome_col}")

    results_subdir = results_dir + outcome_col + "/"
    pkl_prefix = results_subdir + outcome_col

    suffix_possibilities = [
        "",
        "_trt_edu_gen", "_trt_edu", "_trt_gen", "_trt",
        "_edu_gen", "_edu",
        "_gen"
    ]

    reg_all = []
    q0_all = []
    eps_all = []
    R_set_all = []
    R_profiles_all = []

    selected_suffix_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    selected_suffixes = [suffix_possibilities[idx] for idx in selected_suffix_idx]

    for suffix in selected_suffixes:
        outcome_fname = results_subdir + outcome_col + suffix + lambda_str + ".pkl"

        with open(outcome_fname, "rb") as f:
            res_dict = pickle.load(f)

        reg_all.append(res_dict["reg"])
        # q = res_dict["q"]
        q0_all.append(res_dict["q0"])
        eps_all.append(res_dict["eps"])
        # H = res_dict["H"]
        R_set_all.append(res_dict["R_set"])
        R_profiles_all.append(res_dict["R_profiles"])

        print(f"Suffix: {suffix}\t min q: {q0_all[-1]}\t max q: {res_dict['q']}\t num models: {len(res_dict['R_set'])}")

    model_losses = []
    for R_set, R_profiles in zip(R_set_all, R_profiles_all):
        for r, model_r in enumerate(R_set):
            loss_r = 0
            for k, prof_k_id in enumerate(model_r):
                loss_r += R_profiles[k].loss[prof_k_id]
            model_losses.append(loss_r)
    print(f"Total models: {len(model_losses)}")

    best_loss = np.min(model_losses)
    worst_loss = np.max(model_losses)

    eps_original = worst_loss / best_loss - 1

    desired_epsilon = 5e-4
    loss_threshold = best_loss * (1 + desired_epsilon)

    R_set_pruned = []
    q_pruned = []
    q0_pruned = []
    model_losses_pruned = []
    for idx, R_set in enumerate(R_set_all):
        R_set_pruned_i = []
        q_pruned_i = -np.inf
        q0_pruned_i = np.inf
        R_profiles = R_profiles_all[idx]
        for r, model_r in enumerate(R_set):
            loss_r = 0
            for k, prof_k_id in enumerate(model_r):
                loss_r += R_profiles[k].loss[prof_k_id]
            if loss_r <= loss_threshold:
                R_set_pruned_i.append(model_r)
                q_pruned_i = max(q_pruned_i, loss_r)
                q0_pruned_i = max(q0_pruned_i, loss_r)
                model_losses_pruned.append(loss_r)
        R_set_pruned.append(R_set_pruned_i)
        q_pruned.append(q_pruned_i)
        q0_pruned.append(q0_pruned_i)

    eps_new = np.max(model_losses_pruned) / np.min(model_losses_pruned) - 1

    print(f"Original size = {len(model_losses)}. After pruning = {len(model_losses_pruned)}")
    print(f"Origial eps = {eps_original}. After pruning = {eps_new}")

    for idx, suffix in enumerate(selected_suffixes):
        pruned_outcome_fname = results_subdir + outcome_col + suffix + lambda_str + "_pruned.pkl"

        res_dict = {
            "reg": reg_all[idx],
            "q": q_pruned[idx],
            "q0": q0_pruned[idx],
            "eps": eps_new,
            "H": np.inf,
            "R_set": R_set_pruned[idx],
            "R_profiles": R_profiles_all[idx]
        }

        with open(pruned_outcome_fname, "wb") as f:
            pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
