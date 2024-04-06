import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def make_plot(losses, sizes, fname, title):

    sorted_losses = np.sort(losses)
    sorted_indices = np.argsort(model_losses)
    sorted_sizes = model_sizes[sorted_indices]
    sorted_posteriors = np.exp(-sorted_losses)
    map_to_model_ratio = sorted_posteriors / np.max(sorted_posteriors)

    n_range = np.arange(len(sorted_posteriors))

    fig, ax = plt.subplots(figsize=(6, 5))

    # ax.spines[['right', 'top']].set_visible(False)
    ax.spines[['top']].set_visible(False)

    ax2 = ax.twinx()

    ax.plot(n_range, map_to_model_ratio,
            color="dodgerblue",
            zorder=3.1)
    ax2.scatter(n_range, sorted_sizes,
                color="indianred", s=2,
                zorder=3.1, )
    # ax.plot(n_range[:max_idx], map_to_model_ratio[:max_idx],
    #         linewidth=3,
    #         color="forestgreen",
    #        zorder=3.1)

    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$P(\Pi_i | Z) / P(\Pi_0 | Z)$", rotation=90)
    ax2.set_ylabel(r"$|\Pi_i|$", rotation=90)

    # ax.set_yticks([1, 100, 200, 300, 400, 500])

    ax.set_xlim(0)

    ax.set_title(title)

    plt.savefig(fname, dpi=300, bbox_inches="tight")

    print(f"Saved figure to {fname}")

    num_models = np.arange(0, len(model_losses))+1
    # model_errors = sorted_losses * num_models
    est_err = 1 / (num_models * sorted_posteriors)
    sorted_epsilon = sorted_losses / np.min(model_losses) - 1

    print(np.where(sorted_epsilon <= 0.0025)[0])

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.spines[['right', 'top']].set_visible(False)

    ax.plot(sorted_epsilon, est_err,
            color="dodgerblue",
            zorder=3.1)

    # ax.plot(sorted_epsilon[100:], full_error[100:],
    #         color="indianred",
    #        zorder=3.1)

    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\mathcal{O} \left( 1 / \theta |P_{\theta}| \right)$", rotation=90)

    ax.set_ylim(0)

    err_fname = fname[:-4] + "_err.png"
    plt.savefig(err_fname, dpi=300, bbox_inches="tight")

    # plt.show()


if __name__ == "__main__":

    trt_het = True
    edu_het = True
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

    model_losses = np.array(model_losses)
    model_sizes = np.array(model_sizes)

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

    if len(R_set) > 0:
        plot_fname = f"{outcome_col}_{output_fname_suffix}_{reg:.2e}.png"
        plot_fname = results_subdir + plot_fname
        plot_title = f"{outcome_col} {output_fname_suffix}, lambda = {reg:.2e}"
        make_plot(model_losses, model_sizes, plot_fname, plot_title)

    pkl_fname = results_subdir + outcome_col + output_fname_suffix + f"_{reg:.2e}.pkl"
    print(f"Pickling to {pkl_fname}")

    with open(pkl_fname, "wb") as f:
        pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)

    print("Done pickling!")
