import pickle
import numpy as np
import pandas as pd

from rashomon import hasse
from analysis import microfinance_helpers as mh


DATA_PATH = "../Data/banerjee_miracle.csv"
RESULTS_DIR = "../Results/microfinance/"
LAMBDA_STR = "_1.50e-06"
NUM_BINS = 5

ALL_DETAILS_PICKLE_FNAME = f"{RESULTS_DIR}stddev_prob_data.pkl"
COUNTER_FNAME = f"{RESULTS_DIR}counter_results{LAMBDA_STR}_{NUM_BINS}_bins.pkl"
MICROFINANCE_OUTCOMES_CSV = "../Results/microfinance/outcomes.csv"
MICROFINANCE_TREATMENT_EFFECTS_CSV = "../Results/microfinance/te.csv"

M = 7
R = np.array([3, 3, 3, 4, 4, 4, 4])

df = pd.read_csv(DATA_PATH)

cols = df.columns

chosen_covariates_idx = [2, 3, 4, 6, 7, 9, 10]
chosen_covariates = [cols[x] for x in chosen_covariates_idx]
print(f"Covariates used are {chosen_covariates}")

outcome_names = [
    "Any Loan", "Informal Loan", "Female Biz", "Working Hours", "Durables", "Temptation",
    "Expenditure", "Profit", "Revenue", "Employees", "Girls in School", "Biz Assets"
]

profile_labels_on = [
    "None",
    "Reg. Biz",
    "Reg. Debt",
    "Reg. Debt, Reg. Biz",
    "Biz",
    "Biz, Reg. Biz",
    "Biz, Reg. Debt",
    "Biz, Reg. Debt, Reg. Biz",
    "Chld",
    "Chld, Reg. Biz",
    "Chld, Reg. Debt",
    "Chld, Reg. Debt, Reg. Biz",
    "Chld, Biz",
    "Chld, Biz, Reg. Biz",
    "Chld, Biz, Reg. Debt",
    "Chld, Biz, Reg. Debt, Reg. Biz",
]

# profile_labels_on = [x[5:] for x in profile_labels_on]
profile_labels_on = [x.replace(",", " |") for x in profile_labels_on]

suffix_possibilities = [
    "",
    "_trt_edu_gen", "_trt_edu", "_trt_gen", "_trt",
    "_edu_gen", "_edu",
    "_gen"
]

map_arm1 = {-1: "undef", 0: "ctl", 1: "trt"}
map_arm2 = {-1: "undef", 0: "hh_not_edu", 1: "hh_edu"}
map_arm3 = {-1: "undef", 0: "hh_female", 1: "hh_male"}
map_arm4 = {0: "no kids", 1: "1 kid", 2: "2 kids", 3: "> 2 kids"}
map_arm5 = {0: "no biz", 1: "1-2 biz", 2: "3-5 biz", 3: "> 5 biz"}
map_arm6 = {0: "debt_q1", 1: "debt_q2", 2: "debt_q3", 3: "debt_q4"}
map_arm7 = {0: "biz_q1", 1: "biz_q2", 2: "biz_q3", 3: "biz_q4"}

maps = [
    map_arm1,
    map_arm2,
    map_arm3,
    map_arm4,
    map_arm5,
    map_arm6,
    map_arm7,
]

# Calculating Standard Deviations, Creating Bins, and calculating Model Probabilities

all_details = {}

for outcome_col_id in range(14, 26):
    outcome_col = cols[outcome_col_id]
    outcome_title = outcome_names[outcome_col_id-14]

    results_subdir = RESULTS_DIR + outcome_col + "/"
    # results_subdir = RESULTS_DIR + "_archive_lambda_1e-6/" + outcome_col + "/"
    pkl_prefix = results_subdir + outcome_col + LAMBDA_STR

    num_active_arms = 4
    num_active_profiles = 2**num_active_arms

    print(f"Outcome is {outcome_col}")

    for suffix in suffix_possibilities:

        # if suffix != "_trt_gen":
        #     continue

        this_df = df.copy()
        this_R = np.array([2, 2, 2, 4, 4, 4, 4])
        if "trt" not in suffix:
            this_df["treatment"] = this_df["treatment"] + 1
            this_R[0] += 1
        if "edu" not in suffix:
            this_df["hh_edu"] = this_df["hh_edu"] + 1
            this_R[1] += 1
        if "gen" not in suffix:
            this_df["hh_gender"] = this_df["hh_gender"] + 1
            this_R[2] += 1

        # R_set, R_profiles = read_pickle(results_subdir, outcome_col + suffix + "_pruned")
        # R_set, R_profiles = read_pickle(results_subdir, outcome_col + suffix + "_te")
        R_set, R_profiles = mh.read_pickle(results_subdir, outcome_col + suffix + LAMBDA_STR + "_pruned_te")

        if len(R_set) == 0:
            print(f"\tSuffix {suffix} has no models")
            continue
        print(f"\tWorking on suffix {suffix} - {len(R_set)} models")

        all_policies = hasse.enumerate_policies(M, this_R)
        policies_str = []
        for pol in all_policies:
            polx = list(pol)
            pol_str = ""
            for m in range(M):
                if this_R[m] == 3:
                    polx[m] = pol[m] - 1
                pol_str += maps[m][polx[m]] + ","
            policies_str.append(pol_str)
        print("Num policies", len(policies_str))

        X, y = mh.format_data(this_df, outcome_col_id, chosen_covariates_idx)

        # num_data = X.shape[0]
        # print(f"There are {num_data} data points")

        pol_res = mh.get_policy_means(M, this_R, X, y)
        policy_means_profiles = pol_res[0]
        policies_profiles_masked = pol_res[1]
        policies_profiles = pol_res[2]
        policies_means = pol_res[3]
        policies_ids_profiles = pol_res[4]

        effects = mh.get_counts(
            R_set, R_profiles, policies_profiles_masked, policies_profiles, policies_ids_profiles,
            policies_means, this_R, collect_effects=True
        )

        total_prob = effects['total_prob']
        trt_eff_std = np.std(effects['treatment_effects'])
        gen_eff_std = np.std(effects['gender_effects'])
        trt_eff_std_min = np.min(effects['treatment_effects'])
        gen_eff_std_min = np.min(effects['gender_effects'])
        trt_eff_std_max = np.max(effects['treatment_effects'])
        gen_eff_std_max = np.max(effects['gender_effects'])
    # break

    all_details[outcome_col] = {
         "gender_std": gen_eff_std,
         "treatment_std": trt_eff_std,
         "max_treatment": trt_eff_std_max,
         "max_gender": gen_eff_std_max,
         "min_treatment": trt_eff_std_min,
         "min_gender": gen_eff_std_min,
         "total_prob": total_prob
     }

with open(ALL_DETAILS_PICKLE_FNAME, "wb") as f:
    pickle.dump(all_details, f, pickle.HIGHEST_PROTOCOL)

#
# Calculating Effect Counts in each bin
#

bins = {}
for outcome, std_dict in all_details.items():
    if NUM_BINS != 3:
        bin_trt = [std_dict['min_treatment'] - 1] + \
            [2 * std_dict['treatment_std'] * i for i in np.arange(-1, 2, 3/(NUM_BINS - 2))] + \
            [std_dict['max_treatment'] + 1]
        bin_gender = [std_dict['min_gender'] - 1] + \
            [2 * std_dict['gender_std'] * i for i in np.arange(-1, 2, 3/(NUM_BINS - 2))] + \
            [std_dict['max_gender'] + 1]
    else:
        bin_trt = [std_dict['min_treatment'] - 1] + [0] + [std_dict['max_treatment'] + 1]
        bin_gender = [std_dict['min_gender'] - 1] + [0] + [std_dict['max_gender'] + 1]
    bin_dict = {
        "trt": bin_trt[::-1],
        "gender": bin_gender[::-1]
    }
    bins[outcome] = bin_dict

all_results = {}

for outcome_col_id in range(14, 26):
    outcome_col = cols[outcome_col_id]
    outcome_title = outcome_names[outcome_col_id-14]

    results_subdir = RESULTS_DIR + outcome_col + "/"
    # results_subdir = RESULTS_DIR + "_archive_lambda_1e-6/" + outcome_col + "/"
    pkl_prefix = results_subdir + outcome_col

    num_active_arms = 4
    num_active_profiles = 2**num_active_arms

    eff_ctr_matrix = np.zeros((NUM_BINS, num_active_profiles))
    eff_gender_ctr_matrix = np.zeros((NUM_BINS, num_active_profiles))
    total_ctr = 0

    print(f"Outcome is {outcome_col}")

    for suffix in suffix_possibilities:

        # if suffix != "_trt_gen":
        #     continue

        this_df = df.copy()
        this_R = np.array([2, 2, 2, 4, 4, 4, 4])
        if "trt" not in suffix:
            this_df["treatment"] = this_df["treatment"] + 1
            this_R[0] += 1
        if "edu" not in suffix:
            this_df["hh_edu"] = this_df["hh_edu"] + 1
            this_R[1] += 1
        if "gen" not in suffix:
            this_df["hh_gender"] = this_df["hh_gender"] + 1
            this_R[2] += 1

        # R_set, R_profiles = read_pickle(results_subdir, outcome_col + suffix + "_pruned")
        # R_set, R_profiles = read_pickle(results_subdir, outcome_col + suffix + "_te")
        R_set, R_profiles = mh.read_pickle(results_subdir, outcome_col + suffix + LAMBDA_STR + "_pruned_te")

        if len(R_set) == 0:
            print(f"\tSuffix {suffix} has no models")
            continue
        print(f"\tWorking on suffix {suffix} - {len(R_set)} models")

        all_policies = hasse.enumerate_policies(M, this_R)
        policies_str = []
        for pol in all_policies:
            polx = list(pol)
            pol_str = ""
            for m in range(M):
                if this_R[m] == 3:
                    polx[m] = pol[m] - 1
                pol_str += maps[m][polx[m]] + ","
            policies_str.append(pol_str)
        print("Num policies", len(policies_str))

        X, y = mh.format_data(this_df, outcome_col_id, chosen_covariates_idx)
        # num_data = X.shape[0]
        # print(f"There are {num_data} data points")

        pol_res = mh.get_policy_means(M, this_R, X, y)
        policy_means_profiles = pol_res[0]
        policies_profiles_masked = pol_res[1]
        policies_profiles = pol_res[2]
        policies_means = pol_res[3]
        policies_ids_profiles = pol_res[4]

        ctr_res = mh.get_counts(
            R_set, R_profiles, policies_profiles_masked, policies_profiles, policies_ids_profiles,
            policies_means, this_R, bins=bins[outcome_col]
        )

        for i in range(NUM_BINS):
            eff_ctr_matrix[i, :] += ctr_res[f"{i}_pooled"]
            eff_gender_ctr_matrix[i, :] += ctr_res[f"{i}_gender"]

        total_ctr += ctr_res["n_models"]
        beta = ctr_res["beta"]

    all_results[outcome_col] = {
            "n_models": total_ctr,
            "policies": policies_str,
            "beta": beta
        }

    for i in range(NUM_BINS):
        all_results[outcome_col][f"{i}_pooled"] = eff_ctr_matrix[i, :]
        all_results[outcome_col][f"{i}_gender"] = eff_gender_ctr_matrix[i, :]

for k, v in all_results.items():
    new_policies = []
    new_beta = []
    for idx, pol in enumerate(v["policies"]):
        if "undef" not in pol:
            new_policies.append(pol)
            new_beta.append(v["beta"][idx])
    all_results[k]["policies"] = new_policies
    all_results[k]["beta"] = new_beta


with open(COUNTER_FNAME, "wb") as f:
    pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)

df_dict = {}
for k, v in all_results.items():
    for pol, beta_i in zip(v["policies"], v["beta"]):
        if pol in df_dict.keys():
            df_dict[pol].append(beta_i)
        else:
            df_dict[pol] = [beta_i]

cols = chosen_covariates + list(all_results.keys())
df_list = []
for k, v in df_dict.items():
    df_list.append(k.split(",")[:-1] + v)

df = pd.DataFrame(df_list, columns=cols)

df.to_csv(MICROFINANCE_OUTCOMES_CSV)

ctl_df = df[df["treatment"] == "ctl"].copy().reset_index()
trt_df = df[df["treatment"] == "trt"].copy().reset_index()

for k in all_results.keys():
    trt_df[k] = trt_df[k] - ctl_df[k]

trt_df.to_csv(MICROFINANCE_TREATMENT_EFFECTS_CSV)
