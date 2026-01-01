# Save and load the pickle file for the Rashomon set with parameters
#   lambda = 10^{-5}
#   epsilon = 0.004
# Data is pickled as a dictionary with three keys:
#   1. `R_set_pruned`: A list of model indices representing the pruned Rashomon set.
#       Each element is a list indicating which partition/pooling to use for each race profile
#       (or [-1, idx] if all races use the same homogeneous partition).
#   2. `rashomon_profiles``: A list of 3 RashomonSet objects (one per race: White, Black, Hispanic/Other).
#       Each RashomonSet contains:
#           .sigma - List of partition matrices (how policies are pooled together)
#           .Q - Array of loss values for each partition
#           .P_qe - List of pooling structures
#           .H - Number of pools in each partition
#   3. `rashomon_homogeneous`: A single RashomonSet object representing the homogeneous case where all
#       races share the same partition structure (no heterogeneity by race).
# A more heavily pruned dataset is also pruned:
#   R_set: Further pruned list of models
#       (filtered by strict_eps = 0.002, a tighter threshold than the original eps = 0.004)
#   sizes: NumPy array of model complexities (number of pools in each model)
#   losses: NumPy array of loss values for each model
#   pools: List of dictionaries mapping pool IDs to policy lists for each model
#       (precomputed via extract_pools.extract_pools())
#   pool_means: List of dictionaries mapping pool IDs to their mean outcomes
#       (precomputed via loss.compute_pool_means())

import pickle
import numpy as np
import pandas as pd

from copy import deepcopy
from rashomon import hasse, loss
from rashomon import extract_pools
from rashomon.aggregate import (
    RAggregate_profile, find_profile_lower_bound, find_feasible_combinations, remove_unused_poolings
)
from rashomon.sets import RashomonSet


NHANES_DATA_FILE = "../Data/NHANES_telomere.csv"
NHANES_STD_FILE = "../Data/NHANES_telomere_std.csv"
NHANES_RPS_FILE = "../Results/nhanes_r_set_outlier_small_eps.pkl"
NHANES_PRUNED_RPS_FILE = "../Results/nhanes_pruned_results_outlier.pkl"

# With reg = 1e-5, theta = 0.09, H = 60
# Best loss = 0.06423696912419681, Worst loss = 0.06535629544158826, epsilon = 0.01742495532794093
H = np.inf
REG = 1e-5
BEST_LOSS = 0.06423696912419681
EPS = 0.004
THETA = BEST_LOSS * (1 + EPS)

print(f"Using theta = {THETA}, reg = {REG}, H = {H} with eps = {EPS}")

df = pd.read_csv(NHANES_DATA_FILE)
df_std = pd.read_csv(NHANES_STD_FILE)

hours_worked_map = {
    "<=20": 1,
    "21-40": 2,
    ">=41": 3
}

gender_map = {
    "Female": 1,
    "Male": 2
}

age_map = {
    "<=18": 1,
    "19-30": 2,
    "31-50": 3,
    "51-70": 4,
    ">=70": 5
}

race_map = {
    "White": 1,
    "Black": 2,
    "Hispanic": 3,
    "Other": 3
}

education_map = {
    "< GED": 1,
    "GED": 2,
    "College": 3
}

marital_status_map = {
    "Single": 1,
    "Married": 2,
    "Divorced/Widowed": 3
}

income_map = {
    "<20k": 1,
    "20k-45k": 2,
    "45k-75k": 3,
    ">=75k": 4
}


df["HoursWorked"] = df["HoursWorked"].map(hours_worked_map)
df["Gender"] = df["Gender"].map(gender_map)
df["Age"] = df["Age"].map(age_map)
df["Race"] = df["Race"].map(race_map)
df["Education"] = df["Education"].map(education_map)
df["MaritalStatus"] = df["MaritalStatus"].map(marital_status_map)
df["HouseholdIncome"] = df["HouseholdIncome"].map(income_map)

df_std["HoursWorked"] = df_std["HoursWorked"].map(hours_worked_map)
df_std["Gender"] = df_std["Gender"].map(gender_map)
df_std["Age"] = df_std["Age"].map(age_map)
df_std["Race"] = df_std["Race"].map(race_map)
df_std["Education"] = df_std["Education"].map(education_map)
df_std["MaritalStatus"] = df_std["MaritalStatus"].map(marital_status_map)
df_std["HouseholdIncome"] = df_std["HouseholdIncome"].map(income_map)


df2 = df.drop(df[df['Telomean'] > 9].index)
df_std_2 = df_std.drop(df_std[df_std['Telomean'] > 9].index)

Z = df.to_numpy()

# Delete the outlier
Z = np.delete(Z, [4877], axis=0)

X = Z[:, [1, 2, 3, 5]]
y = Z[:, 0]
# y = (y - np.min(y)) / (np.max(y) - np.min(y))
y = y.reshape((-1, 1))

num_data = X.shape[0]
print(f"There are {num_data} data points")

M = 4
R = np.array([4, 3, 6, 4])

num_profiles = 2**M
profiles, profile_map = hasse.enumerate_profiles(M)

all_policies = hasse.enumerate_policies(M, R)
num_policies = len(all_policies)

policies_profiles = {}
policies_profiles_masked = {}
policies_ids_profiles = {}
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

D = np.zeros(shape=y.shape, dtype=np.int64)
profiles_in_data = []
for i in range(num_data):
    policy_i = tuple([int(x) for x in X[i, :]])
    policy_idx = [idx for idx in range(num_policies) if all_policies[idx] == policy_i]
    profiles_in_data.append(hasse.policy_to_profile(policy_i))
    D[i, 0] = int(policy_idx[0])

race_profiles = np.unique(Z[:, 4])

all_active_profile = tuple([1] * M)

policies_temp = [(i, x) for i, x in enumerate(all_policies) if hasse.policy_to_profile(x) == all_active_profile]
unzipped_temp = list(zip(*policies_temp))
policy_race_idx = list(unzipped_temp[0])
policy_race = list(unzipped_temp[1])

D_race = {}
X_race = {}
y_race = {}
policy_means_race = {}
mu_policies_race = {}
true_best_eff_race = {}
policy_means = np.zeros(shape=(len(policy_race), 2))
for race in race_profiles:
    idx = np.where(Z[:, 4] == race)
    X_race[race] = X[idx, :][0]
    y_race[race] = y[idx, :][0]
    num_data_race = X_race[race].shape[0]

    D_race[race] = np.zeros(shape=y_race[race].shape, dtype=np.int64)
    for i in range(num_data_race):
        policy_i = tuple([int(x) for x in X_race[race][i, :]])
        policy_idx = [idx for idx in range(num_policies) if all_policies[idx] == policy_i]
        D_race[race][i, 0] = int(policy_idx[0])

    range_list = list(np.arange(len(policy_race_idx)))
    policy_map = {i: x for i, x in zip(policy_race_idx, range_list)}
    D_race[race] = np.vectorize(policy_map.get)(D_race[race])

    policy_means_race[race] = loss.compute_policy_means(D_race[race], y_race[race], len(policy_race))

    nodata_idx = np.where(policy_means_race[race][:, 1] == 0)[0]
    policy_means_race[race][nodata_idx, 0] = -np.inf
    policy_means_race[race][nodata_idx, 1] = 1
    mu_policies_race[race] = policy_means_race[race][:, 0] / policy_means_race[race][:, 1]
    policy_means_race[race][nodata_idx, 1] = 0

    policy_means_race[race][nodata_idx, 0] = 0
    policy_means += policy_means_race[race]
    policy_means_race[race][nodata_idx, 0] = -np.inf

    true_best_eff_race[race] = np.max(mu_policies_race[race])
    # print(true_best_eff, np.where(mu_policies_race[race] == true_best_eff_race[race]))

nodata_idx = np.where(policy_means[:, 1] == 0)[0]
policy_means[nodata_idx, 0] = -np.inf
policy_means[nodata_idx, 1] = 1
mu_policies = policy_means[:, 0] / policy_means[:, 1]
policy_means[nodata_idx, 1] = 0

true_best_eff = np.max(mu_policies)
# print(true_best_eff, np.where(mu_policies == true_best_eff))

D_remapped = D.copy()
range_list = list(np.arange(len(policy_race_idx)))
policy_map = {i: x for i, x in zip(policy_race_idx, range_list)}
D_remapped = np.vectorize(policy_map.get)(D_remapped)

policy_race_idx_full = {}
for k, race in enumerate(race_profiles):
    policy_race_idx_full[k] = (k*len(range_list) + np.array(range_list)).tolist()

policy_means_race_full = np.concatenate((policy_means_race[1], policy_means_race[2], policy_means_race[3]), axis=0)

# In the best case, every other profile becomes a single pool
# So max number of pools per profile is adjusted accordingly
# H_profile = H - num_profiles + 1
unordered_factors = race_profiles
num_unordered_factors = len(unordered_factors)
H_profile = H - num_unordered_factors + 1

# Subset data by profiles and find equiv policy lower bound
D_profiles = {}
y_profiles = {}
policies_profiles = {}
policy_means_profiles = {}
eq_lb_profiles = np.zeros(shape=(num_unordered_factors,))
for k, profile in enumerate(unordered_factors):

    policy_profiles_idx_k = policy_race_idx
    policies_profiles[k] = policy_race

    D_k = D_race[profile]
    # range_list = list(np.arange(len(policy_profiles_idx_k)))
    # policy_map = {i: x for i, x in zip(policy_profiles_idx_k, range_list)}
    # D_k = np.vectorize(policy_map.get)(D_k)
    y_k = y_race[profile]
    D_profiles[k] = D_k
    y_profiles[k] = y_k

    if D_k is None:
        policy_means_profiles[k] = None
        eq_lb_profiles[k] = 0
        H_profile += 1
    else:
        policy_means_k = policy_means_race[profile]
        policy_means_profiles[k] = policy_means_k
        eq_lb_profiles[k] = find_profile_lower_bound(D_k, y_k, policy_means_k)

eq_lb_profiles /= num_data
eq_lb_sum = np.sum(eq_lb_profiles)
print(eq_lb_profiles, eq_lb_sum)

# # Now solve each profile independently
# # This step can be parallelized
rashomon_profiles: list[RashomonSet] = [None]*num_unordered_factors
feasible = True
for k, profile in enumerate(unordered_factors):
    theta_k = THETA - (eq_lb_sum - eq_lb_profiles[k])
    print(theta_k)
    D_k = D_profiles[k]
    y_k = y_profiles[k]

    policies_k = policies_profiles[k]
    policy_means_k = policy_means_profiles[k]
    profile_mask = list(map(bool, all_active_profile))
    # print(profile_mask)

    # # Mask the empty arms
    for idx, pol in enumerate(policies_k):
        policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
    # R_k = R[profile_mask]
    # M_k = np.sum(profile)
    R_k = R
    M_k = M
    # print(len(policies_k), policy_means_k.shape)

    if D_k is None:
        # TODO: Put all possible sigma matrices here and set loss to 0
        rashomon_profiles[k] = RashomonSet(shape=None)
        rashomon_profiles[k].P_qe = [None]
        rashomon_profiles[k].Q = np.array([0])
        continue

    # Control group is just one policy
    if M_k == 0:
        rashomon_k = RashomonSet(shape=None)
        control_loss = eq_lb_profiles[k] + REG
        rashomon_k.P_qe = [None]
        rashomon_k.Q = np.array([control_loss])
    else:
        rashomon_k = RAggregate_profile(M_k, R_k, H_profile, D_k, y_k, theta_k, all_active_profile, REG,
                                        policies_k, policy_means_k, normalize=num_data)
        rashomon_k.calculate_loss(D_k, y_k, policies_k, policy_means_k, REG, normalize=num_data)

    rashomon_k.sort()
    rashomon_profiles[k] = rashomon_k
    if len(rashomon_k) == 0:
        feasible = False
        print(f"No Rashomon set for race profile {profile}")
        break
    else:
        print(f"Race: {profile}. Found {len(rashomon_profiles[k])} partitions")


# Combine solutions in a feasible way
if feasible:
    R_set = find_feasible_combinations(rashomon_profiles, THETA, H, sorted=True)
else:
    R_set = []
if len(R_set) > 0:
    rashomon_profiles = remove_unused_poolings(R_set, rashomon_profiles)

print(f"Size of RPS = {len(R_set)}")

for k, profile in enumerate(race_profiles):
    print(f"Race: {profile}. Found {len(rashomon_profiles[k])} partitions")

rashomon_homogeneous = RAggregate_profile(M, R, H, D_remapped, y, THETA, all_active_profile, REG,
                                          policy_race, policy_means, normalize=num_data)
rashomon_homogeneous.calculate_loss(D_remapped, y, policy_race, policy_means, REG, normalize=num_data)

print(len(rashomon_homogeneous))

R_set_pruned = []

for r in R_set:
    current_sigma = None
    # homoegeneous = False
    homoegeneous = True
    for race in race_profiles:
        race = int(race) - 1
        sigma_id = r[race]
        sigma_race = rashomon_profiles[race].sigma[sigma_id]
        if current_sigma is None:
            current_sigma = sigma_race
        elif not np.array_equal(current_sigma, sigma_race):
            homoegeneous = False
            break

    if homoegeneous:
        found_in_homogeneous = False
        for idx, r_sigma in enumerate(rashomon_homogeneous.sigma):
            if np.array_equal(r_sigma, current_sigma):
                found_in_homogeneous = True
                R_set_pruned.append([-1, idx])
                break
        if not found_in_homogeneous:
            print("Not found :(")
        else:
            R_set_pruned.append(r)
    else:
        R_set_pruned.append(r)

print(f"Size of R_set_pruned = {len(R_set_pruned)}")

full_rashomon_data = {
    "R_set_pruned": R_set_pruned,
    "rashomon_profiles": rashomon_profiles,
    "rashomon_homogeneous": rashomon_homogeneous
}

with open(NHANES_RPS_FILE, "wb") as f:
    pickle.dump(full_rashomon_data, f, pickle.HIGHEST_PROTOCOL)

print("Done pickling!")

model_losses = []
model_sizes = []

for i, r in enumerate(R_set_pruned):
    if r[0] == -1:
        model_i = r[1]
        loss_i = rashomon_homogeneous.loss[model_i]
        size_i = rashomon_homogeneous.pools[model_i]
    else:
        model_i = r
        loss_i = 0
        size_i = 0
        for race, race_model in enumerate(model_i):
            loss_i += rashomon_profiles[race].loss[race_model]
            size_i += rashomon_profiles[race].num_pools[race_model]
    model_losses.append(loss_i)
    model_sizes.append(size_i)

model_losses = np.array(model_losses)
model_sizes = np.array(model_sizes)
# model_mses = model_losses - reg * model_sizes

best_loss = np.min(model_losses)
worst_loss = np.max(model_losses)

eps = (worst_loss - best_loss) / best_loss
print(f"Best loss = {best_loss}, Worst loss = {worst_loss}, epsilon = {eps}")

model_post_prob = np.exp(-model_losses)
rel_post_prob_ratio = (model_post_prob - np.max(model_post_prob)) / np.max(model_post_prob)

scaled_prob_threshold = 0.6
model_size_threshold = 80

q0 = best_loss
strict_eps = 0.002
loss_threshold = q0 * (1 + strict_eps)

precomputed_pools = {}

R_set_2 = []
model_sizes_2 = []
model_losses_2 = []
model_pools_2 = []
model_pool_means_2 = []
for idx, r_set in enumerate(R_set_pruned):

    if model_losses[idx] > loss_threshold:
        continue

    if r_set[0] != -1:
        pi_policies_profiles_r = {}
        for race, model_race in enumerate(r_set):
            sigma_r_prof = rashomon_profiles[race].sigma[model_race]
            sigma_bytes = sigma_r_prof.tobytes()
            if sigma_bytes in precomputed_pools.keys():
                _, pi_policies_r_k = precomputed_pools[sigma_bytes]
            else:
                pi_pools_r_k, pi_policies_r_k = extract_pools.extract_pools(policy_race, sigma_r_prof)
                precomputed_pools[sigma_bytes] = (pi_pools_r_k, pi_policies_r_k)
            pi_policies_profiles_r[race] = pi_policies_r_k

        pi_pools_r, pi_policies_r = extract_pools.aggregate_pools(pi_policies_profiles_r, policy_race_idx_full)
        pool_means_r = loss.compute_pool_means(policy_means_race_full, pi_pools_r)

    else:
        sigma_r = rashomon_homogeneous.sigma[r_set[1]]
        sigma_bytes = sigma_r.tobytes()
        if sigma_bytes in precomputed_pools.keys():
            pi_pools_r, pi_policies_r = precomputed_pools[sigma_bytes]
        else:
            pi_pools_r, pi_policies_r = extract_pools.extract_pools(policy_race, sigma_r)
            precomputed_pools[sigma_bytes] = (pi_pools_r, pi_policies_r)
        # pi_pools_r, pi_policies_r = extract_pools.extract_pools(policy_race, sigma_r)
        pool_means_r = loss.compute_pool_means(policy_means, pi_pools_r)
        # TODO: Check whether the policy ids are correctly mapped
        # If not, map them here instead of when estimating effects

    R_set_2.append(r_set)
    model_sizes_2.append(model_sizes[idx])
    model_losses_2.append(model_losses[idx])
    model_pools_2.append(pi_pools_r)
    model_pool_means_2.append(pool_means_r)

print(f"RPS after pruning: {len(R_set_2)}")


model_sizes_2 = np.array(model_sizes_2)
model_losses_2 = np.array(model_losses_2)
model_mses_2 = model_losses_2 - REG * model_sizes_2
model_post_prob_2 = np.exp(-model_losses_2)
model_scaled_post_prob_2 = model_post_prob_2 - np.min(model_post_prob_2)
model_scaled_post_prob_2 = model_scaled_post_prob_2 / np.max(model_scaled_post_prob_2)


best_loss = np.min(model_losses_2)
worst_loss = np.max(model_losses_2)

eps = (worst_loss - best_loss) / best_loss
print(f"Best loss = {best_loss}, Worst loss = {worst_loss}, epsilon = {eps}")

rashomon_reduced_data = {
    "R_set": R_set_2,
    "sizes": model_sizes_2,
    "losses": model_losses_2,
    "pools": model_pools_2,
    "pool_means": model_pool_means_2
}

with open(NHANES_PRUNED_RPS_FILE, "wb") as f:
    pickle.dump(rashomon_reduced_data, f, pickle.HIGHEST_PROTOCOL)

print("Done pickling reduced RPS!")
