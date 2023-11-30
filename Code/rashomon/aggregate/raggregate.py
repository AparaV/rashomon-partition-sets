import numpy as np

from sklearn.metrics import mean_squared_error

from .profile import RAggregate_profile
from .utils import find_feasible_sum_subsets

from .. import loss
from ..tva import enumerate_profiles, enumerate_policies, policy_to_profile
from ..sets import RashomonSet


# rashomon aggregation across profiles
# Need to keep track of seen problems and their lower bounds
#   maybe a dictionary?
#   actually no need to do this
# Overall threshold is theta
# For each profile, find the equivalent policy lower bound
#   call this t_i for profile i
# Then, for profile k, the threshold is theta_k = theta - \sum_{i \neq k} t_i
# Get R-sets for each profile, R_i for profile i
# Now, we need to mix and match across profiles

def subset_data(D, y, policy_profiles_idx):
    # The idea here is that values in D corresponds to the index of that policy
    # So we mask and retrieve those values
    mask = np.isin(D, policy_profiles_idx)
    D_profile = np.reshape(D[mask], (-1, 1))
    y_profile = np.reshape(y[mask], (-1, 1))

    # Now remap policies from overall indicies to the indicies within that profile
    range_list = list(np.arange(len(policy_profiles_idx)))
    policy_map = {i: x for i, x in zip(policy_profiles_idx, range_list)}
    D_profile = np.vectorize(policy_map.get)(D_profile)

    return D_profile, y_profile


def find_profile_lower_bound(D_k, y_k, policy_means_k):
    mu = np.float64(policy_means_k[:, 0]) / policy_means_k[:, 1]
    mu_D = mu[list(D_k.reshape((-1,)))]
    mse = mean_squared_error(y_k[:, 0], mu_D)
    return mse


def find_feasible_combinations(rashomon_profiles: list[RashomonSet], theta, sorted=False):

    if not sorted:
        rashomon_profiles = [r.sort() for r in rashomon_profiles]

    S = [r.Q for r in rashomon_profiles]
    feasible_combinations = find_feasible_sum_subsets(S, theta)
    return feasible_combinations


def RAggregate(M, R, H, D, y, theta, reg=1):

    num_profiles = 2**M
    profiles = enumerate_profiles(M)
    all_policies = enumerate_policies(M, R)

    # In the best case, every other profile becomes a single pool
    # So max number of pools per profile is adjusted accordingly
    H_profile = H - num_profiles + 1

    # Subset data by profiles and find equiv policy lower bound
    D_profiles = {}
    y_profiles = {}
    policies_profiles = {}
    policy_means_profiles = {}
    eq_lb_profiles = [0] * num_profiles
    for k, profile in enumerate(profiles):

        policies_temp = [(i, x) for i, x in enumerate(all_policies) if policy_to_profile(x) == profile]
        unzipped_temp = zip(*policies_temp)
        policy_profiles_idx_k = list(unzipped_temp[0])
        policies_profiles[k] = list(unzipped_temp[1])

        D_k, y_k = subset_data(D, y, policy_profiles_idx_k)
        D_profiles[k] = D_k
        y_profiles[k] = y_k

        policy_means_k = loss.compute_policy_means(D_k, y_k, len(policies_profiles[k]))
        policy_means_profiles[k] = policy_means_k
        eq_lb_profiles[k] = find_profile_lower_bound(D_k, y_k, policy_means_k)

    eq_lb_sum = np.sum(eq_lb_profiles)

    # Now solve each profile independently
    # This step can be parallelized
    rashomon_profiles = {}
    for k, profile in enumerate(profiles):
        theta_k = eq_lb_sum - eq_lb_profiles[k]
        D_k = D_profiles[k]
        y_k = y_profiles[k]

        policies_k = policies_profiles[k]
        policy_means_k = policy_means_profiles[k]

        rashomon_k = RAggregate_profile(M, R, H_profile, D_k, y_k, theta_k, profile, reg, policies_k, policy_means_k)
        rashomon_k.calculate_loss(D_k, y_k, policies_k, policy_means_k, reg)
        rashomon_k.sort()

        rashomon_profiles[k] = rashomon_k

    # Combine solutions in a feasible way
    R_set = find_feasible_combinations(rashomon_profiles, theta, sorted=True)

    return R_set
