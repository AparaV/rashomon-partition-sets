import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from .profile_slopes import RAggregate_profile_slopes
from .raggregate import find_feasible_combinations, remove_unused_poolings

from ..hasse import enumerate_profiles, enumerate_policies, policy_to_profile
from ..sets import RashomonSet


def subset_data(D, X, y, policy_profiles_idx):
    # The idea here is that values in D corresponds to the index of that policy
    # So we mask and retrieve those values
    mask = np.isin(D, policy_profiles_idx)
    D_profile = np.reshape(D[mask], (-1, 1))
    y_profile = np.reshape(y[mask], (-1, 1))
    X_profile = X[mask[:, 0], :]

    # Now remap policies from overall indicies to the indicies within that profile
    range_list = list(np.arange(len(policy_profiles_idx)))
    policy_map = {i: x for i, x in zip(policy_profiles_idx, range_list)}
    if len(D_profile) == 0:
        D_profile = None
        y_profile = None
    else:
        D_profile = np.vectorize(policy_map.get)(D_profile)

    return D_profile, X_profile, y_profile


def find_profile_lower_bound_slopes(D_k, X_k, y_k, policies_k):
    n_k = D_k.shape[0]

    y_est = np.zeros(shape=D_k.shape) + np.inf

    for idx, pol_k in enumerate(policies_k):

        # Extract the X matrix
        i_idx = [i for i, p in enumerate(D_k[:, 0]) if p == pol_k]
        X_i = X_k[i_idx, :]
        y_i = y_k[i_idx, :]

        # Run regression and estimate outcomes
        model_i = LinearRegression().fit(X_i, y_i)
        y_est_i = model_i.predict(X_i)
        y_est[i_idx, :] = y_est_i

    mse = mean_squared_error(y_k[:, 0], y_est) * n_k

    return mse


def RAggregate_slopes(M, R, H, D, X, y, theta, reg=1, verbose=False):

    num_profiles = 2**M
    profiles, profile_map = enumerate_profiles(M)
    all_policies = enumerate_policies(M, R)
    num_data = D.shape[0]
    if isinstance(R, int):
        R = np.array([R]*M)

    # In the best case, every other profile becomes a single pool
    # So max number of pools per profile is adjusted accordingly
    H_profile = H - num_profiles + 1

    # Subset data by profiles and find equiv policy lower bound
    D_profiles = {}
    y_profiles = {}
    X_profiles = {}
    policies_profiles = {}
    policy_means_profiles = {}
    eq_lb_profiles = np.zeros(shape=(num_profiles,))
    for k, profile in enumerate(profiles):

        policies_temp = [(i, x) for i, x in enumerate(all_policies) if policy_to_profile(x) == profile]
        unzipped_temp = list(zip(*policies_temp))
        policy_profiles_idx_k = list(unzipped_temp[0])
        policies_profiles[k] = list(unzipped_temp[1])
        policies_idx_k = list(range(len(policy_profiles_idx_k)))

        D_k, X_k, y_k = subset_data(D, X, y, policy_profiles_idx_k)
        D_profiles[k] = D_k
        y_profiles[k] = y_k
        X_profiles[k] = X_k

        if D_k is None:
            policy_means_profiles[k] = None
            eq_lb_profiles[k] = 0
            H_profile += 1
        else:
            eq_lb_profiles[k] = find_profile_lower_bound_slopes(D_k, X_k, y_k, policies_idx_k)

    eq_lb_profiles /= num_data
    eq_lb_sum = np.sum(eq_lb_profiles)

    # Now solve each profile independently
    # This step can be parallelized
    rashomon_profiles: list[RashomonSet] = [None]*num_profiles
    feasible = True
    for k, profile in enumerate(profiles):
        theta_k = theta - (eq_lb_sum - eq_lb_profiles[k])
        D_k = D_profiles[k]
        X_k = X_profiles[k]
        y_k = y_profiles[k]

        policies_k = policies_profiles[k]
        profile_mask = list(map(bool, profile))

        # Mask the empty arms
        for idx, pol in enumerate(policies_k):
            policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
        R_k = R[profile_mask]
        M_k = np.sum(profile)

        if D_k is None:
            # TODO: Put all possible sigma matrices here and set loss to 0
            rashomon_profiles[k] = RashomonSet(shape=None)
            rashomon_profiles[k].P_qe = [None]
            rashomon_profiles[k].Q = np.array([0])
            if verbose:
                print(f"Skipping profile {profile}")
            continue

        # Control group is just one policy
        if verbose:
            print(profile, theta_k)
        if M_k == 0 or (len(R_k) == 1 and R_k[0] == 2):
            rashomon_k = RashomonSet(shape=None)
            control_loss = eq_lb_profiles[k] + reg
            rashomon_k.P_qe = [None]
            rashomon_k.Q = np.array([control_loss])
        else:
            rashomon_k = RAggregate_profile_slopes(
                M_k, R_k, H_profile, D_k, X_k, y_k, theta_k, profile, reg,
                policies_k, normalize=num_data)
            rashomon_k.calculate_loss(
                D_k, y_k, policies_k, policy_means=None, reg=reg, normalize=num_data,
                slopes=True, X=X_k)

        rashomon_k.sort()
        rashomon_profiles[k] = rashomon_k
        if verbose:
            print(len(rashomon_k))
        if len(rashomon_k) == 0:
            feasible = False

    # Combine solutions in a feasible way
    if verbose:
        print("Finding feasible combinations")
    if feasible:
        R_set = find_feasible_combinations(
            rashomon_profiles, theta, H, sorted=True, verbose=verbose)
    else:
        R_set = []
    if len(R_set) > 0:
        rashomon_profiles = remove_unused_poolings(R_set, rashomon_profiles)

    return R_set, rashomon_profiles
