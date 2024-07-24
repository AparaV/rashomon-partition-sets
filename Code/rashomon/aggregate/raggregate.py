import time
import numpy as np

from sklearn.metrics import mean_squared_error

from .profile import RAggregate_profile, _brute_RAggregate_profile
from .utils import find_feasible_sum_subsets

from .. import loss
from ..tva import enumerate_profiles, enumerate_policies, policy_to_profile
from ..sets import RashomonSet


def subset_data(D: np.ndarray, y: np.ndarray,
                policy_profiles_idx: list[int]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Subset data into different profiles

    Arguments:
    D (np.ndarray): Data i.e., policy integers
    y (np.ndarray): Outcomes
    policy_profiles_idx (list[int]): List of policy indices for this profile

    Returns:
    (D_profile, y_profile): Subset of D and y for this profile
        D_profile (np.ndarray): Subset of D for this profile
        y_profile (np.ndarray): Subset of y for this profile
    """
    # The idea here is that values in D corresponds to the index of that policy
    # So we mask and retrieve those values
    mask = np.isin(D, policy_profiles_idx)
    D_profile = np.reshape(D[mask], (-1, 1))
    y_profile = np.reshape(y[mask], (-1, 1))

    # Now remap policies from overall indicies to the indicies within that profile
    range_list = list(np.arange(len(policy_profiles_idx)))
    policy_map = {i: x for i, x in zip(policy_profiles_idx, range_list)}
    if len(D_profile) == 0:
        D_profile = None
        y_profile = None
    else:
        D_profile = np.vectorize(policy_map.get)(D_profile)

    return D_profile, y_profile


def find_profile_lower_bound(D_k: np.ndarray, y_k: np.ndarray, policy_means_k: np.ndarray) -> float:
    """ Find the MSE lower bound for this profile """
    n_k = D_k.shape[0]
    nodata_idx = np.where(policy_means_k[:, 1] == 0)[0]
    policy_means_k[nodata_idx, 1] = 1
    mu = np.float64(policy_means_k[:, 0]) / policy_means_k[:, 1]
    policy_means_k[nodata_idx, 1] = 0
    mu[nodata_idx] = 0
    mu_D = mu[list(D_k.reshape((-1,)))]
    mse = mean_squared_error(y_k[:, 0], mu_D) * n_k
    return mse


def find_feasible_combinations(rashomon_profiles: list[RashomonSet], theta: float, H: int,
                               sorted: bool = False, verbose: bool = False) -> list[list[int]]:
    """
    Find feasible combinations of poolings across profiles based on Rashomon threshold

    Arguments:
    rashomon_profiles (list[RashomonSet]): List of RashomonSet objects for each profile
    theta (float): Rashomon threshold
    H (int): Maximum number of pools
    sorted (bool): Whether the profiles are sorted by loss. Defaults to False
    verbose (bool): Print debug information. Defaults to False

    Returns:
    list[list[int]]: List of feasible combinations of poolings.
        Each list contains the indices of the pools in the RashomonSet
    """

    if not sorted:
        for idx, r in enumerate(rashomon_profiles):
            rashomon_profiles[idx].sort()

    for idx, r in enumerate(rashomon_profiles):
        _ = rashomon_profiles[idx].pools

    losses = [r.loss for r in rashomon_profiles]

    if verbose:
        first_loss = [x[0] for x in losses]
        last_loss = [x[-1] for x in losses]
        print(f"Min = {np.sum(first_loss)}, Max = {np.sum(last_loss)}")

    loss_combinations = find_feasible_sum_subsets(losses, theta)
    # print(f"Found {len(loss_combinations)}")

    # Filter based on pools
    feasible_combinations = []
    for ctr, comb in enumerate(loss_combinations):
        # if (ctr + 1) % 1000 == 0:
        #     print(ctr)
        pools = 0
        for k, idx in enumerate(comb):
            if rashomon_profiles[k].sigma[idx] is None:
                if rashomon_profiles[k].Q[idx] > 0:
                    pools += 1
            else:
                pools += rashomon_profiles[k].pools[idx]
        if pools <= H:
            feasible_combinations.append(comb)

    return feasible_combinations


def remove_unused_poolings(R_set: list[list[int]],
                           rashomon_profiles: list[RashomonSet]) -> list[RashomonSet]:
    """ Remove any pools that are absent in the final RPS solution """
    R_set_np = np.array(R_set)
    sigma_max_id = np.max(R_set_np, axis=0)

    for k, max_id in enumerate(sigma_max_id):
        rashomon_profiles[k].P_qe = rashomon_profiles[k].P_qe[:(max_id+1)]
        rashomon_profiles[k].Q = rashomon_profiles[k].Q[:(max_id+1)]
    return rashomon_profiles


def RAggregate(M: int, R: int | np.ndarray[int], H: int, D: np.ndarray, y: np.ndarray,
               theta: float, reg: float = 1, verbose: bool = False,
               bruteforce: bool = False) -> tuple[list[list[int]], list[RashomonSet]]:
    """
    RPS enumeration algorithm

    Parameters:
    M (int): Number of features
    R (int or np.ndarray of int): Number of factor levels per feature
    H (int): Maximum number of pools in this profile
    D (np.ndarray): Data i.e., policy integers
    y (np.ndarray): Outcomes
    theta (float): Rashomon threshold
    reg (float): Regularization parameter. Defaults to 1
    verbose (bool): Print debug information. Defaults to False
    bruteforce (bool): Use brute force instead of adaptive. Defaults to False

    Returns:
    (R_set, rashomon_profiles): Rashomon Partitions Set
        R_set (list[list[int]]): List of list of indices of pools for each profile
        rashomon_profiles (list[RashomonSet]): Set of Rashomon pools for each profile
    """

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
    policies_profiles = {}
    policy_means_profiles = {}
    eq_lb_profiles = np.zeros(shape=(num_profiles,))
    for k, profile in enumerate(profiles):

        policies_temp = [(i, x) for i, x in enumerate(all_policies) if policy_to_profile(x) == profile]
        unzipped_temp = list(zip(*policies_temp))
        policy_profiles_idx_k = list(unzipped_temp[0])
        policies_profiles[k] = list(unzipped_temp[1])

        D_k, y_k = subset_data(D, y, policy_profiles_idx_k)
        D_profiles[k] = D_k
        y_profiles[k] = y_k

        if D_k is None:
            policy_means_profiles[k] = None
            eq_lb_profiles[k] = 0
            H_profile += 1
        else:
            policy_means_k = loss.compute_policy_means(D_k, y_k, len(policies_profiles[k]))
            policy_means_profiles[k] = policy_means_k
            eq_lb_profiles[k] = find_profile_lower_bound(D_k, y_k, policy_means_k)

    eq_lb_profiles /= num_data
    eq_lb_sum = np.sum(eq_lb_profiles)

    # Now solve each profile independently
    # This step can be parallelized
    rashomon_profiles: list[RashomonSet] = [None]*num_profiles
    feasible = True
    for k, profile in enumerate(profiles):
        theta_k = theta - (eq_lb_sum - eq_lb_profiles[k])
        D_k = D_profiles[k]
        y_k = y_profiles[k]

        policies_k = policies_profiles[k]
        policy_means_k = policy_means_profiles[k]
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
            # print(R_k, np.sum(R_k))
            if not bruteforce:
                if verbose:
                    print("Adaptive")
                    start = time.time()

                rashomon_k = RAggregate_profile(M_k, R_k, H_profile, D_k, y_k, theta_k, profile, reg,
                                                policies_k, policy_means_k, normalize=num_data)
                if verbose:
                    end = time.time()
                    elapsed = end - start
                    print(f"Took {elapsed} s adaptively")

                # start = time.time()
                rashomon_k.calculate_loss(D_k, y_k, policies_k, policy_means_k, reg, normalize=num_data)
                # end = time.time()
                # elapsed = end - start
                # print(f"Took {elapsed} s to calculate loss")
            else:
                if verbose:
                    print("Brute forcing")
                    start = time.time()

                rashomon_k = _brute_RAggregate_profile(
                    M_k, R_k, H_profile, D_k, y_k, theta_k, profile, reg,
                    policies_k, policy_means_k, normalize=num_data)

                if verbose:
                    end = time.time()
                    elapsed = end - start
                    print(f"Took {elapsed} s when brute forcing")

        # start = time.time()
        rashomon_k.sort()
        # end = time.time()
        # elapsed = end - start
        # print(f"Took {elapsed} s to sort")
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
