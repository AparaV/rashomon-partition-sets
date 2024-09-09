import itertools
import numpy as np


def __enumerate_policies_classic__(M: int, R: int) -> list:
    """ Enumerate all features for M arms each with fixed R factor levels """
    intensities = np.arange(R)
    policies = []
    for pol in itertools.product(intensities, repeat=M):
        policies.append(pol)
    return policies


def __enumerate_policies_complex__(R: np.ndarray) -> list:
    """ Enumerate all features with varying R factor levels """
    intensities = []
    for Ri in R:
        intensities.append(np.arange(Ri))
    policies = []
    for pol in itertools.product(*intensities, repeat=1):
        policies.append(pol)
    return policies


def enumerate_policies(M: int, R: int | np.ndarray) -> list:
    """
    Enumerate all possible features

    Arguments:
    M (int): Number of arms
    R (int | np.ndarray): Number of factor levels for each arm.
        If type is of int, all arms have same number of factor levels

    Returns:
    policies (list): List of all possible features
    """
    if isinstance(R, int) or len(R) == 1:
        if not isinstance(R, int):
            R = R[0]
        policies = __enumerate_policies_classic__(M, R)
    else:
        policies = __enumerate_policies_complex__(R)
    return policies


def enumerate_profiles(M: int) -> tuple[list, dict[tuple, int]]:
    """
    Enumerate profiles for M arms

    Arguments:
    M (int): Number of arms

    Returns:
    (profiles, profile_map)
        profiles (list): List of all possible profiles
        profile_map (dict[tuple, int]): Mapping from profile to index in profiles
    """
    profiles = itertools.product([0, 1], repeat=M)
    profiles = [x for x in profiles]
    profile_map = {}
    for i, x in enumerate(profiles):
        profile_map[x] = i
    return profiles, profile_map


def policy_to_profile(policy: tuple) -> tuple:
    """ Identifies the profile that the feature belongs to """
    profile = tuple([int(x > 0) for x in policy])
    return profile


def weakly_dominates(x: tuple, y: tuple) -> bool:
    """ Check if feature x weakly dominates feature y """
    M = len(x)
    dominates = True
    for i in range(M):
        if x[i]+y[i] != 0 and x[i]*y[i] == 0:
            dominates = False
            break
        if x[i] < y[i]:
            dominates = False
            break
    return dominates


def alpha_matrix(policies: list) -> np.ndarray:
    """
    Construct the TVA alpha matrix transformation

    Arguments:
    policies (list): List of features

    Returns:
    G (np.ndarray): Alpha matrix
    """
    n = len(policies)
    G = np.ndarray(shape=(n, n))
    G[:, :] = 0

    for i, pol_i in enumerate(policies):
        j_list = []
        profile_i = policy_to_profile(pol_i)
        for j, pol_j in enumerate(policies):
            profile_j = policy_to_profile(pol_i)
            if profile_j == profile_i and weakly_dominates(pol_i, pol_j):
                j_list.append(j)
        G[i, j_list] = 1
    return G


def get_dummy_matrix(D: np.ndarray, G: np.ndarray, num_policies: int) -> np.ndarray:
    """
    Construct the dummy matrix for TVA

    Arguments:
    D (np.ndarray): Data matrix
    G (np.ndarray): Alpha matrix
    num_policies (int): Number of policies

    Returns:
    D_matrix (np.ndarray): Dummy matrix
    """
    num_data, _ = D.shape
    D_matrix = np.ndarray(shape=(num_data, num_policies))
    D_matrix[:, :] = 0
    for i in range(num_data):
        pol_i = D[i, 0]
        D_matrix[i, :] = G[pol_i, :]
    return D_matrix


def profile_ids_to_univ_ids(pi_pools_0: dict[int, list[int]],
                            univ_pol_id_list: list[int]) -> dict[int, list[int]]:
    """
    Convert policy IDs in pi_pools_0 to global policy IDs

    Arguments:
    pi_pools_0 (dict[int, list[int]]): key is pool ID, value is list of policy IDs within that profile
    univ_pol_id_list (list[int]): List of global policy IDs for profile corresponding to pi_pools_0
        Policy with profile ID `i` has global ID `univ_pol_id_list[i]`

    Returns:
    pi_pools (dict[int, list[int]]): Policy IDs for each pool now correspond to global policy ID
    """
    pi_pools = {}
    for pi_id, pol_idx_list in pi_pools_0.items():
        pol_univ_list = [univ_pol_id_list[idx] for idx in pol_idx_list]
        pi_pools[pi_id] = pol_univ_list
    return pi_pools


def remove_arm(policies_list: list[tuple], arm_idx: int) -> list[tuple]:
    """
    Removes dosage corresponding to arm_idx in policies_list

    Arguments:
    policies_list (list): List of tuples corresponding to policy dosages
    arm_idx (int): Index of arm to remove

    Returns:
    policies_list (list[tuple]): List of tuples corresponding to feature levels
        after arm_idx is removed
    """
    policies_list = [tuple(pol[:arm_idx] + pol[(arm_idx+1):]) for pol in policies_list]
    return policies_list
