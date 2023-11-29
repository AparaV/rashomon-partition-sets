import itertools
import numpy as np


#
# Enumerate policies
#


def __enumerate_policies_classic__(M, R):
    # All arms have the same intensities
    intensities = np.arange(R)
    policies = []
    for pol in itertools.product(intensities, repeat=M):
        policies.append(pol)
    return policies


def __enumerate_policies_complex__(R):
    # Each arm may have different intensities
    intensities = []
    for Ri in R:
        intensities.append(np.arange(Ri))
    policies = []
    for pol in itertools.product(*intensities, repeat=1):
        policies.append(pol)
    return policies


def enumerate_policies(M, R):
    if isinstance(R, int) or len(R) == 1:
        if not isinstance(R, int):
            R = R[0]
        policies = __enumerate_policies_classic__(M, R)
    else:
        policies = __enumerate_policies_complex__(R)
    return policies


def enumerate_profiles(M):
    profiles = itertools.product([0, 1], repeat=M)
    profiles = [x for x in profiles]
    profile_map = {}
    for i, x in enumerate(profiles):
        profile_map[x] = i
    return profiles, profile_map


def policy_to_profile(policy):
    profile = tuple([int(x > 0) for x in policy])
    return profile


def weakly_dominates(x, y):
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


def alpha_matrix(M, R, policies):
    n = len(policies)
    G = np.ndarray(shape=(n, n))
    G[:, :] = 0

    for i, pol_i in enumerate(policies):
        j_list = []
        for j, pol_j in enumerate(policies):
            if weakly_dominates(pol_i, pol_j):
                j_list.append(j)
        G[i, j_list] = 1
    return G


def get_dummy_matrix(D, G, num_policies):
    num_data, _ = D.shape
    D_matrix = np.ndarray(shape=(num_data, num_policies))
    D_matrix[:, :] = 0
    for i in range(num_data):
        pol_i = D[i, 0]
        D_matrix[i, :] = G[pol_i, :]
    return D_matrix
