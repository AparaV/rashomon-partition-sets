import numpy as np

from CTL.causal_tree_learn import CausalTree

from . import extract_pools
from .tva import enumerate_profiles, enumerate_policies, policy_to_profile


def subset_data(D, D_matrix, y, policy_profiles_idx):
    # The idea here is that values in D corresponds to the index of that policy
    # So we mask and retrieve those values
    mask = np.isin(D, policy_profiles_idx)
    mask_idx = np.where(mask)
    D_profile = np.reshape(D[mask], (-1, 1))
    D_matrix_k = D_matrix[mask_idx[0], :]
    y_profile = np.reshape(y[mask], (-1, 1))

    # Now remap policies from overall indicies to the indicies within that profile
    range_list = list(np.arange(len(policy_profiles_idx)))
    policy_map = {i: x for i, x in zip(policy_profiles_idx, range_list)}
    if len(D_profile) == 0:
        D_profile = None
        y_profile = None
    else:
        D_profile = np.vectorize(policy_map.get)(D_profile)

    return D_profile, D_matrix_k, y_profile, mask


def ctl_single_profile(D, y, D_matrix):
    """
    Run causal tree for a single profile
    """

    y_1d = y.reshape((-1,))
    T_1d = 1 + np.zeros(y_1d.shape)
    ct = CausalTree(weight=0.0, split_size=0.0, cont=False,
                    min_size=0)
    ct.fit(D_matrix, y_1d, T_1d)
    ct.prune()
    y_est = ct.predict(D_matrix)

    # Pool policies
    pool_means_k = np.unique(y_est)
    pi_policies = {}
    pi_pools = {}
    for pool_id, pool_means_k_i in enumerate(pool_means_k):
        D_matrix_ids = np.where(y_est == pool_means_k_i)
        policies_ct_i = [x for x in np.unique(D[D_matrix_ids])]
        pi_pools[pool_id] = policies_ct_i
        for policy in policies_ct_i:
            pi_policies[policy] = pool_id

    return pi_pools, pi_policies, y_est


def ctl(M, R, D, y, D_matrix):
    """
    Run causal trees for multiple profiles
    """

    # TODO: Edge case when dosage in one arm is binary
    #       This will fail currently

    profiles, _ = enumerate_profiles(M)
    all_policies = enumerate_policies(M, R)
    trees = {}

    # Subset data by profiles and solve each tree independently
    policies_profiles = {}
    pi_policies_profiles = {}
    policies_ids_profiles = {}
    y_est = np.zeros(y.shape)
    for k, profile in enumerate(profiles):

        policies_temp = [(i, x) for i, x in enumerate(all_policies) if policy_to_profile(x) == profile]
        unzipped_temp = list(zip(*policies_temp))
        policies_ids_k = list(unzipped_temp[0])
        policies_profiles[k] = list(unzipped_temp[1])
        policies_ids_profiles[k] = policies_ids_k

        # Subset data
        D_k, D_matrix_k, y_k, mask_k = subset_data(D, D_matrix, y, policies_ids_k)
        if D_k is None:
            continue

        # Solve for tree
        y_k_1d = y_k.reshape((-1,))
        T_k = 1 + np.zeros(y_k_1d.shape)
        ct = CausalTree(weight=0.0, split_size=0.0, cont=False,
                        min_size=0)
        ct.fit(D_matrix_k, y_k_1d, T_k)
        ct.prune()
        y_est_k = ct.predict(D_matrix_k)
        y_est[mask_k] = y_est_k
        trees[k] = ct

        # Pool policies
        pool_means_k = np.unique(y_est_k)
        pi_policies_k = {}
        for pool_id, pool_means_k_i in enumerate(pool_means_k):
            D_matrix_ids = np.where(y_est_k == pool_means_k_i)
            policies_ct_i = [x for x in np.unique(D_k[D_matrix_ids])]
            for policy in policies_ct_i:
                pi_policies_k[policy] = pool_id
        pi_policies_profiles[k] = pi_policies_k

    pi_pools, pi_policies = extract_pools.aggregate_pools(pi_policies_profiles, policies_ids_profiles)

    return pi_pools, pi_policies, trees, y_est,
