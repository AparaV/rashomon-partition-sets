import collections
import numpy as np

#
# Extract pools from Sigma matrix
#


def lattice_adjacencies(sigma, policies):
    num_policies = len(policies)
    edges = []
    for i in range(num_policies):
        pol1 = policies[i]
        for j in range(i + 1, num_policies):
            pol2 = policies[j]

            diff = np.array(pol1) - np.array(pol2)
            diff = np.sum(np.abs(diff))
            if diff == 1:
                comp = np.not_equal(pol1, pol2)
                arm = np.where(comp)[0][0]
                min_dosage = min(pol1[arm], pol2[arm])
                if sigma[arm, min_dosage - 1] == 1:
                    edges.append((i, j))
    return edges


# Helper to merge components
def __merge_components__(parent, x):
    if parent[x] == x:
        return x
    return __merge_components__(parent, parent[x])


# Disjoint Set Union Algorithm from https://cp-algorithms.com/data_structures/disjoint_set_union.html
# Implementation modified from https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
def connected_components(n, edges):
    # find all parents
    parent = [i for i in range(n)]
    for x in edges:
        comp1 = __merge_components__(parent, x[0])
        comp2 = __merge_components__(parent, x[1])
        parent[comp1] = comp2

    # merge all components
    for i in range(n):
        parent[i] = __merge_components__(parent, parent[i])

    cc_map = collections.defaultdict(list)
    for i in range(n):
        cc_map[parent[i]].append(i)
    conn_comps = [val for key, val in cc_map.items()]

    return conn_comps


def extract_pools(policies, sigma):
    """
    Returns: (pi_pools, pi_policies)
    pi_pools is a dictionary. Key = pool_id, Value = List of policy_id
    pi_policies is a dictionary. Key = policy_id, Value = pool_id
    """
    relations = lattice_adjacencies(sigma, policies)
    pools = connected_components(len(policies), relations)
    pi_pools = {}
    pi_policies = {}
    for i, pool in enumerate(pools):
        pi_pools[i] = pool
        for policy in pool:
            pi_policies[policy] = i
    return (pi_pools, pi_policies)


def aggregate_pools(pi_policies: dict[int, dict[int, int]], policies_ids_profiles) -> tuple[dict, dict]:
    """
    Aggregate partitions across multiple profiles into a unified ID numbering system
    The unified ID numbering system is given by policies_ids_profiles
    pi_policies: key = profile_id, value = dictionary
        pi_policies[k]: key = policy_id within that profile
                        value = pool_id
    policies_ids_profiles: key = profile_id, value is list of policy IDs
        These policy IDs are unique even across profiles
        The i-th policy in profile k aoccroding to pi_policies[k]
        gets mapped to policies_ids_profiles[k][i]
    """
    agg_pi_policies: dict[int, int] = {}
    agg_pi_pools: dict[int, list[int]] = {}
    pool_ctr = 0
    for k, pi_policies_k in pi_policies.items():
        policies_ids_k = policies_ids_profiles[k]
        pool_id_map = {}
        for pol_id, pool_id in pi_policies_k.items():
            if pool_id is None:
                continue
            try:
                agg_pool_id = pool_id_map[pool_id]
            except KeyError:
                agg_pool_id = pool_ctr
                agg_pi_pools[agg_pool_id] = []
                pool_id_map[pool_id] = agg_pool_id
                pool_ctr += 1
            agg_pol_id = policies_ids_k[pol_id]
            agg_pi_policies[agg_pol_id] = agg_pool_id
            agg_pi_pools[agg_pool_id].append(agg_pol_id)

    return (agg_pi_pools, agg_pi_policies)
