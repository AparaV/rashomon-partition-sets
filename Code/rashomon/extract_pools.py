# import time
import collections
import numpy as np

#
# Extract pools from Sigma matrix
#


def lattice_edges(policies):
    """
    Enumerate the Hasse adjacencies
    """
    num_policies = len(policies)
    edges = []
    for i in range(num_policies):
        pol1 = policies[i]
        for j in range(i + 1, num_policies):
            pol2 = policies[j]
            diff = np.array(pol1) - np.array(pol2)
            diff = np.sum(np.abs(diff))
            if diff == 1:
                edges.append((i, j))
    return edges


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


def prune_lattice_edges(sigma, edges, policies):
    pruned_edges = []
    for i, j in edges:
        pol_i = np.array(policies[i])
        pol_j = np.array(policies[j])
        diff = pol_i - pol_j
        arm = np.where(diff != 0)[0][0]
        min_dosage = min(pol_i[arm], pol_j[arm])
        if sigma[arm, min_dosage - 1] == 1:
            pruned_edges.append((i, j))
    return pruned_edges


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


def extract_pools(policies, sigma, lattice_edges=None):
    """
    lattice_edges: List of Hasse adjacencies.
        Policies are indexed by their ID according to position in `policies`
        Defaults to `None` where adjacencies are computed.
        This results in significantly slower runtimes.

    Returns: (pi_pools, pi_policies)
    pi_pools is a dictionary. Key = pool_id, Value = List of policy_id
    pi_policies is a dictionary. Key = policy_id, Value = pool_id
    """

    # t0 = time.time()
    if lattice_edges is None:
        lattice_relations = lattice_adjacencies(sigma, policies)
    else:
        lattice_relations = prune_lattice_edges(sigma, lattice_edges, policies)
    # t1 = time.time()
    pools = connected_components(len(policies), lattice_relations)
    # t2 = time.time()
    pi_pools = {}
    pi_policies = {}
    for i, pool in enumerate(pools):
        pi_pools[i] = pool
        for policy in pool:
            pi_policies[policy] = i
    return (pi_pools, pi_policies)
    # return (pi_pools, pi_policies, t1-t0, t2-t1)


def __aggregator_universalize_policy_ids__(
        pi_policies: dict[int, dict[int, int]],
        policies_ids_profiles) -> dict[int, dict[int, int]]:
    """
    Universalizies policy IDs to follow unified numbering system.
    Does not modify pool IDs within profile
    pi_policies: key = profile_id, value = dictionary
        pi_policies[k]: key = policy_id within that profile
                        value = pool_id within that profile
    policies_ids_profiles: key = profile_id, value is list of policy IDs
        These policy IDs are unique even across profiles
        The i-th policy in profile k aoccroding to pi_policies[k]
        gets mapped to policies_ids_profiles[k][i]
    """

    pi_policies_univ = {}
    for k, pi_policies_k in pi_policies.items():
        pi_policies_k = {}
        policies_ids_k = policies_ids_profiles[k]
        for pol_id, pool_id in pi_policies_k.items():
            agg_pol_id = policies_ids_k[pol_id]
            pi_policies_k[agg_pol_id] = pool_id
        pi_policies_univ[k] = pi_policies_k

    return pi_policies_univ


def __aggregate_pools__(pi_policies: dict[int, dict[int, int]]) -> tuple[dict, dict]:
    """
    Aggregate partitions across multiple profiles into a unified ID numbering system
    Assumes that policy IDs are already universalized.
    pi_policies: key = profile_id, value = dictionary
        pi_policies[k]: key = universal policy_id
                        value = pool_id within profile k
    """
    agg_pi_policies: dict[int, int] = {}
    agg_pi_pools: dict[int, list[int]] = {}
    pool_ctr = 0
    for k, pi_policies_k in pi_policies.items():
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
            agg_pi_policies[pol_id] = agg_pool_id
            agg_pi_pools[agg_pool_id].append(pol_id)

    return (agg_pi_pools, agg_pi_policies)


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

    pi_policies_univ = __aggregator_universalize_policy_ids__(pi_policies, policies_ids_profiles)
    agg_pi_pools, agg_pi_policies = __aggregate_pools__(pi_policies_univ)

    return (agg_pi_pools, agg_pi_policies)


def aggregate_pools_old(pi_policies: dict[int, dict[int, int]], policies_ids_profiles) -> tuple[dict, dict]:
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


def get_trt_ctl_pooled_partition(trt_pools, ctl_pools, sigma_int):
    pools_tmp = {
        "trt": trt_pools.copy(),
        "ctl": ctl_pools.copy(),
        "mix": {}
    }

    mixed_indices = np.where(sigma_int == 1)

    for ctl_i, trt_i in zip(mixed_indices[0], mixed_indices[1]):
        mix_trt_i_pols = pools_tmp["trt"].pop(trt_i)
        mix_ctl_i_pols = pools_tmp["ctl"].pop(ctl_i)
        mix_i_pols = list(set(mix_trt_i_pols + mix_ctl_i_pols))
        mix_id = len(pools_tmp["mix"])
        pools_tmp["mix"][mix_id] = mix_i_pols

    sigma_pools = {}
    sigma_policies = {}
    pool_counter = 0
    for _, dict_i in pools_tmp.items():
        for _, policies_ij in dict_i.items():
            sigma_pools[pool_counter] = policies_ij
            for p in policies_ij:
                sigma_policies[p] = pool_counter
            pool_counter += 1

    return sigma_pools, sigma_policies
