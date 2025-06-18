# import time
import collections
import numpy as np


def _hasse_index_offsets(M: int, R: np.ndarray) -> list[int]:
    """
    Calculate index offsets for each dimension in the Hasse diagram.

    Args:
        M (int): The number of dimensions in each policy.
        R (np.ndarray): An array representing the range of values for each dimension.

    Returns:
        list[int]: A list of offsets for each dimension.
    """
    # Calculate index offsets for each dimension
    # For the last dimension, the offset is 1.
    # For the second-to-last dimension, the offset is the size of the last dimension.
    # For the third-to-last dimension, the offset is the product of the sizes of the last two dimensions.
    # And so on...
    offsets = [1] * M
    for i in reversed(range(M - 1)):
        offsets[i] = offsets[i + 1] * R[i + 1]
    return offsets


def _lattice_edges_memory(M: int, R: np.ndarray | int) -> list[tuple[int, int]]:
    """
    Enumerate the Hasse adjacencies without having to enumerate policies.
    Identifies indices similar to `_lattice_edges_sorted` but without having to enumerate policies
    In cases where enumerating policies is expensive, this can be useful.
    TODO: This is not useful as it is. Need other functionality to traverse policy and index space.

    Args:
        M (int): The number of dimensions in each policy.
        R (np.ndarray | int): An array or integer representing the range of values for each dimension.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the edges in the Hasse diagram.
    """
    if isinstance(R, int):
        R = np.array([R] * M)

    # Compute total number of policies
    total_policies = np.prod(R)

    # Precompute offsets
    offsets = _hasse_index_offsets(M, R)

    edges = []
    for i in range(total_policies):
        for d in range(M):
            neighbor = i + offsets[d]
            # First check that neighbor is within total range
            if neighbor < total_policies:
                # Check if dimension d is not at its boundary
                block_index = i // offsets[d]
                # block_base is the start of that dimension's 'block'
                block_base = (block_index // R[d]) * R[d]
                # If block_index - block_base < R[d] - 1, we haven't reached the boundary
                if (block_index - block_base) < (R[d] - 1):
                    edges.append((i, neighbor))

    return edges


def _lattice_edges_default(policies: list) -> list[tuple[int, int]]:
    """
    Enumerate the Hasse adjacencies the naive way
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


def _lattice_edges_sorted(policies: list, M: int, R: np.ndarray | int) -> list[tuple[int, int]]:
    """
    Enumerate the Hasse adjacencies quickly using the fact that the policies are sorted.

    Args:
        policies (list): A list of policies, where each policy is represented as a list of integers.
        M (int): The number of dimensions in each policy.
        R (np.ndarray | int): An array or integer representing the range of values for each dimension.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the edges in the Hasse diagram.

    The function works as follows:
    1. If R is an integer, convert it to an array of length M with all elements equal to R.
    2. Initialize an offsets list with 1s, which will be used to calculate the index offsets for each dimension.
    3. Calculate the offsets for each dimension in reverse order.
    4. Iterate over each policy and each dimension to find the edges in the Hasse diagram.
    5. If the value of the current dimension is less than the maximum value (R[d] - 1), add an edge to the edges list.
    """
    if isinstance(R, int):
        R = np.array([R] * M)

    offsets = _hasse_index_offsets(M, R)

    edges = []
    for i, pol in enumerate(policies):
        for d in range(M):
            if pol[d] < R[d] - 1:
                edges.append((i, i + offsets[d]))
    return edges


def lattice_edges(policies: list, **kwargs) -> list[tuple[int, int]]:
    """
    Enumerate the Hasse adjacencies for a given list of policies.

    Parameters:
    policies (list): A list of policies where each policy is represented as a list or array.
    **kwargs: Additional keyword arguments.
        sorted (bool): If True, use the sorted method for lattice edge enumeration. Default is False.
        M (int): The length of each policy. Required if sorted is True.
        R (array-like): The maximum values for each dimension of the policies. Required if sorted is True.

    Returns:
    list[tuple[int, int]]: A list of tuples representing the edges in the Hasse diagram.
    """
    # This does not work. TODO: Fix it.
    # sorted = kwargs.get("sorted", False)
    # if sorted:
    #     M = kwargs.get("M", None)
    #     R = kwargs.get("R", None)
    #     if M is None or R is None:
    #         raise ValueError("M and R are required when sorted is True.")
    #     return _lattice_edges_sorted(policies, M, R)
    return _lattice_edges_default(policies)


def lattice_adjacencies(sigma: np.ndarray, policies: int) -> list[tuple[int, int]]:
    """
    Find edges in the Hasse based on partition sigma
    Constructs edges from scratch. More expensive that `prune_lattice_edges`

    Arguments:
    sigma (np.ndarray): Partition matrix
    policies (list): List of policies

    Returns:
    edges (list[tuple[int, int]]): List of edges in the Hasse
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
                comp = np.not_equal(pol1, pol2)
                arm = np.where(comp)[0][0]
                min_dosage = min(pol1[arm], pol2[arm])
                if sigma[arm, min_dosage - 1] == 1:
                    edges.append((i, j))
    return edges


def prune_lattice_edges(sigma: np.ndarray, edges: list[tuple[int, int]],
                        policies: list) -> list[tuple[int, int]]:
    """
    Find edges remaining in the Hasse based on partition sigma.
    Uses list of all possible lattice edges to remove edges based on sigma.

    Arguments:
    sigma (np.ndarray): Partition matrix
    edges (list[tuple[int, int]]): List of all possible edges in the Hasse
    policies (list): List of policies

    Returns:
    pruned_edges (list[tuple[int, int]]): List of edges remaining in the Hasse
    """
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


def __merge_components__(parent, x):
    """ Helper function to merge components in `connected_components` """
    if parent[x] == x:
        return x
    return __merge_components__(parent, parent[x])


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """
    Algorithm to find connected components in a graph
    Disjoint Set Union Algorithm from https://cp-algorithms.com/data_structures/disjoint_set_union.html
    Implementation modified from https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/

    Arguments:
    n (int): Number of nodes
    edges (list[tuple[int, int]]): Edges in graph

    Returns:
    list[list[int]]: List of connected components
    """
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


def iterative_connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Iterative BFS to find connected components."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * n
    components = []

    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            comp = []
            while queue:
                node = queue.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    return components


def extract_pools(policies: list, sigma: np.ndarray,
                  lattice_edges: list[tuple[int, int]] = None) -> tuple[
                      dict[int, list[int]], dict[int, int]
                  ]:
    """
    Extracts pools from sigma matrix

    Arguments:
    policies (list): List of features
    sigma (np.ndarray): Partition matrix
    lattice_edges (list[tuple[int, int]]): List of Hasse adjacencies.
        Policies are indexed by their ID according to position in `policies`
        Defaults to `None` where adjacencies are computed.
        This results in significantly slower runtimes.

    Returns:
    (pi_pools, pi_policies): Extracted pools
        pi_pools (dict[int, list[int]]): Key = pool_id, Value = List of policy_id
        pi_policies (dict[int, int]): Key = policy_id, Value = pool_id
    """

    # t0 = time.time()
    if lattice_edges is None:
        lattice_relations = lattice_adjacencies(sigma, policies)
    else:
        lattice_relations = prune_lattice_edges(sigma, lattice_edges, policies)

    pools = iterative_connected_components(len(policies), lattice_relations)

    pi_pools = {}
    pi_policies = {}
    for i, pool in enumerate(pools):
        pi_pools[i] = pool
        for policy in pool:
            pi_policies[policy] = i
    return (pi_pools, pi_policies)


def __aggregator_universalize_policy_ids__(
        pi_policies: dict[int, dict[int, int]],
        policies_ids_profiles: dict[int, list[int]]) -> dict[int, dict[int, int]]:
    """
    Universalizies policy IDs to follow unified numbering system.
    Does not modify pool IDs within profile

    Arguments:
    pi_policies (dict[int, dict[int, int]]): key = profile_id, value = dictionary
        pi_policies[k]: key = policy_id within that profile
                        value = pool_id within that profile
    policies_ids_profiles (: dict[int, list[int]]): key = profile_id, value is list of policy IDs
        These policy IDs are unique even across profiles
        The i-th policy in profile k aoccroding to pi_policies[k]
        gets mapped to policies_ids_profiles[k][i]

    Returns:
    pi_policies_univ (dict[int, dict[int, int]]): Universalized policy IDs
    """

    pi_policies_univ = {}
    for k, pi_policies_k_0 in pi_policies.items():
        pi_policies_k = {}
        policies_ids_k = policies_ids_profiles[k]
        for pol_id, pool_id in pi_policies_k_0.items():
            if pool_id is None:
                continue
            agg_pol_id = policies_ids_k[pol_id]
            pi_policies_k[agg_pol_id] = pool_id
        pi_policies_univ[k] = pi_policies_k

    return pi_policies_univ


def __aggregate_pools__(pi_policies: dict[int, dict[int, int]]) -> tuple[dict, dict]:
    """
    Aggregate partitions across multiple profiles into a unified ID numbering system
    Assumes that policy IDs are already universalized.

    Arguments:
    pi_policies (dict[int, dict[int, int]]): key = profile_id, value = dictionary
        pi_policies[k]: key = universal policy_id
                        value = pool_id within profile k

    Returns:
    (agg_pi_pools, agg_pi_policies): Aggregated pools and policies
        agg_pi_pools (dict[int, list[int]]): Key = pool_id, Value = List of policy_id
        agg_pi_policies (dict[int, int]): Key = policy_id, Value = pool_id
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


def aggregate_pools(pi_policies: dict[int, dict[int, int]],
                    policies_ids_profiles: dict[int, list[int]]) -> tuple[
                        dict[int, list[int]], dict[int, int]
                    ]:
    """
    Aggregate partitions across multiple profiles into a unified ID numbering system
    The unified ID numbering system is given by policies_ids_profiles

    Arguments:
    pi_policies (dict[int, dict[int, int]]): key = profile_id, value = dictionary
        pi_policies[k]: key = policy_id within that profile
                        value = pool_id
    policies_ids_profiles (dict[int, list[int]]): key = profile_id, value is list of policy IDs
        These policy IDs are unique even across profiles
        The i-th policy in profile k aoccroding to pi_policies[k]
        gets mapped to policies_ids_profiles[k][i]

    Returns:
    (agg_pi_pools, agg_pi_policies): Aggregated pools and policies
        agg_pi_pools (dict[int, list[int]]): Key = pool_id, Value = List of policy_id
        agg_pi_policies (dict[int, int]): Key = policy_id, Value = pool_id
    """

    pi_policies_univ = __aggregator_universalize_policy_ids__(pi_policies, policies_ids_profiles)
    agg_pi_pools, agg_pi_policies = __aggregate_pools__(pi_policies_univ)

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
