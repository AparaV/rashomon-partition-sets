import collections
import numpy as np

##
## Extract pools from Sigma matrix
##

def lattice_adjacencies(sigma, policies):
    num_policies = len(policies)
    edges = []
    for i in range(num_policies):
        pol1 = policies[i]
        for j in range(i+1, num_policies):
            pol2 = policies[j]
            
            diff = np.array(pol1) - np.array(pol2)
            diff = np.sum(np.abs(diff))
            if diff == 1:
                comp = np.not_equal(pol1, pol2)
                arm = np.where(comp)[0][0]
                min_dosage = min(pol1[arm], pol2[arm])
                if sigma[arm, min_dosage-1] == 1:
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