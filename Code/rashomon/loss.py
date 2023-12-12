import numpy as np

from sklearn.metrics import mean_squared_error

from rashomon import counter
from rashomon.extract_pools import extract_pools


def compute_policy_means(D, y, num_policies):
    """
    Returns: policy_means
    policy_means is a np.ndarray of size (num_policies,2)
    policy_means[i, 0] = sum of all y where D[i,0] = i
    policy_means[i, 1] = count of where D[i,0] = i
    """
    policy_means = np.ndarray(shape=(num_policies, 2))
    for policy_id in range(num_policies):
        idx = np.where(D == policy_id)
        policy_means[policy_id, 0] = np.sum(y[idx])
        policy_means[policy_id, 1] = len(idx[0])
    return policy_means


def compute_pool_means(policy_means, pi_pools):
    """
    Returns: mu_pools
    mu_pools is a np.ndarray of size (H,) where H is the number of pools
    mu_pools[i] = mean value in pool i
    """
    H = len(pi_pools.keys())
    mu_pools_temp = np.ndarray(shape=(H, 2))
    for pool_id, pool in pi_pools.items():
        policy_subset = policy_means[pool, :]
        mu_pools_temp[pool_id, :] = np.sum(policy_subset, axis=0)
    # mu_pools = np.float64(mu_pools_temp[:, 0]) / mu_pools_temp[:, 1]
    sums = mu_pools_temp[:, 0]
    counts = mu_pools_temp[:, 1]
    out_format = np.nan + np.zeros_like(sums)
    mu_pools = np.divide(sums, counts, out=out_format, where=counts != 0)
    return mu_pools


def partition_sigma(sigma, i, j):
    """
    Maximally split policies in arm i starting at dosage j
    All other existing splits are maintained
    """
    sigma_fix = np.copy(sigma)
    sigma_fix[i, j:] = 0
    sigma_fix[np.isinf(sigma)] = np.inf
    return sigma_fix


def compute_B(D, y, sigma, i, j, policies, policy_means, reg=1, normalize=0):
    """
    The B function in Theorem 6.3 \ref{thm:rashomon-equivalent-bound}
    """

    # Split maximally in arm i from dosage j
    sigma_fix = partition_sigma(sigma, i, j)
    pi_fixed_pools, pi_fixed_policies = extract_pools(policies, sigma_fix)

    # Compute squared loss for this maximal split
    # This loss is B minus the regularization term
    mu_fixed_pools = compute_pool_means(policy_means, pi_fixed_pools)
    D_pool = [pi_fixed_policies[pol_id] for pol_id in D[:, 0]]
    mu_D = mu_fixed_pools[D_pool]
    mse = mean_squared_error(y[:, 0], mu_D)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    # The least number of pools
    # The number of pools when the splittable policies are pooled maximally
    sigma_fix[i, (j + 1):] = 1
    sigma_fix[np.isinf(sigma)] = np.inf
    h = counter.num_pools(sigma_fix)

    B = mse + reg * h

    return B


def compute_Q(D, y, sigma, policies, policy_means, reg=1, normalize=0):
    """
    Compute the loss Q
    """

    pi_pools, pi_policies = extract_pools(policies, sigma)
    mu_pools = compute_pool_means(policy_means, pi_pools)
    D_pool = [pi_policies[pol_id] for pol_id in D[:, 0]]
    mu_D = mu_pools[D_pool]
    mse = mean_squared_error(y[:, 0], mu_D)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    h = mu_pools.shape[0]
    Q = mse + reg * h

    return Q
