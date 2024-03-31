import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from rashomon import counter
from rashomon.extract_pools import extract_pools, get_trt_ctl_pooled_partition


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
        nodata_idx = np.where(policy_subset[:, 1] == 0)[0]
        policy_subset[:, 0][nodata_idx] = 0
        mu_pools_temp[pool_id, :] = np.sum(policy_subset, axis=0)
    sums = mu_pools_temp[:, 0]
    counts = mu_pools_temp[:, 1]
    nodata_idx = np.where(counts == 0)[0]
    sums[nodata_idx] = 0
    out_format = np.zeros_like(sums)
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


def compute_B(D, y, sigma, i, j, policies, policy_means, reg=1, normalize=0, lattice_edges=None):
    """
    The B function in Theorem 6.3 \ref{thm:rashomon-equivalent-bound}
    """

    # Split maximally in arm i from dosage j
    sigma_fix = partition_sigma(sigma, i, j)
    pi_fixed_pools, pi_fixed_policies = extract_pools(policies, sigma_fix, lattice_edges)

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


def compute_Q(D, y, sigma, policies, policy_means, reg=1, normalize=0, lattice_edges=None):
    """
    Compute the loss Q
    """

    # pi_pools, pi_policies, t1, t2 = extract_pools(policies, sigma, lattice_edges)
    pi_pools, pi_policies = extract_pools(policies, sigma, lattice_edges)
    mu_pools = compute_pool_means(policy_means, pi_pools)
    D_pool = [pi_policies[pol_id] for pol_id in D[:, 0]]
    mu_D = mu_pools[D_pool]
    mse = mean_squared_error(y[:, 0], mu_D)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    h = mu_pools.shape[0]
    Q = mse + reg * h

    return Q
    # return Q, t1, t2


def compute_B_slopes(D, X, y, sigma, i, j, policies, reg=1, normalize=0):
    """
    The B function in Theorem 6.3 \ref{thm:rashomon-equivalent-bound}
    """

    # Split maximally in arm i from dosage j
    sigma_fix = partition_sigma(sigma, i, j)
    pi_fixed_pools, pi_fixed_policies = extract_pools(policies, sigma_fix)

    # Compute squared loss for this maximal split
    y_est = np.zeros(shape=D.shape) + np.inf
    for pi_k, pol_list_k in pi_fixed_pools.items():

        # Extract the X matrix
        k_idx = [i for i, p in enumerate(D[:, 0]) if p in pol_list_k]
        X_k = X[k_idx, :]
        y_k = y[k_idx, :]

        # Run regression and estimate outcomes
        model_k = LinearRegression().fit(X_k, y_k)
        y_est_k = model_k.predict(X_k)
        y_est[k_idx, :] = y_est_k

    mse = mean_squared_error(y[:, 0], y_est)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    # The least number of pools
    # The number of pools when the splittable policies are pooled maximally
    sigma_fix[i, (j + 1):] = 1
    sigma_fix[np.isinf(sigma)] = np.inf
    h = counter.num_pools(sigma_fix)

    B = mse + reg * h

    return B


def compute_Q_slopes(D, X, y, sigma, policies, reg=1, normalize=0):
    """
    Compute the loss Q
    """

    pi_pools, pi_policies = extract_pools(policies, sigma)

    y_est = np.zeros(shape=D.shape) + np.inf
    for pi_k, pol_list_k in pi_pools.items():

        # Extract the X matrix
        k_idx = [i for i, p in enumerate(D[:, 0]) if p in pol_list_k]
        X_k = X[k_idx, :]
        y_k = y[k_idx, :]

        # Run regression and estimate outcomes
        model_k = LinearRegression().fit(X_k, y_k)
        y_est_k = model_k.predict(X_k)
        y_est[k_idx, :] = y_est_k

    mse = mean_squared_error(y[:, 0], y_est)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    h = len(pi_pools.keys())
    Q = mse + reg * h

    return Q


def compute_het_Q(D_tc, y_tc, sigma_int, trt_pools, ctl_pools, policy_means, reg=1, normalize=0):
    """
    Compute the loss after pooling across treatment and control as per sigma_int
    D_tc indices need not be re-indexed. The indicies should match policy_means
    policy_means is for the entire dataset
    """

    sigma_pools, sigma_policies = get_trt_ctl_pooled_partition(trt_pools, ctl_pools, sigma_int)
    mu_pools = compute_pool_means(policy_means, sigma_pools)
    D_tc_pool = [sigma_policies[pol_id] for pol_id in D_tc[:, 0]]
    mu_D = mu_pools[D_tc_pool]
    mse = mean_squared_error(y_tc[:, 0], mu_D)

    if normalize > 0:
        mse = mse * D_tc.shape[0] / normalize

    h = mu_pools.shape[0]
    Q = mse + reg * h

    return Q
