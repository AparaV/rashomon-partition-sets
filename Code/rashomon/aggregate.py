import numpy as np

from collections import deque

from rashomon import loss
from rashomon import counter
from rashomon.tva import enumerate_policies
from rashomon.sets import RashomonSet, RashomonProblemCache, RashomonSubproblemCache


def initialize_sigma(M, R):
    if isinstance(R, int):
        sigma = np.ndarray(shape=(M, R - 2))
        sigma[:, :] = 1
    else:
        sigma = np.ndarray(shape=(M, np.max(R) - 2))
        sigma[:, :] = np.inf
        for idx, R_idx in enumerate(R):
            sigma[idx, :(R_idx-2)] = 1
    return sigma


def RAggregate(M, R, H, D, y, theta, reg=1):
    """
    Aggregation algorithm
    """

    policies = enumerate_policies(M, R)
    policy_means = loss.compute_policy_means(D, y, len(policies))
    sigma = initialize_sigma(M, R)

    P_qe = RashomonSet(sigma.shape)
    Q_seen = RashomonProblemCache(sigma.shape)
    problems = RashomonSubproblemCache(sigma.shape)

    queue = deque([(sigma, 0, 0)])

    while len(queue) > 0:

        # print(P_qe)

        (sigma, i, j) = queue.popleft()
        sigma = np.copy(sigma)

        # Cache problem
        if problems.seen(sigma, i, j):
            continue
        problems.insert(sigma, i, j)

        if counter.num_pools(sigma) > H:
            continue

        B = loss.compute_B(D, y, sigma, i, j, policies, policy_means, reg)
        # print(sigma, B, theta)

        sigma_0 = np.copy(sigma)
        sigma[i, j] = 1
        sigma_0[i, j] = 0

        for m in range(M):
            # if m == i:
            #     continue
            if not problems.seen(sigma, m, 0):
                queue.append((sigma, m, 0))
            if not problems.seen(sigma_0, m, 0):
                queue.append((sigma_0, m, 0))

        if B > theta:
            continue

        # Check if the pooling already satisfies the Rashomon threshold
        if not Q_seen.seen(sigma):
            Q_seen.insert(sigma)
            Q = loss.compute_Q(D, y, sigma, policies, policy_means, reg)
            if Q <= theta:
                # print(sigma)
                P_qe.insert(sigma)

        if not Q_seen.seen(sigma_0) and counter.num_pools(sigma_0) <= H:
            Q_seen.insert(sigma_0)
            Q = loss.compute_Q(D, y, sigma_0, policies, policy_means, reg)
            if Q <= theta:
                # print(sigma_0)
                P_qe.insert(sigma_0)
                # print(P_qe)

        # Add children problems to the queue
        if isinstance(R, int):
            R_i = R
        else:
            R_i = R[i]
        if j < R_i - 3:  # j < R_i - 2 in math notation
            if not problems.seen(sigma, i, j + 1):
                queue.append((sigma, i, j + 1))
            if not problems.seen(sigma_0, i, j + 1):
                queue.append((sigma_0, i, j + 1))

    return P_qe


if __name__ == "__init__":

    # Fix random seed
    np.random.seed(3)

    # Imports for testing only
    from rashomon.extract_pools import extract_pools

    #
    # Setup matrix
    #
    sigma = np.array([[1, 1, 0], [0, 1, 1]], dtype="float64")

    M, n = sigma.shape
    R = n + 2
    num_policies = (R - 1) ** M
    policies = enumerate_policies(M, R)
    pi_pools, pi_policies = extract_pools(policies, sigma)

    #
    # Generate data
    #
    num_pools = len(pi_pools)
    mu = np.random.uniform(0, 4, size=num_pools)
    var = [1] * num_pools

    n_per_pol = 10

    num_data = num_policies * n_per_pol
    X = np.ndarray(shape=(num_data, M))
    D = np.ndarray(shape=(num_data, 1), dtype="int_")
    y = np.ndarray(shape=(num_data, 1))

    for idx, policy in enumerate(policies):
        pool_i = pi_policies[idx]
        mu_i = mu[pool_i]
        var_i = var[pool_i]
        y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

        start_idx = idx * n_per_pol
        end_idx = (idx + 1) * n_per_pol

        X[start_idx:end_idx,] = policy
        D[start_idx:end_idx,] = idx
        y[start_idx:end_idx,] = y_i

    #
    # Aggregate
    #
    P_set = RAggregate(2, 5, 4, D, y, 5, reg=1)
    print(f"There are {P_set.size} poolings in the Rashomon set.")
    print(f"Original matrix in P_set: {P_set.seen(sigma)}.")

    print("The poolings are:")
    for pooling in P_set:
        print(pooling, "--\n")
