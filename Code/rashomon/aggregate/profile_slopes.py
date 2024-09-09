import numpy as np

from collections import deque

from .. import loss
from .. import counter
from ..hasse import enumerate_policies, policy_to_profile
from ..sets import RashomonSet, RashomonProblemCache, RashomonSubproblemCache

from .profile import initialize_sigma


def RAggregate_profile_slopes(M, R, H, D, X, y, theta, profile, reg=1, policies=None, normalize=0):
    """
    Aggregation algorithm for a single profile
    M: int - number of arms
    R: int or list of integers - number of dosage levels per arm (not max dosage)
    H: int - maximum number of pools in this profile
    D - Data i.e., policy integers
    X - Data matrix
    y - Data i.e., outcomes
    theta: float - Rashomon threshold
    profile: tuple - the profile we are considering
    reg: float - regularization parameter
    policies, policy_means - Optional precomputed values when repeatedly calling RAggregate_profile
    """

    if policies is None:
        all_policies = enumerate_policies(M, R)
        policies = [x for x in all_policies if policy_to_profile(x) == profile]
        # policy_means = loss.compute_policy_means(D, y, len(policies))

    if np.max(R) == 2:
        sigma = np.zeros(shape=(M, 1)) + np.inf
        P_qe = RashomonSet(sigma.shape)
        P_qe.insert(sigma)
        return P_qe

    sigma = initialize_sigma(M, R)

    # If R is fixed across, make it a list for compatbility later on
    if isinstance(R, int):
        R = [R] * M

    P_qe = RashomonSet(sigma.shape)
    Q_seen = RashomonProblemCache(sigma.shape)
    problems = RashomonSubproblemCache(sigma.shape)

    for i in range(M):
        if not np.isinf(sigma[i, 0]):
            queue = deque([(sigma, i, 0)])
            break

    while len(queue) > 0:

        (sigma, i, j) = queue.popleft()
        sigma = np.copy(sigma)

        # Cache problem
        if problems.seen(sigma, i, j):
            continue
        problems.insert(sigma, i, j)

        if counter.num_pools(sigma) > H:
            continue

        sigma_0 = np.copy(sigma)
        sigma_1 = np.copy(sigma)
        sigma_1[i, j] = 1
        sigma_0[i, j] = 0

        # Add problem variants to queue
        for m in range(M):
            R_m = R[m]
            j1 = 0
            while problems.seen(sigma_1, m, j1) and j1 < R_m - 3:
                j1 += 1
            if j1 <= R_m - 3:
                queue.append((sigma_1, m, j1))
            j0 = 0
            while problems.seen(sigma_0, m, j0) and j0 < R_m - 3:
                j0 += 1
            if j0 <= R_m - 3:
                queue.append((sigma_0, m, j0))

        # Check if further splits in arm i is feasible
        B = loss.compute_B_slopes(D, X, y, sigma, i, j, policies, reg, normalize)
        if B > theta:
            continue

        # Check if the pooling already satisfies the Rashomon threshold
        if not Q_seen.seen(sigma_1):
            Q_seen.insert(sigma_1)
            Q = loss.compute_Q_slopes(D, X, y, sigma_1, policies, reg, normalize)
            if Q <= theta:
                P_qe.insert(sigma_1)

        if not Q_seen.seen(sigma_0) and counter.num_pools(sigma_0) <= H:
            Q_seen.insert(sigma_0)
            Q = loss.compute_Q_slopes(D, X, y, sigma_0, policies, reg, normalize)
            if Q <= theta:
                P_qe.insert(sigma_0)

        # Add children problems to the queue
        if j < R[i] - 3:  # j < R_i - 2 in math notation
            if not problems.seen(sigma_1, i, j + 1):
                queue.append((sigma_1, i, j + 1))
            if not problems.seen(sigma_0, i, j + 1):
                queue.append((sigma_0, i, j + 1))

    return P_qe
