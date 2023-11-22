import numpy as np


def make_predictions(D, pi_policies, pool_means):
    n, _ = D.shape
    y_pred = np.ndarray(shape=(n,))
    for i in range(n):
        policy_id = D[i, 0]
        pool_id = pi_policies[policy_id]
        y_pred[i] = pool_means[pool_id]
    return y_pred


def intersect_over_union(X, Y):
    XandY = X.intersection(Y)
    XorY = X.union(Y)
    return len(XandY) / len(XorY)


def find_best_policies(D, y_pred):
    y_max = np.max(y_pred)
    pol_max = np.unique(D[np.where(y_pred == y_max), ])
    return pol_max


def find_min_dosage(policy_ids, policies):
    best_dosage = np.inf
    best_policy = []
    for policy_id in policy_ids:
        dosage = np.sum(policies[policy_id])
        if dosage == best_dosage:
            best_policy.append(policy_id)
        if dosage < best_dosage:
            best_policy = [policy_id]
            best_dosage = dosage
    return best_policy


def check_membership(true_x, est_x):
    true_set = set(true_x)
    est_set = set(est_x)
    if len(true_set.intersection(est_set)) > 0:
        return True
    return False


# def min_dosage_best_policy(true_best, est_best):
#     true_best_set = set(true_best)
#     est_best_set = set(est_best)
#     return


def find_best_policy_diff(y_true, y_est):
    return np.max(y_true) - np.max(y_est)
