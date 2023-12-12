import numpy as np

from sklearn.metrics import mean_squared_error

from rashomon import tva


def make_predictions(D, pi_policies, pool_means):
    n, _ = D.shape
    y_pred = np.ndarray(shape=(n,))
    for i in range(n):
        policy_id = D[i, 0]
        pool_id = pi_policies[policy_id]
        y_pred[i] = pool_means[pool_id]
    return y_pred


def find_profiles(subset_policies, all_policies, profile_map):
    """
    Return a list of indicators denoting which profiles are present in subset_policies
    """
    subset_profiles = [tva.policy_to_profile(all_policies[x]) for x in subset_policies]
    subset_profile_ids = [profile_map[x] for x in subset_profiles]
    profile_indicator = [0] * len(profile_map)
    for prof_id in subset_profile_ids:
        profile_indicator[prof_id] = 1
    return profile_indicator


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


def compute_all_metrics(y_true, y_est, D, true_best_policies,
                        all_policies, profile_map, min_dosage_best_policy,
                        true_best_effect):
    # MSE
    sqrd_err = mean_squared_error(y_est, y_true)

    # IOU
    est_best_policies = find_best_policies(D, y_est)
    iou = intersect_over_union(set(true_best_policies), set(est_best_policies))

    # Profiles
    best_profile_indicator = find_profiles(est_best_policies, all_policies, profile_map)

    # Min dosage inclusion
    min_dosage_present = check_membership(min_dosage_best_policy, est_best_policies)

    # Best policy MSE
    best_policy_diff = true_best_effect - np.max(y_est)

    results = {
        "sqrd_err": sqrd_err,
        "iou": iou,
        "best_prof": best_profile_indicator,
        "min_dos_inc": min_dosage_present,
        "best_pol_diff": best_policy_diff
    }

    return results
