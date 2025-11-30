import numpy as np

from sklearn.metrics import mean_squared_error, confusion_matrix

from rashomon import hasse


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
    subset_profiles = [hasse.policy_to_profile(all_policies[x]) for x in subset_policies]
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


def compute_te_het_metrics(te_true, te_est, max_te, max_te_policies,
                           D_trt, univ_pol_id_list):

    # Find MSE in TE
    mse_te = mean_squared_error(te_est, te_true)

    # Find highest TE and error
    max_te_est = np.max(te_est)
    max_te_err = max_te - max_te_est

    # Find IOU of set with highest TE
    max_te_pol = np.unique(D_trt[np.where(te_est == max_te_est), ])
    max_te_pol = [univ_pol_id_list[x] for x in max_te_pol]
    iou = intersect_over_union(set(max_te_policies), set(max_te_pol))

    # Count policies with +, 0, - effects
    te_true_sign = np.sign(te_true)
    te_est_sign = np.sign(te_est)
    conf_mat = confusion_matrix(te_true_sign, te_est_sign, labels=[-1, 0, 1],
                                normalize="true")

    # # Compute overall MSE
    # mse_i = mean_squared_error(y_tc[:, 0], mu_D)

    metrics_results = {
        "mse_te": mse_te,
        "max_te_est": max_te_est,
        "max_te_err": max_te_err,
        "iou": iou,
        "conf_matrix": conf_mat
    }

    return metrics_results


def compute_iou_coverage(coef_samples, X, D, true_best):
    """
    Compute mean IOU across all posterior/bootstrap samples.

    Arguments:
    coef_samples (np.ndarray): Coefficient samples, shape (n_samples, n_features)
    X (np.ndarray): Design matrix
    D (np.ndarray): Policy assignments
    true_best: True best policies

    Returns:
    float: Mean IOU across all samples
    """
    n_samples = coef_samples.shape[0]
    iou_sum = 0.0

    for i in range(n_samples):
        y_pred = np.dot(X, coef_samples[i])
        pred_best = find_best_policies(D, y_pred)
        iou = intersect_over_union(set(true_best), set(pred_best))
        iou_sum += iou

    return iou_sum / n_samples


def compute_min_dosage_coverage(coef_samples, X, D, min_dosage_best_policy):
    """
    Compute mean min dosage coverage: fraction of posterior/bootstrap samples where the minimum
    dosage best policy is included in the predicted best policies.

    Arguments:
    coef_samples (np.ndarray): Coefficient samples, shape (n_samples, n_features)
    X (np.ndarray): Design matrix
    D (np.ndarray): Policy assignments
    min_dosage_best_policy: Minimum dosage policy among true best policies

    Returns:
    float: Mean fraction of samples where min_dosage_best_policy is in predicted best
    """
    n_samples = coef_samples.shape[0]
    coverage_sum = 0.0

    for i in range(n_samples):
        y_pred = np.dot(X, coef_samples[i])
        pred_best = find_best_policies(D, y_pred)
        if check_membership(min_dosage_best_policy, pred_best):
            coverage_sum += 1.0

    return coverage_sum / n_samples
