import numpy as np

from ..sets import RashomonSet
from ..loss import compute_het_Q
from ..extract_pools import extract_pools
from ..hasse import remove_arm, profile_ids_to_univ_ids


def get_intersection_matrix(ctl_pools, trt_pools, all_policies, trt_arm_idx):
    n_ctl_pools = len(ctl_pools.keys())
    n_trt_pools = len(trt_pools.keys())

    sigma_int = np.zeros(shape=(n_ctl_pools, n_trt_pools)) + np.inf
    for ci, pi_ci in ctl_pools.items():
        pi_ci = [all_policies[i] for i in pi_ci]
        pi_ci = remove_arm(pi_ci, trt_arm_idx)
        pi_ci = set(pi_ci)
        for ti, pi_ti in trt_pools.items():
            pi_ti = [all_policies[i] for i in pi_ti]
            pi_ti = remove_arm(pi_ti, trt_arm_idx)
            pi_ti = set(pi_ti)
            if len(pi_ci.intersection(pi_ti)) > 0:
                sigma_int[ci, ti] = 0

    return sigma_int


def het_partition_solver(D_tc, y_tc, policy_means, trt_pools, ctl_pools, col, sigma_int, theta, P_qe, reg,
                         normalize=0):
    ncols = sigma_int.shape[1]
    if col == ncols:
        return P_qe

    Q_0 = compute_het_Q(D_tc, y_tc, sigma_int, trt_pools, ctl_pools, policy_means, reg, normalize=normalize)
    if Q_0 <= theta:
        if not P_qe.seen(sigma_int):
            P_qe.insert(sigma_int.copy())
            P_qe.Q = np.append(P_qe.Q, Q_0)
        P_qe = het_partition_solver(
            D_tc, y_tc, policy_means, trt_pools, ctl_pools, col+1, sigma_int, theta, P_qe, reg,
            normalize=normalize)

    zero_loc = np.where(sigma_int[:, col] == 0)[0]
    nz = len(zero_loc)
    for i in range(nz):
        row = zero_loc[i]

        sigma_tmp = sigma_int.copy()
        sigma_tmp[row, :] = np.inf
        sigma_tmp[:, col] = np.inf
        sigma_tmp[row, col] = 1

        Q_i = compute_het_Q(D_tc, y_tc, sigma_tmp, trt_pools, ctl_pools, policy_means, reg=1e-1, normalize=normalize)
        if Q_i <= theta:
            if not P_qe.seen(sigma_tmp):
                P_qe.insert(sigma_tmp.copy())
                P_qe.Q = np.append(P_qe.Q, Q_i)
            P_qe = het_partition_solver(
                D_tc, y_tc, policy_means, trt_pools, ctl_pools, col+1, sigma_int, theta, P_qe, reg,
                normalize=normalize)

    return P_qe


def find_te_het_partitions(
        sigma_t, sigma_c,
        trt_profile_idx, ctl_profile_idx,
        trt_policies, ctl_policies,
        trt_arm_idx,
        all_policies, policies_ids_profiles,
        D_tc, y_tc, policy_means,
        theta, reg, normalize=0) -> RashomonSet:

    trt_pools_0, _ = extract_pools(trt_policies, sigma_t)
    ctl_pools_0, _ = extract_pools(ctl_policies, sigma_c)

    trt_policies_global_id = policies_ids_profiles[trt_profile_idx]
    ctl_policies_global_id = policies_ids_profiles[ctl_profile_idx]

    trt_pools = profile_ids_to_univ_ids(trt_pools_0, trt_policies_global_id)
    ctl_pools = profile_ids_to_univ_ids(ctl_pools_0, ctl_policies_global_id)

    sigma_int = get_intersection_matrix(ctl_pools, trt_pools, all_policies, trt_arm_idx)

    P_qe = RashomonSet(sigma_int.shape)
    P_qe = het_partition_solver(D_tc, y_tc, policy_means, trt_pools, ctl_pools, 0, sigma_int,
                                theta, P_qe, reg, normalize)

    return P_qe
