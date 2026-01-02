import pickle
import numpy as np
import pandas as pd

from copy import deepcopy
from rashomon import hasse, loss, extract_pools
from rashomon.aggregate import subset_data


def format_data(df: pd.DataFrame, outcome_col_id, chosen_covariates_idx):

    outcome_col = df.columns[outcome_col_id]
    # outcome_col = cols[outcome_col_id]
    df2 = df.copy()
    df2 = df2.dropna(subset=[outcome_col], axis=0)

    Z = df2.to_numpy()

    X = Z[:, chosen_covariates_idx]
    y = Z[:, outcome_col_id]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y = y.reshape((-1, 1))

    return X, y


def read_pickle(results_dir, fname):

    outcome_fname = results_dir + fname + ".pkl"

    with open(outcome_fname, "rb") as f:
        res_dict = pickle.load(f)

    R_set = res_dict["R_set"]
    R_profiles = res_dict["R_profiles"]
    if len(R_set) > 0:
        print(res_dict["q"], res_dict["reg"])

    return R_set, R_profiles


def get_policy_means(M, R, X, y):

    # num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)

    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)

    num_data = X.shape[0]

    # print(num_policies)

    policies_profiles = {}
    policies_profiles_masked = {}
    policies_ids_profiles = {}
    for k, profile in enumerate(profiles):

        policies_temp = [(i, x) for i, x in enumerate(all_policies) if hasse.policy_to_profile(x) == profile]
        unzipped_temp = list(zip(*policies_temp))
        policies_ids_k = list(unzipped_temp[0])
        policies_k = list(unzipped_temp[1])
        policies_profiles[k] = deepcopy(policies_k)
        policies_ids_profiles[k] = policies_ids_k

        profile_mask = list(map(bool, profile))

        # Mask the empty arms
        for idx, pol in enumerate(policies_k):
            policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
        policies_profiles_masked[k] = policies_k

    D = np.zeros(shape=y.shape, dtype=np.int64)
    profiles_in_data = []
    for i in range(num_data):
        policy_i = tuple([int(x) for x in X[i, :]])
        policy_idx = [idx for idx in range(num_policies) if all_policies[idx] == policy_i]
        profiles_in_data.append(hasse.policy_to_profile(policy_i))
        D[i, 0] = int(policy_idx[0])

    policy_means = loss.compute_policy_means(D, y, num_policies)

    policy_means_profiles = {}
    for k, profile in enumerate(profiles):
        D_k, y_k = subset_data(D, y, policies_ids_profiles[k])

        if D_k is None:
            policy_means_profiles[k] = None
        else:
            policy_means_k = policy_means[policies_ids_profiles[k], :]
            policy_means_profiles[k] = policy_means_k

    return (policy_means_profiles, policies_profiles_masked,
            policies_profiles, policy_means, policies_ids_profiles)


def find_gender_trt_con_pairs(policies, subset_indices, trt_idx, trt_label, con_label):
    trt_con_pairs = []
    found = []
    for i in range(len(subset_indices)):
        pol_i_idx = subset_indices[i]
        pol_i = policies[pol_i_idx]
        if pol_i_idx in found:
            continue
        found_match = False
        for j in range(i+1, len(subset_indices)):
            pol_j_idx = subset_indices[j]
            pol_j = policies[pol_j_idx]
            if pol_i[trt_idx] == pol_j[trt_idx]:
                continue
            if pol_i[:trt_idx] == pol_j[:trt_idx] and pol_i[(trt_idx+1):] == pol_j[(trt_idx+1):]:
                if pol_i[trt_idx] == con_label and pol_j[trt_idx] == trt_label:
                    found_match = True
                    trt_con_pairs.append([pol_i_idx, pol_j_idx])
                elif pol_i[trt_idx] == trt_label and pol_j[trt_idx] == con_label:
                    found_match = True
                    trt_con_pairs.append([pol_j_idx, pol_i_idx])
                if found_match:
                    found.append(pol_i_idx)
                    found.append(pol_j_idx)
                    break
    return trt_con_pairs


def assign_label(a, bins):
    if a == 0:
        return (len(bins) - 1) // 2
    else:
        return np.digitize(a, bins) - (a > 0)

# infty, 2sd, 0, -2sd, -infty


def get_counts(
    R_set, R_profiles, policies_profiles_masked, policies_profiles,
    policies_ids_profiles, policy_means, R_vals,
    collect_effects=False, bins=None, collect_effects_by_profile=False
):

    assert (collect_effects or bins is not None or collect_effects_by_profile)

    n_models = len(R_set)

    trt_idx = 0
    trt_label = 1
    con_label = 0
    gen_idx = 2
    mal_label = 1
    fem_label = 0

    if R_vals[0] == 3:
        trt_label += 1
        con_label += 1
    if R_vals[2] == 3:
        mal_label += 1
        fem_label += 1

    M = R_vals.shape[0]
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R_vals)
    num_policies = len(all_policies)

    # trt_indices = [idx for idx, p in enumerate(all_policies) if p[trt_idx] == trt_label]
    # con_indices = [idx for idx, p in enumerate(all_policies) if p[trt_idx] == con_label]
    mal_indices = [idx for idx, p in enumerate(all_policies) if p[gen_idx] == mal_label]
    fem_indices = [idx for idx, p in enumerate(all_policies) if p[gen_idx] == fem_label]

    # Find treatment control pairs for each gender
    # The first element is control and the second element is treatment
    mal_trt_pairs = find_gender_trt_con_pairs(all_policies, mal_indices, trt_idx, trt_label, con_label)
    fem_trt_pairs = find_gender_trt_con_pairs(all_policies, fem_indices, trt_idx, trt_label, con_label)

    # The last four arms are the ones we care about
    active_arms = 4
    active_profiles, _ = hasse.enumerate_profiles(active_arms)
    num_active_profiles = len(active_profiles)

    # Profiles with all the features (but not the treatment)
    profiles_x, _ = hasse.enumerate_profiles(M-1)

    policies_to_active_profiles = {}
    active_profiles_policies_count = np.zeros(num_active_profiles)
    dosages = R_vals - 1
    fixed_poss = np.prod(dosages[:3])
    for i, active_prof in enumerate(active_profiles):
        prof_np = np.array(active_prof)
        dosages_prof_i = dosages[3:][prof_np > 0]
        num_policies_prof_i = fixed_poss * np.prod(dosages_prof_i)
        active_profiles_policies_count[i] = num_policies_prof_i

    active_profiles_policies_count_pooled = active_profiles_policies_count.copy()
    active_profiles_policies_count_gender = active_profiles_policies_count.copy()
    # print(R_vals)

    if R_vals[0] == 3 and R_vals[1] == 3 and R_vals[2] == 3:
        active_profiles_policies_count_pooled /= 2
        active_profiles_policies_count_gender /= 4

    elif R_vals[0] == 2 and R_vals[1] == 3 and R_vals[2] == 2:
        active_profiles_policies_count_pooled *= 2

    elif R_vals[0] == 3 and R_vals[1] == 3 and R_vals[2] == 2:
        active_profiles_policies_count_gender /= 2

    elif R_vals[0] == 2 and R_vals[1] == 2 and R_vals[2] == 2:
        active_profiles_policies_count_pooled[0] *= 4
        active_profiles_policies_count_pooled[1:] *= 4
        active_profiles_policies_count_gender[0] *= 2
        active_profiles_policies_count_gender[1:] *= 2

    elif R_vals[0] == 2 and R_vals[1] == 2 and R_vals[2] == 3:
        active_profiles_policies_count_pooled *= 2

    elif R_vals[0] == 3 and R_vals[1] == 2 and R_vals[2] == 2:
        active_profiles_policies_count_pooled *= 2

    elif R_vals[0] == 3 and R_vals[1] == 2 and R_vals[2] == 3:
        active_profiles_policies_count_gender /= 2

    elif R_vals[0] == 2 and R_vals[1] == 3 and R_vals[2] == 3:
        active_profiles_policies_count_gender /= 2

    for i, pol in enumerate(all_policies):
        last_four_arms = pol[3:]
        for j, prof in enumerate(active_profiles):
            if hasse.policy_to_profile(last_four_arms) == prof:
                policies_to_active_profiles[i] = j

    total_ctr = n_models

    if bins:
        trt_eff_bins = np.zeros(shape=(num_active_profiles, len(bins['trt'])))
        gen_eff_bins = np.zeros(shape=(num_active_profiles, len(bins['gender'])))

    # print("Beta size", num_policies, R_vals)
    beta = np.zeros(shape=(num_policies,))
    norm_const = 0
    total_prob = 0

    hasse_edges = {}
    for k, profile in enumerate(profiles):
        policies_k = [x for x in all_policies if hasse.policy_to_profile(x) == profile]
        hasse_edges[k] = extract_pools.lattice_edges(policies_k)

    if collect_effects:
        gen_trt_eff_array = []
        trt_eff_array = []

    if collect_effects_by_profile:
        # Store effects per profile with their probabilities
        profile_effects = {i: {'treatment': [], 'gender': [], 'probabilities': []}
                           for i in range(num_active_profiles)}

    for r, model_r in enumerate(R_set):
        pi_policies_profiles_r = {}
        loss_r = 0

        for k, profile_idx in enumerate(model_r):

            trt_profile_k = tuple([1] + list(profiles_x[k]))
            trt_profile_k_idx = profile_map[trt_profile_k]
            trt_policies_k = policies_profiles_masked[trt_profile_k_idx]
            trt_policies_k_ids = policies_ids_profiles[trt_profile_k_idx]

            ctl_profile_k = tuple([0] + list(profiles_x[k]))
            ctl_profile_k_idx = profile_map[ctl_profile_k]
            ctl_policies_k = policies_profiles_masked[ctl_profile_k_idx]
            ctl_policies_k_ids = policies_ids_profiles[ctl_profile_k_idx]

            sigma_int_k = R_profiles[k][profile_idx][0]
            sigma_trt_k = R_profiles[k][profile_idx][2]
            sigma_ctl_k = R_profiles[k][profile_idx][3]

            loss_r += R_profiles[k][profile_idx][1]

            # If both are None, then there is no data for either profile
            if sigma_trt_k is None and sigma_ctl_k is None:
                continue

            # If just one is None, then there is no data for that profile
            # So we cannot compute the TE
            if sigma_trt_k is None:
                ctl_pools_k_0, _ = extract_pools.extract_pools(
                    ctl_policies_k, sigma_ctl_k, hasse_edges[ctl_profile_k_idx]
                )
                ctl_pools_k = hasse.profile_ids_to_univ_ids(ctl_pools_k_0, ctl_policies_k_ids)
                sigma_policies_k = {}
                for pi, pol_pi_list in ctl_pools_k.items():
                    for pol_pi_i in pol_pi_list:
                        sigma_policies_k[pol_pi_i] = pi

            elif sigma_ctl_k is None:
                trt_pools_k_0, _ = extract_pools.extract_pools(
                    trt_policies_k, sigma_trt_k, hasse_edges[trt_profile_k_idx]
                )
                trt_pools_k = hasse.profile_ids_to_univ_ids(trt_pools_k_0, trt_policies_k_ids)
                sigma_policies_k = {}
                for pi, pol_pi_list in trt_pools_k.items():
                    for pol_pi_i in pol_pi_list:
                        sigma_policies_k[pol_pi_i] = pi

            else:
                trt_pools_k_0, _ = extract_pools.extract_pools(
                    trt_policies_k, sigma_trt_k, hasse_edges[trt_profile_k_idx]
                )
                ctl_pools_k_0, _ = extract_pools.extract_pools(
                    ctl_policies_k, sigma_ctl_k, hasse_edges[ctl_profile_k_idx]
                )

                trt_pools_k = hasse.profile_ids_to_univ_ids(trt_pools_k_0, trt_policies_k_ids)
                ctl_pools_k = hasse.profile_ids_to_univ_ids(ctl_pools_k_0, ctl_policies_k_ids)

                sigma_pools_k, sigma_policies_k = extract_pools.get_trt_ctl_pooled_partition(
                            trt_pools_k, ctl_pools_k, sigma_int_k
                        )

            pi_policies_profiles_r[k] = sigma_policies_k

        prob_r = np.exp(-loss_r)
        total_prob += prob_r

        # Aggregate pools
        pi_pools_r, pi_policies_r = extract_pools.__aggregate_pools__(pi_policies_profiles_r)
        mu_pools_r = loss.compute_pool_means(policy_means, pi_pools_r)

        policies_outcomes_est = np.zeros(shape=(num_policies,)) + np.inf

        for p, pool_id in pi_policies_r.items():
            policies_outcomes_est[p] = mu_pools_r[pool_id]
            beta[p] += mu_pools_r[pool_id] * np.exp(-loss_r)
        norm_const += np.exp(-loss_r)

        mal_outcome_pairs = np.zeros(shape=(len(mal_trt_pairs), 2))
        fem_outcome_pairs = np.zeros(shape=(len(fem_trt_pairs), 2))

        for idx, pair in enumerate(mal_trt_pairs):
            con_pol_idx = pair[0]
            trt_pol_idx = pair[1]
            mal_outcome_pairs[idx, 0] = policies_outcomes_est[con_pol_idx]
            mal_outcome_pairs[idx, 1] = policies_outcomes_est[trt_pol_idx]

        for idx, pair in enumerate(fem_trt_pairs):
            con_pol_idx = pair[0]
            trt_pol_idx = pair[1]
            fem_outcome_pairs[idx, 0] = policies_outcomes_est[con_pol_idx]
            fem_outcome_pairs[idx, 1] = policies_outcomes_est[trt_pol_idx]

        for i in range(len(mal_trt_pairs)):

            if all_policies[mal_trt_pairs[i][0]][3:] != all_policies[mal_trt_pairs[i][1]][3:]:
                raise RuntimeError("Profile indexing does not match for male treatment-control pairs")
            if all_policies[fem_trt_pairs[i][0]][3:] != all_policies[fem_trt_pairs[i][1]][3:]:
                raise RuntimeError("Profile indexing does not match for female treatment-control pairs")
            if all_policies[fem_trt_pairs[i][0]][3:] != all_policies[mal_trt_pairs[i][1]][3:]:
                raise RuntimeError("Profile indexing does not match for male-female treatment-control pairs")

            y_mal_con = mal_outcome_pairs[i, 0]
            y_mal_trt = mal_outcome_pairs[i, 1]
            y_fem_con = fem_outcome_pairs[i, 0]
            y_fem_trt = fem_outcome_pairs[i, 1]

            if np.isinf(y_mal_con) and np.isinf(y_fem_con) and np.isinf(y_mal_trt) and np.isinf(y_fem_trt):
                continue

            # Find the profile
            mal_con_pol_idx = mal_trt_pairs[i][0]
            profile_i = policies_to_active_profiles[mal_con_pol_idx]

            # Compute effects and then count them
            if np.isinf(y_mal_con) or np.isinf(y_mal_trt):
                trt_eff_mal = 0
            else:
                trt_eff_mal = y_mal_trt - y_mal_con
            if np.isinf(y_fem_trt) or np.isinf(y_fem_con):
                trt_eff_fem = 0
            else:
                trt_eff_fem = y_fem_trt - y_fem_con
            gen_trt_eff = trt_eff_fem - trt_eff_mal

            if collect_effects:
                trt_eff_array += [trt_eff_mal, trt_eff_fem]
                gen_trt_eff_array += [gen_trt_eff]

            if collect_effects_by_profile:
                # Store both male and female treatment effects with probability
                profile_effects[profile_i]['treatment'].append(trt_eff_mal)
                profile_effects[profile_i]['treatment'].append(trt_eff_fem)
                profile_effects[profile_i]['gender'].append(gen_trt_eff)
                # Add probability twice for male and female effects, once for gender effect
                profile_effects[profile_i]['probabilities'].append(prob_r)
                profile_effects[profile_i]['probabilities'].append(prob_r)

            else:
                '''
                vec = np.array(cov_counts_r)
                counts_vec = np.array([assign_label(vec[i], bins) for i in range(len(vec))])
                counts_vec = np.array(
                    [
                        np.dot(np.where(counts_vec == i, 1, 0),
                        all_model_prob_counts[covariate_idx][race_idx][idx]) for i in range(len(bins))
                    ]
                )
                counts_vec /= np.sum(counts_vec)
                signed_counts_race.append(counts_vec.copy())
                '''

                # Treatment effects for male
                trt_label_mal = assign_label(trt_eff_mal, bins["trt"])
                trt_eff_bins[profile_i, trt_label_mal] += prob_r

                trt_label_fem = assign_label(trt_eff_fem, bins["trt"])
                trt_eff_bins[profile_i, trt_label_fem] += prob_r

                # Treatment effects crossed with gender
                gen_label = assign_label(gen_trt_eff, bins["gender"])
                gen_eff_bins[profile_i, gen_label] += prob_r
        # break

    # print(active_profiles_policies_count)
    # print(active_profiles_policies_count_pooled)
    # print(active_profiles_policies_count_gender)

    # Normalize results
    # print("pooled", pos_eff_ctr + zero_eff_ctr + neg_eff_ctr)
    if collect_effects:
        results = {
            "gender_effects": gen_trt_eff_array,
            "treatment_effects": trt_eff_array,
            "total_prob": total_prob
        }
    elif collect_effects_by_profile:
        # Convert lists to arrays and return organized by profile
        for profile_i in range(num_active_profiles):
            profile_effects[profile_i]['treatment'] = np.array(profile_effects[profile_i]['treatment'])
            profile_effects[profile_i]['gender'] = np.array(profile_effects[profile_i]['gender'])
            profile_effects[profile_i]['probabilities'] = np.array(profile_effects[profile_i]['probabilities'])

        results = {
            "profile_effects": profile_effects,
            "total_prob": total_prob
        }
    else:
        assert (bins is not None)

        active_profiles_policies_count_pooled = active_profiles_policies_count_pooled.reshape((-1, 1))
        active_profiles_policies_count_gender = active_profiles_policies_count_gender.reshape((-1, 1))
        trt_eff_bins /= active_profiles_policies_count_pooled
        gen_eff_bins /= active_profiles_policies_count_gender

        beta /= norm_const

        results = {
            "n_models": total_ctr,
            "beta": beta
        }

        # print("trt_eff_bins:", trt_eff_bins)
        # print("gen_eff_bins:", gen_eff_bins)

        for i in range(len(bins['trt'])):
            results[f"{i}_pooled"] = trt_eff_bins[:, i]
            results[f"{i}_gender"] = gen_eff_bins[:, i]

    return results
