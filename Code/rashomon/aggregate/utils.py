import numpy as np


def find_feasible_sum_subsets(S: list[np.ndarray], theta: float) -> list[list[int]]:
    """
    Find all combinations of indices from the list of arrays S such that sum <= theta

    Arguments:
    S (list[np.ndarray]): List of arrays of numbers
    theta (float): Threshold

    Returns:
    list[list[int]]: List of combinations of indices, one from each array in S
        such that their sum is less than or equal to theta
    """
    # See this https://stackoverflow.com/q/62311484/5055644
    # NP Hard? https://link.springer.com/chapter/10.1007/978-3-540-24777-7_11

    num_sets = len(S)
    if num_sets == 0:
        return []

    S1 = S[0]
    S1_feasible_idx = np.where(S1 <= theta)[0]

    if num_sets == 1:
        feasible_combs = [[x] for x in S1_feasible_idx]
        return feasible_combs

    # Since the list is sorted, add the first elements of the remaining sets
    # Then use this to check feasibility of indices in S1_feasible_idx
    first_element_sum = np.sum([s[0] for s in S[1:]])
    feasible_combs = []
    for i in S1_feasible_idx:
        theta_i = theta - S1[i]
        # Safe to stop our search on first infeasible index because of sorting
        if first_element_sum > theta_i:
            break
        subproblem_res_i = find_feasible_sum_subsets(S[1:], theta_i)
        for res in subproblem_res_i:
            feasible_combs.append([i] + res)

    return feasible_combs
