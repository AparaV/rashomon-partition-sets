import numpy as np

from rashomon import hasse
from rashomon import counter

#
# Fix ground truth
#
M = 4
R = np.array([4, 4, 4, 4])

num_profiles = 2**M
profiles, profile_map = hasse.enumerate_profiles(M)

# Fix the partitions
# Profile 0: (0, 0, 0, 0)
sigma_0 = None
mu_0 = np.array([0])
var_0 = np.array([1])

sigma = [sigma_0]
mu = [mu_0]
var = [var_0]

h = 0
for i, profile in enumerate(profiles):
    if i == 0:
        h += 1
        continue

    m = np.sum(profile)
    sigma_i = np.zeros(shape=(m, R[0]-2)) + 1
    num_pol_i = (R[0]-1)**m
    var_fixed = 1
    # var_scale_i = np.sqrt(num_pol_i)
    var_scale_i = 1

    # (1, 0, 1, 0)
    if i == 10:
        # sigma_i[1, 1] = 0
        # mu_i = np.array([0, 4.2])
        # var_i = np.array([1, 2])
        mu_i = np.array([4.5])
        var_i = np.array([var_fixed * var_scale_i])
        var_i = np.array([var_fixed * 1.5])
    # (0, 1, 0, 1)
    elif i == 5:
        # sigma_i[1, 1] = 0
        # mu_i = np.array([0, 3.8])
        # var_i = np.array([1, 1])
        mu_i = np.array([4.45])
        var_i = np.array([var_fixed * var_scale_i])
    # (0, 1, 0, 0)
    elif i == 4:
        # mu_i = np.array([3.8])
        # var_i = np.array([1])
        mu_i = np.array([4.3])
        var_i = np.array([var_fixed * var_scale_i])
    # (0, 0, 0, 1)
    elif i == 1:
        # mu_i = np.array([3.7])
        # var_i = np.array([1])
        mu_i = np.array([4.4])
        var_i = np.array([var_fixed * var_scale_i])
    # (1, 1, 1, 1)
    elif i == 15:
        # sigma_i[3, 1] = 0
        # mu_i = np.array([0, 3.8])
        # var_i = np.array([1, 1])
        mu_i = np.array([4.35])
        var_i = np.array([var_fixed * var_scale_i])
    else:
        # mu_i = np.array([1])
        # var_i = np.array([1])
        mu_i = np.array([0])
        var_i = np.array([var_fixed * var_scale_i])

    h += counter.num_pools(sigma_i)

    # print(i, profile, "\n", sigma_i)

    sigma.append(sigma_i)
    mu.append(mu_i)
    var.append(var_i)

print(h)

H = h + 4
# theta = 4.2
theta = 2.8
reg = 1e-1
# # Lasso
lasso_reg = 5e-3

# # Profile 1: (0, 0, 1)
# sigma_1 = np.array([[1, 1]])
# mu_1 = np.array([0])
# var_1 = np.array([1])

# # Profile 2: (0, 1, 0)
# sigma_2 = np.array([[1, 1]])
# mu_2 = np.array([3.8])
# var_2 = np.array([1])

# # Profile 3: (0, 1, 1)
# sigma_3 = np.array([[1, 1],
#                     [1, 0]])
# mu_3 = np.array([0, 1])
# var_3 = np.array([1, 2])

# # Profile 4: (1, 0, 0)
# sigma_4 = np.array([[1, 1]])
# mu_4 = np.array([1])
# var_4 = np.array([1])

# # Profile 5: (1, 0, 1)
# sigma_5 = np.array([[1, 1],
#                     [1, 0]])
# mu_5 = np.array([0, 4])
# var_5 = np.array([1, 1])

# # Profile 6: (1, 1, 0)
# sigma_6 = np.array([[1, 1],
#                     [1, 0]])
# mu_6 = np.array([0, 1])
# var_6 = np.array([1, 2])

# # Profile 1: (1, 1, 1)
# sigma_7 = np.array([[1, 1],
#                     [1, 1],
#                     [1, 0]])
# mu_7 = np.array([0, 3.8])
# var_7 = np.array([1, 1])

# sigma = [sigma_0, sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7]
# mu = [mu_0, mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7]
# var = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7]
