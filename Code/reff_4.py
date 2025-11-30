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

# print(h)

H = h + 4
# theta = 4.2
theta = 2.8
reg = 1e-1
# # Lasso
lasso_reg = 5e-3

# Bayesian Lasso parameters
# Note: With n=2560, p=256, sampling is slow (~6s per 100 iterations)
# Using reduced iterations for computational feasibility
blasso_n_iter = 5000      # Total iterations (reduced from 2000)
blasso_burnin = 2000      # Burn-in (reduced from 500)
blasso_thin = 2          # Thinning
blasso_n_chains = 3      # Number of chains (reduced from 4)
blasso_lambda = 5e-1      # Regularization strength (stronger for high-dim)
blasso_tau2_a = 1e-1      # Inverse-Gamma prior shape
blasso_tau2_b = 1e-1     # Inverse-Gamma prior scale

# Bootstrap Lasso parameters
bootstrap_n_iter = 1000          # Number of bootstrap iterations
bootstrap_alpha = 5e-3           # Lasso regularization (same as lasso_reg)
bootstrap_confidence_level = 0.95  # Confidence level for intervals
bootstrap_random_state = None    # Random state (None uses sim_i)
