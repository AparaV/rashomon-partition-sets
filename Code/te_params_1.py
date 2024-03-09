import numpy as np

M = 4
R = np.array([2, 4, 5, 5])


# (0, 1, 1, 1)
sigma_0 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                    ])
mu_0 = np.array([0])
var_0 = np.array([1]) * 0.5

# (1, 1, 1)
sigma_1 = np.array([[np.inf, np.inf, np.inf],
                    [0, 0, np.inf],
                    [1, 0, 1],
                    [1, 1, 0],
                    ])
mu_1 = np.array([2, 4, 2, 0, 3, 5, 7, 1, 1, -1, -1, -2])
var_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 0.5


interested_profiles = [(0, 1, 1, 1), (1, 1, 1, 1)]

sigma_tmp = [sigma_0, sigma_1]
mu_tmp = [mu_0, mu_1]
var_tmp = [var_0, var_1]

trt_arm_idx = 0

ctl_profile_idx = 7
trt_profile_idx = 15

ctl_profile = (0, 1, 1, 1)
trt_profile = (1, 1, 1, 1)

H = np.inf
theta = 2
reg = 1e-1

lasso_reg = 1e-2
