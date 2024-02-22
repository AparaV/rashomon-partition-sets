import numpy as np

#
# Fix ground truth
#
M = 3
R = np.array([3, 4, 6])

# Profile 0: (0, 0, 0)
sigma_0 = None
beta_0 = np.array([[0, 0, 0, 0]])
var_0 = np.array([1]) * 0.

# Profile 1: (0, 0, 1)
sigma_1 = np.array([[1, 1, 1, 1]])
beta_1 = np.array([[0, 0, 0, 0]])
var_1 = np.array([1]) * 0

# Profile 2: (0, 1, 0)
sigma_2 = np.array([[1, 1]])
beta_2 = np.array([[0, 0, 0, 0]])
var_2 = np.array([1]) * 0

# Profile 3: (0, 1, 1)
sigma_3 = np.array([[1, 1, np.inf, np.inf],
                    [1, 1, 1, 1]])
beta_3 = np.array([[0, 0, 0, 0]])
var_3 = np.array([1]) * 0

# Profile 4: (1, 0, 0)
sigma_4 = np.array([[1]])
beta_4 = np.array([[0, 0, 0, 0]])
var_4 = np.array([1]) * 0

# Profile 5: (1, 0, 1)
sigma_5 = np.array([[1, np.inf, np.inf, np.inf],
                    [1, 1, 1, 1]])
beta_5 = np.array([[0, 0, 0, 0]])
var_5 = np.array([1]) * 0

# Profile 6: (1, 1, 0)
sigma_6 = np.array([[1, np.inf],
                    [1, 1]])
beta_6 = np.array([[0, 0, 0, 0]])
var_6 = np.array([1]) * 0

# Profile 1: (1, 1, 1)
sigma_7 = np.array([[0, np.inf, np.inf, np.inf],
                    [0, 0, np.inf, np.inf],
                    [1, 0, 1, 1]])
beta_7 = np.array([[0, -1, 0, 1],   # 1
                   [3, -4, 0, 1],   # 2
                   [0, -1, 0, 1],   # 3
                   [2, 0, 0, 0],  # 4
                   [4, -2, -1, 1],  # 5
                   [1, 1, 1, -1],   # 6
                   [-3, 2, -3, 1],   # 7
                   [0, 0, 0, 0],   # 8
                   [4, 2, -3, -1],   # 9
                   [0, 0, 0, 0],  # 10
                   [5, 2, -3, 0],
                   [5, -1, 0, -1]])
var_7 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 0.01

sigma = [sigma_0, sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7]
beta = [beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7]
var = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7]

H = np.inf
# theta = 0.58
theta = 0.022
reg = 1e-3
