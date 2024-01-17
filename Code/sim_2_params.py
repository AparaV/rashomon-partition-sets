import numpy as np

#
# Fix ground truth
#
M = 3
R = np.array([4, 4, 4])

# Fix the partitions
# Profile 0: (0, 0)
sigma_0 = None
mu_0 = np.array([0])
var_0 = np.array([1])

# Profile 1: (0, 0, 1)
sigma_1 = np.array([[1, 1]])
mu_1 = np.array([1])
var_1 = np.array([1])

# Profile 2: (0, 1, 0)
sigma_2 = np.array([[1, 0]])
mu_2 = np.array([0.5, 3.8])
var_2 = np.array([1, 1])

# Profile 3: (0, 1, 1)
sigma_3 = np.array([[1, 1],
                    [1, 1]])
mu_3 = np.array([1.5])
var_3 = np.array([1])

# Profile 4: (1, 0, 0)
sigma_4 = np.array([[1, 1]])
mu_4 = np.array([1])
var_4 = np.array([1])

# Profile 5: (1, 0, 1)
sigma_5 = np.array([[0, 1],
                    [1, 1]])
mu_5 = np.array([2, 3])
var_5 = np.array([1, 1])

# Profile 6: (1, 1, 0)
sigma_6 = np.array([[1, 1],
                    [1, 1]])
mu_6 = np.array([2.5])
var_6 = np.array([1])

# Profile 1: (1, 1, 1)
sigma_7 = np.array([[1, 1],
                    [1, 1],
                    [1, 0]])
mu_7 = np.array([3, 4])
var_7 = np.array([1, 1])

sigma = [sigma_0, sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7]
mu = [mu_0, mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7]
var = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7]

H = 15
theta = 2.2
reg = 1e-1
