"""
Test script for PPMx implementation.

Validates PPMx on simple synthetic data.
"""

import numpy as np
from baselines import PPMx
from sklearn.metrics import mean_squared_error

# Set seed for reproducibility
np.random.seed(42)

# Generate simple synthetic data
# 4 policies with 2 features each
# Policies 0,1 should be in one cluster (mean=0)
# Policies 2,3 should be in another cluster (mean=2)

n_policies = 4
n_obs_per_policy = 20
n_features = 2

# Policy features
X = np.array([
    [0.0, 0.0],  # Policy 0
    [0.1, 0.1],  # Policy 1 (similar to 0)
    [1.0, 1.0],  # Policy 2
    [1.1, 1.1],  # Policy 3 (similar to 2)
])

# Generate outcomes
D = np.repeat(np.arange(n_policies), n_obs_per_policy)
y = np.zeros(len(D))

# Policies 0,1 have mean 0, policies 2,3 have mean 2
for i in range(len(D)):
    policy_id = D[i]
    if policy_id < 2:
        y[i] = np.random.normal(0.0, 0.5)
    else:
        y[i] = np.random.normal(2.0, 0.5)

y = y.reshape(-1, 1)
D = D.reshape(-1, 1)

print("="*60)
print("Testing PPMx Implementation")
print("="*60)
print(f"Data shape: {len(D)} observations, {n_policies} policies")
print(f"Policy features shape: {X.shape}")
print("True structure: Policies [0,1] in cluster 1 (mean≈0), Policies [2,3] in cluster 2 (mean≈2)")
print()

# Test 1: PPMx without covariates
print("Test 1: PPMx without covariates (similarity_weight=0)")
print("-"*60)

ppmx_no_cov = PPMx(
    n_iter=2000,
    burnin=500,
    thin=2,
    alpha=1.0,
    cohesion='gaussian',
    similarity_weight=0.0,
    random_state=42,
    verbose=False
)

ppmx_no_cov.fit(X, y, D)

y_pred_no_cov = ppmx_no_cov.predict(X)
mse_no_cov = mean_squared_error(y, y_pred_no_cov)

print(f"MSE: {mse_no_cov:.4f}")
print(f"Mean number of clusters: {np.mean(ppmx_no_cov.n_clusters_samples_):.2f}")
print(f"Acceptance rate: {ppmx_no_cov.acceptance_rate_:.3f}")

# Get MAP partition
map_partition = ppmx_no_cov.get_map_partition()
print(f"MAP partition: {map_partition}")
print()

# Test 2: PPMx with covariates
print("Test 2: PPMx with covariates (similarity_weight=0.5)")
print("-"*60)

ppmx_with_cov = PPMx(
    n_iter=2000,
    burnin=500,
    thin=2,
    alpha=1.0,
    cohesion='gaussian',
    similarity_weight=0.5,
    similarity_bandwidth=0.5,
    random_state=42,
    verbose=False
)

ppmx_with_cov.fit(X, y, D)

y_pred_with_cov = ppmx_with_cov.predict(X)
mse_with_cov = mean_squared_error(y, y_pred_with_cov)

print(f"MSE: {mse_with_cov:.4f}")
print(f"Mean number of clusters: {np.mean(ppmx_with_cov.n_clusters_samples_):.2f}")
print(f"Acceptance rate: {ppmx_with_cov.acceptance_rate_:.3f}")

map_partition_cov = ppmx_with_cov.get_map_partition()
print(f"MAP partition: {map_partition_cov}")
print()

# Test 3: Check if it recovers true structure
print("Test 3: Partition structure analysis")
print("-"*60)


# Check if policies 0,1 are in same cluster and 2,3 are in same cluster
def check_structure(partition):
    """Check if partition matches true structure."""
    cluster_01_same = partition[0] == partition[1]
    cluster_23_same = partition[2] == partition[3]
    cluster_01_diff_23 = partition[0] != partition[2]

    return cluster_01_same and cluster_23_same and cluster_01_diff_23


correct_structure_no_cov = check_structure(map_partition)
correct_structure_with_cov = check_structure(map_partition_cov)

print(f"Without covariates - Correct structure: {correct_structure_no_cov}")
print(f"With covariates - Correct structure: {correct_structure_with_cov}")
print()

# Distribution of cluster counts
print("Test 4: Posterior distribution of cluster counts")
print("-"*60)
print("Without covariates:")
for n_clust in range(1, n_policies + 1):
    count = np.sum(ppmx_no_cov.n_clusters_samples_ == n_clust)
    if count > 0:
        pct = 100 * count / len(ppmx_no_cov.n_clusters_samples_)
        print(f"  {n_clust} clusters: {count} samples ({pct:.1f}%)")

print("\nWith covariates:")
for n_clust in range(1, n_policies + 1):
    count = np.sum(ppmx_with_cov.n_clusters_samples_ == n_clust)
    if count > 0:
        pct = 100 * count / len(ppmx_with_cov.n_clusters_samples_)
        print(f"  {n_clust} clusters: {count} samples ({pct:.1f}%)")

print()
print("="*60)
print("Tests completed successfully!")
print("="*60)
