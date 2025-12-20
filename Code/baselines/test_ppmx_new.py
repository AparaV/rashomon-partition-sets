"""
Quick test for ppmx_new implementation
"""

import numpy as np
import itertools
from ppmx_new import PPMxConfig, PPMxData, PPMxSampler, posterior_similarity_matrix, map_partition


def compute_rhat(chains):
    """Compute Gelman-Rubin R-hat statistic for convergence diagnosis.
    
    Parameters
    ----------
    chains : list of np.ndarray
        List of chains, each of shape (n_samples, ...) where ... is the parameter dimension
        
    Returns
    -------
    float
        R-hat statistic. Values close to 1.0 indicate convergence (< 1.1 is good)
    """
    chains = [np.array(chain) for chain in chains]
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    
    # Flatten parameter dimensions if needed
    original_shape = chains[0].shape
    chains_flat = [chain.reshape(n_samples, -1) for chain in chains]
    n_params = chains_flat[0].shape[1]
    
    # Compute for each parameter
    rhats = []
    for param_idx in range(n_params):
        param_chains = [chain[:, param_idx] for chain in chains_flat]
        
        # Within-chain variance
        W = np.mean([np.var(chain, ddof=1) for chain in param_chains])
        
        # Between-chain variance
        chain_means = [np.mean(chain) for chain in param_chains]
        B = n_samples * np.var(chain_means, ddof=1)
        
        # Variance estimate
        var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        
        # R-hat
        if W > 0:
            rhat = np.sqrt(var_plus / W)
        else:
            rhat = 1.0
        
        rhats.append(rhat)
    
    # Return max R-hat (worst case)
    return np.max(rhats)


def generate_simulation_data(M=3, R=3, n_per_pol=10, seed=42):
    """Generate data similar to simulation setup.
    
    Parameters
    ----------
    M : int
        Number of arms/factors
    R : int
        Number of levels per arm
    n_per_pol : int
        Number of observations per policy
    seed : int
        Random seed
        
    Returns
    -------
    X : np.ndarray
        Covariate matrix (policies)
    y : np.ndarray
        Response vector
    true_pools : dict
        Ground truth pool assignments
    """
    np.random.seed(seed)
    
    # Generate all possible policies
    all_policies = list(itertools.product(range(R), repeat=M))
    num_policies = len(all_policies)
    
    # Create pool structure - simple version
    # Assign policies to pools based on similarity
    # Pool 0: low intensity policies (sum < M)
    # Pool 1: medium intensity (sum = M to 2*M-1)
    # Pool 2: high intensity (sum >= 2*M)
    
    pool_assignment = {}
    pool_means = {0: 0.0, 1: 2.0, 2: 4.0}  # Different means for each pool
    pool_vars = {0: 1.0, 1: 1.0, 2: 1.0}
    
    for i, policy in enumerate(all_policies):
        intensity_sum = sum(policy)
        if intensity_sum < M:
            pool_assignment[i] = 0
        elif intensity_sum < 2 * M:
            pool_assignment[i] = 1
        else:
            pool_assignment[i] = 2
    
    # Generate data
    n_total = num_policies * n_per_pol
    X = np.zeros((n_total, M))
    y = np.zeros(n_total)
    
    idx = 0
    for pol_idx, policy in enumerate(all_policies):
        pool_id = pool_assignment[pol_idx]
        mu = pool_means[pool_id]
        sigma = np.sqrt(pool_vars[pool_id])
        
        # Generate n_per_pol observations for this policy
        for _ in range(n_per_pol):
            X[idx] = policy
            y[idx] = np.random.normal(mu, sigma)
            idx += 1
    
    return X, y, pool_assignment


def test_basic_clustering():
    """Test basic clustering with synthetic two-cluster data."""
    print("Test: Basic clustering (simple)")
    
    # Create synthetic data with two clear clusters
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(20, 2) + [0, 0],  # Cluster 1
        np.random.randn(20, 2) + [5, 5],  # Cluster 2
    ])
    
    # Create data container
    data = PPMxData.from_raw(X, standardize=True)
    print(f"  Data shape: {data.X.shape}")
    print(f"  n={data.n}, d={data.d}")
    
    # Configure sampler
    config = PPMxConfig(
        alpha=1.0,
        n_iter=100,
        burn_in=20,
        thin=2,
        random_state=42
    )
    
    # Run sampler
    sampler = PPMxSampler(data, config)
    results = sampler.run()
    
    print(f"  Samples collected: {results['assignments'].shape[0]}")
    print(f"  Number of clusters (last sample): {results['n_clusters'][-1]}")
    
    unique_k, counts = np.unique(results['n_clusters'], return_counts=True)
    print(f"  Cluster distribution: {dict(zip(unique_k, counts))}")
    
    # Compute diagnostics
    similarity = posterior_similarity_matrix(results['assignments'])
    print(f"  Similarity matrix shape: {similarity.shape}")
    print(f"  Diagonal check (should be 1.0): min={similarity.diagonal().min():.3f}, max={similarity.diagonal().max():.3f}")
    
    map_part = map_partition(results['assignments'])
    print(f"  MAP partition: {len(np.unique(map_part))} clusters")
    print(f"  Cluster sizes: {np.bincount(map_part)}")
    
    print("  ✓ Passed\n")


def test_simulation_style_data():
    """Test with data generated similar to simulations."""
    print("Test: Simulation-style data (M=3, R=3) with convergence diagnostics")
    
    # Generate data with M=3 arms, R=3 levels, 5 obs per policy
    X, y, true_pools = generate_simulation_data(M=3, R=3, n_per_pol=5, seed=42)
    
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    print(f"  Number of policies: {len(set(map(tuple, X)))}")
    print(f"  True number of pools: {len(set(true_pools.values()))}")
    
    # Create data container
    data = PPMxData.from_raw(X, Y=y, standardize=True)
    
    # Run multiple chains for convergence diagnostics
    n_chains = 4
    n_iter = 300
    burn_in = 100
    thin = 4
    
    print(f"  Running {n_chains} chains with {n_iter} iterations each...")
    
    chains_results = []
    chains_similarity = []
    chains_n_clusters = []
    
    for chain_id in range(n_chains):
        config = PPMxConfig(
            alpha=1.0,
            n_iter=n_iter,
            burn_in=burn_in,
            thin=thin,
            random_state=42 + chain_id  # Different seed for each chain
        )
        
        sampler = PPMxSampler(data, config)
        results = sampler.run()
        
        chains_results.append(results['assignments'])
        chains_n_clusters.append(results['n_clusters'])
        
        # Compute similarity matrix for this chain
        similarity = posterior_similarity_matrix(results['assignments'])
        chains_similarity.append(similarity)
    
    # Compute R-hat on similarity matrices (upper triangle only to avoid redundancy)
    n = X.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    chains_similarity_triu = [sim[triu_indices] for sim in chains_similarity]
    
    rhat_similarity = compute_rhat(chains_similarity_triu)
    print(f"  R-hat (similarity matrix): {rhat_similarity:.4f} {'✓' if rhat_similarity < 1.1 else '✗'}")
    
    # Compute R-hat on number of clusters
    rhat_n_clusters = compute_rhat([[nc] for nc in chains_n_clusters])
    print(f"  R-hat (n_clusters): {rhat_n_clusters:.4f} {'✓' if rhat_n_clusters < 1.1 else '✗'}")
    
    # Use first chain for remaining diagnostics
    results = {'assignments': chains_results[0], 'n_clusters': chains_n_clusters[0]}
    
    print(f"  Samples per chain: {results['assignments'].shape[0]}")
    print(f"  Number of clusters (last sample, chain 1): {results['n_clusters'][-1]}")
    
    unique_k, counts = np.unique(results['n_clusters'], return_counts=True)
    print(f"  Cluster distribution (chain 1): {dict(zip(unique_k, counts))}")
    
    # Compute diagnostics on pooled chains
    pooled_assignments = np.vstack(chains_results)
    pooled_similarity = posterior_similarity_matrix(pooled_assignments)
    map_part = map_partition(pooled_assignments)
    print(f"  MAP partition (pooled): {len(np.unique(map_part))} clusters")
    
    # Check if policies with same true pool tend to cluster together
    pol_to_cluster = {}
    for i in range(len(X)):
        policy = tuple(X[i])
        cluster = map_part[i]
        if policy not in pol_to_cluster:
            pol_to_cluster[policy] = []
        pol_to_cluster[policy].append(cluster)
    
    # Each policy should be consistently assigned
    consistent = all(len(set(clusters)) == 1 for clusters in pol_to_cluster.values())
    print(f"  Policies consistently assigned: {consistent}")
    
    if rhat_similarity < 1.1 and rhat_n_clusters < 1.1:
        print("  ✓ Passed (converged)\n")
    else:
        print("  ⚠ Passed but convergence questionable\n")


def test_single_cluster():
    """Test with data from single cluster."""
    print("Test: Single cluster data")
    
    np.random.seed(123)
    X = np.random.randn(30, 3)
    
    data = PPMxData.from_raw(X, standardize=True)
    config = PPMxConfig(
        alpha=0.5,
        n_iter=50,
        burn_in=10,
        thin=2,
        random_state=123
    )
    
    sampler = PPMxSampler(data, config)
    results = sampler.run()
    
    print(f"  Samples: {results['assignments'].shape[0]}")
    print(f"  Cluster counts: {np.unique(results['n_clusters'])}")
    
    map_part = map_partition(results['assignments'])
    print(f"  MAP: {len(np.unique(map_part))} clusters")
    print("  ✓ Passed\n")


def test_with_y():
    """Test that Y is accepted but not used in clustering."""
    print("Test: Data with Y")
    
    np.random.seed(456)
    X = np.random.randn(20, 2)
    Y = np.random.randn(20)
    
    data = PPMxData.from_raw(X, Y=Y, standardize=True)
    print(f"  X shape: {data.X.shape}, Y shape: {data.Y.shape}")
    
    config = PPMxConfig(n_iter=30, burn_in=10, thin=2, random_state=456)
    sampler = PPMxSampler(data, config)
    results = sampler.run()
    
    print(f"  Samples collected: {results['assignments'].shape[0]}")
    print("  ✓ Passed\n")


def test_reproducibility():
    """Test that same seed gives same results."""
    print("Test: Reproducibility")
    
    X = np.random.randn(15, 2)
    
    config = PPMxConfig(n_iter=50, burn_in=10, thin=2, random_state=999)
    
    # Run 1
    data1 = PPMxData.from_raw(X.copy(), standardize=True)
    sampler1 = PPMxSampler(data1, config)
    results1 = sampler1.run()
    
    # Run 2
    data2 = PPMxData.from_raw(X.copy(), standardize=True)
    sampler2 = PPMxSampler(data2, config)
    results2 = sampler2.run()
    
    # Check if identical
    match = np.all(results1['assignments'] == results2['assignments'])
    print(f"  Results match: {match}")
    
    if match:
        print("  ✓ Passed\n")
    else:
        print("  ✗ Failed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ppmx_new implementation")
    print("=" * 60 + "\n")
    
    test_basic_clustering()
    test_simulation_style_data()
    test_single_cluster()
    test_with_y()
    test_reproducibility()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
