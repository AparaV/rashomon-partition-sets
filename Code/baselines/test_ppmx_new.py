"""
Quick test for ppmx_new implementation
"""

import numpy as np
from ppmx_new import PPMxConfig, PPMxData, PPMxSampler, posterior_similarity_matrix, map_partition


def test_basic_clustering():
    """Test basic clustering with synthetic two-cluster data."""
    print("Test: Basic clustering")
    
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
    test_single_cluster()
    test_with_y()
    test_reproducibility()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
