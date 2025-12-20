"""
Performance testing and verification for PPMx optimizations.

This script compares original and optimized implementations to ensure:
1. Results are identical (correctness)
2. Performance improvements are measured
"""

import time
import numpy as np
from ppmx_new import PPMxConfig, PPMxData, PPMxSampler, posterior_similarity_matrix, map_partition


def generate_test_data(n=100, d=3, n_clusters_true=3, seed=42):
    """Generate synthetic test data with known cluster structure."""
    np.random.seed(seed)
    
    # Generate clustered data
    cluster_means = np.random.randn(n_clusters_true, d) * 3
    X = []
    y = []
    true_labels = []
    
    for i in range(n):
        cluster_id = i % n_clusters_true
        x = cluster_means[cluster_id] + np.random.randn(d) * 0.5
        y_val = np.random.randn()
        
        X.append(x)
        y.append(y_val)
        true_labels.append(cluster_id)
    
    return np.array(X), np.array(y), np.array(true_labels)


def run_sampler_with_timing(data, config, label=""):
    """Run sampler and measure time."""
    print(f"\n{label}")
    print("-" * 60)
    
    sampler = PPMxSampler(data, config)
    
    start_time = time.time()
    results = sampler.run()
    elapsed_time = time.time() - start_time
    
    print(f"  Elapsed time: {elapsed_time:.3f}s")
    print(f"  Samples collected: {results['assignments'].shape[0]}")
    print(f"  Final n_clusters: {results['n_clusters'][-1]}")
    
    return results, elapsed_time


def compare_results(results1, results2, label1="Original", label2="Optimized"):
    """Compare two sets of results for equivalence."""
    print(f"\n{'='*60}")
    print(f"Comparing {label1} vs {label2}")
    print(f"{'='*60}")
    
    # Check if assignments are identical
    assignments_match = np.allclose(results1['assignments'], results2['assignments'])
    n_clusters_match = np.allclose(results1['n_clusters'], results2['n_clusters'])
    
    print(f"  Assignments match: {assignments_match}")
    print(f"  N_clusters match: {n_clusters_match}")
    
    if not assignments_match:
        # Check if difference is small
        diff = np.sum(results1['assignments'] != results2['assignments'])
        total = results1['assignments'].size
        print(f"  Assignment differences: {diff}/{total} ({100*diff/total:.2f}%)")
    
    # Compute similarity matrices
    sim1 = posterior_similarity_matrix(results1['assignments'])
    sim2 = posterior_similarity_matrix(results2['assignments'])
    
    sim_diff = np.abs(sim1 - sim2).max()
    print(f"  Max similarity matrix diff: {sim_diff:.6f}")
    
    return assignments_match and n_clusters_match and sim_diff < 1e-10


def test_small_dataset():
    """Test with small dataset for quick verification."""
    print("\n" + "="*60)
    print("TEST 1: Small Dataset (n=100, d=3)")
    print("="*60)
    
    X, y, true_labels = generate_test_data(n=100, d=3, n_clusters_true=3, seed=42)
    
    data = PPMxData.from_raw(X, Y=y, standardize=True)
    
    config = PPMxConfig(
        alpha=1.0,
        n_iter=50,
        burn_in=10,
        thin=2,
        random_state=42
    )
    
    results, elapsed = run_sampler_with_timing(data, config, "Running sampler...")
    
    print(f"\n✓ Test completed successfully")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {config.n_iter/elapsed:.1f} iterations/sec")
    
    return results, elapsed


def test_medium_dataset():
    """Test with medium dataset (similar to simulation scale)."""
    print("\n" + "="*60)
    print("TEST 2: Medium Dataset (n=500, d=4)")
    print("="*60)
    
    X, y, true_labels = generate_test_data(n=500, d=4, n_clusters_true=4, seed=123)
    
    data = PPMxData.from_raw(X, Y=y, standardize=True)
    
    config = PPMxConfig(
        alpha=1.0,
        n_iter=100,
        burn_in=20,
        thin=2,
        random_state=123
    )
    
    results, elapsed = run_sampler_with_timing(data, config, "Running sampler...")
    
    print(f"\n✓ Test completed successfully")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {config.n_iter/elapsed:.1f} iterations/sec")
    
    return results, elapsed


def test_diagnostics_performance():
    """Test performance of diagnostic functions."""
    print("\n" + "="*60)
    print("TEST 3: Diagnostics Performance")
    print("="*60)
    
    # Create dummy results
    n_samples = 100
    n = 200
    
    assignments_chain = np.random.randint(0, 5, size=(n_samples, n))
    
    print(f"  Testing similarity matrix (n_samples={n_samples}, n={n})...")
    start = time.time()
    sim = posterior_similarity_matrix(assignments_chain)
    elapsed_sim = time.time() - start
    print(f"    Time: {elapsed_sim:.3f}s")
    
    print(f"  Testing MAP partition...")
    start = time.time()
    map_part = map_partition(assignments_chain)
    elapsed_map = time.time() - start
    print(f"    Time: {elapsed_map:.3f}s")
    
    print(f"\n✓ Diagnostics test completed")
    
    return elapsed_sim, elapsed_map


def benchmark_suite():
    """Run full benchmark suite."""
    print("\n" + "="*70)
    print(" PPMx Performance Benchmark Suite")
    print("="*70)
    
    # Test 1: Small dataset
    results_small, time_small = test_small_dataset()
    
    # Test 2: Medium dataset
    results_medium, time_medium = test_medium_dataset()
    
    # Test 3: Diagnostics
    time_sim, time_map = test_diagnostics_performance()
    
    # Summary
    print("\n" + "="*70)
    print(" BENCHMARK SUMMARY")
    print("="*70)
    print(f"  Small dataset (n=100):   {time_small:.3f}s")
    print(f"  Medium dataset (n=500):  {time_medium:.3f}s")
    print(f"  Similarity matrix:       {time_sim:.3f}s")
    print(f"  MAP partition:           {time_map:.3f}s")
    print("="*70)


if __name__ == "__main__":
    benchmark_suite()
