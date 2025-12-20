"""
Test and benchmark comparison between Python and R PPMx backends.

This script tests both PPMx implementations on the same data to verify:
1. Compatibility (both produce valid results)
2. Performance (timing comparison)
3. Convergence (R-hat diagnostics)
4. Accuracy (MSE, IOU metrics)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rashomon import hasse
from rashomon import metrics
from rashomon import extract_pools
from baselines import PPMx

# Try to import R backend
try:
    from baselines.ppmx_r import PPMxR
    HAS_PPMX_R = True
except (ImportError, RuntimeError) as e:
    print(f"R backend not available: {e}")
    HAS_PPMX_R = False

# Import parameters
import reff_4 as params


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare Python and R PPMx backends")
    parser.add_argument("--n-per-pol", type=int, default=5,
                        help="Number of observations per policy")
    parser.add_argument("--n-iter", type=int, default=500,
                        help="Number of MCMC iterations")
    parser.add_argument("--burnin", type=int, default=100,
                        help="Burn-in period")
    parser.add_argument("--n-chains", type=int, default=2,
                        help="Number of MCMC chains")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    return parser.parse_args()


def generate_test_data(n_per_pol, M, R, mu, var, sigma, profiles, seed=42):
    """Generate test data for comparison using simulations.py approach."""
    np.random.seed(seed)
    
    # Generate policies and profiles
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)
    profiles, profile_map = hasse.enumerate_profiles(M)
    
    # Group policies by profile
    policies_profiles = {}
    pi_policies = {}
    
    for k, profile in enumerate(profiles):
        # Find all policies belonging to this profile
        policies_temp = [(i, x) for i, x in enumerate(all_policies) 
                        if hasse.policy_to_profile(x) == profile]
        if policies_temp:
            unzipped_temp = list(zip(*policies_temp))
            policies_k = list(unzipped_temp[1])
            policies_profiles[k] = policies_k
            
            # Mask empty arms for pool extraction
            profile_mask = list(map(bool, profile))
            policies_k_masked = []
            for pol in policies_k:
                masked_pol = tuple([pol[i] for i in range(M) if profile_mask[i]])
                policies_k_masked.append(masked_pol)
            
            # Extract pools
            if np.sum(profile) > 0:
                _, pi_policies_k = extract_pools.extract_pools(policies_k_masked, sigma[k])
                pi_policies[k] = pi_policies_k
            else:
                pi_policies[k] = {0: 0}
        else:
            policies_profiles[k] = []
            pi_policies[k] = {0: 0}
    
    # Generate data (ensure all policies get data)
    num_data = num_policies * n_per_pol
    X = np.zeros((num_data, M))
    D = np.zeros((num_data, 1), dtype=int)
    y = np.zeros((num_data, 1))
    
    idx_ctr = 0
    for k, profile in enumerate(profiles):
        policies_k = policies_profiles.get(k, [])
        
        for pol_idx, policy in enumerate(policies_k):
            policy_global_idx = all_policies.index(policy)
            pool_id = pi_policies[k].get(pol_idx, 0)
            mu_i = mu[k][pool_id]
            var_i = var[k][pool_id]
            y_i = np.random.normal(mu_i, np.sqrt(var_i), size=(n_per_pol, 1))
            
            start_idx = idx_ctr * n_per_pol
            end_idx = (idx_ctr + 1) * n_per_pol
            
            X[start_idx:end_idx, :] = policy
            D[start_idx:end_idx, :] = policy_global_idx
            y[start_idx:end_idx, :] = y_i
            
            idx_ctr += 1
    
    return X, D, y, all_policies, profile_map


def run_backend(backend_name, PPMxClass, X, y, D, n_iter, burnin, n_chains, seed, verbose):
    """Run a single PPMx backend and collect results."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {backend_name} backend")
        print(f"{'='*60}")
    
    # Initialize model
    model = PPMxClass(
        n_iter=n_iter,
        burnin=burnin,
        thin=2,
        alpha=1.0,
        cohesion='gaussian',
        similarity_weight=0.5,
        random_state=seed,
        verbose=verbose
    )
    
    # Time the fit
    start_time = time.time()
    model.fit(X, y, D, n_chains=n_chains)
    elapsed_time = time.time() - start_time
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Compute MSE
    mse = np.mean((y - y_pred) ** 2)
    
    # Get convergence info
    converged = model.converged_
    max_rhat = np.max(model.rhat_)
    mean_rhat = np.mean(model.rhat_)
    
    # Get cluster info
    mean_n_clusters = np.mean(model.n_clusters_samples_)
    
    results = {
        'backend': backend_name,
        'elapsed_time': elapsed_time,
        'mse': mse,
        'converged': converged,
        'max_rhat': max_rhat,
        'mean_rhat': mean_rhat,
        'mean_n_clusters': mean_n_clusters,
        'n_samples': len(model.partition_samples_),
        'model': model
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print(f"  MSE: {mse:.6f}")
        print(f"  Converged: {converged}")
        print(f"  Max R-hat: {max_rhat:.4f}")
        print(f"  Mean R-hat: {mean_rhat:.4f}")
        print(f"  Mean clusters: {mean_n_clusters:.2f}")
        print(f"  Total samples: {results['n_samples']}")
    
    return results


def main():
    args = parse_arguments()
    
    # Load parameters
    M = params.M
    R = params.R
    sigma = params.sigma
    mu = params.mu
    var = params.var
    
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"PPMx Backend Comparison")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Observations per policy: {args.n_per_pol}")
        print(f"  MCMC iterations: {args.n_iter}")
        print(f"  Burn-in: {args.burnin}")
        print(f"  Number of chains: {args.n_chains}")
        print(f"  Random seed: {args.seed}")
    
    # Generate test data
    if args.verbose:
        print(f"\nGenerating test data...")
    
    profiles, profile_map = hasse.enumerate_profiles(M)
    X, D, y, all_policies, profile_map = generate_test_data(
        args.n_per_pol, M, R, mu, var, sigma, profiles, seed=args.seed
    )
    
    if args.verbose:
        print(f"  Data shape: X={X.shape}, y={y.shape}, D={D.shape}")
        print(f"  Number of policies: {len(all_policies)}")
    
    # Test backends
    results_list = []
    
    # Python backend
    if args.verbose:
        print(f"\nTesting Python backend...")
    
    try:
        results_python = run_backend(
            'Python', PPMx, X, y, D,
            args.n_iter, args.burnin, args.n_chains, args.seed, args.verbose
        )
        results_list.append(results_python)
    except Exception as e:
        print(f"Python backend failed: {e}")
        import traceback
        traceback.print_exc()
    
    # R backend
    if HAS_PPMX_R:
        if args.verbose:
            print(f"\nTesting R backend...")
        
        try:
            results_r = run_backend(
                'R', PPMxR, X, y, D,
                args.n_iter, args.burnin, args.n_chains, args.seed, args.verbose
            )
            results_list.append(results_r)
        except Exception as e:
            print(f"R backend failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nR backend not available - skipping")
    
    # Compare results
    if len(results_list) >= 2:
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Comparison")
            print(f"{'='*60}")
        
        python_results = results_list[0]
        r_results = results_list[1]
        
        speedup = python_results['elapsed_time'] / r_results['elapsed_time']
        mse_diff = abs(python_results['mse'] - r_results['mse'])
        mse_rel_diff = mse_diff / python_results['mse'] * 100
        
        print(f"\nPerformance:")
        print(f"  Python time: {python_results['elapsed_time']:.2f}s")
        print(f"  R time: {r_results['elapsed_time']:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        print(f"\nAccuracy:")
        print(f"  Python MSE: {python_results['mse']:.6f}")
        print(f"  R MSE: {r_results['mse']:.6f}")
        print(f"  Absolute difference: {mse_diff:.6f}")
        print(f"  Relative difference: {mse_rel_diff:.2f}%")
        
        print(f"\nConvergence:")
        print(f"  Python converged: {python_results['converged']}, R-hat: {python_results['max_rhat']:.4f}")
        print(f"  R converged: {r_results['converged']}, R-hat: {r_results['max_rhat']:.4f}")
        
        print(f"\nClusters:")
        print(f"  Python mean clusters: {python_results['mean_n_clusters']:.2f}")
        print(f"  R mean clusters: {r_results['mean_n_clusters']:.2f}")
        
        # Save comparison results
        comparison_df = pd.DataFrame(results_list)
        comparison_df = comparison_df.drop('model', axis=1)
        output_path = "../Results/ppmx_backend_comparison.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        
        if args.verbose:
            print(f"\nSaved comparison to: {output_path}")
    
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"Test complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
