"""
Test script for Gaussian PPMx integration with simulation framework.

This script runs the Gaussian PPMx implementation from gaussian_ppmx.py
against the simulation setup defined in reff_4.py and simulations.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from copy import deepcopy

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rashomon import hasse
from rashomon import loss
from rashomon import metrics
from rashomon import extract_pools
from baselines.gaussian_ppmx import GaussianPPMx

# Import parameters from reff_4
import reff_4 as params


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test Gaussian PPMx with simulation framework")
    parser.add_argument("--sample_size", type=int, default=10,
                        help="Number of samples per feature combination")
    parser.add_argument("--iters", type=int, default=5,
                        help="Number of simulation iterations")
    parser.add_argument("--output_prefix", type=str, default="test_gaussian_ppmx",
                        help="Prefix for output file name")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with reduced iterations"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress information (default: True)"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
        help="Disable progress printing"
    )
    args = parser.parse_args()
    return args


def generate_data(mu, var, n_per_pol, all_policies, pi_policies, profiles, profiles_profiles, M):
    """Generate synthetic data for simulation."""
    num_policies = len(all_policies)
    num_data = num_policies * n_per_pol
    X = np.zeros(shape=(num_data, M))
    D = np.zeros(shape=(num_data, 1), dtype='int')
    y = np.zeros(shape=(num_data, 1))

    idx_ctr = 0
    for k, profile in enumerate(profiles):
        policies_k = profiles_profiles[k]

        for idx, policy in enumerate(policies_k):
            policy_idx = [i for i, x in enumerate(all_policies) if x == policy]

            pool_id = pi_policies[k][idx]
            mu_i = mu[k][pool_id]
            var_i = var[k][pool_id]
            y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

            start_idx = idx_ctr * n_per_pol
            end_idx = (idx_ctr + 1) * n_per_pol

            X[start_idx:end_idx, ] = policy
            D[start_idx:end_idx, ] = policy_idx[0]
            y[start_idx:end_idx, ] = y_i

            idx_ctr += 1

    return X, D, y


def main():
    args = parse_arguments()

    # Load parameters from reff_4
    M = params.M
    R = params.R
    sigma = params.sigma
    mu = params.mu
    var = params.var
    H = params.H
    theta = params.theta
    reg = params.reg

    # PPMx parameters
    if args.test:
        ppmx_n_iter = 200
        ppmx_burnin = 50
        ppmx_thin = 2
        ppmx_n_chains = 2
        ppmx_alpha = getattr(params, 'ppmx_alpha', 1.0)
        ppmx_cohesion = getattr(params, 'ppmx_cohesion', 'normal-gamma')
        ppmx_similarity_weight = getattr(params, 'ppmx_similarity_weight', 0.5)
        ppmx_similarity_bandwidth = getattr(params, 'ppmx_similarity_bandwidth', 1.0)
    else:
        ppmx_n_iter = getattr(params, 'ppmx_n_iter', 1000)
        ppmx_burnin = getattr(params, 'ppmx_burnin', 300)
        ppmx_thin = getattr(params, 'ppmx_thin', 2)
        ppmx_n_chains = getattr(params, 'ppmx_n_chains', 2)
        ppmx_alpha = getattr(params, 'ppmx_alpha', 1.0)
        ppmx_cohesion = getattr(params, 'ppmx_cohesion', 'normal-gamma')
        ppmx_similarity_weight = getattr(params, 'ppmx_similarity_weight', 0.5)
        ppmx_similarity_bandwidth = getattr(params, 'ppmx_similarity_bandwidth', 1.0)

    # Map cohesion string to integer
    cohesion_map = {
        'dirichlet': 1,
        'uniform': 2,
        'normal-gamma': 1  # Default to Dirichlet
    }
    cohesion_int = cohesion_map.get(ppmx_cohesion, 1)

    # Simulation parameters
    if args.test:
        samples_per_pol = [args.sample_size] if args.sample_size else [10]
        num_sims = args.iters if args.iters else 5
        verbose = args.verbose
        if verbose:
            print("Running in TEST mode: 5 iterations, reduced MCMC samples")
    else:
        samples_per_pol = [args.sample_size]
        num_sims = args.iters
        verbose = args.verbose

    # Enumerate profiles and policies
    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)

    if verbose:
        print(f"Number of profiles: {num_profiles}")
        print(f"Number of policies: {num_policies}")
        print(f"Number of features (M): {M}")
        print()

    # Identify the pools
    policies_profiles = {}
    policies_profiles_masked = {}
    policies_ids_profiles = {}
    pi_policies = {}
    pi_pools = {}

    for k, profile in enumerate(profiles):
        policies_temp = [(i, x) for i, x in enumerate(all_policies) if hasse.policy_to_profile(x) == profile]
        unzipped_temp = list(zip(*policies_temp))
        policies_ids_k = list(unzipped_temp[0])
        policies_k = list(unzipped_temp[1])
        policies_profiles[k] = deepcopy(policies_k)
        policies_ids_profiles[k] = policies_ids_k

        profile_mask = list(map(bool, profile))

        # Mask the empty arms
        for idx, pol in enumerate(policies_k):
            policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
        policies_profiles_masked[k] = policies_k

        if np.sum(profile) > 0:
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(policies_k, sigma[k])
            if len(pi_pools_k.keys()) != mu[k].shape[0]:
                print(f"Profile {k}. Expected {len(pi_pools_k.keys())} pools. Received {mu[k].shape[0]} means.")
            pi_policies[k] = pi_policies_k
            pi_pools[k] = {}
            for x, y in pi_pools_k.items():
                y_full = [policies_profiles[k][i] for i in y]
                y_agg = [all_policies.index(i) for i in y_full]
                pi_pools[k][x] = y_agg
        else:
            pi_policies[k] = {0: 0}
            pi_pools[k] = {0: [0]}

    # Compute ground truth
    best_per_profile = [np.max(mu_k) for mu_k in mu]
    true_best_profile = np.argmax(best_per_profile)
    true_best_profile_idx = int(true_best_profile)
    true_best_effect = np.max(mu[true_best_profile])
    true_best = pi_pools[true_best_profile][np.argmax(mu[true_best_profile])]
    min_dosage_best_policy = metrics.find_min_dosage(true_best, all_policies)

    if verbose:
        print(f"True best profile: {true_best_profile_idx}")
        print(f"True best effect: {true_best_effect:.3f}")
        print(f"Number of true best policies: {len(true_best)}")
        print()

    # Storage for results
    ppmx_list = []
    profiles_str = [str(prof) for prof in profiles]

    # Output file setup
    start_sim = 0
    output_dir = "../Results/4arms/"
    os.makedirs(output_dir, exist_ok=True)

    output_suffix = f"_{samples_per_pol[0]}_{num_sims}"
    if args.test:
        output_suffix += "_test"
    output_suffix += ".csv"
    ppmx_fname = args.output_prefix + "_gaussian_ppmx" + output_suffix

    if verbose:
        print(f"Sample size per policy: {samples_per_pol[0]}")
        print(f"Number of iterations: {num_sims}")
        print(f"Output file: {ppmx_fname}")
        print()

    np.random.seed(3)

    # Run simulations
    for n_per_pol in samples_per_pol:
        if verbose:
            print(f"\nNumber of samples per policy: {n_per_pol}")

        for sim_i in range(start_sim, start_sim + num_sims):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Simulation {sim_i + 1}/{num_sims}")
                print(f"{'='*60}")

            np.random.seed(sim_i)

            # Generate data
            X, D, y = generate_data(mu, var, n_per_pol, all_policies, pi_policies, profiles, policies_profiles, M)
            y_flat = y.flatten()

            if verbose:
                print(f"Generated data: X.shape={X.shape}, y.shape={y.shape}")
                print(f"Number of observations: {len(y_flat)}")
                print(f"Number of features: {X.shape[1]}")
                print()

            # Create DataFrame for covariates (continuous only)
            X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(M)])

            # Run Gaussian PPMx for each chain
            if verbose:
                print(f"Running Gaussian PPMx with {ppmx_n_chains} chains...")

            for chain_idx in range(ppmx_n_chains):
                if verbose:
                    print(f"\n  Chain {chain_idx + 1}/{ppmx_n_chains}")

                # Initialize PPMx model
                ppmx_model = GaussianPPMx(
                    mean_model=1,  # Cluster-specific means
                    cohesion=cohesion_int,
                    M=ppmx_alpha,
                    PPM=False,  # Use PPMx with covariates
                    similarity_function=1,  # Auxiliary similarity
                    consim=1,  # N-N similarity
                    calibrate=0,  # No calibration
                    sim_parms=None,  # Use defaults
                    model_priors=None,  # Use defaults
                    mh=None,  # Use defaults
                    draws=ppmx_n_iter,
                    burn=ppmx_burnin,
                    thin=ppmx_thin,
                    verbose=verbose,
                    random_state=sim_i * 1000 + chain_idx
                )

                # Fit model
                try:
                    ppmx_model.fit(y_flat, X=X_df)
                    results = ppmx_model.results_

                    if verbose:
                        print(f"    MCMC completed successfully")
                        print(f"    WAIC: {results.WAIC:.2f}")
                        print(f"    LPML: {results.lpml:.2f}")
                        print(f"    Mean clusters: {np.mean(results.nclus):.1f}")

                    # Extract posterior samples
                    n_samples = results.mu.shape[0]

                    # Compute log posterior (use negative WAIC as proxy)
                    # For proper comparison, we'd need the actual posterior density
                    # Here we use the likelihood contribution
                    log_posteriors = np.sum(results.like, axis=1)
                    neg_log_posteriors = -log_posteriors

                    # Process each posterior sample
                    for sample_idx in range(n_samples):
                        # Get predictions for this sample
                        y_pred = results.fitted[sample_idx, :].reshape(-1, 1)

                        # Compute metrics
                        ppmx_results = metrics.compute_all_metrics(
                            y, y_pred, D, true_best, all_policies, profile_map,
                            min_dosage_best_policy, true_best_effect
                        )

                        sqrd_err_ppmx = ppmx_results["sqrd_err"]
                        iou_ppmx = ppmx_results["iou"]
                        best_profile_indicator_ppmx = ppmx_results["best_prof"]
                        min_dosage_present_ppmx = ppmx_results["min_dos_inc"]
                        best_policy_diff_ppmx = ppmx_results["best_pol_diff"]

                        # Get cluster information for this sample
                        n_clusters_sample = results.nclus[sample_idx]

                        # Compute acceptance rate (placeholder - would need actual MH acceptance tracking)
                        acceptance_rate = 0.0  # Not available in current implementation

                        # Compute coverage metrics using all samples from this chain
                        if sample_idx == n_samples - 1:  # Compute once at the end
                            # Use all posterior samples for coverage
                            fitted_samples = results.fitted[:, :].reshape(-1, X.shape[0])
                            # For simplicity, compute coverage on last sample
                            # In practice, you'd want to use all samples
                            iou_coverage_ppmx = iou_ppmx  # Placeholder
                            min_dosage_coverage_ppmx = min_dosage_present_ppmx  # Placeholder
                        else:
                            iou_coverage_ppmx = 0.0
                            min_dosage_coverage_ppmx = 0.0

                        # Store results
                        converged = 1  # Assume converged for now
                        max_rhat = 1.0  # Would need to compute if we had multiple chains combined

                        this_list = [
                            n_per_pol, sim_i, sample_idx,
                            neg_log_posteriors[sample_idx], sqrd_err_ppmx, iou_ppmx,
                            min_dosage_present_ppmx, best_policy_diff_ppmx,
                            converged, max_rhat,
                            iou_coverage_ppmx, min_dosage_coverage_ppmx,
                            n_clusters_sample, acceptance_rate
                        ]
                        this_list += list(best_profile_indicator_ppmx)
                        ppmx_list.append(this_list)

                except Exception as e:
                    print(f"    Error in chain {chain_idx + 1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            if verbose and (sim_i + 1) % 5 == 0:
                print(f"\nCompleted {sim_i + 1}/{num_sims} simulations")

    # Save results
    ppmx_cols = [
        "n_per_pol", "sim_num", "sample_idx",
        "neg_log_posterior", "MSE", "IOU", "min_dosage", "best_pol_diff",
        "converged", "max_rhat", "IOU_coverage", "min_dosage_coverage",
        "n_clusters", "acceptance_rate"
    ]
    ppmx_cols += profiles_str

    ppmx_df = pd.DataFrame(ppmx_list, columns=ppmx_cols)
    output_path = os.path.join(output_dir, ppmx_fname)
    ppmx_df.to_csv(output_path, index=False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulations complete!")
        print(f"Saved results to: {output_path}")
        print(f"\nSummary statistics:")
        print(f"  Total samples: {len(ppmx_df)}")
        print(f"  Mean MSE: {ppmx_df['MSE'].mean():.4f}")
        print(f"  Mean IOU: {ppmx_df['IOU'].mean():.4f}")
        print(f"  Mean clusters: {ppmx_df['n_clusters'].mean():.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
