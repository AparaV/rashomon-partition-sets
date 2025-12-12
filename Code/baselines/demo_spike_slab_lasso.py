"""
Demo: Spike and Slab Lasso for Bayesian Variable Selection

This script demonstrates the use of SpikeSlabLasso for variable selection
in sparse regression problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add Code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import SpikeSlabLasso


def demo_basic_usage():
    """Basic usage example."""
    print("="*70)
    print("DEMO: Basic Usage of Spike and Slab Lasso")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 10
    
    # Sparse true coefficients (only 3 non-zero)
    true_beta = np.zeros(n_features)
    true_beta[2] = 2.0   # Strong positive effect
    true_beta[5] = -1.5  # Strong negative effect
    true_beta[8] = 0.8   # Moderate positive effect
    
    X = np.random.randn(n_samples, n_features)
    y = X @ true_beta + 0.3 * np.random.randn(n_samples)
    
    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True non-zero coefficients at indices: [2, 5, 8]")
    print(f"True coefficients: {true_beta}")
    
    # Fit Spike and Slab Lasso
    print("\nFitting Spike and Slab Lasso...")
    model = SpikeSlabLasso(
        n_iter=2000,
        burnin=500,
        thin=2,
        lambda0=15.0,      # Strong spike (shrinkage near zero)
        lambda1=0.5,       # Moderate slab (non-zero regularization)
        theta_init=0.3,    # Prior: expect 30% features to be non-zero
        update_theta=True, # Adapt theta from data
        random_state=42,
        verbose=False
    )
    model.fit(X, y, n_chains=4)
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Converged: {model.converged_}")
    print(f"Max R-hat: {np.max(model.rhat_):.4f}")
    print(f"Posterior theta (inclusion probability): {model.theta_:.4f}")
    print(f"\nPosterior mean coefficients:")
    print(model.coef_)
    print(f"\nPosterior inclusion probabilities:")
    print(model.inclusion_probs_)
    
    # Variable selection
    selected = model.get_selected_features(threshold=0.5)
    print(f"\nSelected features (p > 0.5): {selected}")
    
    # Compare with truth
    print(f"\n{'Feature':<10} {'True β':<10} {'Est. β':<10} {'P(included)':<12} {'Selected':<10}")
    print("-"*60)
    for j in range(n_features):
        selected_mark = "✓" if j in selected else ""
        print(f"{j:<10} {true_beta[j]:<10.3f} {model.coef_[j]:<10.3f} "
              f"{model.inclusion_probs_[j]:<12.4f} {selected_mark:<10}")
    
    # Prediction
    y_pred = model.predict(X)
    y_pred_map = model.predict_map(X)
    mse_mean = np.mean((y - y_pred) ** 2)
    mse_map = np.mean((y - y_pred_map) ** 2)
    print(f"\nMSE (posterior mean): {mse_mean:.6f}")
    print(f"MSE (MAP estimate):   {mse_map:.6f}")
    
    return model, X, y, true_beta


def demo_comparison_with_bayesian_lasso():
    """Compare Spike-Slab Lasso with Bayesian Lasso."""
    print("\n" + "="*70)
    print("DEMO: Comparison with Bayesian Lasso")
    print("="*70)
    
    from baselines import BayesianLasso
    
    # Generate very sparse data
    np.random.seed(123)
    n_samples, n_features = 80, 20
    
    true_beta = np.zeros(n_features)
    true_beta[3] = 2.5
    true_beta[10] = -2.0
    true_beta[17] = 1.5
    
    X = np.random.randn(n_samples, n_features)
    y = X @ true_beta + 0.2 * np.random.randn(n_samples)
    
    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True sparsity: 3/{n_features} non-zero coefficients")
    print(f"Non-zero indices: [3, 10, 17]")
    
    # Fit Spike-Slab Lasso
    print("\nFitting Spike-Slab Lasso...")
    ssl = SpikeSlabLasso(
        n_iter=1500,
        burnin=300,
        lambda0=20.0,
        lambda1=0.5,
        theta_init=0.2,
        update_theta=True,
        random_state=123,
        verbose=False
    )
    ssl.fit(X, y, n_chains=3)
    
    # Fit Bayesian Lasso
    print("Fitting Bayesian Lasso...")
    blasso = BayesianLasso(
        n_iter=1500,
        burnin=300,
        lambda_prior=1.0,
        random_state=123,
        verbose=False
    )
    blasso.fit(X, y, n_chains=3)
    
    # Compare variable selection
    ssl_selected = ssl.get_selected_features(threshold=0.5)
    blasso_thresh = np.abs(blasso.coef_) > 0.1  # Threshold for BLasso
    blasso_selected = np.where(blasso_thresh)[0]
    
    print(f"\n{'='*70}")
    print("VARIABLE SELECTION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Feature':<10} {'True β':<10} {'SSL β':<10} {'BLasso β':<10} "
          f"{'P(incl)':<10} {'SSL Sel':<10} {'BL Sel':<10}")
    print("-"*80)
    
    for j in range(n_features):
        ssl_mark = "✓" if j in ssl_selected else ""
        bl_mark = "✓" if j in blasso_selected else ""
        true_mark = "*" if true_beta[j] != 0 else ""
        
        print(f"{j:<10} {true_beta[j]:<10.3f} {ssl.coef_[j]:<10.3f} "
              f"{blasso.coef_[j]:<10.3f} {ssl.inclusion_probs_[j]:<10.4f} "
              f"{ssl_mark:<10} {bl_mark:<10} {true_mark}")
    
    # Prediction performance
    ssl_mse = np.mean((y - ssl.predict(X)) ** 2)
    bl_mse = np.mean((y - blasso.predict(X)) ** 2)
    
    print(f"\nPrediction MSE:")
    print(f"  Spike-Slab Lasso: {ssl_mse:.6f}")
    print(f"  Bayesian Lasso:   {bl_mse:.6f}")
    
    # Sparsity
    ssl_sparsity = len(ssl_selected) / n_features
    bl_sparsity = len(blasso_selected) / n_features
    true_sparsity = 3 / n_features
    
    print(f"\nSparsity:")
    print(f"  True:             {true_sparsity:.2%} ({3}/{n_features})")
    print(f"  Spike-Slab Lasso: {ssl_sparsity:.2%} ({len(ssl_selected)}/{n_features})")
    print(f"  Bayesian Lasso:   {bl_sparsity:.2%} ({len(blasso_selected)}/{n_features})")


def demo_high_dimensional():
    """Demonstrate p > n scenario."""
    print("\n" + "="*70)
    print("DEMO: High-Dimensional Setting (p > n)")
    print("="*70)
    
    # High-dimensional setting: more features than samples
    np.random.seed(456)
    n_samples, n_features = 50, 100  # p = 100 > n = 50
    
    # Very sparse
    true_beta = np.zeros(n_features)
    true_beta[10] = 2.0
    true_beta[25] = -1.8
    true_beta[50] = 1.5
    true_beta[75] = -1.2
    true_beta[90] = 1.0
    
    X = np.random.randn(n_samples, n_features)
    y = X @ true_beta + 0.25 * np.random.randn(n_samples)
    
    print(f"\nData: {n_samples} samples, {n_features} features (p > n)")
    print(f"True sparsity: 5/{n_features} non-zero coefficients")
    print(f"Non-zero indices: [10, 25, 50, 75, 90]")
    
    # Fit model
    print("\nFitting Spike-Slab Lasso...")
    model = SpikeSlabLasso(
        n_iter=2000,
        burnin=500,
        thin=2,
        lambda0=25.0,      # Very strong spike for high-dim
        lambda1=0.3,       # Moderate slab
        theta_init=0.1,    # Expect very sparse (10%)
        update_theta=True,
        random_state=456,
        verbose=False
    )
    model.fit(X, y, n_chains=4)
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Converged: {model.converged_}")
    print(f"Posterior theta: {model.theta_:.4f}")
    
    selected = model.get_selected_features(threshold=0.5)
    print(f"\nNumber of selected features: {len(selected)}/{n_features}")
    print(f"Selected features: {selected}")
    
    # Check recovery of true non-zeros
    true_nonzero = np.array([10, 25, 50, 75, 90])
    recovered = np.isin(true_nonzero, selected)
    print(f"\nRecovered true non-zeros: {np.sum(recovered)}/5")
    print(f"Recovered indices: {true_nonzero[recovered]}")
    
    # Top features by inclusion probability
    top_k = 10
    top_indices = np.argsort(model.inclusion_probs_)[::-1][:top_k]
    print(f"\nTop {top_k} features by inclusion probability:")
    print(f"{'Index':<10} {'True β':<10} {'Est. β':<10} {'P(included)':<12}")
    print("-"*50)
    for idx in top_indices:
        mark = "*" if true_beta[idx] != 0 else ""
        print(f"{idx:<10} {true_beta[idx]:<10.3f} {model.coef_[idx]:<10.3f} "
              f"{model.inclusion_probs_[idx]:<12.4f} {mark}")
    
    # Prediction
    mse = np.mean((y - model.predict(X)) ** 2)
    print(f"\nPrediction MSE: {mse:.6f}")


if __name__ == "__main__":
    # Run all demos
    print("\n" + "#"*70)
    print("# SPIKE AND SLAB LASSO DEMONSTRATION")
    print("#"*70)
    
    model, X, y, true_beta = demo_basic_usage()
    
    demo_comparison_with_bayesian_lasso()
    
    demo_high_dimensional()
    
    print("\n" + "#"*70)
    print("# END OF DEMONSTRATION")
    print("#"*70 + "\n")
