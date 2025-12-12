#!/usr/bin/env python
"""Quick test script to verify SpikeSlabLasso implementation."""

import numpy as np
import sys
import os

# Add Code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import SpikeSlabLasso

def test_basic_functionality():
    """Test basic fit and predict."""
    print("Testing basic functionality...")
    
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1.0, -0.5, 0.8, 0.0, -0.3])
    y = X @ true_beta + 0.1 * np.random.randn(n_samples)

    # Fit model
    model = SpikeSlabLasso(
        n_iter=1000,  # More iterations for better convergence
        burnin=200, 
        thin=2, 
        lambda0=10.0,
        lambda1=1.0,
        random_state=42, 
        verbose=True
    )
    model.fit(X, y, n_chains=3)  # More chains

    # Predict
    y_pred = model.predict(X)
    y_pred_map = model.predict_map(X)

    # Check results
    mse = np.mean((y - y_pred) ** 2)
    mse_map = np.mean((y - y_pred_map) ** 2)
    
    print(f"\nResults:")
    print(f"  True coefficients:     {true_beta}")
    print(f"  Estimated coefficients: {model.coef_}")
    print(f"  MAP coefficients:       {model.coef_map_}")
    print(f"  Inclusion probabilities: {model.inclusion_probs_}")
    print(f"  Posterior theta:        {model.theta_:.4f}")
    print(f"  MSE (posterior mean):   {mse:.6f}")
    print(f"  MSE (MAP):              {mse_map:.6f}")
    print(f"  Converged:              {model.converged_}")
    print(f"  Max R-hat:              {np.max(model.rhat_):.4f}")
    
    # Get selected features
    selected = model.get_selected_features(threshold=0.5)
    print(f"  Selected features (p>0.5): {selected}")
    
    assert y_pred.shape == (n_samples,)
    assert model.coef_.shape == (n_features,)
    assert mse < 1.0
    
    print("\n✓ Basic functionality test passed!")
    return True


def test_variable_selection():
    """Test variable selection on sparse problem."""
    print("\n" + "="*60)
    print("Testing variable selection...")
    
    np.random.seed(123)
    n_samples, n_features = 100, 10
    
    # Create sparse true coefficients (only 3 non-zero)
    true_beta = np.zeros(n_features)
    true_beta[0] = 2.0
    true_beta[3] = -1.5
    true_beta[7] = 1.0
    
    X = np.random.randn(n_samples, n_features)
    y = X @ true_beta + 0.2 * np.random.randn(n_samples)

    # Fit model with spike and slab
    model = SpikeSlabLasso(
        n_iter=1000,
        burnin=200,
        thin=2,
        lambda0=20.0,  # Strong spike
        lambda1=0.5,   # Moderate slab
        theta_init=0.3,  # Low prior inclusion probability
        update_theta=True,
        random_state=123,
        verbose=True
    )
    model.fit(X, y, n_chains=3)

    print(f"\nResults:")
    print(f"  True non-zero indices: [0, 3, 7]")
    print(f"  True coefficients:     {true_beta}")
    print(f"  Estimated coefficients: {model.coef_}")
    print(f"  Inclusion probabilities: {model.inclusion_probs_}")
    print(f"  Posterior theta:        {model.theta_:.4f}")
    
    selected = model.get_selected_features(threshold=0.5)
    print(f"  Selected features (p>0.5): {selected}")
    
    # Check performance
    true_nonzero = np.array([0, 3, 7])
    for idx in true_nonzero:
        print(f"  Feature {idx} (non-zero): inclusion prob = {model.inclusion_probs_[idx]:.4f}")
    
    print("\n✓ Variable selection test passed!")
    return True


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_variable_selection()
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
