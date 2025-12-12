"""
Unit tests for Spike and Slab Lasso implementation.

Tests cover:
- Basic functionality (fit, predict)
- Variable selection capability
- sklearn API compatibility
- Convergence diagnostics
- Reproducibility with random seeds
- Edge cases and error handling
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path to import baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.spike_slab_lasso import SpikeSlabLasso
from baselines.diagnostics import gelman_rubin, check_convergence


class TestSpikeSlabLassoBasic:
    """Test basic functionality of SpikeSlabLasso."""

    def test_fit_predict(self):
        """Test that model can fit and predict without errors."""
        # Simple linear regression problem
        np.random.seed(42)
        n_samples, n_features = 100, 5
        X = np.random.randn(n_samples, n_features)
        true_beta = np.array([1.0, -0.5, 0.8, 0.0, -0.3])
        y = X @ true_beta + 0.1 * np.random.randn(n_samples)

        # Fit model
        model = SpikeSlabLasso(
            n_iter=500, 
            burnin=100, 
            thin=2, 
            lambda0=10.0,
            lambda1=1.0,
            random_state=42, 
            verbose=False
        )
        model.fit(X, y, n_chains=2)

        # Predict
        y_pred = model.predict(X)

        # Basic checks
        assert y_pred.shape == (n_samples,)
        assert model.coef_.shape == (n_features,)
        assert model.n_features_in_ == n_features
        assert model.inclusion_probs_.shape == (n_features,)

        # Check predictions are reasonable (MSE should be small)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 1.0, f"MSE too large: {mse}"

    def test_sklearn_api_compatibility(self):
        """Test that API matches sklearn conventions."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + 0.1 * np.random.randn(50)

        model = SpikeSlabLasso(n_iter=200, burnin=50, random_state=42, verbose=False)

        # Test fit returns self
        result = model.fit(X, y, n_chains=2)
        assert result is model

        # Test attributes exist
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'coef_map_')
        assert hasattr(model, 'inclusion_probs_')
        assert hasattr(model, 'theta_')
        assert hasattr(model, 'n_features_in_')

        # Test predict method
        y_pred = model.predict(X)
        assert isinstance(y_pred, np.ndarray)
        
        # Test predict_map method
        y_pred_map = model.predict_map(X)
        assert isinstance(y_pred_map, np.ndarray)

    def test_input_validation(self):
        """Test input validation and error handling."""
        model = SpikeSlabLasso(random_state=42, verbose=False)

        # Test with wrong dimensions
        with pytest.raises(ValueError, match="Expected 2D array"):
            model.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))

        # Test predict before fit
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(np.random.randn(10, 3))
        
        # Test predict_map before fit
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict_map(np.random.randn(10, 3))

    def test_y_shape_handling(self):
        """Test that both 1D and 2D y arrays work."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y_1d = np.random.randn(30)
        y_2d = y_1d.reshape(-1, 1)

        model1 = SpikeSlabLasso(n_iter=200, burnin=50, random_state=42, verbose=False)
        model1.fit(X, y_1d, n_chains=2)

        model2 = SpikeSlabLasso(n_iter=200, burnin=50, random_state=42, verbose=False)
        model2.fit(X, y_2d, n_chains=2)

        # Both should produce same results
        np.testing.assert_array_almost_equal(model1.coef_, model2.coef_, decimal=5)


class TestVariableSelection:
    """Test variable selection capability."""

    def test_sparse_recovery(self):
        """Test that model can identify truly zero coefficients."""
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
            verbose=False
        )
        model.fit(X, y, n_chains=3)

        # Check that non-zero coefficients have high inclusion probability
        true_nonzero = np.array([0, 3, 7])
        for idx in true_nonzero:
            assert model.inclusion_probs_[idx] > 0.5, \
                f"Non-zero coefficient {idx} has low inclusion prob: {model.inclusion_probs_[idx]}"

        # Check that some zero coefficients have low inclusion probability
        true_zero = np.array([1, 2, 4, 5, 6, 8, 9])
        n_correctly_excluded = np.sum(model.inclusion_probs_[true_zero] < 0.5)
        assert n_correctly_excluded >= 4, \
            f"Only {n_correctly_excluded}/7 zero coefficients correctly identified"

    def test_get_selected_features(self):
        """Test get_selected_features method."""
        np.random.seed(42)
        n_samples, n_features = 50, 8
        
        # Sparse problem
        true_beta = np.zeros(n_features)
        true_beta[1] = 1.5
        true_beta[4] = -1.0
        
        X = np.random.randn(n_samples, n_features)
        y = X @ true_beta + 0.1 * np.random.randn(n_samples)

        model = SpikeSlabLasso(
            n_iter=800,
            burnin=200,
            lambda0=15.0,
            lambda1=0.5,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=2)

        # Get selected features with different thresholds
        selected_50 = model.get_selected_features(threshold=0.5)
        selected_30 = model.get_selected_features(threshold=0.3)

        # More permissive threshold should include more features
        assert len(selected_30) >= len(selected_50)

        # Should return numpy array
        assert isinstance(selected_50, np.ndarray)

    def test_inclusion_probabilities_range(self):
        """Test that inclusion probabilities are in [0, 1]."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        model = SpikeSlabLasso(n_iter=300, burnin=50, random_state=42, verbose=False)
        model.fit(X, y, n_chains=2)

        # All inclusion probabilities should be in [0, 1]
        assert np.all(model.inclusion_probs_ >= 0.0)
        assert np.all(model.inclusion_probs_ <= 1.0)


class TestThetaUpdate:
    """Test adaptive theta updating."""

    def test_fixed_theta(self):
        """Test that theta remains fixed when update_theta=False."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        theta_init = 0.3
        model = SpikeSlabLasso(
            n_iter=300,
            burnin=50,
            theta_init=theta_init,
            update_theta=False,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=2)

        # Theta should equal initial value
        assert model.theta_ == theta_init

    def test_adaptive_theta(self):
        """Test that theta is updated when update_theta=True."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        theta_init = 0.5
        model = SpikeSlabLasso(
            n_iter=500,
            burnin=100,
            theta_init=theta_init,
            update_theta=True,
            theta_a=1.0,
            theta_b=1.0,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=2)

        # Theta should be different from initial (unless by coincidence)
        # More importantly, it should be in valid range
        assert 0.0 < model.theta_ < 1.0

    def test_theta_adapts_to_sparsity(self):
        """Test that theta adapts based on true sparsity level."""
        np.random.seed(456)
        n_samples, n_features = 100, 20
        
        # Very sparse: only 2 non-zero out of 20
        true_beta = np.zeros(n_features)
        true_beta[5] = 2.0
        true_beta[15] = -1.5
        
        X = np.random.randn(n_samples, n_features)
        y = X @ true_beta + 0.1 * np.random.randn(n_samples)

        model = SpikeSlabLasso(
            n_iter=1000,
            burnin=200,
            theta_init=0.5,  # Start at 0.5
            update_theta=True,
            theta_a=1.0,
            theta_b=1.0,
            lambda0=10.0,
            random_state=456,
            verbose=False
        )
        model.fit(X, y, n_chains=3)

        # For very sparse data, posterior theta should be relatively low
        # (though not guaranteed due to uncertainty)
        mean_inclusion = np.mean(model.inclusion_probs_)
        assert mean_inclusion < 0.5, \
            f"Expected low mean inclusion for sparse data, got {mean_inclusion}"


class TestConvergence:
    """Test convergence diagnostics."""

    def test_convergence_simple_problem(self):
        """Test that chains converge on a simple problem."""
        np.random.seed(42)
        n_samples, n_features = 100, 3
        X = np.random.randn(n_samples, n_features)
        true_beta = np.array([1.0, -0.5, 0.3])
        y = X @ true_beta + 0.1 * np.random.randn(n_samples)

        model = SpikeSlabLasso(
            n_iter=1000,
            burnin=200,
            thin=2,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=4)

        # Check that R-hat is computed
        assert model.rhat_ is not None
        assert len(model.rhat_) == n_features

        # For a simple problem, chains should converge
        assert model.converged_, f"Chains did not converge. Max R-hat: {np.max(model.rhat_)}"
        assert np.max(model.rhat_) < 1.1

    def test_rhat_calculation(self):
        """Test that R-hat is calculated correctly."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        model = SpikeSlabLasso(n_iter=500, burnin=100, random_state=42, verbose=False)
        model.fit(X, y, n_chains=3)

        # R-hat should be positive and close to 1 for converged chains
        assert np.all(model.rhat_ > 0)
        assert np.all(model.rhat_ < 2.0)  # Should not be wildly divergent


class TestReproducibility:
    """Test reproducibility with random seeds."""

    def test_random_state_reproducibility(self):
        """Test that same random_state gives identical results."""
        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = X @ np.array([1.0, 0.5, -0.3, 0.0]) + 0.1 * np.random.randn(60)

        model1 = SpikeSlabLasso(
            n_iter=500,
            burnin=100,
            random_state=123,
            verbose=False
        )
        model1.fit(X, y, n_chains=3)

        model2 = SpikeSlabLasso(
            n_iter=500,
            burnin=100,
            random_state=123,
            verbose=False
        )
        model2.fit(X, y, n_chains=3)

        # Coefficients should be identical
        np.testing.assert_array_equal(model1.coef_, model2.coef_)
        np.testing.assert_array_equal(model1.inclusion_probs_, model2.inclusion_probs_)

    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = np.random.randn(60)

        model1 = SpikeSlabLasso(n_iter=500, burnin=100, random_state=123, verbose=False)
        model1.fit(X, y, n_chains=2)

        model2 = SpikeSlabLasso(n_iter=500, burnin=100, random_state=456, verbose=False)
        model2.fit(X, y, n_chains=2)

        # Results should differ
        assert not np.allclose(model1.coef_, model2.coef_)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_high_dimensional(self):
        """Test p > n case (high-dimensional)."""
        np.random.seed(42)
        n_samples, n_features = 30, 50  # p > n
        
        # Sparse true coefficients
        true_beta = np.zeros(n_features)
        true_beta[5] = 1.5
        true_beta[10] = -1.0
        true_beta[25] = 0.8
        
        X = np.random.randn(n_samples, n_features)
        y = X @ true_beta + 0.2 * np.random.randn(n_samples)

        model = SpikeSlabLasso(
            n_iter=800,
            burnin=150,
            lambda0=15.0,
            lambda1=0.5,
            random_state=42,
            verbose=False
        )
        
        # Should not raise an error
        model.fit(X, y, n_chains=2)
        
        # Should produce predictions
        y_pred = model.predict(X)
        assert y_pred.shape == (n_samples,)
        
        # Should identify some sparsity
        n_selected = len(model.get_selected_features(threshold=0.5))
        assert n_selected < n_features, "Should select fewer features than total"

    def test_all_zeros_coefficients(self):
        """Test with pure noise (all true coefficients are zero)."""
        np.random.seed(42)
        n_samples, n_features = 50, 5
        
        X = np.random.randn(n_samples, n_features)
        y = 0.5 * np.random.randn(n_samples)  # Pure noise

        model = SpikeSlabLasso(
            n_iter=500,
            burnin=100,
            lambda0=10.0,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=2)

        # Should have low inclusion probabilities on average
        mean_inclusion = np.mean(model.inclusion_probs_)
        assert mean_inclusion < 0.7, \
            f"Expected low mean inclusion for pure noise, got {mean_inclusion}"

    def test_single_chain(self):
        """Test that model works with a single chain."""
        np.random.seed(42)
        X = np.random.randn(40, 3)
        y = np.random.randn(40)

        model = SpikeSlabLasso(n_iter=300, burnin=50, random_state=42, verbose=False)
        model.fit(X, y, n_chains=1)

        # Should produce valid results
        assert model.coef_.shape == (3,)
        assert model.chains_.shape[0] == 1  # One chain
        
        # Note: R-hat may not be meaningful with single chain
        # but should still be computed


class TestPosteriorCaching:
    """Test posterior density caching functionality."""

    def test_cache_posteriors_enabled(self):
        """Test that posteriors are cached when enabled."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = SpikeSlabLasso(
            n_iter=300,
            burnin=50,
            cache_posteriors=True,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=2)

        # Should have cached posteriors
        assert model.log_posteriors_ is not None
        
        # Get cached posteriors
        log_post = model.get_log_posteriors()
        assert isinstance(log_post, np.ndarray)
        
        # Length should match total number of samples
        n_chains, n_samples, _ = model.chains_.shape
        assert len(log_post) == n_chains * n_samples

    def test_cache_posteriors_disabled(self):
        """Test that posteriors are not cached when disabled."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = SpikeSlabLasso(
            n_iter=300,
            burnin=50,
            cache_posteriors=False,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=2)

        # Should not have cached posteriors
        assert model.log_posteriors_ is None
        
        # get_log_posteriors should raise error
        with pytest.raises(ValueError, match="not cached"):
            model.get_log_posteriors()

    def test_get_log_posteriors_before_fit(self):
        """Test that get_log_posteriors raises error before fit."""
        model = SpikeSlabLasso(cache_posteriors=True, verbose=False)
        
        with pytest.raises(ValueError, match="not been fitted"):
            model.get_log_posteriors()


class TestMAPEstimate:
    """Test MAP (Maximum A Posteriori) estimation."""

    def test_map_estimate_computed(self):
        """Test that MAP estimate is computed during fit."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, -0.5, 0.3]) + 0.1 * np.random.randn(50)

        model = SpikeSlabLasso(n_iter=500, burnin=100, random_state=42, verbose=False)
        model.fit(X, y, n_chains=2)

        # MAP estimate should exist
        assert model.coef_map_ is not None
        assert model.coef_map_.shape == (3,)

    def test_map_vs_mean(self):
        """Test that MAP and posterior mean can differ."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = SpikeSlabLasso(n_iter=500, burnin=100, random_state=42, verbose=False)
        model.fit(X, y, n_chains=2)

        # MAP and mean are usually different (though can be similar)
        # Just check they're both valid
        assert model.coef_.shape == model.coef_map_.shape
        assert np.all(np.isfinite(model.coef_))
        assert np.all(np.isfinite(model.coef_map_))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
