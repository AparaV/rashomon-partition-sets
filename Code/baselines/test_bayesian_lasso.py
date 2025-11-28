"""
Unit tests for Bayesian Lasso implementation.

Tests cover:
- Basic functionality (fit, predict)
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

from baselines import BayesianLasso, gelman_rubin, check_convergence


class TestBayesianLassoBasic:
    """Test basic functionality of BayesianLasso."""
    
    def test_fit_predict(self):
        """Test that model can fit and predict without errors."""
        # Simple linear regression problem
        np.random.seed(42)
        n_samples, n_features = 100, 5
        X = np.random.randn(n_samples, n_features)
        true_beta = np.array([1.0, -0.5, 0.8, 0.0, -0.3])
        y = X @ true_beta + 0.1 * np.random.randn(n_samples)
        
        # Fit model
        model = BayesianLasso(n_iter=500, burnin=100, thin=2, random_state=42, verbose=False)
        model.fit(X, y, n_chains=2)
        
        # Predict
        y_pred = model.predict(X)
        
        # Basic checks
        assert y_pred.shape == (n_samples,)
        assert model.coef_.shape == (n_features,)
        assert model.n_features_in_ == n_features
        
        # Check predictions are reasonable (MSE should be small)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 1.0, f"MSE too large: {mse}"
    
    def test_sklearn_api_compatibility(self):
        """Test that API matches sklearn conventions."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + 0.1 * np.random.randn(50)
        
        model = BayesianLasso(n_iter=200, burnin=50, random_state=42, verbose=False)
        
        # Test fit returns self
        result = model.fit(X, y, n_chains=2)
        assert result is model
        
        # Test attributes exist
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'n_features_in_')
        
        # Test predict method
        y_pred = model.predict(X)
        assert isinstance(y_pred, np.ndarray)
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        model = BayesianLasso(random_state=42, verbose=False)
        
        # Test with wrong dimensions
        with pytest.raises(ValueError, match="Expected 2D array"):
            model.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))
        
        # Test predict before fit
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(np.random.randn(10, 3))
    
    def test_y_shape_handling(self):
        """Test that both 1D and 2D y arrays work."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y_1d = np.random.randn(30)
        y_2d = y_1d.reshape(-1, 1)
        
        model1 = BayesianLasso(n_iter=200, burnin=50, random_state=42, verbose=False)
        model1.fit(X, y_1d, n_chains=2)
        
        model2 = BayesianLasso(n_iter=200, burnin=50, random_state=42, verbose=False)
        model2.fit(X, y_2d, n_chains=2)
        
        # Both should produce same results
        np.testing.assert_array_almost_equal(model1.coef_, model2.coef_, decimal=5)


class TestConvergence:
    """Test convergence diagnostics."""
    
    def test_simple_convergence(self):
        """Test that chains converge on a simple problem."""
        np.random.seed(42)
        n_samples, n_features = 200, 3
        X = np.random.randn(n_samples, n_features)
        true_beta = np.array([2.0, -1.5, 1.0])
        y = X @ true_beta + 0.5 * np.random.randn(n_samples)
        
        # Fit with sufficient iterations
        model = BayesianLasso(
            n_iter=1000,
            burnin=200,
            thin=2,
            lambda_prior=0.1,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=4)
        
        # Check convergence
        assert model.converged_, f"Model did not converge. Max R-hat: {np.max(model.rhat_):.4f}"
        assert np.all(model.rhat_ < 1.1), f"Some R-hat values too high: {model.rhat_}"
        
        # Check posterior mean is close to true values
        for i, (est, true) in enumerate(zip(model.coef_, true_beta)):
            error = np.abs(est - true)
            assert error < 0.5, f"β_{i}: estimated {est:.3f}, true {true:.3f}, error {error:.3f}"
    
    def test_gelman_rubin_calculation(self):
        """Test Gelman-Rubin statistic calculation."""
        # Create fake chains with known properties
        np.random.seed(42)
        n_chains, n_samples, n_params = 4, 100, 3
        
        # Converged chains: all centered around same value
        converged_chains = np.random.randn(n_chains, n_samples, n_params) * 0.1
        for i in range(n_chains):
            converged_chains[i] += np.array([1.0, 2.0, 3.0])
        
        rhat = gelman_rubin(converged_chains)
        
        # R-hat should be close to 1 for converged chains
        assert np.all(rhat < 1.05), f"R-hat too high for converged chains: {rhat}"
        
        # Check convergence function
        assert check_convergence(converged_chains, threshold=1.1)
    
    def test_gelman_rubin_diverged_chains(self):
        """Test that G-R detects non-converged chains."""
        np.random.seed(42)
        n_chains, n_samples, n_params = 4, 100, 2
        
        # Diverged chains: each centered at different values
        diverged_chains = np.zeros((n_chains, n_samples, n_params))
        for i in range(n_chains):
            diverged_chains[i] = np.random.randn(n_samples, n_params) * 0.1 + i * 2
        
        rhat = gelman_rubin(diverged_chains)
        
        # R-hat should be much larger than 1
        assert np.all(rhat > 1.1), f"R-hat should indicate non-convergence: {rhat}"
        
        # Check convergence function
        assert not check_convergence(diverged_chains, threshold=1.1)


class TestReproducibility:
    """Test reproducibility with random seeds."""
    
    def test_random_state_reproducibility(self):
        """Test that same random_state gives same results."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = X @ np.array([1.0, -1.0, 0.5, 0.0]) + 0.2 * np.random.randn(50)
        
        # Fit two models with same random state
        model1 = BayesianLasso(n_iter=300, burnin=50, random_state=123, verbose=False)
        model1.fit(X, y, n_chains=2)
        
        model2 = BayesianLasso(n_iter=300, burnin=50, random_state=123, verbose=False)
        model2.fit(X, y, n_chains=2)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(model1.coef_, model2.coef_, decimal=10)
        np.testing.assert_array_almost_equal(model1.chains_, model2.chains_, decimal=10)
    
    def test_different_seeds_different_results(self):
        """Test that different random_state gives different results."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = X @ np.array([1.0, -1.0, 0.5, 0.0]) + 0.2 * np.random.randn(50)
        
        model1 = BayesianLasso(n_iter=300, burnin=50, random_state=123, verbose=False)
        model1.fit(X, y, n_chains=2)
        
        model2 = BayesianLasso(n_iter=300, burnin=50, random_state=456, verbose=False)
        model2.fit(X, y, n_chains=2)
        
        # Results should be different (but both reasonable)
        assert not np.allclose(model1.coef_, model2.coef_)


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_orthogonal_design(self):
        """Test with orthogonal design matrix (X'X = I)."""
        np.random.seed(42)
        n_samples, n_features = 100, 5
        
        # Create orthogonal design via QR decomposition
        X_raw = np.random.randn(n_samples, n_features)
        X, _ = np.linalg.qr(X_raw)
        X = X * np.sqrt(n_samples)  # Scale so X'X ≈ I
        
        true_beta = np.array([1.0, -1.0, 0.5, -0.5, 0.8])
        y = X @ true_beta + 0.1 * np.random.randn(n_samples)
        
        model = BayesianLasso(
            n_iter=800,
            burnin=200,
            lambda_prior=0.1,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=3)
        
        # Should recover true coefficients well
        mse = np.mean((model.coef_ - true_beta) ** 2)
        assert mse < 0.1, f"MSE too high for orthogonal design: {mse}"
    
    def test_high_dimensional(self):
        """Test with p > n (more features than samples)."""
        np.random.seed(42)
        n_samples, n_features = 30, 50
        X = np.random.randn(n_samples, n_features)
        
        # Only first 5 features are active
        true_beta = np.zeros(n_features)
        true_beta[:5] = [2.0, -1.5, 1.0, -0.5, 0.8]
        y = X @ true_beta + 0.5 * np.random.randn(n_samples)
        
        model = BayesianLasso(
            n_iter=500,
            burnin=100,
            lambda_prior=1.0,  # Stronger regularization
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=2)
        
        # Model should run without error
        assert model.coef_.shape == (n_features,)
        
        # Should identify some of the active features
        predicted = model.predict(X)
        mse = np.mean((y - predicted) ** 2)
        assert mse < 2.0, f"MSE too high: {mse}"
    
    def test_sparse_solution(self):
        """Test that strong regularization produces sparse solutions."""
        np.random.seed(42)
        n_samples, n_features = 100, 10
        X = np.random.randn(n_samples, n_features)
        
        # Sparse true coefficients
        true_beta = np.array([2.0, 0, 0, -1.5, 0, 0, 0, 1.0, 0, 0])
        y = X @ true_beta + 0.3 * np.random.randn(n_samples)
        
        model = BayesianLasso(
            n_iter=800,
            burnin=200,
            lambda_prior=2.0,  # Strong regularization
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=2)
        
        # Count near-zero coefficients
        n_near_zero = np.sum(np.abs(model.coef_) < 0.2)
        assert n_near_zero >= 5, f"Expected sparse solution, got {n_near_zero} near-zero coefficients"


class TestChainStorage:
    """Test chain storage and retrieval."""
    
    def test_chains_shape(self):
        """Test that stored chains have correct shape."""
        np.random.seed(42)
        X = np.random.randn(40, 3)
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.2 * np.random.randn(40)
        
        n_iter = 400
        burnin = 100
        thin = 2
        n_chains = 3
        
        model = BayesianLasso(
            n_iter=n_iter,
            burnin=burnin,
            thin=thin,
            random_state=42,
            verbose=False
        )
        model.fit(X, y, n_chains=n_chains)
        
        # Calculate expected number of samples
        n_samples = (n_iter - burnin) // thin
        
        # Check shape
        assert model.chains_.shape == (n_chains, n_samples, 3)
    
    def test_posterior_mean_calculation(self):
        """Test that coef_ is computed as posterior mean."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = X @ np.array([1.0, -1.0, 0.5, 0.0]) + 0.2 * np.random.randn(50)
        
        model = BayesianLasso(n_iter=300, burnin=50, thin=1, random_state=42, verbose=False)
        model.fit(X, y, n_chains=2)
        
        # Manually compute posterior mean
        manual_mean = np.mean(model.chains_.reshape(-1, 4), axis=0)
        
        # Should match model.coef_
        np.testing.assert_array_almost_equal(model.coef_, manual_mean, decimal=10)


def run_all_tests():
    """Run all tests and report results."""
    print("Running BayesianLasso tests...\n")
    
    # Run pytest programmatically
    exit_code = pytest.main([__file__, '-v', '--tb=short'])
    
    return exit_code


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
