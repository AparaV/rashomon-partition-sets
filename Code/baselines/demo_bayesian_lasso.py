"""
Simple demonstration of the BayesianLasso implementation.

This script shows basic usage and validates the implementation works correctly.
"""

import numpy as np

from baselines import BayesianLasso, plot_traces, plot_posterior_hist


def demo_basic_usage():
    """Demonstrate basic usage of BayesianLasso."""
    print("=" * 70)
    print("Bayesian Lasso Demo: Basic Usage")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)

    # True coefficients (sparse)
    true_beta = np.array([2.0, -1.5, 0.0, 1.0, 0.0])
    print(f"\nTrue coefficients: {true_beta}")

    # Generate response with noise
    noise_std = 0.5
    y = X @ true_beta + noise_std * np.random.randn(n_samples)

    # Fit Bayesian Lasso
    print("\nFitting Bayesian Lasso...")
    model = BayesianLasso(
        n_iter=1500,
        burnin=300,
        thin=2,
        lambda_prior=1.0,
        random_state=42,
        verbose=True
    )
    model.fit(X, y, n_chains=4)

    # Display results
    print(f"\nEstimated coefficients: {model.coef_}")
    print(f"Estimation error: {np.linalg.norm(model.coef_ - true_beta):.4f}")
    print(f"\nConvergence status: {model.converged_}")
    print(f"R-hat values: {model.rhat_}")
    print(f"Max R-hat: {np.max(model.rhat_):.4f}")

    # Predictions
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    print(f"\nMean Squared Error: {mse:.4f}")
    print(f"True noise std: {noise_std:.4f}")

    return model, X, y, true_beta


def demo_visualization(model, true_beta):
    """Demonstrate visualization utilities."""
    print("\n" + "=" * 70)
    print("Generating Diagnostic Plots")
    print("=" * 70)

    # Create output directory for plots
    import os
    plot_dir = "../Figures/blasso_demo"
    os.makedirs(plot_dir, exist_ok=True)

    # Trace plots
    print("\nGenerating trace plots...")
    param_names = [f'β_{i}' for i in range(len(true_beta))]
    plot_traces(
        model.chains_,
        param_names=param_names,
        save_path=f"{plot_dir}/traces.png"
    )
    print(f"  Saved to {plot_dir}/traces.png")

    # Posterior histograms for each parameter
    print("\nGenerating posterior histograms...")
    for i in range(len(true_beta)):
        plot_posterior_hist(
            model.chains_,
            param_idx=i,
            param_name=param_names[i],
            true_value=true_beta[i],
            save_path=f"{plot_dir}/posterior_beta_{i}.png"
        )
    print(f"  Saved {len(true_beta)} histograms to {plot_dir}/")

    print("\nVisualization complete!")


def demo_comparison_with_ols():
    """Compare Bayesian Lasso with OLS."""
    print("\n" + "=" * 70)
    print("Comparison: Bayesian Lasso vs OLS")
    print("=" * 70)

    np.random.seed(123)
    n_samples, n_features = 50, 10
    X = np.random.randn(n_samples, n_features)

    # Sparse true coefficients
    true_beta = np.zeros(n_features)
    true_beta[[0, 2, 5, 8]] = [3.0, -2.0, 1.5, -1.0]
    print(f"\nTrue coefficients (only 4 non-zero): {true_beta}")

    y = X @ true_beta + 0.3 * np.random.randn(n_samples)

    # Fit Bayesian Lasso
    print("\nFitting Bayesian Lasso (strong regularization)...")
    blasso = BayesianLasso(
        n_iter=1000,
        burnin=200,
        lambda_prior=2.0,
        random_state=123,
        verbose=False
    )
    blasso.fit(X, y, n_chains=3)

    # Fit OLS for comparison
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X, y)

    print("\nResults:")
    print(f"{'Feature':<10} {'True':<10} {'BLasso':<10} {'OLS':<10}")
    print("-" * 40)
    for i in range(n_features):
        print(f"β_{i:<8} {true_beta[i]:>8.3f} {blasso.coef_[i]:>10.3f} {ols.coef_[i]:>10.3f}")

    # Count near-zero coefficients
    blasso_sparse = np.sum(np.abs(blasso.coef_) < 0.3)
    ols_sparse = np.sum(np.abs(ols.coef_) < 0.3)
    print("\nNear-zero coefficients:")
    print(f"  Bayesian Lasso: {blasso_sparse}/10")
    print(f"  OLS: {ols_sparse}/10")
    print("  (True model has 6 zero coefficients)")

    # MSE comparison
    blasso_mse = np.mean((y - blasso.predict(X)) ** 2)
    ols_mse = np.mean((y - ols.predict(X)) ** 2)
    print("\nMean Squared Error:")
    print(f"  Bayesian Lasso: {blasso_mse:.4f}")
    print(f"  OLS: {ols_mse:.4f}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("BAYESIAN LASSO IMPLEMENTATION DEMO")
    print("=" * 70)

    # Run demos
    model, X, y, true_beta = demo_basic_usage()

    try:
        demo_visualization(model, true_beta)
    except Exception as e:
        print(f"\nVisualization skipped (plotting error): {e}")

    demo_comparison_with_ols()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
