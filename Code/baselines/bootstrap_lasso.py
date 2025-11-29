"""
Bootstrap Lasso for uncertainty quantification in policy evaluation.

This module implements a bootstrap-based Lasso regression that resamples
training data to generate an empirical distribution of treatment effects.
"""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BootstrapLasso:
    """
    Bootstrap Lasso regression for uncertainty quantification.

    This estimator fits Lasso models on bootstrap resamples of the training
    data to generate an empirical distribution of coefficients and predictions.
    The final predictions use aggregated coefficients (mean across bootstrap samples).

    Parameters
    ----------
    n_bootstrap : int, default=1000
        Number of bootstrap iterations to perform.

    alpha : float, default=1.0
        Regularization strength for Lasso. Must be a positive float.
        Larger values specify stronger regularization.

    confidence_level : float, default=0.95
        Confidence level for computing coefficient confidence intervals.
        Must be between 0 and 1.

    fit_intercept : bool, default=False
        Whether to fit an intercept term in the model.

    random_state : int or None, default=None
        Random seed for reproducibility of bootstrap sampling.

    verbose : bool, default=False
        Whether to print progress during fitting.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Mean of bootstrap coefficient estimates. Used for predictions.

    bootstrap_coefs_ : ndarray of shape (n_bootstrap, n_features)
        All bootstrap coefficient samples.

    coef_ci_ : ndarray of shape (n_features, 2)
        Confidence intervals for each coefficient.
        Column 0 contains lower bounds, column 1 contains upper bounds.

    coef_median_ : ndarray of shape (n_features,)
        Median of bootstrap coefficient estimates (more robust alternative).

    n_features_in_ : int
        Number of features seen during fit.

    coverage_ : float
        Proportion of bootstrap iterations where at least one coefficient
        was non-zero (model complexity indicator).

    Examples
    --------
    >>> from bootstrap_lasso import BootstrapLasso
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> bootstrap = BootstrapLasso(n_bootstrap=500, alpha=0.1, random_state=42)
    >>> bootstrap.fit(X, y)
    >>> predictions = bootstrap.predict(X)
    """

    def __init__(
        self,
        n_bootstrap=1000,
        alpha=1.0,
        confidence_level=0.95,
        fit_intercept=False,
        random_state=None,
        verbose=False
    ):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the Bootstrap Lasso model.

        Performs bootstrap resampling and fits a Lasso model on each resample
        to generate an empirical distribution of coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (design matrix).

        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Flatten y before validation to avoid warning
        if hasattr(y, 'ndim') and y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Initialize storage for bootstrap samples
        self.bootstrap_coefs_ = np.zeros((self.n_bootstrap, n_features))

        # Set up random number generator
        rng = np.random.RandomState(self.random_state)

        # Perform bootstrap iterations
        for i in range(self.n_bootstrap):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"Bootstrap iteration {i + 1}/{self.n_bootstrap}")

            # Resample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Ensure y_boot is 1D to avoid sklearn warning
            if y_boot.ndim > 1:
                y_boot = y_boot.ravel()

            # Fit Lasso on bootstrap sample
            lasso = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=None,  # Use same seed for all iterations
                max_iter=5000,
                tol=1e-4
            )
            lasso.fit(X_boot, y_boot)

            # Store coefficients
            self.bootstrap_coefs_[i] = lasso.coef_

        # Compute aggregated coefficients
        self.coef_ = np.mean(self.bootstrap_coefs_, axis=0)
        self.coef_median_ = np.median(self.bootstrap_coefs_, axis=0)

        # Compute confidence intervals using percentile method
        alpha_lower = (1 - self.confidence_level) / 2
        alpha_upper = 1 - alpha_lower
        self.coef_ci_ = np.percentile(
            self.bootstrap_coefs_,
            [alpha_lower * 100, alpha_upper * 100],
            axis=0
        ).T  # Shape: (n_features, 2)

        # Compute coverage: proportion of iterations with non-zero coefficients
        nonzero_counts = np.sum(np.abs(self.bootstrap_coefs_) > 1e-10, axis=1)
        self.coverage_ = np.mean(nonzero_counts > 0)

        return self

    def predict(self, X):
        """
        Make predictions using the fitted Bootstrap Lasso model.

        Uses the mean of bootstrap coefficient estimates for predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ['coef_', 'bootstrap_coefs_'])
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but BootstrapLasso "
                f"was fitted with {self.n_features_in_} features."
            )

        # Use mean coefficients for prediction
        return X @ self.coef_

    def predict_with_ci(self, X, confidence_level=None):
        """
        Make predictions with confidence intervals.

        Generates predictions for each bootstrap sample and computes
        confidence intervals using the percentile method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        confidence_level : float or None, default=None
            Confidence level for prediction intervals. If None, uses
            the confidence_level from initialization.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Mean predicted values across bootstrap samples.

        y_ci : ndarray of shape (n_samples, 2)
            Confidence intervals for predictions.
            Column 0 contains lower bounds, column 1 contains upper bounds.
        """
        check_is_fitted(self, ['coef_', 'bootstrap_coefs_'])
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but BootstrapLasso "
                f"was fitted with {self.n_features_in_} features."
            )

        if confidence_level is None:
            confidence_level = self.confidence_level

        # Generate predictions for all bootstrap samples
        y_bootstrap = X @ self.bootstrap_coefs_.T  # Shape: (n_samples, n_bootstrap)

        # Compute mean and confidence intervals
        y_pred = np.mean(y_bootstrap, axis=1)

        alpha_lower = (1 - confidence_level) / 2
        alpha_upper = 1 - alpha_lower
        y_ci = np.percentile(
            y_bootstrap,
            [alpha_lower * 100, alpha_upper * 100],
            axis=1
        ).T  # Shape: (n_samples, 2)

        return y_pred, y_ci

    def get_feature_importance(self, threshold=1e-10):
        """
        Get feature importance based on bootstrap stability.

        Importance is measured as the proportion of bootstrap iterations
        where the coefficient was non-zero (above threshold).

        Parameters
        ----------
        threshold : float, default=1e-10
            Minimum absolute coefficient value to consider as non-zero.

        Returns
        -------
        importance : ndarray of shape (n_features,)
            Proportion of bootstrap iterations where each feature was selected.
        """
        check_is_fitted(self, ['bootstrap_coefs_'])

        is_nonzero = np.abs(self.bootstrap_coefs_) > threshold
        importance = np.mean(is_nonzero, axis=0)

        return importance
