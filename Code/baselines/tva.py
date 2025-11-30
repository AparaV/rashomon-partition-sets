"""
TVA (Treatment Variance Analysis) using Puffer Transform.

This module implements the Puffer transform for treatment effect estimation
with adaptive regularization.
"""

import numpy as np
from sklearn.linear_model import Lasso


class TVA:
    """
    Treatment Variance Analysis using Puffer Transform.

    The Puffer transform decorrelates the design matrix to improve Lasso
    estimation, particularly useful for treatment effect estimation with
    sample size adaptive regularization.

    Parameters
    ----------
    alpha : float, default=1e-3
        Regularization parameter for Lasso.
    gamma : float, default=-1
        Scaling exponent for sample-size adaptive regularization.
        Effective alpha = alpha * n^gamma.
    fit_intercept : bool, default=False
        Whether to fit an intercept term.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    coef_ : np.ndarray
        Estimated coefficients in original space.
    U_ : np.ndarray
        Left singular vectors from SVD.
    S_ : np.ndarray
        Singular values from SVD.
    Vh_ : np.ndarray
        Right singular vectors from SVD.
    F_ : np.ndarray
        Puffer transform matrix.
    F_inv_ : np.ndarray
        Inverse of Puffer transform matrix.
    lasso_ : Lasso
        Fitted Lasso model in transformed space.
    """

    def __init__(self, alpha=1e-3, gamma=-1, fit_intercept=False, random_state=None):
        self.alpha = alpha
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _puffer_transform(self, X, y):
        """
        Apply Puffer transform to decorrelate design matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix
        y : np.ndarray, shape (n_samples, 1) or (n_samples,)
            Target values

        Returns
        -------
        X_transformed : np.ndarray
            Transformed design matrix
        y_transformed : np.ndarray
            Transformed target values
        """
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # SVD decomposition
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        S_inv = np.diag(1 / S)
        S_mat = np.diag(S)

        # Compute Puffer transform matrix F
        # F = U @ S_inv @ U.T
        F = np.matmul(S_inv, U.T)
        F = np.matmul(U, F)

        # Compute inverse transform matrix
        # F_inv = U @ S @ U.T
        F_inv = np.matmul(U, S_mat)
        F_inv = np.matmul(F_inv, U.T)

        # Apply transforms
        y_transformed = np.matmul(F, y)
        X_transformed = np.matmul(U, Vh)

        # Store for later use
        self.U_ = U
        self.S_ = S
        self.Vh_ = Vh
        self.F_ = F
        self.F_inv_ = F_inv

        return X_transformed, y_transformed

    def fit(self, X, y):
        """
        Fit TVA model using Puffer transform and Lasso.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix
        y : np.ndarray, shape (n_samples, 1) or (n_samples,)
            Target values

        Returns
        -------
        self
        """
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]

        # Apply Puffer transform
        X_transformed, y_transformed = self._puffer_transform(X, y)

        # Compute adaptive regularization
        scaling = n_samples ** self.gamma
        effective_alpha = self.alpha * scaling

        # Fit Lasso in transformed space
        self.lasso_ = Lasso(
            alpha=effective_alpha,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state
        )
        self.lasso_.fit(X_transformed, y_transformed.ravel())

        # Store coefficients (same in original and transformed space)
        self.coef_ = self.lasso_.coef_

        return self

    def predict(self, X):
        """
        Predict using fitted TVA model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values in original space
        """
        # Predictions in original space: y_pred = X @ coef
        predictions = np.dot(X, self.coef_)
        return predictions.ravel()

    def score(self, X, y):
        """
        Compute R^2 score.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix
        y : np.ndarray, shape (n_samples,) or (n_samples, 1)
            True values

        Returns
        -------
        score : float
            R^2 score
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        if y.ndim > 1:
            y = y.ravel()
        return r2_score(y, y_pred)
