"""
Bayesian Lasso implementation using Gibbs sampling.

Based on Park & Casella (2008) "The Bayesian Lasso"
Journal of the American Statistical Association, 103(482), 681-686.

The model is:
    y | X, β, τ² ~ N(Xβ, τ²I)
    β_j | τ², λ_j² ~ N(0, τ²λ_j²)  [Laplace prior via scale mixture of normals]
    λ_j² ~ Exponential(λ²/2)
    τ² ~ InverseGamma(a, b)
"""

import numpy as np
import warnings

from typing import Optional

# Try to import numba for speedup, fall back if not available
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # No-op decorator if numba not available

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

try:
    from scipy.stats import invgauss
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. BayesianLasso requires scipy for inverse-Gaussian sampling.")


class BayesianLasso:
    """
    Bayesian Lasso regression with Laplace priors via Gibbs sampling.

    Parameters
    ----------
    n_iter : int, default=2000
        Number of MCMC iterations per chain
    burnin : int, default=500
        Number of burn-in iterations to discard
    thin : int, default=2
        Thinning parameter - keep every thin-th sample
    lambda_prior : float, default=1.0
        Scale parameter for the Laplace prior (regularization strength)
    tau2_a : float, default=0.1
        Shape parameter for inverse-gamma prior on τ²
    tau2_b : float, default=0.1
        Scale parameter for inverse-gamma prior on τ²
    fit_intercept : bool, default=False
        Whether to fit an intercept term
    random_state : int or None, default=None
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Posterior mean of regression coefficients
    chains_ : ndarray of shape (n_chains, n_samples, n_features)
        MCMC samples for all chains (after burn-in and thinning)
    rhat_ : ndarray of shape (n_features,)
        Gelman-Rubin convergence statistic for each coefficient
    converged_ : bool
        Whether all chains converged (max R-hat < 1.1)
    n_features_in_ : int
        Number of features seen during fit
    """

    def __init__(
        self,
        n_iter: int = 2000,
        burnin: int = 500,
        thin: int = 2,
        lambda_prior: float = 1.0,
        tau2_a: float = 0.1,
        tau2_b: float = 0.1,
        fit_intercept: bool = False,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        if not SCIPY_AVAILABLE:
            raise ImportError("BayesianLasso requires scipy. Install with: pip install scipy")

        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.lambda_prior = lambda_prior
        self.tau2_a = tau2_a
        self.tau2_b = tau2_b
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

        # Attributes set during fit
        self.coef_ = None
        self.chains_ = None
        self.rhat_ = None
        self.converged_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_chains: int = 4) -> 'BayesianLasso':
        """
        Fit Bayesian Lasso model using Gibbs sampling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,) or (n_samples, 1)
            Target values
        n_chains : int, default=4
            Number of independent chains to run for convergence diagnostics

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Input validation
        X = self._validate_data(X)
        y = self._validate_target(y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.verbose:
            print(f"Running {n_chains} chains with {self.n_iter} iterations each...")

        # Run multiple chains
        all_chains = []
        for chain_idx in range(n_chains):
            if self.verbose and chain_idx > 0:
                print(f"  Chain {chain_idx + 1}/{n_chains}")

            # Set seed for this chain
            if self.random_state is not None:
                seed = self.random_state + chain_idx
            else:
                seed = None

            chain_samples = self._run_chain(X, y, seed)
            all_chains.append(chain_samples)

        # Stack chains: (n_chains, n_samples, n_features)
        self.chains_ = np.array(all_chains)

        # Compute posterior mean across all chains
        self.coef_ = np.mean(self.chains_.reshape(-1, n_features), axis=0)

        # Compute Gelman-Rubin diagnostic
        from .diagnostics import gelman_rubin, check_convergence
        self.rhat_ = gelman_rubin(self.chains_)
        self.converged_ = check_convergence(self.chains_, threshold=1.1)

        if self.verbose:
            print(f"  Convergence: {self.converged_} (max R-hat: {np.max(self.rhat_):.4f})")
            if not self.converged_:
                warnings.warn(
                    f"Chains may not have converged. Max R-hat = {np.max(self.rhat_):.4f} > 1.1"
                )

        return self

    def _run_chain(self, X: np.ndarray, y: np.ndarray, seed: Optional[int]) -> np.ndarray:
        """Run a single Gibbs sampling chain."""
        rng = np.random.default_rng(seed)
        n_samples, n_features = X.shape

        # Pre-compute for efficiency
        XtX = X.T @ X
        Xty = X.T @ y

        # Initialize parameters
        beta = np.zeros(n_features)
        tau2 = 1.0
        lambda2 = np.ones(n_features)

        # Storage for samples (after burn-in and thinning)
        n_keep = (self.n_iter - self.burnin) // self.thin
        beta_samples = np.zeros((n_keep, n_features))
        sample_idx = 0

        # Gibbs sampling
        for iter_idx in range(self.n_iter):
            # Sample β | τ², λ², y
            beta = self._sample_beta(X, y, XtX, Xty, tau2, lambda2, rng)

            # Sample τ² | β, y
            tau2 = self._sample_tau2(y, X, beta, n_samples, n_features, rng)

            # Sample λ_j² | β_j, τ²
            lambda2 = self._sample_lambda2(beta, tau2, rng)

            # Store samples after burn-in with thinning
            if iter_idx >= self.burnin and (iter_idx - self.burnin) % self.thin == 0:
                beta_samples[sample_idx] = beta
                sample_idx += 1

        return beta_samples

    def _sample_beta(
        self,
        X: np.ndarray,
        y: np.ndarray,
        XtX: np.ndarray,
        Xty: np.ndarray,
        tau2: float,
        lambda2: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample β from its conditional posterior distribution.

        β | τ², λ², y ~ N(μ_β, Σ_β)
        where Σ_β = τ²(X'X + D_λ^(-1))^(-1)
              μ_β = Σ_β X'y / τ²
        and D_λ = diag(λ_1², ..., λ_p²)
        """
        n_features = len(lambda2)

        # Covariance matrix: Σ_β = τ²(X'X + D_λ^(-1))^(-1)
        D_lambda_inv = np.diag(1.0 / lambda2)
        A = XtX + D_lambda_inv

        # Use Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(A)
            # Solve for Σ_β
            Sigma_beta = np.linalg.solve(L @ L.T, np.eye(n_features)) * tau2
            # Mean: μ_β = Σ_β X'y / τ²
            mu_beta = Sigma_beta @ Xty / tau2
        except np.linalg.LinAlgError:
            # Fall back to regularized version if Cholesky fails
            warnings.warn("Cholesky decomposition failed, using regularized inversion")
            A_reg = A + 1e-6 * np.eye(n_features)
            Sigma_beta = np.linalg.inv(A_reg) * tau2
            mu_beta = Sigma_beta @ Xty / tau2

        # Sample from multivariate normal
        beta = rng.multivariate_normal(mu_beta, Sigma_beta)

        return beta

    def _sample_tau2(
        self,
        y: np.ndarray,
        X: np.ndarray,
        beta: np.ndarray,
        n_samples: int,
        n_features: int,
        rng: np.random.Generator
    ) -> float:
        """
        Sample τ² from its conditional posterior distribution.

        τ² | β, y ~ InverseGamma(a + n/2 + p/2, b + RSS/2 + β'D_λ^(-1)β/2)
        where RSS = ||y - Xβ||²
        """
        # Residual sum of squares
        residuals = y - X @ beta
        rss = np.sum(residuals ** 2)

        # Posterior parameters
        shape = self.tau2_a + (n_samples + n_features) / 2.0
        scale = self.tau2_b + rss / 2.0

        # Sample from inverse-gamma: sample from gamma then take reciprocal
        tau2 = 1.0 / rng.gamma(shape, 1.0 / scale)

        return tau2

    def _sample_lambda2(
        self,
        beta: np.ndarray,
        tau2: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample λ_j² from its conditional posterior distribution.

        λ_j² | β_j, τ² ~ InverseGaussian(μ = λ/|β_j|, λ = λ²)
        where λ is the prior scale parameter
        """
        n_features = len(beta)
        lambda2 = np.zeros(n_features)

        for j in range(n_features):
            # Parameters for inverse-Gaussian
            mu = self.lambda_prior / (np.abs(beta[j]) + 1e-10)  # Add small constant for stability
            shape = self.lambda_prior ** 2

            # Sample from inverse-Gaussian using scipy
            lambda2[j] = invgauss.rvs(mu / shape, scale=shape, random_state=rng)

        return lambda2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the posterior mean of coefficients.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = self._validate_data(X, reset=False)
        return X @ self.coef_

    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        if not reset and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with {self.n_features_in_} features"
            )

        return X

    def _validate_target(self, y: np.ndarray) -> np.ndarray:
        """Validate target array."""
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError(f"Expected 1D array for y, got {y.ndim}D array instead")
        return y
