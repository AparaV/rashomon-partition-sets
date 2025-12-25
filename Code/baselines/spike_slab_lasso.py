"""
Spike and Slab Lasso implementation using Gibbs sampling.

Based on Ročková & George (2018) "The Spike-and-Slab LASSO"
Journal of the American Statistical Association, 113(521), 431-444.

The model is:
    y | X, β, τ² ~ N(Xβ, τ²I)
    β_j | γ_j, τ² ~ γ_j·N(0, τ²/λ₀²) + (1-γ_j)·Laplace(0, λ₁)  [spike-slab mixture]
    γ_j ~ Bernoulli(θ)  [inclusion indicator]
    θ ~ Beta(a_θ, b_θ)  [adaptive prior inclusion probability]
    τ² ~ InverseGamma(a, b)

The spike component (γ_j=1) uses a tight normal prior (λ₀ large → strong shrinkage near zero).
The slab component (γ_j=0) uses a Laplace prior for non-zero coefficients.
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


class SpikeSlabLasso:
    """
    Spike and Slab Lasso regression with variable selection via Gibbs sampling.

    Parameters
    ----------
    n_iter : int, default=2000
        Number of MCMC iterations per chain
    burnin : int, default=500
        Number of burn-in iterations to discard
    thin : int, default=2
        Thinning parameter - keep every thin-th sample
    lambda0 : float, default=10.0
        Spike precision parameter (large values → strong shrinkage near zero)
    lambda1 : float, default=1.0
        Slab scale parameter for Laplace prior (regularization for non-zero coefficients)
    theta_init : float, default=0.5
        Initial prior inclusion probability P(γ_j = 1)
    update_theta : bool, default=True
        Whether to update theta from data using Beta(a_θ, b_θ) prior
    theta_a : float, default=1.0
        Shape parameter a for Beta prior on θ (used if update_theta=True)
    theta_b : float, default=1.0
        Shape parameter b for Beta prior on θ (used if update_theta=True)
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
    cache_posteriors : bool, default=True
        Whether to cache log posterior densities during fit for reuse

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Posterior mean of regression coefficients
    coef_map_ : ndarray of shape (n_features,)
        MAP (Maximum A Posteriori) estimate of regression coefficients
    inclusion_probs_ : ndarray of shape (n_features,)
        Posterior inclusion probabilities P(γ_j = 1 | data) for each feature
    theta_ : float
        Posterior mean of prior inclusion probability (if update_theta=True)
    chains_ : ndarray of shape (n_chains, n_samples, n_features)
        MCMC samples for all chains (after burn-in and thinning)
    gamma_chains_ : ndarray of shape (n_chains, n_samples, n_features)
        MCMC samples of inclusion indicators for all chains
    rhat_ : ndarray of shape (n_features,)
        Gelman-Rubin convergence statistic for each coefficient
    converged_ : bool
        Whether all chains converged (max R-hat < 1.1)
    n_features_in_ : int
        Number of features seen during fit
    log_posteriors_ : ndarray of shape (n_samples,) or None
        Cached log posterior densities for all MCMC samples (if cache_posteriors=True)
    """

    def __init__(
        self,
        n_iter: int = 2000,
        burnin: int = 500,
        thin: int = 2,
        lambda0: float = 10.0,
        lambda1: float = 1.0,
        theta_init: float = 0.5,
        update_theta: bool = True,
        theta_a: float = 1.0,
        theta_b: float = 1.0,
        tau2_a: float = 0.1,
        tau2_b: float = 0.1,
        fit_intercept: bool = False,
        random_state: Optional[int] = None,
        verbose: bool = False,
        cache_posteriors: bool = True
    ):
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.theta_init = theta_init
        self.update_theta = update_theta
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.tau2_a = tau2_a
        self.tau2_b = tau2_b
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.cache_posteriors = cache_posteriors
        self._A_workspace = None  # Workspace matrix for optimization

        # Attributes set during fit
        self.coef_ = None
        self.coef_map_ = None
        self.inclusion_probs_ = None
        self.theta_ = None
        self.chains_ = None
        self.gamma_chains_ = None
        self.rhat_ = None
        self.converged_ = None
        self.n_features_in_ = None
        self.log_posteriors_ = None
        self._X_fit = None  # Reference to training data for cache validation
        self._y_fit = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_chains: int = 4) -> 'SpikeSlabLasso':
        """
        Fit Spike and Slab Lasso model using Gibbs sampling.

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

        # Pre-allocate workspace matrix (optimization)
        self._A_workspace = np.empty((n_features, n_features))

        if self.verbose:
            print(f"Running {n_chains} chains with {self.n_iter} iterations each...")

        # Run multiple chains
        all_chains = []
        all_gamma_chains = []
        all_theta_samples = []
        
        for chain_idx in range(n_chains):
            if self.verbose and chain_idx > 0:
                print(f"  Chain {chain_idx + 1}/{n_chains}")

            # Set seed for this chain
            if self.random_state is not None:
                seed = self.random_state + chain_idx
            else:
                seed = None

            chain_samples, gamma_samples, theta_samples = self._run_chain(X, y, seed)
            all_chains.append(chain_samples)
            all_gamma_chains.append(gamma_samples)
            all_theta_samples.append(theta_samples)

        # Stack chains: (n_chains, n_samples, n_features)
        self.chains_ = np.array(all_chains)
        self.gamma_chains_ = np.array(all_gamma_chains)

        # Compute posterior mean across all chains
        self.coef_ = np.mean(self.chains_.reshape(-1, n_features), axis=0)
        
        # Compute posterior inclusion probabilities
        self.inclusion_probs_ = np.mean(self.gamma_chains_.reshape(-1, n_features), axis=0)
        
        # Compute posterior mean of theta (if updated)
        if self.update_theta:
            all_theta = np.concatenate(all_theta_samples)
            self.theta_ = np.mean(all_theta)
        else:
            self.theta_ = self.theta_init
        
        # Compute MAP estimate (coefficient vector with highest posterior density)
        all_samples = self.chains_.reshape(-1, n_features)
        posterior_densities = self._compute_posterior_densities(X, y, all_samples)
        map_idx = np.argmax(posterior_densities)
        self.coef_map_ = all_samples[map_idx]
        
        # Cache posterior densities and training data references if enabled
        if self.cache_posteriors:
            self.log_posteriors_ = posterior_densities
            self._X_fit = X
            self._y_fit = y

        # Compute Gelman-Rubin diagnostic
        from .diagnostics import gelman_rubin, check_convergence
        if self.chains_.shape[0] < 2:
            # warnings.warn(
            #     "Gelman-Rubin diagnostic requires at least 2 chains. "
            #     "Skipping convergence check."
            # )
            self.rhat_ = np.array([np.nan] * n_features)
            self.converged_ = False
        else:
            self.rhat_ = gelman_rubin(self.chains_)
            self.converged_ = check_convergence(self.chains_, threshold=1.1)

        if self.verbose:
            print(f"  Convergence: {self.converged_} (max R-hat: {np.max(self.rhat_):.4f})")
            print(f"  Mean inclusion probability: {np.mean(self.inclusion_probs_):.4f}")
            print(f"  Posterior theta: {self.theta_:.4f}")
            if not self.converged_:
                warnings.warn(
                    f"Chains may not have converged. Max R-hat = {np.max(self.rhat_):.4f} > 1.1"
                )

        return self

    def _run_chain(self, X: np.ndarray, y: np.ndarray, seed: Optional[int]) -> tuple:
        """Run a single Gibbs sampling chain."""
        rng = np.random.default_rng(seed)
        n_samples, n_features = X.shape

        # Pre-compute for efficiency
        XtX = X.T @ X
        Xty = X.T @ y

        # Initialize parameters
        beta = np.zeros(n_features)
        gamma = (rng.uniform(0, 1, n_features) < self.theta_init).astype(int)
        tau2 = 1.0
        theta = self.theta_init
        lambda2_slab = np.ones(n_features)  # For Laplace prior in slab

        # Storage for samples (after burn-in and thinning)
        n_keep = (self.n_iter - self.burnin) // self.thin
        beta_samples = np.zeros((n_keep, n_features))
        gamma_samples = np.zeros((n_keep, n_features))
        theta_samples = []
        sample_idx = 0

        # Gibbs sampling
        for iter_idx in range(self.n_iter):
            # Sample β | γ, τ², λ²_slab, y
            beta = self._sample_beta_given_gamma(y, XtX, Xty, tau2, gamma, lambda2_slab, rng)

            # Sample λ²_j for slab components (γ_j = 0) - Laplace prior via scale mixture
            lambda2_slab = self._sample_lambda2_slab(beta, gamma, tau2, rng)

            # Sample γ | β, τ², θ, y
            gamma = self._sample_gamma(beta, tau2, theta, rng)

            # Compute residuals once for tau2 sampling (optimization)
            residuals = y - X @ beta

            # Sample τ² | β, y
            tau2 = self._sample_tau2(residuals, n_samples, n_features, rng)

            # Sample θ | γ (if update_theta is True)
            if self.update_theta:
                theta = self._sample_theta(gamma, rng)

            # Store samples after burn-in with thinning
            if iter_idx >= self.burnin and (iter_idx - self.burnin) % self.thin == 0 and sample_idx < n_keep:
                beta_samples[sample_idx] = beta
                gamma_samples[sample_idx] = gamma
                if self.update_theta:
                    theta_samples.append(theta)
                sample_idx += 1

        return beta_samples, gamma_samples, theta_samples

    def _sample_beta_given_gamma(
        self,
        y: np.ndarray,
        XtX: np.ndarray,
        Xty: np.ndarray,
        tau2: float,
        gamma: np.ndarray,
        lambda2_slab: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample β from its conditional posterior distribution given γ.

        For spike (γ_j=1): β_j ~ N(0, τ²/λ₀²) [tight prior, strong shrinkage]
        For slab (γ_j=0): β_j ~ Laplace(0, λ₁) via scale mixture with λ²_j

        Combined conditional posterior:
        β | γ, τ², λ²_slab, y ~ N(μ_β, Σ_β)
        where Σ_β = τ²(X'X + D_γ)^(-1)
              μ_β = Σ_β X'y / τ²
        and D_γ has diagonal elements: λ₀²/τ² if γ_j=1, else 1/(τ²·λ²_j) for slab
        """
        n_features = len(gamma)
        beta = np.zeros(n_features)

        # Build precision diagonal: D_γ
        D_diag = np.zeros(n_features)
        for j in range(n_features):
            if gamma[j] == 1:  # Spike: use tight normal prior
                D_diag[j] = (self.lambda0 ** 2) / tau2
            else:  # Slab: use Laplace via scale mixture
                D_diag[j] = 1.0 / (tau2 * lambda2_slab[j])

        # A = X'X + D_γ - use workspace to avoid allocation
        A = self._A_workspace
        np.copyto(A, XtX)
        A.flat[::n_features + 1] += D_diag  # Add to diagonal elements

        # Use Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(A)

            # Solve for mean: L @ L.T @ μ = X'y / τ²
            v = Xty / tau2
            w = np.linalg.solve(L, v)  # Forward solve: L @ w = v
            mu_beta = np.linalg.solve(L.T, w)  # Backward solve: L.T @ μ = w

            # Sample: β = μ + L^(-T) @ (√τ² * z) where z ~ N(0, I)
            z = rng.standard_normal(n_features) * np.sqrt(tau2)
            beta = mu_beta + np.linalg.solve(L.T, z)

        except np.linalg.LinAlgError:
            # Fall back to regularized version if Cholesky fails
            warnings.warn("Cholesky decomposition failed, using regularized inversion")
            A_reg = A + 1e-6 * np.eye(n_features)
            L = np.linalg.cholesky(A_reg)
            v = Xty / tau2
            w = np.linalg.solve(L, v)
            mu_beta = np.linalg.solve(L.T, w)
            z = rng.standard_normal(n_features) * np.sqrt(tau2)
            beta = mu_beta + np.linalg.solve(L.T, z)

        return beta

    def _sample_gamma(
        self,
        beta: np.ndarray,
        tau2: float,
        theta: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample γ from its conditional posterior distribution.

        γ_j | β_j, τ², θ ~ Bernoulli(p_j)
        
        where p_j = P(γ_j = 1 | β_j, τ², θ)
                  = p₁ / (p₁ + p₀)
        
        p₁ = P(γ_j = 1) * P(β_j | γ_j = 1, τ²) = θ * N(β_j; 0, τ²/λ₀²)
        p₀ = P(γ_j = 0) * P(β_j | γ_j = 0, τ²) = (1-θ) * Laplace(β_j; 0, λ₁)
        
        Using log probabilities for numerical stability.
        """
        n_features = len(beta)
        gamma = np.zeros(n_features, dtype=int)

        for j in range(n_features):
            # Log probability under spike (γ_j = 1): log N(β_j; 0, τ²/λ₀²)
            spike_var = tau2 / (self.lambda0 ** 2)
            log_p1 = (
                np.log(theta) 
                - 0.5 * np.log(2 * np.pi * spike_var)
                - 0.5 * (beta[j] ** 2) / spike_var
            )

            # Log probability under slab (γ_j = 0): log Laplace(β_j; 0, λ₁)
            log_p0 = (
                np.log(1 - theta)
                - np.log(2 * self.lambda1)
                - np.abs(beta[j]) / self.lambda1
            )

            # Compute posterior probability (using log-sum-exp trick)
            max_log_p = max(log_p1, log_p0)
            log_p1_normalized = log_p1 - max_log_p
            log_p0_normalized = log_p0 - max_log_p

            p1 = np.exp(log_p1_normalized)
            p0 = np.exp(log_p0_normalized)
            
            prob_spike = p1 / (p1 + p0)

            # Sample γ_j
            gamma[j] = 1 if rng.uniform(0, 1) < prob_spike else 0

        return gamma

    def _sample_lambda2_slab(
        self,
        beta: np.ndarray,
        gamma: np.ndarray,
        tau2: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample λ²_j for slab components (γ_j = 0) from conditional posterior.

        For Laplace prior via scale mixture of normals:
        λ²_j | β_j, τ², γ_j=0 ~ InverseGaussian(μ = λ₁/|β_j|, λ = λ₁²)

        For spike components (γ_j = 1), we don't use λ²_j, so set to 1.
        
        Uses vectorized sampling via chi-squared relationship (same as BayesianLasso).
        """
        n_features = len(beta)
        lambda2 = np.ones(n_features)

        # Only sample for slab components (γ_j = 0)
        slab_mask = (gamma == 0)
        if not np.any(slab_mask):
            return lambda2

        # Parameters for inverse-Gaussian: μ = λ₁/|β_j|, shape = λ₁²
        beta_slab = beta[slab_mask]
        n_slab = len(beta_slab)
        
        mu = self.lambda1 / (np.abs(beta_slab) + 1e-10)  # Vectorized
        lam = self.lambda1 ** 2

        # Sample using chi-squared relationship (vectorized)
        nu = rng.chisquare(1, size=n_slab)  # χ²(1) samples
        y = mu + (mu**2 * nu) / (2 * lam) - (mu / (2 * lam)) * np.sqrt(4 * mu * lam * nu + mu**2 * nu**2)

        # Accept with probability μ/(μ+y), otherwise return μ²/y
        u = rng.uniform(0, 1, size=n_slab)
        accept = u <= mu / (mu + y)

        # Compute rejected values safely (avoid division by zero)
        y_safe = np.maximum(y, 1e-12)  # Ensure y > 0
        rejected_vals = mu**2 / y_safe

        lambda2_slab = np.where(accept, y, rejected_vals)
        # Clip to reasonable range to avoid numerical issues
        lambda2_slab = np.clip(lambda2_slab, 1e-10, 1e10)

        # Update only slab components
        lambda2[slab_mask] = lambda2_slab

        return lambda2

    def _sample_tau2(
        self,
        residuals: np.ndarray,
        n_samples: int,
        n_features: int,
        rng: np.random.Generator
    ) -> float:
        """
        Sample τ² from its conditional posterior distribution.

        τ² | β, y ~ InverseGamma(a + n/2, b + RSS/2)
        where RSS = ||y - Xβ||²

        OPTIMIZED: Accepts pre-computed residuals to avoid recomputation.
        """
        # Residual sum of squares
        rss = np.sum(residuals ** 2)

        # Posterior parameters (simplified - not including beta prior term)
        shape = self.tau2_a + n_samples / 2.0
        scale = self.tau2_b + rss / 2.0

        # Sample from inverse-gamma: sample from gamma then take reciprocal
        tau2 = 1.0 / rng.gamma(shape, 1.0 / scale)

        return tau2

    def _sample_theta(
        self,
        gamma: np.ndarray,
        rng: np.random.Generator
    ) -> float:
        """
        Sample θ from its conditional posterior distribution.

        θ | γ ~ Beta(a_θ + ∑γ_j, b_θ + p - ∑γ_j)
        
        where p is the number of features and ∑γ_j is the number of included features.
        """
        n_features = len(gamma)
        n_included = np.sum(gamma)

        # Posterior parameters for Beta distribution
        alpha = self.theta_a + n_included
        beta = self.theta_b + (n_features - n_included)

        # Sample from Beta distribution
        theta = rng.beta(alpha, beta)

        return theta

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

    def predict_map(self, X: np.ndarray) -> np.ndarray:
        """Predict using the MAP estimate of coefficients.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values using MAP estimate
        """
        if self.coef_map_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = self._validate_data(X, reset=False)
        return X @ self.coef_map_

    def get_selected_features(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get indices of selected features based on posterior inclusion probabilities.

        Parameters
        ----------
        threshold : float, default=0.5
            Inclusion probability threshold for feature selection

        Returns
        -------
        selected : ndarray of shape (n_selected,)
            Indices of features with inclusion probability >= threshold
        """
        if self.inclusion_probs_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return np.where(self.inclusion_probs_ >= threshold)[0]

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
    
    def _compute_posterior_densities(self, X: np.ndarray, y: np.ndarray, 
                                     samples: np.ndarray) -> np.ndarray:
        """Compute (unnormalized) log posterior density for each sample.
        
        For Spike and Slab Lasso, the posterior is proportional to:
        p(β|y) ∝ p(y|β) * p(β|γ) * p(γ)
        
        Where:
        - p(y|β) = N(Xβ, τ²I) [likelihood]
        - p(β|γ) = mixture of spike and slab priors
        - p(γ) = product of Bernoulli(θ) priors
        
        OPTIMIZED: Vectorized computation across all samples.
        Uses cached values if available and data matches.
        
        Note: We marginalize over γ by computing the mixture density for each β_j.
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Design matrix
        y : ndarray of shape (n,)
            Response vector
        samples : ndarray of shape (n_samples, p)
            MCMC samples of coefficients
        
        Returns
        -------
        log_posteriors : ndarray of shape (n_samples,)
            Log posterior density for each sample
        """
        # Check if we can use cached posteriors
        if (self.cache_posteriors and 
            self.log_posteriors_ is not None and
            self._X_fit is X and 
            self._y_fit is y and
            samples.shape[0] == self.log_posteriors_.shape[0]):
            # Verify samples match (check if samples is reshaped chains)
            all_samples = self.chains_.reshape(-1, self.n_features_in_)
            if samples is all_samples or np.array_equal(samples, all_samples):
                return self.log_posteriors_
        
        # Vectorized predictions: (n, n_samples) = (n, p) @ (p, n_samples)
        predictions = X @ samples.T  # Shape: (n, n_samples)
        
        # Vectorized residuals: broadcast y to (n, n_samples)
        residuals = y[:, np.newaxis] - predictions  # Shape: (n, n_samples)
        
        # Log likelihood: -0.5 * ||residuals||^2 for each sample
        # Sum over observations (axis=0) to get (n_samples,)
        log_likelihood = -0.5 * np.sum(residuals ** 2, axis=0)
        
        # Log prior: mixture of spike and slab for each coefficient
        # For computational efficiency, we approximate using the slab (Laplace) component
        # since the spike is concentrated near zero
        log_prior = -self.lambda1 * np.sum(np.abs(samples), axis=1)
        
        log_posteriors = log_likelihood + log_prior
        
        return log_posteriors
    
    def get_log_posteriors(self) -> np.ndarray:
        """Get cached log posterior densities for MCMC samples.
        
        Returns the cached log posterior densities computed during fit.
        This is useful for external analysis without recomputing.
        
        Returns
        -------
        log_posteriors : ndarray of shape (n_samples,)
            Log posterior density for each MCMC sample
        
        Raises
        ------
        ValueError
            If model has not been fitted or posteriors were not cached
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if not self.cache_posteriors:
            raise ValueError(
                "Posteriors were not cached. Set cache_posteriors=True when "
                "initializing the model to enable caching."
            )
        
        if self.log_posteriors_ is None:
            raise ValueError("Cached posteriors not available.")
        
        return self.log_posteriors_
