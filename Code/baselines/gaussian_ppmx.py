"""
Gaussian PPMx Model - Python Implementation (Continuous Covariates Only)

This module provides a Python implementation of the Gaussian Product Partition Model
with covariates (PPMx) as described in Mueller, Quintana, and Rosner (2011).

The PPMx model is a Bayesian nonparametric clustering method that incorporates
covariate information into the partition model through a similarity function.

Note: This implementation supports continuous covariates only. For categorical 
covariates, they must be converted to numeric form (e.g., dummy variables) before use.

References:
- Mueller, P., Quintana, F. A., & Rosner, G. L. (2011). A product partition model with 
  regression on covariates. Journal of Computational and Graphical Statistics.
- Page, G. L., Quintana, F. A. (2018). Calibrating covariate informed product partition models.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial.distance import pdist, squareform


class PPMxError(Exception):
    """Base exception for Gaussian PPMx errors."""
    pass


class InvalidCovariateError(PPMxError):
    """Raised when covariates are not valid continuous variables."""
    pass


@dataclass
class PPMxResults:
    """Container for Gaussian PPMx MCMC results."""
    mu: NDArray[np.float64]
    sig2: NDArray[np.float64]
    Si: NDArray[np.int32]
    mu0: NDArray[np.float64]
    sig20: NDArray[np.float64]
    nclus: NDArray[np.int32]
    like: NDArray[np.float64]
    fitted: NDArray[np.float64]
    WAIC: float
    lpml: float
    beta: Optional[NDArray[np.float64]] = None
    ppred: Optional[NDArray[np.float64]] = None
    predclass: Optional[NDArray[np.int32]] = None
    rbpred: Optional[NDArray[np.float64]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            key: value for key, value in self.__dict__.items()
            if value is not None
        }


class GaussianPPMx:
    """
    Gaussian Product Partition Model with covariates (PPMx).
    
    This class implements a Bayesian nonparametric clustering model for Gaussian data
    that incorporates covariate information to inform the partition structure.
    
    Parameters
    ----------
    mean_model : int, default=1
        Type of mean model:
        - 1: cluster-specific means with no covariates in likelihood
        - 2: cluster-specific intercepts with global regression (Xbeta)
    
    cohesion : int, default=1
        Type of cohesion function:
        - 1: Dirichlet process style c(S) = M * (|S| - 1)!
        - 2: Uniform cohesion c(S) = 1
    
    M : float, default=1.0
        Precision parameter for the cohesion function
    
    PPM : bool, default=False
        If True, use PPM (without covariates). If False, use PPMx (with covariates)
    
    similarity_function : int, default=1
        Type of similarity function:
        - 1: Auxiliary similarity
        - 2: Double dipper similarity
    
    Note: This implementation only supports continuous covariates.
    
    consim : int, default=1
        Type of continuous similarity:
        - 1: N-N(m0, s20, v) model
        - 2: N-NIG(m0, k0, nu0, s20) model
    
    calibrate : int, default=0
        Calibration option:
        - 0: no calibration
        - 1: standardize similarity for each covariate
        - 2: coarsening (raise to 1/p power)
    
    sim_parms : list, default=[0.0, 1.0, 0.1, 1.0, 2.0, 0.1, 1.0]
        Similarity function parameters [m0, s20, v2, k0, nu0, a0, alpha]
    
    model_priors : list, default=[0, 10000, 1, 1]
        Data model priors [mu0_mean, mu0_var, sig2_lower, sig2_upper]
    
    mh : list, default=[0.5, 0.5]
        Metropolis-Hastings tuning parameters for sig2 and sig20
    
    draws : int, default=1100
        Total number of MCMC iterations
    
    burn : int, default=100
        Number of burn-in iterations to discard
    
    thin : int, default=1
        Thinning interval
    
    verbose : bool, default=False
        Print progress information
    
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        mean_model: int = 1,
        cohesion: int = 1,
        M: float = 1.0,
        PPM: bool = False,
        similarity_function: int = 1,
        consim: int = 1,
        calibrate: int = 0,
        sim_parms: List[float] = None,
        model_priors: List[float] = None,
        mh: List[float] = None,
        draws: int = 1100,
        burn: int = 100,
        thin: int = 1,
        verbose: bool = False,
        random_state: Optional[int] = None
    ):
        self.mean_model = mean_model
        self.cohesion = cohesion
        self.M = M
        self.PPM = PPM
        self.similarity_function = similarity_function
        self.consim = consim
        self.calibrate = calibrate
        
        # Default similarity parameters
        if sim_parms is None:
            sim_parms = [0.0, 1.0, 0.1, 1.0, 2.0, 0.1, 1.0]
        self.sim_parms = sim_parms
        
        # Default model priors
        if model_priors is None:
            model_priors = [0, 10000, 1, 1]
        self.model_priors = model_priors
        
        # Default MH tuning
        if mh is None:
            mh = [0.5, 0.5]
        self.mh = mh
        
        self.draws = draws
        self.burn = burn
        self.thin = thin
        self.verbose = verbose
        self.random_state = random_state
        
        # Storage for MCMC results
        self.results_: Optional[PPMxResults] = None
        self.n_obs_: Optional[int] = None
        self.n_pred_: Optional[int] = None
        self.covariate_names_: Optional[List[str]] = None
    
    @property
    def n_iterations(self) -> int:
        """Number of MCMC iterations saved after burn-in and thinning."""
        return (self.draws - self.burn) // self.thin
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self.results_ is not None
    
    @property
    def n_covariates(self) -> int:
        """Number of covariates."""
        return len(self.covariate_names_) if self.covariate_names_ else 0
        
    def _process_covariates(
        self, 
        X: Optional[pd.DataFrame],
        X_pred: Optional[pd.DataFrame] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Process and standardize continuous covariates.
        
        Parameters
        ----------
        X : DataFrame, optional
            Training covariates
        X_pred : DataFrame, optional
            Prediction covariates
        
        Returns
        -------
        Xcon : Standardized training covariates
        Xconp : Standardized prediction covariates
        ncon : Number of continuous covariates
        
        Raises
        ------
        InvalidCovariateError
            If covariates contain non-numeric data
        """
        if X is None:
            return np.zeros((1, 1)), np.zeros((1, 1)), 0
        
        n_obs = len(X)
        n_pred = len(X_pred) if X_pred is not None else 0
        
        # Combine data for consistent preprocessing
        X_all = pd.concat([X, X_pred], ignore_index=True) if X_pred is not None else X.copy()
        
        # Validate numeric data
        try:
            X_all = X_all.astype(float)
        except (ValueError, TypeError) as e:
            raise InvalidCovariateError(
                "All covariates must be continuous (numeric). "
                "Categorical covariates are not supported."
            ) from e
        
        ncon = X_all.shape[1]
        if ncon == 0:
            return np.zeros((n_obs, 1)), np.zeros((max(1, n_pred), 1)), 0
        
        # Standardize: mean=0, std=1
        X_values = X_all.values
        mean = np.nanmean(X_values, axis=0)
        std = np.nanstd(X_values, axis=0)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
        
        X_standardized = (X_values - mean) / std
        
        Xcon = X_standardized[:n_obs]
        Xconp = X_standardized[n_obs:] if n_pred > 0 else np.zeros((1, ncon))
        
        return Xcon, Xconp, ncon
    
    def _compute_gower_distance(
        self,
        X: pd.DataFrame,
        X_pred: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gower dissimilarity matrix.
        
        Returns
        -------
        dissim_train : dissimilarity matrix for training data
        dissim_test : dissimilarity matrix between test and training data
        """
        from sklearn.metrics import pairwise_distances
        
        # Combine data
        if X_pred is not None:
            X_all = pd.concat([X, X_pred], axis=0, ignore_index=True)
        else:
            X_all = X.copy()
        
        n_obs = len(X)
        
        # Compute Gower distance (simplified version)
        n_features = X_all.shape[1]
        n_total = len(X_all)
        
        # Compute dissimilarity for each feature and average
        dissim_matrices = []
        for col in X_all.columns:
            values = X_all[col].values.astype(float)
            col_range = np.nanmax(values) - np.nanmin(values)
            
            if col_range > 0:
                col_dissim = np.abs(np.subtract.outer(values, values)) / col_range
            else:
                col_dissim = np.zeros((n_total, n_total))
            
            dissim_matrices.append(col_dissim)
        
        dissim_all = np.mean(dissim_matrices, axis=0)
        
        dissim_train = dissim_all[:n_obs, :n_obs]
        dissim_test = dissim_all[n_obs:, :n_obs] if X_pred is not None else np.zeros((1, n_obs))
        
        return dissim_train, dissim_test
    
    def _compute_similarity(
        self,
        j: int,
        cluster_idx: np.ndarray,
        Xcon: np.ndarray,
        ncon: int
    ) -> float:
        """
        Compute similarity between subject j and a cluster (continuous covariates only).
        
        Parameters
        ----------
        j : index of subject
        cluster_idx : indices of subjects in the cluster
        Xcon : continuous covariates
        ncon : number of continuous covariates
        
        Returns
        -------
        similarity : similarity value
        """
        if self.PPM or len(cluster_idx) == 0:
            return 0.0
        
        if self.PPM or len(cluster_idx) == 0 or ncon == 0 or Xcon.shape[1] <= 1:
            return 0.0
        
        m0, s20, v2, k0, nu0, *_ = self.sim_parms
        n_c = len(cluster_idx)
        
        # Compute similarity for each covariate
        if self.consim == 1:
            # N-N model
            similarities = [
                self._compute_nn_similarity(Xcon[j, p], Xcon[cluster_idx, p], m0, s20, v2, n_c)
                for p in range(ncon)
            ]
        else:
            # N-NIG model
            similarities = [
                self._compute_nig_similarity(Xcon[j, p], Xcon[cluster_idx, p], m0, s20, k0, nu0, n_c)
                for p in range(ncon)
            ]
        
        return sum(similarities)
    
    @staticmethod
    def _compute_nn_similarity(
        x_j: float, 
        x_cluster: NDArray[np.float64], 
        m0: float, 
        s20: float, 
        v2: float, 
        n_c: int
    ) -> float:
        """Compute N-N model similarity for a single covariate."""
        var_post = 1.0 / (1.0/s20 + n_c/v2)
        mean_post = var_post * (m0/s20 + np.sum(x_cluster)/v2)
        return stats.norm.logpdf(x_j, mean_post, np.sqrt(var_post + v2))
    
    @staticmethod
    def _compute_nig_similarity(
        x_j: float, 
        x_cluster: NDArray[np.float64], 
        m0: float, 
        s20: float, 
        k0: float, 
        nu0: float, 
        n_c: int
    ) -> float:
        """Compute N-NIG model similarity for a single covariate."""
        x_bar = np.mean(x_cluster)
        
        # Update parameters
        kn = k0 + n_c
        nun = nu0 + n_c
        mn = (k0 * m0 + n_c * x_bar) / kn
        
        ss = np.sum((x_cluster - x_bar)**2) if n_c > 1 else 0.0
        s2n = (nu0 * s20 + ss + k0 * n_c * (x_bar - m0)**2 / kn) / nun
        
        # Predictive distribution is Student's t
        scale = np.sqrt(s2n * (1 + 1/kn))
        return stats.t.logpdf(x_j, nun, mn, scale)
    
    def _cohesion_function(self, cluster_size: int) -> float:
        """
        Compute cohesion function value.
        
        Parameters
        ----------
        cluster_size : number of subjects in cluster
        
        Returns
        -------
        cohesion : cohesion value
        """
        if self.cohesion == 1:
            # Dirichlet process style: c(S) = M * (|S| - 1)!
            if cluster_size == 0:
                return 0.0
            if cluster_size == 1:
                return self.M
            # Use gammaln to avoid overflow: gamma(n) = (n-1)!
            from scipy.special import gammaln
            log_cohesion = np.log(self.M) + gammaln(cluster_size)
            return np.exp(min(log_cohesion, 700))  # Cap at exp(700) to avoid overflow
        elif self.cohesion == 2:
            # Uniform cohesion
            return 1.0
        else:
            return 1.0
    
    def _update_partition(
        self,
        y: np.ndarray,
        Si: np.ndarray,
        muh: np.ndarray,
        sig2h: np.ndarray,
        Xcon: np.ndarray,
        ncon: int,
        beta: np.ndarray
    ) -> np.ndarray:
        """
        Update partition assignments using Algorithm 8 from Neal (2000).
        
        Returns
        -------
        Si : updated cluster assignments
        """
        n_obs = len(y)
        Si_new = Si.copy()
        
        for j in range(n_obs):
            # Remove j from current cluster
            current_cluster = Si_new[j]
            cluster_counts = np.bincount(Si_new, minlength=n_obs)
            cluster_counts[current_cluster] -= 1
            
            # Compute probabilities for each cluster
            unique_clusters = np.where(cluster_counts > 0)[0]
            n_clusters = len(unique_clusters)
            
            # Add probability for new cluster
            log_probs = np.zeros(n_clusters + 1)
            
            # Existing clusters
            for idx, k in enumerate(unique_clusters):
                cluster_members = np.where(Si_new == k)[0]
                cluster_members = cluster_members[cluster_members != j]
                
                # Cohesion
                n_k = len(cluster_members)
                log_cohesion = np.log(self._cohesion_function(n_k))
                
                # Similarity
                similarity = self._compute_similarity(
                    j, cluster_members, Xcon, ncon
                )
                
                # Likelihood - use first member of cluster to get parameters
                if len(cluster_members) > 0:
                    representative_idx = cluster_members[0]
                    mean_j = muh[representative_idx]
                    sig2_j = sig2h[representative_idx]
                else:
                    # Empty cluster, should not happen but handle it
                    mean_j = 0.0
                    sig2_j = 1.0
                
                if self.mean_model == 2 and beta is not None:
                    # Add regression component
                    mean_j += np.dot(Xcon[j], beta)
                
                log_like = stats.norm.logpdf(y[j], mean_j, np.sqrt(sig2_j))
                
                log_probs[idx] = log_cohesion + similarity + log_like
            
            # New cluster
            log_cohesion = np.log(self.M)
            
            # Sample new parameters for new cluster
            mu_new = np.random.normal(self.model_priors[0], np.sqrt(self.model_priors[1]))
            sig2_new = np.random.uniform(self.model_priors[2], self.model_priors[3])
            
            mean_j = mu_new
            if self.mean_model == 2 and beta is not None:
                mean_j += np.dot(Xcon[j], beta)
            
            log_like = stats.norm.logpdf(y[j], mean_j, np.sqrt(sig2_new))
            log_probs[n_clusters] = log_cohesion + log_like
            
            # Normalize and sample with numerical stability
            log_probs_max = np.max(log_probs)
            if np.isnan(log_probs_max) or np.isinf(log_probs_max):
                # If all log probs are -inf or nan, use uniform distribution
                probs = np.ones(n_clusters + 1) / (n_clusters + 1)
            else:
                log_probs = log_probs - log_probs_max
                # Cap log_probs to avoid overflow
                log_probs = np.clip(log_probs, -700, 700)
                probs = np.exp(log_probs)
                probs_sum = np.sum(probs)
                if probs_sum == 0 or np.isnan(probs_sum) or np.isinf(probs_sum):
                    # Fallback to uniform if normalization fails
                    probs = np.ones(n_clusters + 1) / (n_clusters + 1)
                else:
                    probs = probs / probs_sum
            
            new_assignment = np.random.choice(n_clusters + 1, p=probs)
            
            if new_assignment < n_clusters:
                Si_new[j] = unique_clusters[new_assignment]
            else:
                # Assign to new cluster
                Si_new[j] = n_obs + np.random.randint(1000)
        
        # Relabel clusters to be consecutive
        unique_labels = np.unique(Si_new)
        for new_label, old_label in enumerate(unique_labels):
            Si_new[Si_new == old_label] = new_label
        
        return Si_new
    
    def _update_cluster_parameters(
        self,
        y: np.ndarray,
        Si: np.ndarray,
        beta: Optional[np.ndarray],
        Xcon: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Update cluster-specific parameters (means and variances).
        
        Returns
        -------
        muh : cluster means
        sig2h : cluster variances
        mu0 : hyperparameter mu0
        sig20 : hyperparameter sig20
        """
        n_obs = len(y)
        muh = np.zeros(n_obs)
        sig2h = np.zeros(n_obs)
        
        # Update each cluster's parameters
        for k in np.unique(Si):
            cluster_idx = np.where(Si == k)[0]
            n_k = len(cluster_idx)
            
            # Get residuals
            y_k = y[cluster_idx].copy()
            if self.mean_model == 2 and beta is not None:
                # Vectorized residual computation
                y_k -= Xcon[cluster_idx] @ beta
            
            # Update variance (MH step)
            current_sig2 = sig2h[cluster_idx[0]] if len(cluster_idx) > 0 else 1.0
            if current_sig2 == 0:
                current_sig2 = 1.0
            
            proposal_sig2 = current_sig2 * np.exp(np.random.normal(0, self.mh[0]))
            proposal_sig2 = np.clip(proposal_sig2, self.model_priors[2], self.model_priors[3])
            
            # Compute acceptance ratio
            log_like_current = np.sum(stats.norm.logpdf(y_k, muh[cluster_idx[0]], np.sqrt(current_sig2)))
            log_like_proposal = np.sum(stats.norm.logpdf(y_k, muh[cluster_idx[0]], np.sqrt(proposal_sig2)))
            
            log_ratio = log_like_proposal - log_like_current
            
            if np.log(np.random.random()) < log_ratio:
                sig2_k = proposal_sig2
            else:
                sig2_k = current_sig2
            
            # Update mean (conjugate update)
            prior_mean = self.model_priors[0]
            prior_var = self.model_priors[1]
            
            post_var = 1.0 / (1.0/prior_var + n_k/sig2_k)
            post_mean = post_var * (prior_mean/prior_var + np.sum(y_k)/sig2_k)
            
            mu_k = np.random.normal(post_mean, np.sqrt(post_var))
            
            # Assign to all members of cluster
            muh[cluster_idx] = mu_k
            sig2h[cluster_idx] = sig2_k
        
        # Update hyperparameters
        mu0 = np.random.normal(self.model_priors[0], np.sqrt(self.model_priors[1]))
        sig20 = np.random.uniform(self.model_priors[2], self.model_priors[3])
        
        return muh, sig2h, mu0, sig20
    
    def _update_beta(
        self,
        y: np.ndarray,
        Si: np.ndarray,
        muh: np.ndarray,
        sig2h: np.ndarray,
        Xcon: np.ndarray,
        ncon: int
    ) -> np.ndarray:
        """
        Update regression coefficients beta (if mean_model == 2).
        
        Returns
        -------
        beta : regression coefficients
        """
        if self.mean_model != 2:
            return None
        
        n_obs = len(y)
        n_cov = ncon
        
        if n_cov == 0:
            return None
        
        # Construct design matrix
        X = Xcon
        
        # Compute residuals
        y_resid = y - muh
        
        # Bayesian linear regression update
        # Prior: beta ~ N(0, 100^2 * I)
        prior_var = 100**2
        
        # Likelihood variance
        sig2_vec = sig2h
        
        # Weighted least squares
        try:
            W = np.diag(1.0 / sig2_vec)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y_resid
            
            # Posterior covariance
            post_cov = np.linalg.inv(XtWX + np.eye(n_cov) / prior_var)
            post_mean = post_cov @ XtWy
            
            # Sample from posterior
            beta = np.random.multivariate_normal(post_mean, post_cov)
        except:
            # If singular, use prior
            beta = np.random.normal(0, 10, n_cov)
        
        return beta
    
    def fit(
        self,
        y: np.ndarray,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        X_pred: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> 'GaussianPPMx':
        """
        Fit the Gaussian PPMx model using MCMC.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Response variable
        
        X : DataFrame or array-like, shape (n_samples, n_features), optional
            Covariate matrix for training data
        
        X_pred : DataFrame or array-like, shape (n_pred, n_features), optional
            Covariate matrix for prediction
        
        Returns
        -------
        self : object
            Returns self with fitted parameters stored in results_
        """
        # Convert to appropriate types
        y = np.array(y).flatten()
        
        if X is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if X_pred is not None and not isinstance(X_pred, pd.DataFrame):
            if X is not None:
                X_pred = pd.DataFrame(X_pred, columns=X.columns)
            else:
                X_pred = pd.DataFrame(X_pred)
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_obs_ = len(y)
        self.n_pred_ = len(X_pred) if X_pred is not None else 0
        
        # Process covariates
        Xcon, Xconp, ncon = self._process_covariates(X, X_pred)
        self.covariate_names_ = list(X.columns) if X is not None else []
        
        # Compute Gower distance if needed
        dissim_train = dissim_test = None
        if self.similarity_function == 4 and X is not None:
            dissim_train, dissim_test = self._compute_gower_distance(X, X_pred)
        
        # Initialize MCMC storage
        n_out = self.n_iterations
        
        # Initialize result arrays
        base_results = {
            'mu': np.zeros((n_out, self.n_obs_)),
            'sig2': np.zeros((n_out, self.n_obs_)),
            'Si': np.zeros((n_out, self.n_obs_), dtype=np.int32),
            'mu0': np.zeros(n_out),
            'sig20': np.zeros(n_out),
            'nclus': np.zeros(n_out, dtype=np.int32),
            'like': np.zeros((n_out, self.n_obs_)),
            'fitted': np.zeros((n_out, self.n_obs_)),
        }
        
        # Add optional results
        optional_results = {}
        if self.mean_model == 2 and ncon > 0:
            optional_results['beta'] = np.zeros((n_out, ncon))
        
        if self.n_pred_ > 0:
            optional_results.update({
                'ppred': np.zeros((n_out, self.n_pred_)),
                'predclass': np.zeros((n_out, self.n_pred_), dtype=np.int32),
                'rbpred': np.zeros((n_out, self.n_pred_))
            })
        
        results = {**base_results, **optional_results}
        
        # Initialize parameters
        Si = np.random.randint(0, min(3, self.n_obs_), size=self.n_obs_)
        muh = np.random.normal(0, 1, self.n_obs_)
        sig2h = np.random.gamma(1, 1, self.n_obs_)
        mu0 = 0.0
        sig20 = 1.0
        beta = None
        if self.mean_model == 2 and ncon > 0:
            beta = np.zeros(ncon)
        
        # MCMC loop
        save_idx = 0
        for iteration in range(self.draws):
            if self.verbose and iteration % 100 == 0:
                progress = (iteration / self.draws) * 100
                print(f"MCMC iteration {iteration}/{self.draws} ({progress:.1f}%)")
            
            # Update partition
            Si = self._update_partition(y, Si, muh, sig2h, Xcon, ncon, beta)
            
            # Update cluster parameters
            muh, sig2h, mu0, sig20 = self._update_cluster_parameters(
                y, Si, beta, Xcon
            )
            
            # Update regression coefficients
            if self.mean_model == 2:
                beta = self._update_beta(y, Si, muh, sig2h, Xcon, ncon)
            
            # Save results after burn-in
            if iteration >= self.burn and (iteration - self.burn) % self.thin == 0:
                results['mu'][save_idx] = muh
                results['sig2'][save_idx] = sig2h
                results['Si'][save_idx] = Si
                results['mu0'][save_idx] = mu0
                results['sig20'][save_idx] = sig20
                results['nclus'][save_idx] = len(np.unique(Si))
                
                # Compute likelihood
                for j in range(self.n_obs_):
                    mean_j = muh[j]
                    if self.mean_model == 2 and beta is not None:
                        mean_j += np.dot(Xcon[j], beta)
                    results['like'][save_idx, j] = np.exp(
                        stats.norm.logpdf(y[j], mean_j, np.sqrt(sig2h[j]))
                    )
                    results['fitted'][save_idx, j] = mean_j
                
                if self.mean_model == 2 and beta is not None:
                    results['beta'][save_idx] = beta
                
                # Predictions
                if self.n_pred_ > 0:
                    for p in range(self.n_pred_):
                        # Simple prediction: sample from existing clusters weighted by size
                        cluster_labels, cluster_counts = np.unique(Si, return_counts=True)
                        probs = cluster_counts / np.sum(cluster_counts)
                        pred_cluster = np.random.choice(cluster_labels, p=probs)
                        
                        results['predclass'][save_idx, p] = pred_cluster
                        
                        mean_pred = muh[np.where(Si == pred_cluster)[0][0]]
                        sig2_pred = sig2h[np.where(Si == pred_cluster)[0][0]]
                        
                        if self.mean_model == 2 and beta is not None:
                            mean_pred += np.dot(Xconp[p], beta)
                        
                        results['ppred'][save_idx, p] = np.random.normal(mean_pred, np.sqrt(sig2_pred))
                        results['rbpred'][save_idx, p] = mean_pred
                
                save_idx += 1
        
        # Compute WAIC and LPML
        waic = self._compute_waic(results['like'])
        lpml = self._compute_lpml(results['like'])
        
        # Convert to dataclass
        self.results_ = PPMxResults(
            mu=results['mu'],
            sig2=results['sig2'],
            Si=results['Si'],
            mu0=results['mu0'],
            sig20=results['sig20'],
            nclus=results['nclus'],
            like=results['like'],
            fitted=results['fitted'],
            WAIC=waic,
            lpml=lpml,
            beta=results.get('beta'),
            ppred=results.get('ppred'),
            predclass=results.get('predclass'),
            rbpred=results.get('rbpred')
        )
        return self
    
    def _compute_waic(self, like: np.ndarray) -> float:
        """
        Compute Watanabe-Akaike Information Criterion.
        
        Parameters
        ----------
        like : array of shape (n_iter, n_obs)
            Likelihood values
        
        Returns
        -------
        waic : WAIC value
        """
        log_like = np.log(like + 1e-300)
        lppd = np.sum(np.log(np.mean(like, axis=0)))
        p_waic = np.sum(np.var(log_like, axis=0))
        waic = -2 * (lppd - p_waic)
        return waic
    
    def _compute_lpml(self, like: np.ndarray) -> float:
        """
        Compute Log Pseudo Marginal Likelihood.
        
        Parameters
        ----------
        like : array of shape (n_iter, n_obs)
            Likelihood values
        
        Returns
        -------
        lpml : LPML value
        """
        cpo = 1.0 / np.mean(1.0 / (like + 1e-300), axis=0)
        lpml = np.sum(np.log(cpo + 1e-300))
        return lpml
    
    def predict(self, X_new: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> NDArray[np.float64]:
        """Generate predictions for new data.
        
        Parameters
        ----------
        X_new : DataFrame or array-like, optional
            New covariate values. If None, returns fitted values.
        
        Returns
        -------
        Posterior mean predictions
        
        Raises
        ------
        PPMxError
            If model not fitted or predictions not available
        """
        if self.results_ is None:
            raise PPMxError("Model must be fitted before making predictions")
        
        if X_new is None:
            return np.mean(self.results_.fitted, axis=0)
        
        if self.results_.ppred is None:
            raise PPMxError(
                "No predictions available. Fit model with X_pred argument."
            )
        
        return np.mean(self.results_.ppred, axis=0)
    
    def get_partition(self, method: str = 'last') -> NDArray[np.int32]:
        """
        Get partition/clustering from MCMC samples.
        
        Parameters
        ----------
        method : {'last', 'mode'}
            Method to extract partition
        
        Returns
        -------
        Cluster assignments
        
        Raises
        ------
        PPMxError
            If model not fitted or method unknown
        """
        if self.results_ is None:
            raise PPMxError("Model must be fitted first")
        
        valid_methods = {'last', 'mode'}
        if method not in valid_methods:
            raise PPMxError(f"Unknown method '{method}'. Choose from {valid_methods}")
        
        if method == 'mode':
            warnings.warn(
                "Mode method not implemented, returning last partition",
                UserWarning
            )
        
        return self.results_.Si[-1]


def gaussian_ppmx(
    y: np.ndarray,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    X_pred: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    mean_model: int = 1,
    cohesion: int = 1,
    M: float = 1.0,
    PPM: bool = False,
    similarity_function: int = 1,
    consim: int = 1,
    calibrate: int = 0,
    sim_parms: List[float] = None,
    model_priors: List[float] = None,
    mh: List[float] = None,
    draws: int = 1100,
    burn: int = 100,
    thin: int = 1,
    verbose: bool = False,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit Gaussian PPMx model (functional interface).
    
    This function provides a convenient functional interface to the GaussianPPMx class,
    similar to the R implementation.
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : DataFrame or array-like, optional
        Covariate matrix for training
    X_pred : DataFrame or array-like, optional
        Covariate matrix for prediction
    mean_model : int, default=1
        Type of mean model (1 or 2)
    cohesion : int, default=1
        Type of cohesion function (1 or 2)
    M : float, default=1.0
        Precision parameter
    PPM : bool, default=False
        Use PPM instead of PPMx
    similarity_function : int, default=1
        Type of similarity function (1-4)
    consim : int, default=1
        Type of continuous similarity (1 or 2)
    calibrate : int, default=0
        Calibration option (0-2)
    sim_parms : list, optional
        Similarity parameters
    model_priors : list, optional
        Model priors
    mh : list, optional
        MH tuning parameters
    draws : int, default=1100
        Total MCMC iterations
    burn : int, default=100
        Burn-in iterations
    thin : int, default=1
        Thinning interval
    verbose : bool, default=False
        Print progress
    random_state : int, optional
        Random seed
    
    Returns
    -------
    results : dict
        Dictionary containing MCMC samples and model fit statistics
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gaussian_ppmx import gaussian_ppmx
    >>> 
    >>> # Generate synthetic data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = pd.DataFrame({
    ...     'x1': np.random.normal(0, 1, n),
    ...     'x2': np.random.choice(['A', 'B', 'C'], n)
    ... })
    >>> y = 2 * X['x1'] + np.where(X['x2'] == 'A', 0, 1) + np.random.normal(0, 0.5, n)
    >>> 
    >>> # Fit model
    >>> results = gaussian_ppmx(
    ...     y=y,
    ...     X=X,
    ...     draws=500,
    ...     burn=100,
    ...     verbose=True
    ... )
    >>> 
    >>> # Extract results
    >>> posterior_means = results['mu']
    >>> cluster_labels = results['Si']
    >>> n_clusters = results['nclus']
    """
    model = GaussianPPMx(
        mean_model=mean_model,
        cohesion=cohesion,
        M=M,
        PPM=PPM,
        similarity_function=similarity_function,
        consim=consim,
        calibrate=calibrate,
        sim_parms=sim_parms,
        model_priors=model_priors,
        mh=mh,
        draws=draws,
        burn=burn,
        thin=thin,
        verbose=verbose,
        random_state=random_state
    )
    
    model.fit(y, X, X_pred)
    
    return model.results_.to_dict()


if __name__ == "__main__":
    # Example usage
    print("Gaussian PPMx - Python Implementation (Continuous Covariates Only)")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    
    # Create continuous covariates only
    X = pd.DataFrame({
        'continuous1': np.random.normal(0, 1, n),
        'continuous2': np.random.normal(0, 1, n),
        'continuous3': np.random.normal(0, 1, n)
    })
    
    # Generate response with 3 clusters
    true_clusters = np.random.choice([0, 1, 2], n)
    cluster_means = [0, 3, -2]
    y = np.array([cluster_means[c] for c in true_clusters])
    y += X['continuous1'] * 0.5 + np.random.normal(0, 0.5, n)
    
    print(f"Generated {n} observations with 3 true clusters")
    print(f"True cluster distribution: {np.bincount(true_clusters)}")
    print()
    
    # Fit model
    print("Fitting Gaussian PPMx model...")
    results = gaussian_ppmx(
        y=y,
        X=X,
        draws=200,
        burn=50,
        thin=1,
        mean_model=1,
        M=1.0,
        verbose=True,
        random_state=42
    )
    
    print()
    print("Model Results:")
    print("-" * 50)
    print(f"WAIC: {results['WAIC']:.2f}")
    print(f"LPML: {results['lpml']:.2f}")
    print(f"Mean number of clusters: {np.mean(results['nclus']):.1f}")
    print(f"Final partition: {results['nclus'][-1]} clusters")
    print()
    print("Posterior mean parameters (first 5 observations):")
    print(f"mu: {np.mean(results['mu'], axis=0)[:5]}")
    print(f"sig2: {np.mean(results['sig2'], axis=0)[:5]}")
