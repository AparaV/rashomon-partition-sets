"""
PPMx: Product Partition Model with Covariates

Single-file implementation following the spec.
Public API: PPMxConfig, PPMxData, PPMxSampler, posterior_similarity_matrix, map_partition
"""

import numpy as np
from dataclasses import dataclass
from scipy.special import gammaln, logsumexp


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass(frozen=True)
class PPMxConfig:
    """Configuration for PPMx sampler.
    
    Parameters
    ----------
    alpha : float
        Concentration parameter for Dirichlet process
    mu0 : float
        Prior mean for Normal
    kappa0 : float
        Prior precision scaling for Normal
    a0 : float
        Prior shape for Inverse-Gamma
    b0 : float
        Prior scale for Inverse-Gamma
    n_iter : int
        Total number of MCMC iterations
    burn_in : int
        Number of burn-in iterations to discard
    thin : int
        Thinning interval for stored samples
    random_state : int | None
        Random seed for reproducibility
    """
    alpha: float = 1.0
    mu0: float = 0.0
    kappa0: float = 0.05
    a0: float = 2.0
    b0: float = 1.0
    n_iter: int = 2000
    burn_in: int = 500
    thin: int = 5
    random_state: int | None = None


# ==============================================================================
# Data container
# ==============================================================================

class PPMxData:
    """Container for PPMx data.
    
    Attributes
    ----------
    X : np.ndarray
        Covariate matrix (n x d)
    Y : np.ndarray | None
        Response vector (optional, not used in clustering)
    n : int
        Number of observations
    d : int
        Number of covariates
    """
    
    def __init__(self, X: np.ndarray, Y: np.ndarray | None, n: int, d: int):
        self.X = X
        self.Y = Y
        self.n = n
        self.d = d
    
    @classmethod
    def from_raw(cls, X, Y=None, standardize=True):
        """Create PPMxData from raw arrays.
        
        Parameters
        ----------
        X : array-like
            Covariate matrix
        Y : array-like, optional
            Response vector
        standardize : bool
            Whether to standardize X columnwise
            
        Returns
        -------
        PPMxData
            Validated and processed data container
            
        Raises
        ------
        ValueError
            If X contains NaNs or if shapes are mismatched
        """
        X = np.asarray(X, dtype=float)
        
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n, d = X.shape
        
        if Y is not None:
            Y = np.asarray(Y, dtype=float)
            if Y.shape[0] != n:
                raise ValueError(f"X and Y have mismatched lengths: {n} vs {Y.shape[0]}")
            if np.any(np.isnan(Y)):
                raise ValueError("Y contains NaN values")
        
        if standardize:
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0, ddof=1)
            X_std[X_std == 0] = 1.0  # Avoid division by zero
            X = (X - X_mean) / X_std
        
        return cls(X=X, Y=Y, n=n, d=d)


# ==============================================================================
# Internal utilities
# ==============================================================================

def _softmax_log(log_probs: np.ndarray) -> np.ndarray:
    """Convert log probabilities to probabilities using stable softmax.
    
    Parameters
    ----------
    log_probs : np.ndarray
        Array of log probabilities
        
    Returns
    -------
    np.ndarray
        Normalized probabilities
    """
    log_probs = log_probs - logsumexp(log_probs)
    return np.exp(log_probs)


def _set_random_seed(seed: int | None):
    """Set numpy random seed if provided.
    
    Parameters
    ----------
    seed : int | None
        Random seed, or None to skip
    """
    if seed is not None:
        np.random.seed(seed)


def _relabel_clusters(assignments: np.ndarray) -> np.ndarray:
    """Relabel cluster assignments to be compact (0, 1, 2, ...).
    
    Parameters
    ----------
    assignments : np.ndarray
        Original cluster assignments
        
    Returns
    -------
    np.ndarray
        Relabeled assignments
    """
    unique_labels = np.unique(assignments)
    mapping = {old: new for new, old in enumerate(unique_labels)}
    return np.array([mapping[label] for label in assignments])


# ==============================================================================
# Gaussian sufficient statistics
# ==============================================================================

@dataclass
class _GaussianStats:
    """Sufficient statistics for multivariate Gaussian with diagonal covariance.
    
    Attributes
    ----------
    n : int
        Number of observations in cluster
    sum_x : np.ndarray
        Sum of observations (d-dimensional)
    sum_x2 : np.ndarray
        Sum of squared observations (d-dimensional)
    """
    n: int
    sum_x: np.ndarray
    sum_x2: np.ndarray


def _empty_stats(d: int) -> _GaussianStats:
    """Create empty sufficient statistics.
    
    Parameters
    ----------
    d : int
        Dimensionality
        
    Returns
    -------
    _GaussianStats
        Empty statistics
    """
    return _GaussianStats(
        n=0,
        sum_x=np.zeros(d),
        sum_x2=np.zeros(d)
    )


def _add_point(stats: _GaussianStats, x_i: np.ndarray) -> _GaussianStats:
    """Add a point to sufficient statistics.
    
    Parameters
    ----------
    stats : _GaussianStats
        Current statistics
    x_i : np.ndarray
        Point to add
        
    Returns
    -------
    _GaussianStats
        Updated statistics
    """
    return _GaussianStats(
        n=stats.n + 1,
        sum_x=stats.sum_x + x_i,
        sum_x2=stats.sum_x2 + x_i**2
    )


def _remove_point(stats: _GaussianStats, x_i: np.ndarray) -> _GaussianStats:
    """Remove a point from sufficient statistics.
    
    Parameters
    ----------
    stats : _GaussianStats
        Current statistics
    x_i : np.ndarray
        Point to remove
        
    Returns
    -------
    _GaussianStats
        Updated statistics
    """
    return _GaussianStats(
        n=stats.n - 1,
        sum_x=stats.sum_x - x_i,
        sum_x2=stats.sum_x2 - x_i**2
    )


# ==============================================================================
# Cohesion function
# ==============================================================================

def _log_cohesion(size: int, alpha: float) -> float:
    """Log cohesion function for existing cluster.
    
    Parameters
    ----------
    size : int
        Current cluster size
    alpha : float
        Concentration parameter
        
    Returns
    -------
    float
        Log cohesion value
    """
    return np.log(size)


# ==============================================================================
# Covariate similarity
# ==============================================================================

def _log_marginal_x(stats: _GaussianStats, config: PPMxConfig) -> float:
    """Compute log marginal likelihood of observations in a cluster.
    
    Uses Normal-Inverse-Gamma conjugate prior with independence across dimensions.
    
    Parameters
    ----------
    stats : _GaussianStats
        Sufficient statistics for cluster
    config : PPMxConfig
        Prior hyperparameters
        
    Returns
    -------
    float
        Log marginal likelihood
    """
    if stats.n == 0:
        return 0.0
    
    n = stats.n
    d = len(stats.sum_x)
    
    # Posterior parameters
    kappa_n = config.kappa0 + n
    a_n = config.a0 + n / 2
    
    # Compute for each dimension independently and sum
    log_lik = 0.0
    
    for j in range(d):
        # Sample mean
        x_bar_j = stats.sum_x[j] / n if n > 0 else 0.0
        
        # Sum of squared deviations
        ss_j = stats.sum_x2[j] - n * x_bar_j**2
        
        # Posterior scale parameter
        b_n_j = config.b0 + 0.5 * ss_j + 0.5 * (config.kappa0 * n / kappa_n) * (x_bar_j - config.mu0)**2
        
        # Log marginal likelihood for dimension j
        log_lik_j = (
            0.5 * np.log(config.kappa0 / kappa_n)
            + config.a0 * np.log(config.b0)
            - a_n * np.log(b_n_j)
            + gammaln(a_n)
            - gammaln(config.a0)
            - 0.5 * n * np.log(2 * np.pi)
        )
        
        log_lik += log_lik_j
    
    return log_lik


def _log_predictive_x(x_i: np.ndarray, stats: _GaussianStats, config: PPMxConfig) -> float:
    """Compute log predictive probability of adding x_i to a cluster.
    
    Parameters
    ----------
    x_i : np.ndarray
        Point to add
    stats : _GaussianStats
        Current cluster statistics
    config : PPMxConfig
        Prior hyperparameters
        
    Returns
    -------
    float
        Log predictive probability
    """
    # Compute marginal likelihood before and after adding point
    log_marg_before = _log_marginal_x(stats, config)
    stats_after = _add_point(stats, x_i)
    log_marg_after = _log_marginal_x(stats_after, config)
    
    return log_marg_after - log_marg_before


# ==============================================================================
# Partition representation
# ==============================================================================

class _Partition:
    """Internal representation of a partition with sufficient statistics.
    
    Attributes
    ----------
    assignments : np.ndarray
        Cluster assignment for each observation
    clusters : dict[int, list[int]]
        Mapping from cluster label to list of observation indices
    stats : dict[int, _GaussianStats]
        Mapping from cluster label to sufficient statistics
    """
    
    def __init__(self, n: int, d: int):
        """Initialize empty partition.
        
        Parameters
        ----------
        n : int
            Number of observations
        d : int
            Dimensionality
        """
        self.n = n
        self.d = d
        self.assignments = np.zeros(n, dtype=int)
        self.clusters = {}
        self.stats = {}
    
    def remove(self, i: int):
        """Remove observation i from its current cluster.
        
        Parameters
        ----------
        i : int
            Observation index
        """
        k = self.assignments[i]
        self.clusters[k].remove(i)
        
        # Update stats (will be done by caller who has access to X)
        # This method just updates the cluster membership
        
    def add(self, i: int, k: int, x_i: np.ndarray):
        """Add observation i to cluster k.
        
        Parameters
        ----------
        i : int
            Observation index
        k : int
            Cluster label
        x_i : np.ndarray
            Observation covariates
        """
        self.assignments[i] = k
        
        if k not in self.clusters:
            self.clusters[k] = []
            self.stats[k] = _empty_stats(self.d)
        
        self.clusters[k].append(i)
        self.stats[k] = _add_point(self.stats[k], x_i)
    
    def create_new(self, i: int, x_i: np.ndarray) -> int:
        """Create a new cluster for observation i.
        
        Parameters
        ----------
        i : int
            Observation index
        x_i : np.ndarray
            Observation covariates
            
        Returns
        -------
        int
            New cluster label
        """
        # Find next available label
        if len(self.clusters) == 0:
            k_new = 0
        else:
            k_new = max(self.clusters.keys()) + 1
        
        self.add(i, k_new, x_i)
        return k_new
    
    def cleanup(self):
        """Remove empty clusters and relabel to be compact."""
        # Remove empty clusters
        empty_clusters = [k for k, members in self.clusters.items() if len(members) == 0]
        for k in empty_clusters:
            del self.clusters[k]
            del self.stats[k]
        
        # Relabel to be compact
        if len(self.clusters) == 0:
            return
        
        old_labels = sorted(self.clusters.keys())
        if old_labels == list(range(len(old_labels))):
            return  # Already compact
        
        # Create mapping
        label_map = {old: new for new, old in enumerate(old_labels)}
        
        # Update assignments
        for i in range(self.n):
            self.assignments[i] = label_map[self.assignments[i]]
        
        # Update clusters and stats
        new_clusters = {}
        new_stats = {}
        for old_k, new_k in label_map.items():
            new_clusters[new_k] = self.clusters[old_k]
            new_stats[new_k] = self.stats[old_k]
        
        self.clusters = new_clusters
        self.stats = new_stats


# ==============================================================================
# Sampler
# ==============================================================================

class PPMxSampler:
    """PPMx Gibbs sampler for Bayesian clustering.
    
    Attributes
    ----------
    data : PPMxData
        Data container
    config : PPMxConfig
        Configuration
    partition : _Partition
        Current partition state
    assignments_chain : list[np.ndarray]
        Stored partition assignments (post burn-in, thinned)
    n_clusters_chain : list[int]
        Stored cluster counts
    """
    
    def __init__(self, data: PPMxData, config: PPMxConfig):
        """Initialize sampler.
        
        Parameters
        ----------
        data : PPMxData
            Data container
        config : PPMxConfig
            Configuration
        """
        self.data = data
        self.config = config
        self.partition = _Partition(data.n, data.d)
        self.assignments_chain = []
        self.n_clusters_chain = []
        
        _set_random_seed(config.random_state)
    
    def initialize(self):
        """Initialize partition with each observation in its own cluster."""
        for i in range(self.data.n):
            self.partition.create_new(i, self.data.X[i])
    
    def step(self):
        """Perform one Gibbs sampling iteration."""
        # Iterate over all observations in random order
        indices = np.random.permutation(self.data.n)
        
        for i in indices:
            x_i = self.data.X[i]
            
            # Remove observation i
            k_old = self.partition.assignments[i]
            self.partition.remove(i)
            self.partition.stats[k_old] = _remove_point(self.partition.stats[k_old], x_i)
            
            # Compute log probabilities for each existing cluster and new cluster
            log_probs = []
            cluster_labels = []
            
            # Existing clusters
            for k, members in self.partition.clusters.items():
                if len(members) == 0:
                    continue  # Skip empty clusters
                
                size = len(members)
                log_cohesion = _log_cohesion(size, self.config.alpha)
                log_predictive = _log_predictive_x(x_i, self.partition.stats[k], self.config)
                
                log_probs.append(log_cohesion + log_predictive)
                cluster_labels.append(k)
            
            # New cluster
            log_cohesion_new = np.log(self.config.alpha)
            empty_stats = _empty_stats(self.data.d)
            log_predictive_new = _log_predictive_x(x_i, empty_stats, self.config)
            
            log_probs.append(log_cohesion_new + log_predictive_new)
            cluster_labels.append(-1)  # Sentinel for new cluster
            
            # Sample cluster assignment
            log_probs = np.array(log_probs)
            probs = _softmax_log(log_probs)
            k_new_idx = np.random.choice(len(probs), p=probs)
            k_new = cluster_labels[k_new_idx]
            
            # Add observation to selected cluster
            if k_new == -1:
                self.partition.create_new(i, x_i)
            else:
                self.partition.add(i, k_new, x_i)
            
        # Cleanup empty clusters
        self.partition.cleanup()
    
    def run(self):
        """Run the Gibbs sampler for specified iterations.
        
        Returns
        -------
        dict
            Dictionary with:
            - 'assignments': np.ndarray of shape (n_samples, n)
            - 'n_clusters': np.ndarray of shape (n_samples,)
        """
        self.initialize()
        
        for iteration in range(self.config.n_iter):
            self.step()
            
            # Store samples after burn-in with thinning
            if iteration >= self.config.burn_in and (iteration - self.config.burn_in) % self.config.thin == 0:
                self.assignments_chain.append(self.partition.assignments.copy())
                self.n_clusters_chain.append(len(self.partition.clusters))
        
        return {
            'assignments': np.array(self.assignments_chain),
            'n_clusters': np.array(self.n_clusters_chain)
        }


# ==============================================================================
# Diagnostics
# ==============================================================================

def posterior_similarity_matrix(assignments_chain: np.ndarray) -> np.ndarray:
    """Compute posterior co-clustering probability matrix.
    
    Parameters
    ----------
    assignments_chain : np.ndarray
        Array of shape (n_samples, n) containing cluster assignments
        
    Returns
    -------
    np.ndarray
        Matrix of shape (n, n) where entry (i, j) is the proportion of samples
        in which observations i and j are in the same cluster
    """
    n_samples, n = assignments_chain.shape
    similarity = np.zeros((n, n))
    
    for sample in range(n_samples):
        assignments = assignments_chain[sample]
        for i in range(n):
            for j in range(i, n):
                if assignments[i] == assignments[j]:
                    similarity[i, j] += 1
                    if i != j:
                        similarity[j, i] += 1
    
    similarity /= n_samples
    return similarity


def map_partition(assignments_chain: np.ndarray) -> np.ndarray:
    """Find Maximum A Posteriori (MAP) partition.
    
    Parameters
    ----------
    assignments_chain : np.ndarray
        Array of shape (n_samples, n) containing cluster assignments
        
    Returns
    -------
    np.ndarray
        MAP partition assignment of shape (n,)
    """
    n_samples = assignments_chain.shape[0]
    
    # Count occurrences of each unique partition
    # We'll use a string representation for hashing
    partition_counts = {}
    
    for sample in range(n_samples):
        # Relabel to canonical form
        partition = _relabel_clusters(assignments_chain[sample])
        partition_key = tuple(partition)
        
        if partition_key not in partition_counts:
            partition_counts[partition_key] = 0
        partition_counts[partition_key] += 1
    
    # Find most frequent partition
    map_key = max(partition_counts, key=partition_counts.get)
    return np.array(map_key)
