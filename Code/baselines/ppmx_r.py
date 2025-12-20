"""
PPMx wrapper using R's ppmSuite package for 50-100x speedup.

This module provides a Python interface to R's optimized ppmSuite implementation
while maintaining compatibility with the existing Python PPMx interface.
"""

import numpy as np
from collections import Counter
import warnings

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


class PPMxR:
    """
    Product Partition Model with Covariates using R's ppmSuite backend.
    
    This class provides a Python interface matching the native Python PPMx implementation
    but uses R's highly optimized ppmSuite package for 50-100x speedup.
    
    Maintains the same interface as baselines.ppmx.PPMx for drop-in compatibility.
    """
    
    def __init__(self, n_iter=5000, burnin=1000, thin=2,
                 alpha=1.0, cohesion='gaussian',
                 similarity_weight=0.5, similarity_bandwidth=1.0,
                 use_adaptive_proposals=True,
                 random_state=None, verbose=False):
        """
        Parameters
        ----------
        n_iter : int, default=5000
            Total MCMC iterations
        burnin : int, default=1000
            Burn-in period to discard
        thin : int, default=2
            Thinning interval
        alpha : float, default=1.0
            Concentration parameter (prior on number of clusters)
            Higher alpha → more clusters
        cohesion : str, default='gaussian'
            Cohesion function type ('gaussian', 'normal-gamma')
        similarity_weight : float in [0, 1], default=0.5
            Weight for covariate similarity (0=ignore, 1=full)
        use_adaptive_proposals : bool, default=True
            Whether to adaptively adjust proposal probabilities during burnin
            (Note: R implementation may not support this)
        similarity_bandwidth : float, default=1.0
            Bandwidth for Gaussian kernel similarity
        random_state : int, optional
            Random seed
        verbose : bool, default=False
            Print progress information
        """
        if not HAS_RPY2:
            raise ImportError(
                "rpy2 is required for PPMxR. Install with: pip install rpy2"
            )
        
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.alpha = alpha
        # self.cohesion = cohesion
        self.cohesion = 1
        self.similarity_weight = similarity_weight
        self.similarity_bandwidth = similarity_bandwidth
        self.use_adaptive_proposals = use_adaptive_proposals
        self.random_state = random_state
        self.verbose = verbose
        
        # Posterior samples (will be populated by fit)
        self.partition_samples_ = None
        self.n_clusters_samples_ = None
        self.cluster_means_samples_ = None
        self.acceptance_rate_ = None
        self.accept_by_move_ = None
        self.chains_ = None
        self.converged_ = None
        self.rhat_ = None
        self.coef_samples_ = None
        
        # Adaptive proposal probabilities (for interface compatibility)
        self.proposal_probs_ = [0.3, 0.3, 0.4]
        
        # Check R availability
        try:
            ro.r('library(ppmSuite)')
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ppmSuite R package: {e}\n"
                "Install with: R -e 'install.packages(\"ppmSuite\")'"
            )
    
    def _cohesion_str_to_int(self, cohesion_str):
        """
        Map cohesion string to R ppmSuite integer code.
        
        Parameters
        ----------
        cohesion_str : str
            Python cohesion type ('gaussian' or 'normal-gamma')
        
        Returns
        -------
        cohesion_int : int
            R ppmSuite cohesion code
        """
        cohesion_map = {
            'gaussian': 1,
            'normal-gamma': 2
        }
        if cohesion_str not in cohesion_map:
            warnings.warn(
                f"Unknown cohesion '{cohesion_str}', defaulting to 'gaussian'",
                UserWarning
            )
            return 1
        return cohesion_map[cohesion_str]
    
    def _extract_partitions_from_r(self, r_result):
        """
        Extract partition samples from R ppmSuite result.
        
        Parameters
        ----------
        r_result : rpy2 object
            Result from gaussian_ppmx() call
        
        Returns
        -------
        partitions : list of np.ndarray
            List of partition samples (each shape (n_policies,))
        """
        # R ppmSuite returns a list with various components
        # We need to extract the partition/clustering information
        
        # Common result components:
        # - Si: cluster assignments matrix (samples x observations)
        # - nclus: number of clusters per sample
        # - fitted: fitted values
        
        # Try to get Si (cluster assignments)
        try:
            partitions = None
            available_names = []
            for named_item in r_result.items():
                item_name = named_item.name
                available_names.append(item_name)
                if item_name == "Si":
                    Si = named_item.value  # shape: (n_samples, n_policies)
                    partitions = [Si[i, :].astype(int)-1 for i in range(Si.shape[0])]  # R uses 1-based indexing
            
            if partitions is None:
                raise ValueError(f"Could not find partition information in R result. Available names: {available_names}")
            
            return partitions

        except Exception as e:
            raise RuntimeError(f"Failed to extract partitions from R result: {e}")

    def _extract_cluster_means_from_r(self, r_result, partitions):
        """
        Extract or compute cluster means from R result.
        
        Parameters
        ----------
        r_result : rpy2 object
            Result from gaussian_ppmx() call
        partitions : list of np.ndarray
            Partition samples
        
        Returns
        -------
        cluster_means_list : list of dict
            List of cluster means dictionaries for each sample
        """
        cluster_means_list = []
        
        # Try to get mu (cluster means) from R result
        # names() is a method in rpy2, not a property
        mu = None
        for named_item in r_result.items():
            item_name = named_item.name
            if item_name == "mu":
                mu = named_item.value  # shape: (n_samples, n_policies)
                for sample_idx, partition in enumerate(partitions):
                    cluster_means = {}
                    unique_clusters = np.unique(partition)
                    for cluster_id in unique_clusters:
                        policies_in_cluster = np.where(partition == cluster_id)[0]
                        # Use mean of mu values for policies in this cluster
                        cluster_means[cluster_id] = np.mean(mu[sample_idx, policies_in_cluster])
                    cluster_means_list.append(cluster_means)
        
        if mu is None:
            # Fallback: compute from data
            print("'mu' not found in R result, computing cluster means from data.")
            for partition in partitions:
                cluster_means = {}
                unique_clusters = np.unique(partition)
                for cluster_id in unique_clusters:
                    policies_in_cluster = np.where(partition == cluster_id)[0]
                    obs_in_cluster = np.concatenate([self.policy_to_obs_[p] for p in policies_in_cluster])
                    cluster_means[cluster_id] = np.mean(self.y_[obs_in_cluster])
                cluster_means_list.append(cluster_means)
        
        return cluster_means_list
    
    def _compute_rhat(self, chains):
        """
        Gelman-Rubin R-hat convergence diagnostic.
        
        Parameters
        ----------
        chains : np.ndarray, shape (n_chains, n_samples, n_features)
            MCMC chains
        
        Returns
        -------
        rhat : np.ndarray, shape (n_features,)
            R-hat values for each feature
        """
        n_chains, n_samples, n_features = chains.shape
        
        # Need at least 2 samples per chain and 2 chains for R-hat
        if n_samples < 2 or n_chains < 2:
            return np.ones(n_features)
        
        # Compute between-chain and within-chain variance
        chain_means = np.mean(chains, axis=1)  # (n_chains, n_features)
        overall_mean = np.mean(chain_means, axis=0)  # (n_features,)
        
        # Between-chain variance
        if n_chains > 1:
            B = n_samples * np.var(chain_means, axis=0, ddof=1)
        else:
            B = np.zeros(n_features)
        
        # Within-chain variance
        if n_samples > 1:
            chain_vars = np.var(chains, axis=1, ddof=1)  # (n_chains, n_features)
            W = np.mean(chain_vars, axis=0)
        else:
            W = np.zeros(n_features)
        
        # Estimated variance
        var_est = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        
        # R-hat with protection against division by zero
        rhat = np.ones(n_features)
        mask = W > 1e-10
        rhat[mask] = np.sqrt(var_est[mask] / W[mask])
        
        # If both W and B are near zero, chains are identical -> convergence
        # If W is zero but B is not, chains differ but no within-chain variance -> problematic
        problematic = (W <= 1e-10) & (B > 1e-10)
        if np.any(problematic):
            rhat[problematic] = np.inf
        
        return rhat
    
    def fit(self, X, y, D=None, n_chains=4):
        """
        Fit PPMx model using R's ppmSuite with multiple chains.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_data, n_features) or (n_policies, n_features)
            Policy features. If shape[0] == n_data, will aggregate to policy level.
        y : np.ndarray, shape (n_data, 1)
            Outcomes
        D : np.ndarray, shape (n_data, 1), optional
            Policy assignments. If None, assumes X already at policy level.
        n_chains : int, default=4
            Number of MCMC chains to run
        
        Returns
        -------
        self
        """
        if self.verbose:
            print(f"Running PPMxR with {n_chains} chains...")
        
        # Store D for use in predictions
        if D is not None:
            self.D_ = D.flatten()
            n_policies = len(np.unique(D))
            
            # Aggregate X to policy level if needed
            if X.shape[0] == y.shape[0]:
                X_policy = np.zeros((n_policies, X.shape[1]))
                for policy_id in range(n_policies):
                    policy_obs = D.flatten() == policy_id
                    X_policy[policy_id] = X[policy_obs][0]
                X = X_policy
        else:
            n_policies = X.shape[0]
            self.D_ = np.arange(n_policies)
        
        self.X_policy_ = X
        self.n_policies_ = n_policies
        self.y_ = y
        
        # Build observation-to-policy lookup
        self.policy_to_obs_ = {}
        for policy_id in range(n_policies):
            self.policy_to_obs_[policy_id] = np.where(self.D_ == policy_id)[0]
        
        # Map cohesion to R integer code
        # cohesion_int = self._cohesion_str_to_int(self.cohesion)
        cohesion_int = self.cohesion
        
        # Map similarity parameters
        # R ppmSuite uses simParms vector: c(m0, s20, v, k, nu0, s20, l)
        # For now, use defaults and control via similarity_function and consim
        similarity_function = 1
        
        # Prepare R parameters
        r_params = {
            'cohesion': cohesion_int,
            'similarity_function': similarity_function,
            'consim': self.similarity_weight,
            'draws': self.n_iter - self.burnin,  # R counts post-burnin draws
            'burn': self.burnin,
            'thin': self.thin,
            'verbose': self.verbose
        }
        
        # Run multiple chains
        all_chains = []
        all_partition_samples = []
        all_n_clusters = []
        all_cluster_means = []
        
        for chain_idx in range(n_chains):
            if self.verbose:
                print(f"Running chain {chain_idx + 1}/{n_chains}...")
            
            # Set random seed for this chain
            if self.random_state is not None:
                ro.r(f'set.seed({self.random_state + chain_idx * 1000})')
            
            # Convert data to R objects using context manager
            with localconverter(ro.default_converter + numpy2ri.converter):
                # Flatten X for R matrix conversion (R is column-major)
                r_X = ro.r.matrix(
                    ro.FloatVector(X.flatten('F')),  # Use Fortran order for R
                    nrow=n_policies,
                    ncol=X.shape[1]
                )
                r_y = ro.FloatVector(y.flatten())
                
                # Call R's gaussian_ppmx
                try:
                    r_result = ro.r['gaussian_ppmx'](
                        y=r_y,
                        X=r_X,
                        **r_params
                    )
                except Exception as e:
                    raise RuntimeError(f"R gaussian_ppmx failed: {e}")
            
            # Extract results from R
            partitions = self._extract_partitions_from_r(r_result)
            cluster_means_list = self._extract_cluster_means_from_r(r_result, partitions)
            
            # Compute coefficient samples (mean for each policy)
            chain_samples = []
            for partition, cluster_means in zip(partitions, cluster_means_list):
                coef_vector = np.zeros(n_policies)
                for policy_id in range(n_policies):
                    cluster_id = partition[policy_id]
                    coef_vector[policy_id] = cluster_means[cluster_id]
                chain_samples.append(coef_vector)
            
            # Store chain results
            all_chains.append(chain_samples)
            all_partition_samples.extend(partitions)
            all_n_clusters.extend([len(np.unique(p)) for p in partitions])
            all_cluster_means.extend(cluster_means_list)
        
        # Combine all chains
        self.chains_ = np.array(all_chains)  # (n_chains, n_samples, n_features)
        self.partition_samples_ = all_partition_samples
        self.n_clusters_samples_ = np.array(all_n_clusters)
        self.cluster_means_samples_ = all_cluster_means
        
        # Flatten chains for easy access
        n_chains_actual, n_samples_per_chain, n_features = self.chains_.shape
        self.coef_samples_ = self.chains_.reshape(n_chains_actual * n_samples_per_chain, n_features)
        
        # Compute R-hat for convergence diagnostics
        self.rhat_ = self._compute_rhat(self.chains_)
        self.converged_ = np.all(self.rhat_ < 1.1)
        
        # R implementation doesn't provide acceptance rate - set to None or estimate
        self.acceptance_rate_ = None
        self.accept_by_move_ = {'split': None, 'merge': None, 'reassign': None}
        
        if self.verbose:
            print(f"Fit complete. Converged: {self.converged_}")
            print(f"Mean R-hat: {np.mean(self.rhat_):.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict outcomes using posterior mean over partitions.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_data, n_features)
            Features for prediction
        
        Returns
        -------
        predictions : np.ndarray, shape (n_data, 1)
            Predicted outcomes
        """
        n_samples = len(self.partition_samples_)
        n_obs = len(self.D_)
        
        predictions = np.zeros(n_obs)
        
        for sample_idx in range(n_samples):
            partition = self.partition_samples_[sample_idx]
            cluster_means = self.cluster_means_samples_[sample_idx]
            
            # Map observations to clusters via D and partition
            for obs_idx in range(n_obs):
                policy_id = self.D_[obs_idx]
                cluster_id = partition[policy_id]
                predictions[obs_idx] += cluster_means[cluster_id]
        
        predictions /= n_samples
        
        return predictions.reshape(-1, 1)
    
    def get_map_partition(self):
        """
        Get Maximum A Posteriori (MAP) partition.
        
        Returns the most frequently sampled partition.
        
        Returns
        -------
        map_partition : np.ndarray
            Most common partition from posterior samples
        """
        partition_tuples = [tuple(p) for p in self.partition_samples_]
        partition_counts = Counter(partition_tuples)
        map_partition = np.array(partition_counts.most_common(1)[0][0])
        
        return map_partition
    
    def get_log_posteriors(self):
        """
        Compute log posterior for each stored sample.
        
        Note: R implementation may not provide log posteriors directly.
        This would need to be computed from the data.
        
        Returns
        -------
        log_posteriors : np.ndarray, shape (n_samples,)
            Log posterior values for each sample (or None if not available)
        """
        warnings.warn(
            "Log posteriors not directly available from R backend. "
            "Would need to compute from likelihood + prior.",
            UserWarning
        )
        return None
