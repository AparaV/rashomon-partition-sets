"""
PPMx wrapper using R's ppmSuite package for 50-100x speedup.

This module provides a Python interface to R's optimized ppmSuite implementation
while maintaining compatibility with the existing Python PPMx interface.
"""

import numpy as np
from collections import Counter
# import warnings
from scipy.special import gammaln


try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.rlike.container import NamedList
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

CONSIM_ENUM = {
    'nn': 1
}


class PPMxR:
    """
    Product Partition Model with Covariates using R's ppmSuite backend.

    This class provides a Python interface matching the native Python PPMx implementation
    but uses R's highly optimized ppmSuite package for 50-100x speedup.

    Maintains the same interface as baselines.ppmx.PPMx for drop-in compatibility.
    """

    def __init__(self, n_iter=5000, burnin=1000, thin=2,
                 alpha=1.0, cohesion=1,
                 similarity_weight=0.5, similarity_bandwidth=1.0,
                 consim='nn',
                 use_adaptive_proposals=True,
                 M=1.0,
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
        cohesion : int, default=1
            Cohesion function type:
            1 = Dirichlet Process: c(S) = M * (|S| - 1)!
            2 = Uniform: c(S) = 1
        similarity_weight : float in [0, 1], default=1
            Weight for covariate similarity (0=ignore, 1=full)
        consim: str, default='nn'
            Values are 'nn' and 'nig'
        use_adaptive_proposals : bool, default=True
            Whether to adaptively adjust proposal probabilities during burnin
            (Note: R implementation may not support this)
        similarity_bandwidth : float, default=1.0
            Bandwidth for Gaussian kernel similarity
        M : float, default=1.0
            Cohesion precision parameter for prior computation
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
        self.cohesion = cohesion
        # self.cohesion = 2
        # self.similarity_weight = similarity_weight
        self.similarity_weight = 1
        self.consim = CONSIM_ENUM[consim]
        self.similarity_bandwidth = similarity_bandwidth
        self.use_adaptive_proposals = use_adaptive_proposals
        self.M = M
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
        self.log_likelihoods_ = None
        self.similarity_matrix_ = None

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

    # def _cohesion_str_to_int(self, cohesion_str):
    #     """
    #     Map cohesion string to R ppmSuite integer code.

    #     Parameters
    #     ----------
    #     cohesion_str : str
    #         Python cohesion type ('gaussian' or 'normal-gamma')

    #     Returns
    #     -------
    #     cohesion_int : int
    #         R ppmSuite cohesion code
    #     """
    #     cohesion_map = {
    #         'gaussian': 1,
    #         'normal-gamma': 2
    #     }
    #     if cohesion_str not in cohesion_map:
    #         warnings.warn(
    #             f"Unknown cohesion '{cohesion_str}', defaulting to 'gaussian'",
    #             UserWarning
    #         )
    #         return 1
    #     return cohesion_map[cohesion_str]

    def _extract_partitions_and_likelihoods_from_r(self, r_result):
        """
        Extract partition samples and log likelihoods from R ppmSuite result.

        R's PPMx operates on observations and returns observation-level partitions.
        We convert these to policy-level partitions by taking the cluster assignment
        of the first observation for each policy.

        Parameters
        ----------
        r_result : rpy2 object
            Result from gaussian_ppmx() call

        Returns
        -------
        partitions : list of np.ndarray
            List of policy-level partition samples (each shape (n_policies,))
        log_likelihoods : np.ndarray
            Log likelihoods for each sample (shape (n_samples,))
        """
        try:
            partitions_obs = None
            like_matrix = None
            available_names = []

            for named_item in r_result.items():
                item_name = named_item.name
                available_names.append(item_name)
                if item_name == "Si":
                    Si = named_item.value  # shape: (n_samples, n_observations)
                    partitions_obs = [Si[i, :].astype(int)-1 for i in range(Si.shape[0])]  # R uses 1-based indexing
                elif item_name == "like":
                    like_matrix = named_item.value  # shape: (n_samples, n_observations)

            if partitions_obs is None:
                raise ValueError(f"Could not find partition information (Si) in R result. Available names: {available_names}")

            if like_matrix is None:
                raise ValueError(f"Could not find likelihood matrix (like) in R result. Available names: {available_names}")

            # Convert observation-level partitions to policy-level partitions
            partitions = []
            for partition_obs in partitions_obs:
                partition_policy = np.zeros(self.n_policies_, dtype=int)
                for policy_id in range(self.n_policies_):
                    obs_indices = self.policy_to_obs_[policy_id]
                    if len(obs_indices) > 0:
                        partition_policy[policy_id] = partition_obs[obs_indices[0]]
                partitions.append(partition_policy)

            # Sum log likelihoods across observations for each sample
            log_likelihoods = np.sum(np.log(like_matrix + 1e-300), axis=1)

            return partitions, log_likelihoods

        except Exception as e:
            raise RuntimeError(f"Failed to extract partitions and likelihoods from R result: {e}")

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
        # overall_mean = np.mean(chain_means, axis=0)  # (n_features,)

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

    def _cohesion_function(self, cluster_size):
        """
        Compute log cohesion function c(S) = M * (|S| - 1)!.

        Parameters
        ----------
        cluster_size : int
            Size of the cluster

        Returns
        -------
        cohesion : float
            Log cohesion value
        """
        if cluster_size == 0:
            return 0.0
        elif cluster_size == 1:
            return self.M
        else:
            if self.cohesion == 1:
                # Dirichlet Process cohesion
                # Use log-gamma for numerical stability: log(c(S)) = log(M) + log((|S|-1)!)
                log_cohesion = np.log(self.M) + gammaln(cluster_size)
            elif self.cohesion == 2:
                log_cohesion = 0.0  # log(1) = 0
            # Cap to prevent overflow
            return min(log_cohesion, 700)

    def _compute_similarity_matrix(self, X_policy):
        """
        Compute pairwise similarity matrix using Gaussian kernel.

        Parameters
        ----------
        X_policy : np.ndarray, shape (n_policies, n_features)
            Policy features

        Returns
        -------
        similarity_matrix : np.ndarray, shape (n_policies, n_policies)
            Symmetric similarity matrix with diagonal = 1
        """
        n_policies = X_policy.shape[0]
        similarity_matrix = np.zeros((n_policies, n_policies))

        for i in range(n_policies):
            for j in range(i, n_policies):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    dist_sq = np.sum((X_policy[i] - X_policy[j]) ** 2)
                    sim = np.exp(-dist_sq / (2 * self.similarity_bandwidth ** 2))
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        return similarity_matrix

    def _compute_prior_for_partition(self, partition, X_policy):
        """
        Compute log prior for a given partition.

        log p(partition | X) = k * log(alpha) + sum_i log(c(S_i)) + similarity_terms

        Parameters
        ----------
        partition : np.ndarray, shape (n_policies,)
            Cluster assignments
        X_policy : np.ndarray, shape (n_policies, n_features)
            Policy features

        Returns
        -------
        log_prior : float
            Log prior value
        """
        unique_clusters = np.unique(partition)
        k = len(unique_clusters)

        # Term 1: k * log(alpha)
        log_prior = k * np.log(self.alpha)

        # Term 2: Sum of log cohesion for each cluster
        for cluster_id in unique_clusters:
            cluster_size = np.sum(partition == cluster_id)
            log_cohesion = self._cohesion_function(cluster_size)
            log_prior += log_cohesion

        # Term 3: Similarity terms (if weight > 0)
        # if self.similarity_weight > 0:
        for cluster_id in unique_clusters:
            policies_in_cluster = np.where(partition == cluster_id)[0]
            if len(policies_in_cluster) > 1:
                # Add pairwise similarity for all pairs in cluster
                for i in range(len(policies_in_cluster)):
                    for j in range(i + 1, len(policies_in_cluster)):
                        pi = policies_in_cluster[i]
                        pj = policies_in_cluster[j]
                        sim = self.similarity_matrix_[pi, pj]
                        # log_prior += self.similarity_weight * np.log(sim + 1e-300)
                        log_prior += np.log(sim + 1e-300)

        return log_prior

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

        X_standardized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
        self.X_policy_ = X
        self.X_standardized_ = X_standardized
        self.n_policies_ = n_policies
        self.y_ = y

        # Compute and cache similarity matrix for prior computation
        self.similarity_matrix_ = self._compute_similarity_matrix(X_standardized)

        # Build observation-to-policy lookup
        self.policy_to_obs_ = {}
        for policy_id in range(n_policies):
            self.policy_to_obs_[policy_id] = np.where(self.D_ == policy_id)[0]

        # Map cohesion to R integer code
        # cohesion_int = self._cohesion_str_to_int(self.cohesion)
        # cohesion_int = self.cohesion

        # Map similarity parameters
        # R ppmSuite uses simParms vector: c(m0, s20, v, k, nu0, s20, l)
        # For now, use defaults and control via similarity_function and consim
        similarity_function = 1

        simParams_m0 = 0.0
        simParams_s20 = 1000.0
        simParams_v = 10000.0
        simParams_list = NamedList(
            names=['mu0', 's20', 'v'],
            seq=[simParams_m0, simParams_s20, simParams_v]
        )

        # Prepare R parameters
        r_params = {
            'cohesion': int(self.cohesion),
            'similarity_function': similarity_function,
            'consim': self.consim,
            'M': self.M,
            'draws': self.n_iter,  # R counts post-burnin draws
            'burn': self.burnin,
            'thin': self.thin,
            'simParms': simParams_list,
            'verbose': self.verbose
        }

        # Run multiple chains
        all_chains = []
        all_partition_samples = []
        all_n_clusters = []
        all_cluster_means = []
        all_log_likelihoods = []

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
                    ro.FloatVector(X_standardized.flatten('F')),  # Use Fortran order for R
                    nrow=n_policies,
                    ncol=X_standardized.shape[1]
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
            partitions, log_likelihoods = self._extract_partitions_and_likelihoods_from_r(r_result)
            cluster_means_list = self._extract_cluster_means_from_r(r_result, partitions)
            all_log_likelihoods.append(log_likelihoods)

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

        # Concatenate log likelihoods from all chains
        self.log_likelihoods_ = np.concatenate(all_log_likelihoods)

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

        log p(partition | y, X) = log p(y | partition) + log p(partition | X)
                                 = log_likelihood + log_prior

        The log likelihood comes from R's ppmSuite computation.
        The log prior is computed in Python using cohesion and similarity.

        Returns
        -------
        log_posteriors : np.ndarray, shape (n_samples,)
            Log posterior values for each sample
        """
        if self.log_likelihoods_ is None:
            raise ValueError("Model must be fit before computing log posteriors")

        n_samples = len(self.partition_samples_)
        log_posteriors = np.zeros(n_samples)

        for i in range(n_samples):
            partition = self.partition_samples_[i]
            log_likelihood = self.log_likelihoods_[i]
            log_prior = self._compute_prior_for_partition(partition, self.X_policy_)
            log_posteriors[i] = log_likelihood + log_prior

        return log_posteriors
