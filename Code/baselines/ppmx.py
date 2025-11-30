"""
Product Partition Model with Covariates (PPMx) for policy clustering.

PPMx is a Bayesian nonparametric method that partitions policies into clusters
based on outcome similarity while incorporating policy features as covariates.
"""

import numpy as np
from scipy.special import gammaln
from collections import Counter


class PPMx:
    """
    Product Partition Model with Covariates for policy clustering.

    Partitions policies into pools based on outcome similarity,
    incorporating policy features as covariates.
    """

    def __init__(self, n_iter=5000, burnin=1000, thin=2,
                 alpha=1.0, cohesion='gaussian',
                 similarity_weight=0.5, similarity_bandwidth=1.0,
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
        similarity_bandwidth : float, default=1.0
            Bandwidth for Gaussian kernel similarity
        random_state : int, optional
            Random seed
        verbose : bool, default=False
            Print progress information
        """
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.alpha = alpha
        self.cohesion = cohesion
        self.similarity_weight = similarity_weight
        self.similarity_bandwidth = similarity_bandwidth
        self.random_state = random_state
        self.verbose = verbose

        # Posterior samples
        self.partition_samples_ = None
        self.n_clusters_samples_ = None
        self.cluster_means_samples_ = None
        self.acceptance_rate_ = None

    def _cohesion_function(self, y_cluster):
        """
        Compute cohesion for a cluster (higher = more cohesive).

        Parameters
        ----------
        y_cluster : np.ndarray, shape (n_obs,)
            Outcomes for policies in this cluster

        Returns
        -------
        cohesion : float
            Log cohesion value
        """
        n = len(y_cluster)
        if n == 0:
            return 0.0

        if self.cohesion == 'gaussian':
            # Simple Gaussian cohesion
            # C(S) = -(n/2) * log(2π) - (n-1)/2 * log(s^2) - n/2

            if n == 1:
                return 0.0

            y_mean = np.mean(y_cluster)
            ss = np.sum((y_cluster - y_mean) ** 2)

            # Add small constant to avoid log(0)
            cohesion = -(n/2) * np.log(2*np.pi) - ((n-1)/2) * np.log(ss/n + 1e-10) - n/2

        elif self.cohesion == 'normal-gamma':
            # Conjugate Normal-Gamma prior
            mu_0 = 0.0
            kappa_0 = 0.01
            a_0 = 2.0
            b_0 = 1.0

            kappa_n = kappa_0 + n
            # mu_n = (kappa_0 * mu_0 + n * np.mean(y_cluster)) / kappa_n
            a_n = a_0 + n

            ss = np.sum((y_cluster - np.mean(y_cluster)) ** 2) if n > 1 else 0
            b_n = b_0 + 0.5 * ss + (kappa_0 * n * (np.mean(y_cluster) - mu_0)**2) / (2 * kappa_n)

            cohesion = (gammaln(a_n/2) - gammaln(a_0/2)
                        + (a_0/2)*np.log(b_0 + 1e-10) - (a_n/2)*np.log(b_n + 1e-10)
                        + 0.5*np.log(kappa_0/(kappa_n + 1e-10)) - (n/2)*np.log(2*np.pi))
        else:
            raise ValueError(f"Unknown cohesion type: {self.cohesion}")

        return cohesion

    def _similarity_kernel(self, X_i, X_j):
        """
        Compute similarity between two policies based on features.

        Parameters
        ----------
        X_i, X_j : np.ndarray, shape (n_features,)
            Policy features

        Returns
        -------
        similarity : float in [0, 1]
            Higher = more similar
        """
        dist_sq = np.sum((X_i - X_j) ** 2)
        similarity = np.exp(-dist_sq / (2 * self.similarity_bandwidth**2))
        return similarity

    def _partition_log_probability(self, partition, y, X):
        """
        Compute log probability of a partition.

        log p(partition | y, X) ∝ log p(y | partition) + log p(partition | X)

        Parameters
        ----------
        partition : np.ndarray, shape (n_policies,)
            Cluster assignments for each policy
        y : np.ndarray, shape (n_data, 1)
            Outcomes
        X : np.ndarray, shape (n_policies, n_features)
            Policy features

        Returns
        -------
        log_prob : float
            Log probability
        """
        unique_clusters = np.unique(partition)
        n_clusters = len(unique_clusters)

        # Prior on partition: p(partition) ∝ α^k
        log_prior = n_clusters * np.log(self.alpha)

        # Likelihood: product of cohesions
        log_likelihood = 0.0

        for cluster_id in unique_clusters:
            # Get policies in this cluster
            policies_in_cluster = np.where(partition == cluster_id)[0]

            # Get outcomes for observations from these policies
            obs_in_cluster = np.where(np.isin(self.D_, policies_in_cluster))[0]
            y_cluster = y[obs_in_cluster].flatten()

            # Add cohesion for this cluster
            cohesion = self._cohesion_function(y_cluster)
            log_likelihood += cohesion

            # Add similarity weights if using covariates
            if self.similarity_weight > 0 and len(policies_in_cluster) > 1:
                for i in range(len(policies_in_cluster)):
                    for j in range(i+1, len(policies_in_cluster)):
                        sim = self._similarity_kernel(
                            X[policies_in_cluster[i]],
                            X[policies_in_cluster[j]]
                        )
                        log_likelihood += self.similarity_weight * np.log(sim + 1e-10)

        return log_prior + log_likelihood

    def _relabel_partition(self, partition):
        """Relabel cluster IDs to be contiguous 0, 1, 2, ..."""
        unique_clusters = np.unique(partition)
        relabeled = np.zeros_like(partition)
        for new_id, old_id in enumerate(unique_clusters):
            relabeled[partition == old_id] = new_id
        return relabeled

    def _propose_split(self, partition, cluster_to_split):
        """Propose splitting a cluster into two."""
        policies_in_cluster = np.where(partition == cluster_to_split)[0]

        if len(policies_in_cluster) < 2:
            return partition, 0.0

        new_partition = partition.copy()
        new_cluster_id = np.max(partition) + 1

        # Randomly assign to two new clusters
        split_mask = np.random.rand(len(policies_in_cluster)) < 0.5
        # Ensure at least one policy in each new cluster
        if not np.any(split_mask) or np.all(split_mask):
            split_mask[0] = True
            split_mask[1] = False

        new_partition[policies_in_cluster[split_mask]] = new_cluster_id
        new_partition = self._relabel_partition(new_partition)

        return new_partition, 0.0

    def _propose_merge(self, partition, cluster_i, cluster_j):
        """Propose merging two clusters."""
        new_partition = partition.copy()
        new_partition[partition == cluster_j] = cluster_i
        new_partition = self._relabel_partition(new_partition)

        return new_partition, 0.0

    def _gibbs_step(self, partition, y, X):
        """
        Single Gibbs sampling step with split-merge moves.

        Alternates between split, merge, and reassignment moves.
        """
        unique_clusters = np.unique(partition)
        n_clusters = len(unique_clusters)

        # Choose move type
        if n_clusters == 1:
            move_type = 'split'
        elif n_clusters >= len(partition):
            move_type = 'merge'
        else:
            move_type = np.random.choice(['split', 'merge', 'reassign'], p=[0.3, 0.3, 0.4])

        if move_type == 'split' and n_clusters < len(partition):
            cluster_to_split = np.random.choice(unique_clusters)
            proposed_partition, log_q = self._propose_split(partition, cluster_to_split)

        elif move_type == 'merge' and n_clusters > 1:
            cluster_i, cluster_j = np.random.choice(unique_clusters, size=2, replace=False)
            proposed_partition, log_q = self._propose_merge(partition, cluster_i, cluster_j)

        elif move_type == 'reassign':
            policy_idx = np.random.randint(len(partition))
            proposed_partition = partition.copy()

            # Randomly choose to create new cluster or join existing
            if np.random.rand() < 0.2 and n_clusters < len(partition):
                # Create new cluster
                new_cluster = np.max(partition) + 1
                proposed_partition[policy_idx] = new_cluster
            else:
                # Join existing cluster
                new_cluster = np.random.choice(unique_clusters)
                proposed_partition[policy_idx] = new_cluster

            proposed_partition = self._relabel_partition(proposed_partition)
            log_q = 0.0
        else:
            return partition, False

        # Metropolis-Hastings acceptance
        log_prob_current = self._partition_log_probability(partition, y, X)
        log_prob_proposed = self._partition_log_probability(proposed_partition, y, X)

        log_accept_ratio = log_prob_proposed - log_prob_current + log_q

        if np.log(np.random.rand()) < log_accept_ratio:
            return proposed_partition, True
        else:
            return partition, False

    def fit(self, X, y, D=None):
        """
        Fit PPMx model using MCMC sampling.

        Parameters
        ----------
        X : np.ndarray, shape (n_data, n_features) or (n_policies, n_features)
            Policy features. If shape[0] == n_data, will aggregate to policy level.
        y : np.ndarray, shape (n_data, 1)
            Outcomes
        D : np.ndarray, shape (n_data, 1), optional
            Policy assignments. If None, assumes X already at policy level.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Store D for use in cohesion computation
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

        # Initialize partition: all policies in one cluster
        partition = np.zeros(n_policies, dtype=int)

        # Storage for posterior samples
        partition_samples = []
        n_clusters_samples = []
        cluster_means_samples = []
        n_accepted = 0

        # MCMC sampling
        for iter_i in range(self.n_iter):
            # Gibbs step
            partition, accepted = self._gibbs_step(partition, y, X)
            if accepted:
                n_accepted += 1

            # Store sample after burn-in with thinning
            if iter_i >= self.burnin and (iter_i - self.burnin) % self.thin == 0:
                partition_samples.append(partition.copy())
                n_clusters_samples.append(len(np.unique(partition)))

                # Compute cluster means for this partition
                unique_clusters = np.unique(partition)
                cluster_means = {}
                for cluster_id in unique_clusters:
                    policies_in_cluster = np.where(partition == cluster_id)[0]
                    obs_in_cluster = np.where(np.isin(self.D_, policies_in_cluster))[0]
                    cluster_means[cluster_id] = np.mean(y[obs_in_cluster])
                cluster_means_samples.append(cluster_means)

            if self.verbose and (iter_i + 1) % 500 == 0:
                print(f"Iteration {iter_i + 1}/{self.n_iter}, "
                      f"n_clusters={len(np.unique(partition))}, "
                      f"acceptance_rate={n_accepted/(iter_i+1):.3f}")

        self.partition_samples_ = partition_samples
        self.n_clusters_samples_ = np.array(n_clusters_samples)
        self.cluster_means_samples_ = cluster_means_samples
        self.acceptance_rate_ = n_accepted / self.n_iter

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
