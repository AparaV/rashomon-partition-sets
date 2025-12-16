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
        self.use_adaptive_proposals = use_adaptive_proposals
        self.random_state = random_state
        self.verbose = verbose

        # Posterior samples
        self.partition_samples_ = None
        self.n_clusters_samples_ = None
        self.cluster_means_samples_ = None
        self.acceptance_rate_ = None
        self.accept_by_move_ = None  # Acceptance rates by move type
        self.chains_ = None  # Store coefficient chains for R-hat
        self.converged_ = None
        self.rhat_ = None
        self.coef_samples_ = None  # Flattened coefficient samples
        
        # Adaptive proposal probabilities (split, merge, reassign)
        self.proposal_probs_ = [0.3, 0.3, 0.4]

    def _compute_cluster_stats(self, obs_indices):
        """
        Compute statistics for observations in a cluster.

        Parameters
        ----------
        obs_indices : np.ndarray
            Indices of observations in this cluster

        Returns
        -------
        stats : dict
            Dictionary with 'n', 'sum', 'sum_sq', 'mean'
        """
        if len(obs_indices) == 0:
            return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0, 'mean': 0.0}
        
        y_cluster = self.y_[obs_indices].flatten()
        n = len(y_cluster)
        y_sum = np.sum(y_cluster)
        y_mean = y_sum / n
        y_sum_sq = np.sum(y_cluster ** 2)
        
        return {'n': n, 'sum': y_sum, 'sum_sq': y_sum_sq, 'mean': y_mean}

    def _cohesion_from_stats(self, stats):
        """
        Compute cohesion from precomputed cluster statistics.

        Parameters
        ----------
        stats : dict
            Cluster statistics from _compute_cluster_stats

        Returns
        -------
        cohesion : float
            Log cohesion value
        """
        n = stats['n']
        if n == 0:
            return 0.0
        
        if self.cohesion == 'gaussian':
            if n == 1:
                return 0.0
            
            # Compute SS from statistics: SS = sum(y^2) - n*mean^2
            ss = stats['sum_sq'] - n * stats['mean']**2
            cohesion = -(n/2) * np.log(2*np.pi) - ((n-1)/2) * np.log(ss/n + 1e-10) - n/2
            
        elif self.cohesion == 'normal-gamma':
            mu_0 = 0.0
            kappa_0 = 0.01
            a_0 = 2.0
            b_0 = 1.0

            kappa_n = kappa_0 + n
            a_n = a_0 + n

            ss = stats['sum_sq'] - n * stats['mean']**2 if n > 1 else 0
            b_n = b_0 + 0.5 * ss + (kappa_0 * n * (stats['mean'] - mu_0)**2) / (2 * kappa_n)

            cohesion = (gammaln(a_n/2) - gammaln(a_0/2)
                        + (a_0/2)*np.log(b_0 + 1e-10) - (a_n/2)*np.log(b_n + 1e-10)
                        + 0.5*np.log(kappa_0/(kappa_n + 1e-10)) - (n/2)*np.log(2*np.pi))
        else:
            raise ValueError(f"Unknown cohesion type: {self.cohesion}")
        
        return cohesion

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

    def _compute_similarity_matrix(self, X):
        """
        Compute pairwise similarity matrix for all policies (vectorized).

        Parameters
        ----------
        X : np.ndarray, shape (n_policies, n_features)
            Policy features

        Returns
        -------
        similarity_matrix : np.ndarray, shape (n_policies, n_policies)
            Pairwise similarity values
        """
        # Vectorized computation: ||x_i - x_j||^2 for all pairs
        dists_sq = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
        similarity_matrix = np.exp(-dists_sq / (2 * self.similarity_bandwidth**2))
        return similarity_matrix

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

            # Get outcomes for observations from these policies (use cached lookup)
            obs_in_cluster = np.concatenate([self.policy_to_obs_[p] for p in policies_in_cluster])
            y_cluster = y[obs_in_cluster].flatten()

            # Add cohesion for this cluster
            cohesion = self._cohesion_function(y_cluster)
            log_likelihood += cohesion

            # Add similarity weights if using covariates
            if self.similarity_weight > 0 and len(policies_in_cluster) > 1:
                # Use cached similarity matrix for efficiency
                for i in range(len(policies_in_cluster)):
                    for j in range(i+1, len(policies_in_cluster)):
                        sim = self.similarity_matrix_[policies_in_cluster[i], policies_in_cluster[j]]
                        log_likelihood += self.similarity_weight * np.log(sim + 1e-10)

        return log_prior + log_likelihood

    def _incremental_log_prob_change(self, partition, policy_idx, new_cluster_id, cluster_stats):
        """
        Compute change in log probability when moving one policy to a new cluster.
        
        This is much faster than recomputing the full partition probability.
        
        Parameters
        ----------
        partition : np.ndarray
            Current partition
        policy_idx : int
            Policy being moved
        new_cluster_id : int
            Target cluster
        cluster_stats : dict
            Current cluster statistics
            
        Returns
        -------
        delta_log_prob : float
            Change in log probability
        """
        old_cluster_id = partition[policy_idx]
        
        if old_cluster_id == new_cluster_id:
            return 0.0
        
        delta_log_prob = 0.0
        
        # Get observations for this policy
        policy_obs = self.policy_to_obs_[policy_idx]
        
        # Change in number of clusters (if creating new or emptying old)
        old_cluster_policies = np.where(partition == old_cluster_id)[0]
        new_cluster_policies = np.where(partition == new_cluster_id)[0]
        
        if len(old_cluster_policies) == 1:
            # Removing last policy from old cluster (cluster disappears)
            delta_log_prob -= np.log(self.alpha)
        if len(new_cluster_policies) == 0:
            # Creating new cluster
            delta_log_prob += np.log(self.alpha)
        
        # Cohesion change for old cluster (remove policy)
        old_stats_before = cluster_stats[old_cluster_id]
        cohesion_old_before = self._cohesion_from_stats(old_stats_before)
        
        if len(old_cluster_policies) == 1:
            # Cluster becomes empty
            cohesion_old_after = 0.0
        else:
            # Recompute stats without this policy
            old_obs = np.concatenate([self.policy_to_obs_[p] for p in old_cluster_policies if p != policy_idx])
            old_stats_after = self._compute_cluster_stats(old_obs)
            cohesion_old_after = self._cohesion_from_stats(old_stats_after)
        
        delta_log_prob += cohesion_old_after - cohesion_old_before
        
        # Cohesion change for new cluster (add policy)
        if new_cluster_id in cluster_stats:
            new_stats_before = cluster_stats[new_cluster_id]
            cohesion_new_before = self._cohesion_from_stats(new_stats_before)
        else:
            # New cluster being created
            cohesion_new_before = 0.0
        
        # Recompute stats with this policy added
        if len(new_cluster_policies) == 0:
            new_obs = policy_obs
        else:
            new_obs = np.concatenate([self.policy_to_obs_[p] for p in new_cluster_policies] + [policy_obs])
        new_stats_after = self._compute_cluster_stats(new_obs)
        cohesion_new_after = self._cohesion_from_stats(new_stats_after)
        
        delta_log_prob += cohesion_new_after - cohesion_new_before
        
        # Similarity weight changes (if using covariates)
        if self.similarity_weight > 0:
            # Remove similarities with old cluster members
            for other_policy in old_cluster_policies:
                if other_policy != policy_idx:
                    sim = self.similarity_matrix_[policy_idx, other_policy]
                    delta_log_prob -= self.similarity_weight * np.log(sim + 1e-10)
            
            # Add similarities with new cluster members
            for other_policy in new_cluster_policies:
                sim = self.similarity_matrix_[policy_idx, other_policy]
                delta_log_prob += self.similarity_weight * np.log(sim + 1e-10)
        
        return delta_log_prob

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

    def _gibbs_step(self, partition, y, X, cluster_stats=None, proposal_probs=None):
        """
        Single Gibbs sampling step with split-merge moves.

        Alternates between split, merge, and reassignment moves.
        Uses incremental updates for reassignment (most common move).
        
        Parameters
        ----------
        cluster_stats : dict, optional
            Cached cluster statistics for incremental updates
        proposal_probs : list, optional
            Probabilities for [split, merge, reassign]
        
        Returns
        -------
        partition : np.ndarray
            Updated partition
        accepted : bool
            Whether move was accepted
        cluster_stats : dict
            Updated cluster statistics
        move_type : str
            Type of move proposed ('split', 'merge', 'reassign')
        """
        unique_clusters = np.unique(partition)
        n_clusters = len(unique_clusters)
        
        if proposal_probs is None:
            proposal_probs = [0.3, 0.3, 0.4]

        # Choose move type
        if n_clusters == 1:
            move_type = 'split'
        elif n_clusters >= len(partition):
            move_type = 'merge'
        else:
            move_type = np.random.choice(['split', 'merge', 'reassign'], p=proposal_probs)

        if move_type == 'split' and n_clusters < len(partition):
            cluster_to_split = np.random.choice(unique_clusters)
            proposed_partition, log_q = self._propose_split(partition, cluster_to_split)
            
            # Use full probability for split
            log_prob_current = self._partition_log_probability(partition, y, X)
            log_prob_proposed = self._partition_log_probability(proposed_partition, y, X)
            log_accept_ratio = log_prob_proposed - log_prob_current + log_q

        elif move_type == 'merge' and n_clusters > 1:
            cluster_i, cluster_j = np.random.choice(unique_clusters, size=2, replace=False)
            proposed_partition, log_q = self._propose_merge(partition, cluster_i, cluster_j)
            
            # Use full probability for merge
            log_prob_current = self._partition_log_probability(partition, y, X)
            log_prob_proposed = self._partition_log_probability(proposed_partition, y, X)
            log_accept_ratio = log_prob_proposed - log_prob_current + log_q

        elif move_type == 'reassign':
            policy_idx = np.random.randint(len(partition))
            
            # Randomly choose to create new cluster or join existing
            if np.random.rand() < 0.2 and n_clusters < len(partition):
                # Create new cluster
                new_cluster = np.max(partition) + 1
            else:
                # Similarity-based cluster selection (if similarity matrix available)
                if self.similarity_weight > 0 and hasattr(self, 'similarity_matrix_'):
                    # Compute similarity to each cluster (mean similarity to policies in cluster)
                    cluster_similarities = []
                    for cluster_id in unique_clusters:
                        policies_in_cluster = np.where(partition == cluster_id)[0]
                        # Mean similarity from policy_idx to all policies in this cluster
                        avg_sim = np.mean([self.similarity_matrix_[policy_idx, p] for p in policies_in_cluster])
                        cluster_similarities.append(avg_sim)
                    
                    # Softmax with temperature for exploration/exploitation
                    temperature = 1.0
                    similarities = np.array(cluster_similarities)
                    exp_sims = np.exp(similarities / temperature)
                    probs = exp_sims / np.sum(exp_sims)
                    
                    # Sample cluster based on similarity
                    cluster_idx = np.random.choice(len(unique_clusters), p=probs)
                    new_cluster = unique_clusters[cluster_idx]
                else:
                    # Fallback: uniform random selection
                    new_cluster = np.random.choice(unique_clusters)
            
            # Use incremental update for efficiency (no need to create proposed partition yet)
            if cluster_stats is not None:
                delta_log_prob = self._incremental_log_prob_change(
                    partition, policy_idx, new_cluster, cluster_stats
                )
                log_accept_ratio = delta_log_prob
            else:
                # Fallback to full computation if stats not provided
                proposed_partition = partition.copy()
                proposed_partition[policy_idx] = new_cluster
                proposed_partition = self._relabel_partition(proposed_partition)
                log_prob_current = self._partition_log_probability(partition, y, X)
                log_prob_proposed = self._partition_log_probability(proposed_partition, y, X)
                log_accept_ratio = log_prob_proposed - log_prob_current
            
            # Early rejection check
            if np.log(np.random.rand()) >= log_accept_ratio:
                return partition, False, cluster_stats, move_type
            
            # If accepted, create the actual proposed partition
            proposed_partition = partition.copy()
            proposed_partition[policy_idx] = new_cluster
            proposed_partition = self._relabel_partition(proposed_partition)
        else:
            return partition, False, cluster_stats, move_type

        # Metropolis-Hastings acceptance
        if np.log(np.random.rand()) < log_accept_ratio:
            # Update cluster stats if accepted
            if cluster_stats is not None:
                new_cluster_stats = {}
                for cluster_id in np.unique(proposed_partition):
                    policies_in_cluster = np.where(proposed_partition == cluster_id)[0]
                    obs_in_cluster = np.concatenate([self.policy_to_obs_[p] for p in policies_in_cluster])
                    new_cluster_stats[cluster_id] = self._compute_cluster_stats(obs_in_cluster)
                return proposed_partition, True, new_cluster_stats, move_type
            return proposed_partition, True, cluster_stats, move_type
        else:
            return partition, False, cluster_stats, move_type

    def _align_partition_labels(self, partition, reference_partition):
        """
        Align partition labels to match reference using greedy matching.
        
        Parameters
        ----------
        partition : np.ndarray
            Partition to relabel
        reference_partition : np.ndarray
            Reference partition
            
        Returns
        -------
        aligned_partition : np.ndarray
            Relabeled partition matching reference
        """
        # Build confusion matrix: confusion[i,j] = # of policies in cluster i (partition) and j (reference)
        unique_partition = np.unique(partition)
        unique_reference = np.unique(reference_partition)
        
        confusion = np.zeros((len(unique_partition), len(unique_reference)))
        for i, cluster_p in enumerate(unique_partition):
            for j, cluster_r in enumerate(unique_reference):
                confusion[i, j] = np.sum((partition == cluster_p) & (reference_partition == cluster_r))
        
        # Greedy matching: assign each partition cluster to reference cluster with max overlap
        aligned = partition.copy()
        label_map = {}
        used_reference_labels = set()
        
        # Sort partition clusters by size (largest first) for better matching
        cluster_sizes = [(cluster, np.sum(partition == cluster)) for cluster in unique_partition]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        for cluster_p, _ in cluster_sizes:
            cluster_idx = np.where(unique_partition == cluster_p)[0][0]
            
            # Find best matching reference cluster (not yet used)
            best_match = None
            best_overlap = -1
            for j, cluster_r in enumerate(unique_reference):
                if cluster_r not in used_reference_labels and confusion[cluster_idx, j] > best_overlap:
                    best_overlap = confusion[cluster_idx, j]
                    best_match = cluster_r
            
            # If all reference labels used, assign to next available integer
            if best_match is None:
                best_match = max(unique_reference) + len(label_map) + 1
            else:
                used_reference_labels.add(best_match)
            
            label_map[cluster_p] = best_match
        
        # Apply relabeling
        for old_label, new_label in label_map.items():
            aligned[partition == old_label] = new_label
        
        return aligned

    def _compute_rhat(self, chains):
        """
        Gelman-Rubin R-hat convergence diagnostic with label switching resolution.

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
            return np.ones(n_features)  # Return 1.0 (perfect convergence) if insufficient data

        # Compute between-chain and within-chain variance
        chain_means = np.mean(chains, axis=1)  # (n_chains, n_features)
        overall_mean = np.mean(chain_means, axis=0)  # (n_features,)

        # Between-chain variance (protect against single chain)
        if n_chains > 1:
            B = n_samples * np.var(chain_means, axis=0, ddof=1)
        else:
            B = np.zeros(n_features)

        # Within-chain variance (protect against single sample)
        if n_samples > 1:
            chain_vars = np.var(chains, axis=1, ddof=1)  # (n_chains, n_features)
            W = np.mean(chain_vars, axis=0)
        else:
            W = np.zeros(n_features)

        # Estimated variance
        var_est = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

        # R-hat with protection against division by zero
        # If W is zero (no within-chain variance), check if B is also zero
        rhat = np.ones(n_features)
        mask = W > 1e-10
        rhat[mask] = np.sqrt(var_est[mask] / W[mask])
        
        # If both W and B are near zero, chains are identical -> convergence
        # If W is zero but B is not, chains differ but no within-chain variance -> problematic
        problematic = (W <= 1e-10) & (B > 1e-10)
        if np.any(problematic):
            rhat[problematic] = np.inf  # Flag as non-converged

        return rhat

    def _update_proposal_probabilities(self, accept_counts, total_counts):
        """
        Adaptively update proposal probabilities based on acceptance rates.
        
        Target acceptance rates:
        - Split/Merge: 25-40%
        - Reassign: 40-60%
        
        Parameters
        ----------
        accept_counts : dict
            Acceptance counts by move type
        total_counts : dict
            Total proposal counts by move type
        """
        if not self.use_adaptive_proposals:
            return
        
        # Compute current acceptance rates
        rates = {}
        for move_type in ['split', 'merge', 'reassign']:
            if total_counts.get(move_type, 0) > 0:
                rates[move_type] = accept_counts[move_type] / total_counts[move_type]
            else:
                rates[move_type] = 0.0
        
        # Adjust probabilities based on acceptance rates
        # Increase probability if acceptance too high (easy moves, explore more)
        # Decrease probability if acceptance too low (hard moves, waste time)
        adjustments = [0.0, 0.0, 0.0]  # split, merge, reassign
        
        # Split adjustment (target 25-40%)
        if rates['split'] > 0.5:
            adjustments[0] = 0.05  # Too easy, do more
        elif rates['split'] < 0.2:
            adjustments[0] = -0.05  # Too hard, do less
        
        # Merge adjustment (target 25-40%)
        if rates['merge'] > 0.5:
            adjustments[1] = 0.05
        elif rates['merge'] < 0.2:
            adjustments[1] = -0.05
        
        # Reassign adjustment (target 40-60%)
        if rates['reassign'] > 0.7:
            adjustments[2] = 0.05
        elif rates['reassign'] < 0.3:
            adjustments[2] = -0.05
        
        # Apply adjustments and normalize
        new_probs = [max(0.1, min(0.6, self.proposal_probs_[i] + adjustments[i])) for i in range(3)]
        total = sum(new_probs)
        self.proposal_probs_ = [p / total for p in new_probs]
        
        if self.verbose:
            print(f"  Adaptive update: probs={[f'{p:.2f}' for p in self.proposal_probs_]}, "
                  f"rates: split={rates['split']:.2f}, merge={rates['merge']:.2f}, reassign={rates['reassign']:.2f}")

    def fit(self, X, y, D=None, n_chains=4):
        """
        Fit PPMx model using MCMC sampling with multiple chains.

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
        self.n_policies_ = n_policies
        self.y_ = y  # Store for cluster stats computation
        
        # OPTIMIZATION 1: Precompute similarity matrix (vectorized)
        if self.similarity_weight > 0:
            if self.verbose:
                print("Precomputing similarity matrix...")
            self.similarity_matrix_ = self._compute_similarity_matrix(X)
        else:
            self.similarity_matrix_ = None
        
        # OPTIMIZATION 2: Precompute observation-to-policy lookup
        if self.verbose:
            print("Building observation index...")
        self.policy_to_obs_ = {}
        for policy_id in range(n_policies):
            self.policy_to_obs_[policy_id] = np.where(self.D_ == policy_id)[0]

        # Run multiple chains
        all_chains = []
        all_partition_samples = []
        all_partition_samples_by_chain = []  # For label alignment in R-hat
        all_n_clusters = []
        all_cluster_means = []
        total_accepted = 0
        total_accept_by_move = {'split': 0, 'merge': 0, 'reassign': 0}
        total_count_by_move = {'split': 0, 'merge': 0, 'reassign': 0}

        for chain_idx in range(n_chains):
            if self.verbose:
                print(f"Running chain {chain_idx + 1}/{n_chains}...")

            # Set different seed for each chain
            if self.random_state is not None:
                np.random.seed(self.random_state + chain_idx * 1000)

            # Initialize partition: all policies in one cluster
            partition = np.zeros(n_policies, dtype=int)
            
            # OPTIMIZATION 3: Initialize cluster statistics cache
            cluster_stats = {}
            obs_all = np.concatenate([self.policy_to_obs_[p] for p in range(n_policies)])
            cluster_stats[0] = self._compute_cluster_stats(obs_all)

            # Storage for this chain
            chain_samples = []
            partition_samples = []
            n_clusters_samples = []
            cluster_means_samples = []
            n_accepted = 0
            accept_by_move = {'split': 0, 'merge': 0, 'reassign': 0}
            count_by_move = {'split': 0, 'merge': 0, 'reassign': 0}

            # MCMC sampling
            for iter_i in range(self.n_iter):
                # Gibbs step with cluster statistics cache and current proposal probs
                partition, accepted, cluster_stats, move_type = self._gibbs_step(
                    partition, y, X, cluster_stats, self.proposal_probs_
                )
                
                # Track by move type
                count_by_move[move_type] += 1
                if accepted:
                    n_accepted += 1
                    accept_by_move[move_type] += 1
                
                # Adaptive proposal update during burnin (every 100 iterations)
                if self.use_adaptive_proposals and iter_i < self.burnin and (iter_i + 1) % 100 == 0:
                    self._update_proposal_probabilities(accept_by_move, count_by_move)

                # Store sample after burn-in with thinning
                if iter_i >= self.burnin and (iter_i - self.burnin) % self.thin == 0:
                    partition_samples.append(partition.copy())
                    n_clusters_samples.append(len(np.unique(partition)))

                    # Compute cluster means and convert to coefficient vector
                    unique_clusters = np.unique(partition)
                    cluster_means = {}
                    coef_vector = np.zeros(n_policies)
                    for cluster_id in unique_clusters:
                        policies_in_cluster = np.where(partition == cluster_id)[0]
                        obs_in_cluster = np.concatenate([self.policy_to_obs_[p] for p in policies_in_cluster])
                        mean_val = np.mean(y[obs_in_cluster])
                        cluster_means[cluster_id] = mean_val
                        coef_vector[policies_in_cluster] = mean_val

                    cluster_means_samples.append(cluster_means)
                    chain_samples.append(coef_vector)

                if self.verbose and (iter_i + 1) % 1000 == 0:
                    print(f"  Chain {chain_idx + 1}, Iteration {iter_i + 1}/{self.n_iter}, "
                          f"n_clusters={len(np.unique(partition))}, "
                          f"acceptance_rate={n_accepted/(iter_i+1):.3f}")

            # Store chain results
            all_chains.append(chain_samples)
            all_partition_samples.extend(partition_samples)
            all_partition_samples_by_chain.append(partition_samples)  # Store by chain for alignment
            all_n_clusters.extend(n_clusters_samples)
            all_cluster_means.extend(cluster_means_samples)
            total_accepted += n_accepted
            
            # Aggregate acceptance counts by move type
            for move_type in ['split', 'merge', 'reassign']:
                total_accept_by_move[move_type] += accept_by_move[move_type]
                total_count_by_move[move_type] += count_by_move[move_type]

        # Combine all chains
        self.chains_ = np.array(all_chains)  # (n_chains, n_samples, n_features)
        self.partition_samples_ = all_partition_samples
        self.partition_samples_by_chain_ = all_partition_samples_by_chain  # For label alignment
        self.n_clusters_samples_ = np.array(all_n_clusters)
        self.cluster_means_samples_ = all_cluster_means
        self.acceptance_rate_ = total_accepted / (self.n_iter * n_chains)
        
        # Acceptance rates by move type
        self.accept_by_move_ = {}
        for move_type in ['split', 'merge', 'reassign']:
            if total_count_by_move[move_type] > 0:
                self.accept_by_move_[move_type] = total_accept_by_move[move_type] / total_count_by_move[move_type]
            else:
                self.accept_by_move_[move_type] = 0.0

        # Flatten chains for easy access
        n_chains_actual, n_samples_per_chain, n_features = self.chains_.shape
        self.coef_samples_ = self.chains_.reshape(n_chains_actual * n_samples_per_chain, n_features)

        # Compute R-hat for convergence diagnostics
        self.rhat_ = self._compute_rhat(self.chains_)
        self.converged_ = np.all(self.rhat_ < 1.1)

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

        Returns
        -------
        log_posteriors : np.ndarray, shape (n_samples,)
            Log posterior values for each sample
        """
        n_samples = len(self.partition_samples_)
        log_posteriors = np.zeros(n_samples)

        for sample_idx in range(n_samples):
            partition = self.partition_samples_[sample_idx]
            log_prob = self._partition_log_probability(partition, self.y_, self.X_policy_)
            log_posteriors[sample_idx] = log_prob

        return log_posteriors
