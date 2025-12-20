# PPMx Performance Bottlenecks Analysis

## Optimization Status

### ✅ COMPLETED
- **Fix #3: Conditional cleanup** - Added flag to only cleanup when empty clusters exist (minimal overhead reduction)
- **Fix #2: Vectorize marginal likelihood** - Removed Python loop over dimensions, vectorized computations (modest speedup)
- **Fix #4: Vectorize similarity matrix** - Used broadcasting for co-clustering matrix (**75x speedup**: 0.603s → 0.008s)

### 🔄 IN PROGRESS  
- None

### ⏳ TODO
- **Fix #1: Closed-form predictive** (CRITICAL - This is the main bottleneck, 10-100x speedup expected for main sampling loop)

## Performance Summary (Benchmarks)

| Optimization | Small (n=100) | Medium (n=500) | Similarity Matrix |
|--------------|---------------|----------------|-------------------|
| Baseline     | 1.385s        | 23.727s        | 0.603s           |
| After #3+#2  | 1.764s        | 24.761s        | 0.587s           |
| After #3+#2+#4 | 1.756s      | 24.661s        | **0.008s** (75x) |

**Note:** Main sampling loop (n=500) hasn't improved significantly yet because the critical bottleneck (Fix #1) remains unaddressed.

---

## Critical Bottlenecks (Major Impact)

### 1. **`_log_predictive_x` computes marginal likelihood twice per cluster** ⚠️⚠️⚠️
**Location:** Lines 360-367
**Issue:** For each observation and each cluster, we compute the full marginal likelihood twice:
- Once for `stats` (before adding point)
- Once for `stats_after` (after adding point)

**Impact:** O(n × k × d) per iteration, with expensive operations (log, gammaln, etc.)

**Current code:**
```python
def _log_predictive_x(x_i: np.ndarray, stats: _GaussianStats, config: PPMxConfig) -> float:
    log_marg_before = _log_marginal_x(stats, config)
    stats_after = _add_point(stats, x_i)
    log_marg_after = _log_marginal_x(stats_after, config)
    return log_marg_after - log_marg_before
```

**Solution:** Derive the closed-form predictive density directly without computing full marginal likelihoods.

### 2. **Python loop over dimensions in `_log_marginal_x`** ⚠️⚠️
**Location:** Lines 318-340
**Issue:** Non-vectorized loop over d dimensions

**Current code:**
```python
for j in range(d):
    x_bar_j = stats.sum_x[j] / n if n > 0 else 0.0
    ss_j = stats.sum_x2[j] - n * x_bar_j**2
    b_n_j = config.b0 + 0.5 * ss_j + 0.5 * (config.kappa0 * n / kappa_n) * (x_bar_j - config.mu0)**2
    log_lik_j = (...)
    log_lik += log_lik_j
```

**Solution:** Vectorize operations across dimensions using numpy arrays.

### 3. **`cleanup()` called every iteration** ⚠️⚠️
**Location:** Line 589 in `step()`
**Issue:** Full partition relabeling after every Gibbs iteration, including:
- Iterating over all n observations to update assignments
- Rebuilding cluster and stats dictionaries

**Impact:** O(n × k) every iteration, unnecessary when clusters don't change

**Solution:** Only cleanup when necessary (e.g., when empty clusters exist), or defer until end.

### 4. **`posterior_similarity_matrix` has O(n²) nested Python loops** ⚠️
**Location:** Lines 636-645
**Issue:** Triple nested Python loop (samples × n × n)

**Current code:**
```python
for sample in range(n_samples):
    assignments = assignments_chain[sample]
    for i in range(n):
        for j in range(i, n):
            if assignments[i] == assignments[j]:
                similarity[i, j] += 1
```

**Impact:** Very slow for large n (e.g., n=2560 in reff_4)

**Solution:** Vectorize using broadcasting or pairwise comparison.

## Moderate Bottlenecks

### 5. **List iteration over clusters in Gibbs step**
**Location:** Lines 558-567
**Issue:** Building Python lists for every observation in every iteration

**Solution:** Pre-allocate arrays or use more efficient data structures.

### 6. **`map_partition` uses tuple hashing**
**Location:** Lines 665-672
**Issue:** Creating tuple for each sample is slow for large n

**Solution:** Use integer-based hashing or comparison methods.

## Optimization Priority

1. **CRITICAL:** Fix `_log_predictive_x` to use closed-form (10-100x speedup)
2. **HIGH:** Vectorize `_log_marginal_x` across dimensions (2-5x speedup)
3. **HIGH:** Make `cleanup()` conditional (2-3x speedup)
4. **MEDIUM:** Vectorize `posterior_similarity_matrix` (10x speedup for diagnostics)
5. **LOW:** Optimize Gibbs loop structure

## Estimated Overall Speedup
Implementing fixes 1-3 could provide **20-50x overall speedup** for typical use cases.

## Implementation Notes

### For Fix #1 (Closed-form predictive):
The Student-t predictive density for Normal-Inverse-Gamma is:
```
log p(x_new | stats) = log_t(x_new; μ_n, Σ_n, 2*a_n)
```
where the predictive is a multivariate t-distribution. For diagonal covariance:
```
log p(x_new) = Σ_j log_t(x_new[j]; μ_n[j], σ²_n[j], 2*a_n)
```

This eliminates the need to compute full marginal likelihoods.

### For Fix #2 (Vectorize marginal):
```python
x_bar = stats.sum_x / n  # vectorized
ss = stats.sum_x2 - n * x_bar**2  # vectorized
b_n = config.b0 + 0.5 * ss + 0.5 * (config.kappa0 * n / kappa_n) * (x_bar - config.mu0)**2  # vectorized
log_lik = np.sum(0.5 * np.log(config.kappa0 / kappa_n) + ...)  # sum at end
```

### For Fix #3 (Conditional cleanup):
```python
def step(self):
    has_empty = False
    for i in indices:
        # ... Gibbs step ...
        if len(self.partition.clusters[k_old]) == 0:
            has_empty = True
    
    if has_empty:
        self.partition.cleanup()
```
