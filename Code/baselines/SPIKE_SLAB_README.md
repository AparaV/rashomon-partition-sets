# Spike and Slab Lasso Implementation

## Overview

Implemented `SpikeSlabLasso` as a new baseline method for Bayesian variable selection in the Rashomon TVA project.

## Implementation Details

### Model Specification

The Spike and Slab Lasso uses a mixture prior for variable selection:

```
y | X, β, τ² ~ N(Xβ, τ²I)
β_j | γ_j, τ² ~ γ_j·N(0, τ²/λ₀²) + (1-γ_j)·Laplace(0, λ₁)
γ_j ~ Bernoulli(θ)
θ ~ Beta(a_θ, b_θ)  [adaptive]
τ² ~ InverseGamma(a, b)
```

**Key components:**
- **Spike** (γ_j=1): Tight normal prior N(0, τ²/λ₀²) for shrinkage near zero
- **Slab** (γ_j=0): Laplace prior for non-zero coefficients (via scale mixture)
- **Adaptive θ**: Prior inclusion probability updated from data using Beta prior

### Gibbs Sampling Algorithm

1. Sample β | γ, τ², λ²_slab, y using Cholesky decomposition
2. Sample λ²_j for slab components (γ_j=0) via inverse-Gaussian
3. Sample γ | β, τ², θ using posterior inclusion probabilities
4. Sample τ² | β, y from inverse-gamma
5. Sample θ | γ from Beta posterior (if update_theta=True)

### Key Features

- **sklearn-compatible API**: `fit()`, `predict()`, `predict_map()`
- **Multi-chain MCMC**: Runs multiple chains for convergence diagnostics (Gelman-Rubin R-hat)
- **Variable selection**: Tracks posterior inclusion probabilities P(γ_j=1|data)
- **Adaptive theta**: Learns sparsity level from data
- **MAP estimation**: Finds coefficient vector with highest posterior density
- **Posterior caching**: Stores log posteriors for downstream analysis
- **Efficient sampling**: Uses Cholesky decomposition and vectorization

### Class Attributes

After fitting:
- `coef_`: Posterior mean of coefficients
- `coef_map_`: MAP estimate
- `inclusion_probs_`: Posterior P(γ_j=1|data) for each feature
- `theta_`: Posterior mean of inclusion probability
- `chains_`: MCMC samples (n_chains, n_samples, n_features)
- `gamma_chains_`: Inclusion indicator samples
- `rhat_`: Gelman-Rubin convergence statistics
- `converged_`: Boolean convergence flag
- `log_posteriors_`: Cached log posterior densities (if enabled)

### Methods

- `fit(X, y, n_chains=4)`: Fit model via Gibbs sampling
- `predict(X)`: Predict using posterior mean
- `predict_map(X)`: Predict using MAP estimate
- `get_selected_features(threshold=0.5)`: Get features with P(included) ≥ threshold
- `get_log_posteriors()`: Retrieve cached log posteriors

## Usage Example

```python
from baselines import SpikeSlabLasso

# Fit model
ssl = SpikeSlabLasso(
    n_iter=2000,
    burnin=500,
    thin=2,
    lambda0=15.0,      # Spike precision
    lambda1=1.0,       # Slab scale
    theta_init=0.5,    # Initial inclusion probability
    update_theta=True, # Adapt theta from data
    random_state=42
)
ssl.fit(X, y, n_chains=4)

# Variable selection
selected = ssl.get_selected_features(threshold=0.5)
print(f"Selected features: {selected}")
print(f"Inclusion probabilities: {ssl.inclusion_probs_}")

# Predictions
y_pred = ssl.predict(X_test)
y_pred_map = ssl.predict_map(X_test)

# Convergence diagnostics
print(f"Converged: {ssl.converged_}")
print(f"Max R-hat: {np.max(ssl.rhat_)}")
```

## Hyperparameters

### MCMC Parameters
- `n_iter=2000`: Total iterations per chain
- `burnin=500`: Burn-in iterations to discard
- `thin=2`: Keep every thin-th sample
- `n_chains=4`: Number of independent chains

### Prior Parameters
- `lambda0=10.0`: Spike precision (larger → stronger shrinkage)
- `lambda1=1.0`: Slab scale (Laplace regularization strength)
- `theta_init=0.5`: Initial prior inclusion probability
- `update_theta=True`: Whether to adapt theta from data
- `theta_a=1.0`, `theta_b=1.0`: Beta prior hyperparameters for θ
- `tau2_a=0.1`, `tau2_b=0.1`: Inverse-gamma prior for error variance

### Tuning Guidelines

**For sparse problems (few non-zero coefficients):**
- Increase `lambda0` (15-25) for stronger spike
- Decrease `theta_init` (0.1-0.3) for low prior inclusion
- Use `update_theta=True` to learn sparsity

**For dense problems (many non-zero coefficients):**
- Decrease `lambda0` (5-10) for weaker spike  
- Increase `theta_init` (0.5-0.7) for higher prior inclusion
- Adjust `lambda1` for regularization strength

**For high-dimensional (p > n):**
- Use strong spike: `lambda0` ≥ 20
- Low initial inclusion: `theta_init` < 0.2
- More iterations: `n_iter` ≥ 2000

## Testing

Comprehensive test suite in `test_spike_slab_lasso.py`:

```bash
pytest baselines/test_spike_slab_lasso.py -v
```

Test coverage:
- Basic functionality (fit, predict)
- Variable selection accuracy
- Adaptive theta updating
- Convergence diagnostics
- High-dimensional settings (p > n)
- Edge cases (pure noise, single chain)
- Posterior caching
- Reproducibility with random seeds

## Integration with Simulation Pipeline

To use in simulations, add to parameter files (e.g., `slopes_sim_params.py`):

```python
# Spike-Slab Lasso parameters
ssl_n_iter = 2000
ssl_burnin = 500
ssl_thin = 2
ssl_n_chains = 4
ssl_lambda0 = 15.0
ssl_lambda1 = 1.0
ssl_theta_init = 0.5
ssl_update_theta = True
ssl_theta_a = 1.0
ssl_theta_b = 1.0
```

## Files Created

1. `Code/baselines/spike_slab_lasso.py` - Main implementation (580 lines)
2. `Code/baselines/test_spike_slab_lasso.py` - Test suite (540 lines)
3. `Code/baselines/demo_spike_slab_lasso.py` - Demo script (350 lines)
4. `Code/baselines/__init__.py` - Updated to export `SpikeSlabLasso`

## Performance Notes

- **Convergence**: May require more iterations than BayesianLasso due to discrete γ sampling
- **Speed**: Similar to BayesianLasso (Cholesky decomposition is main bottleneck)
- **Memory**: Stores both β and γ chains, ~2x memory of BayesianLasso
- **Parallelization**: Chains run sequentially (can be parallelized externally)

## Comparison with Other Baselines

| Method | Variable Selection | Uncertainty | Speed | Best For |
|--------|-------------------|-------------|-------|----------|
| Lasso | Via threshold | Bootstrap only | Fast | Quick baseline |
| BayesianLasso | Soft (L1 prior) | Full posterior | Medium | Continuous shrinkage |
| **SpikeSlabLasso** | **Hard (mixture)** | **Full posterior** | **Medium** | **Sparse problems** |
| BootstrapLasso | Via threshold | Empirical | Slow | Non-Bayesian UQ |

**Advantages of Spike-Slab:**
- Explicit variable selection via γ indicators
- Posterior inclusion probabilities P(γ_j=1|data)
- Adapts to unknown sparsity level (via θ updating)
- Better for very sparse problems than pure L1

**Disadvantages:**
- More hyperparameters to tune
- Slower convergence (discrete + continuous sampling)
- May struggle if sparsity level is misspecified

## References

Ročková, V., & George, E. I. (2018). The Spike-and-Slab LASSO. *Journal of the American Statistical Association*, 113(521), 431-444.

Park, T., & Casella, G. (2008). The Bayesian Lasso. *Journal of the American Statistical Association*, 103(482), 681-686.
