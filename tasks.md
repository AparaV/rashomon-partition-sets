1. Add Rashomon to the HPD posterior plot
2. Fix PPMx implementation
    - tried a fresh implementation. found a bug in the old implementation where cohesion was a function of outcome which is odd.
      but fixing this bug made things a lot slower
    - tried translating R code from ppmSuite to python. Still very slow
      Subsampled to 500 observations. This took 10 minutes for one iteration (1000 iters, 300 burn-in, 2 chains)
      MSE is high but not as bad as the old implementation
    - Next solution is to see if numba can make this fast
    - Other solution is to pipe the data to R
3. Run SSL for sim 2
4. Run PPMx for sim 1
5. Run PPMx for sim 2
6. Run blasso and SSL for longer chains in sim 1



## ppmSuite Plan

Plan: Integrate R's ppmSuite via rpy2
Create an R-backed PPMx wrapper that maintains the existing Python interface while leveraging the optimized R implementation for 50-100x speedup.

Steps
Add dependencies and verify R setup

Add rpy2 to requirements.txt
Create install script verifying R is available with ppmSuite package
Test basic rpy2 connectivity
Create PPMxR wrapper class in baselines/ppmx_r.py

Implement __init__, fit, predict matching ppmx.py:75-150
Handle numpy↔R conversions (arrays to data.frames, 0/1-based indexing)
Map Python parameters to R ppmSuite equivalents (cohesion string→int, parameter names)
Extract R output into Python attributes (chains_, coef_samples_, n_clusters_samples_, rhat_, converged_)
Add R backend selection in simulations.py:641-740

Add --ppmx-backend argument (choices: 'python', 'r')
Import and instantiate PPMxR when backend='r'
Ensure results structure matches existing code expectations
Test equivalence and performance

Run test_gaussian_ppmx_simulation.py with both backends on n=100 subsample
Compare MSE, IOU, cluster counts between Python/R implementations
Benchmark timing (expect 10-50x speedup)
Update test script for full-scale runs

Remove 500-observation subsampling in test_gaussian_ppmx_simulation.py:254-260
Set default backend to 'r' for production runs
Add fallback to Python backend if R unavailable
Further Considerations
R package availability: Will users/systems have R + ppmSuite installed? Option: Include conda environment spec with R dependencies / Option: Keep Python fallback for CI/testing
Multiple chains: Does ppmSuite support n_chains parameter or need separate calls? May need to run chains sequentially and combine
Missing diagnostics: If R doesn't return log posteriors, compute manually using likelihood + prior / If no R-hat provided, compute in Python from chains