- Run SSL for sim 2
- Run PPMx for sim 2
- Make plots for sim 2
- Clean plots
- Add plots to overleaf
- Update discussion
- Clean up the code
    - Include R code and instructions
    - Clean up notebooks keeping only necessary things
    - Remove all python ppmx code
    - Remove all test scripts created by copilot
- Run blasso and SSL for longer chains in sim 1



## ppmSuite Plan

Plan: Parallelize PPMxR Chain Execution
Parallelize the 4 independent MCMC chains in ppmx_r.py using multiprocessing.Pool to reduce wall-clock time by ~4x (with 4 chains). Each chain calls R's ppmSuite independently with different random seeds.

Steps
1. Create standalone worker function outside the class (required for pickling) that executes one chain: accepts chain parameters (X, y, n_policies, r_params, random_seed), initializes R/ppmSuite, runs gaussian_ppmx, extracts results, returns processed chain data (samples, partitions, clusters).
2. Add n_jobs parameter to ppmx_r.py:46-95 with default -1 (use all cores) or 1 (sequential fallback), matching scikit-learn conventions.
3. Refactor ppmx_r.py:265-345 to use multiprocessing.Pool: prepare worker arguments list with unique seeds per chain, dispatch via pool.starmap() (following rashomon/aggregate.py pattern), collect and combine results into existing all_chains, all_partition_samples structures.
4. Add platform-specific initialization: wrap worker function with R library reload (ro.r('library(ppmSuite)')) to ensure each forked/spawned process has valid R session, handle Windows spawn context explicitly if needed.
5. Update progress reporting: replace per-chain verbose prints with total elapsed time since parallel execution hides individual chain progress, optionally add progress callback if n_jobs=1.

Further Considerations
1. Default parallelization vs. user control? Set n_jobs=-1 (parallel by default) for speed, or n_jobs=1 (sequential) for debugging? Recommend parallel default with easy override.
2. Windows compatibility? Test with multiprocessing.set_start_method('spawn') on Windows since fork unavailable - may require explicit data passing. macOS/Linux should work with fork.
3. Memory overhead? Each process duplicates X, y data (~minimal for typical sizes) but gains 4x speedup. Document memory consideration for very large datasets (>1GB).