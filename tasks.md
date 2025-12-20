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
