# Minimize Entropy Optimization — Performance & Data Profiles

## Problem Setup
- Depolarization Channel with $p=0.1$
- Future work: Werner-Holevo channel

## Benchmarking Framework
- Follows More–Wild definitions of *performance* and *data profiles*.
- Convergence test:
  $$f_\text{target} = f_L + \tau(f_0 - f_L), \quad \tau = 10^{-7}$$
- $t_{p,s}$: number of function evaluations needed by solver *s* to reach $f(x_k) \le f_\text{target}$.
- Profiles:
  - **Performance Profile:** compares relative speed across solvers.
  - **Data Profile:** normalizes evaluations by problem dimension ($n_p = 2n^2$).

## Parameters
- Dimensions tested: $n = 2, 5, 10$
- Depolarizing parameter: $p = 0.10$
- Number of random initializations per dimension: $10$
- Function evaluation budget per solver: $500$
- tolerance: $\tau = 10^{-7}$

## Solvers and Results
- **Powell**(Line search, derivative-free) performs best in both performance and data profiles.  
- **BFGS**(Gradient-based, quasi-Newton) converges accurately but with high function-evaluation cost.  
- **Nelder–Mead**(Simplex heuristic) stagnates even at low dimensions.
- Plots of data & performance profile can be found at `plots`
