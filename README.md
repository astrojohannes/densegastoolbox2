# Dense Gas Toolbox #
All versions DOI: 10.5281/zenodo.3686329
Created by Johannes Puschnig
www.jpuschnig.com

# Aim
Calculate density and temperature from observed molecular emission lines,
using radiative transfer models.

# Method
Our models assume that the molecular emission lines emerge from a
multi-density medium rather than from a single density alone.
The density distribution is assumed to be log-normal or log-normal with
a power-law tail.
The parameters (density, temperature, width of density distribution and line optical depths)
are inferred using Bayesian statistics, i.e. Markov chain Monte Carlo (MCMC).

# Results
Given an ascii table of observed molecular intensities [K km/s],
the results (mass-weighted mean density, temperature, width of the density
distribution and line optical depths) are saved in an output ascii file.
Furthermore, diagnostic plots are created to assess the quality of the
fit/derived parameters.

---

# VERSION HISTORY

- Jun 11, 2026 | Version 2.1

    * Updated the code base for compatibility with Python 3.12 and Python 3.13.

    * Updated the Python requirements to support the current Scientific Python stack, including NumPy 2.x.

    * PTMCMCSampler and mpi4py are no longer imported unconditionally. They are only required when the optional parallel-tempered MCMC mode is used.

    * Improved robustness of the model-grid loading procedure.

    * Removed shell-dependent result formatting.

    * Fixed several runtime issues exposed by newer Python versions.

    * Added updated installation notes for Python >3.11.

- Dec 15, 2024

   * Updated model grids, also including (4-3) transitions

   * MCMC bugfix by Soren to avoid walkers getting stuck in low-prob. regions

- May 24, 2024 | Version 2.0 (first v2 release)

   * Updated model grid (emmisivities) stored in 2 pickle files (32GB each)

   * New models allow to vary line optical depths or leave optical depths as free parameter that is inferred

   * Models available for the following molecular lines (up to dJ=3-2): 12CO, HCN, HCO+, 13CO, C18O

   * Temperatures range from 10 to 30K (in steps of 5K)

   * The widths of the density distributions range from 0.2 to 0.8dex (in steps of 0.1dex)

   * Available line optical depths (tau) are:
         Low: [0.1,0.2,0.3] for 13CO and C18O
         Mid: [0.8,1.1,1.5] for HCN and HCO+
         High: [5.0,6.5,8.0] for 12CO

   * See "example.py" for how to use the Dense Gas Toolbox. It's easy!
 
---
