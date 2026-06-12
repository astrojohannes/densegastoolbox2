# Dense Gas Toolbox #
All versions DOI: 10.5281/zenodo.3686329
Created by Johannes Puschnig
www.jpuschnig.com

# Aim
Calculate density, temperature and line optical depths from observed molecular emission lines, using radiative transfer model grids.

# Method
The Dense Gas Toolbox is based on the assumption that molecular emission lines emerge from a multi-density medium rather than from gas at a single density alone. The gas density distribution is described either by a log-normal distribution or by a log-normal distribution with an additional power-law tail.

The physical parameters are inferred by comparing observed molecular line intensities and line ratios to pre-computed radiative transfer model grids. The inferred parameters include the mass-weighted mean density, gas temperature, the width of the density distribution, and, starting with version 2, the optical depths of the observed molecular lines. The parameter inference can be performed using Bayesian statistics, i.e. Markov chain Monte Carlo (MCMC).

# Results
Given an ASCII table of observed molecular intensities in units of K km/s, the Dense Gas Toolbox saves the inferred physical parameters to an output ASCII file. These include the mass-weighted mean density, gas temperature, width of the density distribution, and, for version 2 model grids, the line optical depths. Furthermore, diagnostic plots are created to assess the quality of the fit and the robustness of the derived parameters.

---

# VERSION HISTORY

- Jun 12, 2026 | Version 2.2

    * Added RAM-aware model-grid loading for large pickle files

    * Model grids are now reduced earlier to the requested transitions, temperature, width, and tau constraints

    * Temporary reduced chunks are written during loading to reduce peak memory usage

    * Added clearer RAM and disk-space checks for large model files

    * Improved robustness of model downloads

    * Added optional SHA256 verification for downloaded model files

    * tau_fiducial(type_of_models) now returns fiducial tau values matching the selected model grid

    * Updated the example script to use fixed fiducial tau values by default.

    * Improved compatibility with historical DGT v1.x fiducial-tau results

    * Increased the initial MCMC burn-in phase for improved walker stability

    * Disabled do_model_test by default in the example script

    * Included updates to test input and example data

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
