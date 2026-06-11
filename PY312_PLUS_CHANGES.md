# Python > 3.11 upgrade changes

This copy was adjusted for Python 3.12/3.13-era environments.

## Dependency changes

- Replaced old strict pins with Python 3.12/3.13-compatible ranges in `requirements.txt`.
- Added missing runtime dependencies: `pandas`, `psutil`, `requests`, `h5py`, `tabulate`, `packaging`.
- Kept `PTMCMCSampler` and `mpi4py` optional. They are needed only if `use_pt=True`.

## Code changes

- `mcmc.py`
  - Made `PTMCMCSampler` an optional import.
  - Fixed fixed-tau emcee initial positions to shape `(nwalkers, ndim)`.
  - Fixed PTMCMC branch to use `nsims` instead of undefined `nsteps`.
  - Allows `n_cpus=1` without creating a multiprocessing pool.

- `dgt2.py`
  - Added missing `importlib.util` import.
  - Uses Matplotlib `Agg` backend by default for headless/server-safe plot generation.
  - Removed `from pylab import *`.
  - Replaced shell-based `sed` cleanup with pure Python file handling.
  - Creates `results2/` as needed.
  - Downloads `models_<type>/dgt_config.py` if it is missing.

- `read_grid_ndist2.py`
  - No hard import-time dependency on `dask`; falls back to pandas concat if dask is not installed.
  - Streams downloads and checks HTTP errors.
  - Uses atomic `.download` temp files.
  - Ensures `tmp/` exists and clears stale chunk files before chunking.
  - Closes pickle files via context managers.

- `mcmc_corner_plot2.py`
  - Uses Matplotlib `Agg` backend by default.
  - Fixed KDE failure path for nearly constant MCMC samples.
  - Closes generated figures.
  - Fixed PTMCMC chain path building for non-string pixel ids.

- `autocorr.py`
  - Handles constant chains without dividing by zero.

## Quick install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary -r requirements.txt
```
