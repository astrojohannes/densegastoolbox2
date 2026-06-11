# Installation for Python > 3.11

Tested target: Python 3.12 / 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary -r requirements.txt
```

For an even stricter check that avoids local source builds:

```bash
python -m pip install --only-binary=:all: -r requirements.txt
```

Optional PTMCMC mode:

```bash
# Fedora examples
sudo dnf install python3.13-devel openmpi-devel mpich-devel

source .venv/bin/activate
python -m pip install mpi4py PTMCMCSampler
```

`PTMCMCSampler` is no longer imported unconditionally. It is only required when `use_pt=True`.
