# Explore Source Nature Using KM3NeT in Gammapy

This repository implements a full analysis pipeline to investigate the
physical nature of high-energy astrophysical sources using KM3NeT
Instrument Response Functions (IRFs) within the Gammapy framework.  
The scientific goal of the project is to determine the *hadronic fraction*
of the source emission by performing a likelihood fit of spectral models,
including both leptonic and hadronic components.

The workflow is structured so that each directory corresponds to one
logical step of the analysis: from IRFs → models → datasets → likelihood
analysis. All components follow Gammapy standards and are designed to be
reproducible, modular, and extensible.

---

## Repository Structure and Workflow

### `irfs/`
Contains the Instrument Response Functions (IRFs) used in the analysis.
These include KM3NeT IRFs.
They define the detector performance:  
effective area, energy dispersion, PSF, and background rates.  
These files represent the starting point of the analysis chain.

### `models/`
Defines source models used in the analysis (spectral, spatial, temporal),
built using Gammapy objects.  
These represent the physical hypotheses tested in the likelihood fit,
particularly regarding the hadronic fraction.

### `datasets/`
Contains datasets generated from the IRFs and source models.
Each dataset corresponds to a simulated or real observation,
including background definitions and all metadata required by Gammapy.

### `analysis/`
Notebooks and scripts performing:
- model fitting  
- parameter estimation  
- hypothesis testing  
- visualization of spectra, likelihood profiles, and hadronic contribution  

This is the main section where scientific results are produced.

---
# Environment Setup

This project is designed to run inside the **Gammapy 2.0** framework.

------------------------------------------------------------------------

# 1. Using Conda

## Create the environment

``` bash
conda env create -f environment.yml
```

## Activate the environment

``` bash
conda activate gammapy_acme
```

## Add the environment as a Jupyter kernel

``` bash
python -m ipykernel install --user --name gammapy_acme
```

------------------------------------------------------------------------

# 2. Using Micromamba (recommended if Conda is slow)

## Install micromamba (one time only)

### Download micromamba (macOS Intel)

``` bash
curl -L https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xvj bin/micromamba
chmod +x ~/bin/micromamba
```

### Initialize micromamba in the current shell

``` bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$HOME/bin/micromamba" shell hook --shell zsh)"
```

### Verify installation

``` bash
micromamba --version
```

## Create the environment

``` bash
micromamba create -f environment.yml -n gammapy_acme
```

## Activate the environment

``` bash
micromamba activate gammapy_acme
```

### Important: In every new terminal session, run:

``` bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$HOME/bin/micromamba" shell hook --shell zsh)"
micromamba activate gammapy_acme
```

## Add the environment as a Jupyter kernel

``` bash
python -m ipykernel install --user --name gammapy_acme
```

------------------------------------------------------------------------

# 3. Using Only pip + venv

This method uses the built-in Python `venv` module and avoids
Conda/Mamba entirely.

## Prerequisites

-   Python **3.10+** (recommended: 3.11 or 3.12)

Check your Python version:

``` bash
python3 --version
```

## Create a new virtual environment

``` bash
python3 -m venv gammapy-acme
```

## Activate the environment

macOS / Linux:

``` bash
source gammapy-acme/bin/activate
```

Windows (PowerShell):

``` bash
.\gammapy-acme\Scripts\Activate.ps1
```

## Upgrade pip (recommended)

``` bash
python -m pip install --upgrade pip
```

## Install dependencies

``` bash
python -m pip install -r requirements.txt
```

## Start Jupyter

Modern interface:

``` bash
python -m jupyter lab
```

Classic interface:

``` bash
python -m notebook
```

License
--------

This project is released under the MIT License
(or whichever license you prefer).

Contact
-------
If you have any questions, please contact:

Antonio Condorelli
condorelli@apc.in2p3.fr

or better: open a ticket on ACME! :)
https://support.acme-astro.eu/
