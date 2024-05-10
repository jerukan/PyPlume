# PyPlume

[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/9735287/tree/v1)

A collection of notebooks and methods made unifying the process of loading two-dimensional oceanic current vector fields from models and observations, simulating trajectory models, and analyzing and visualizing particle trajectories.

The core library used in this project is [OceanParcels](https://oceanparcels.org/), which is used for particle advection math.

[Link to corresponding paper](https://doi.org/10.1016/j.envsoft.2023.105783)

## Environment setup

It is highly recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up the Python environment.

The easiest way to install the dependencies is with the `environment.yml` file.
```shell
conda env create -f environment.yml
```
This will create a Conda environment named `py3-parcels`.

To make the Conda environment useable in Jupyter, run these two commands
```shell
conda activate py3-parcels
python -m ipykernel install --user --name py3-parcels --display-name "py3-parcels"
```

## Data processing

More information about how to format the data can be found in [`data/README.md`](data/README.md).

## Using the notebooks

### Running simulations

1. Set up config files in [`configs/`](configs).
	- **(More information on creating config files can be found in the [config README](configs/README.md))**
2. Choose config files in the [`simulation_runner.ipynb`](simulation_runner.ipynb) notebook for the OceanParcels simulation. This notebook outputs a new netcdf file with particle simulation data and creates a sequence of snapshots of particle movement stitch together into a gif.

### Saving netcdf files from Thredds

Use the [`download_data.ipynb`](download_data.ipynb) notebook to save a specified region from some online source.

### Gapfilling

Use [`gapfilling.ipynb`](gapfilling.ipynb) to gapfill a netcdf file. Instructions are in the notebook.

## Possible issues and fixes

### `FieldOutOfBoundError` or other Field errors during simulation

If particles are intented to go out of bounds during the simulation, add `pyplume.kernels.DeleteStatusOutOfBounds` as a kernel to the list of kernels in the configuration. Other error codes like `ErrorTimeExtrapolation` can be ignored using `pyplume.kernels.DeleteStatusError`.

### How to run simulations backward in time?

The only thing that needs to change is a negative simulation delta time, which can be set in the `simulation_dt` field when configuring.
