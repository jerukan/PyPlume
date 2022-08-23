# parcels_westcoast

A collection of notebooks with the purpose of taking ocean current data from the
[HFRNet Thredds Data Server](https://hfrnet-tds.ucsd.edu/) (and other files) and using
[OceanParcels](https://oceanparcels.org/) to simulate particle movement.

## NOTE: Running on HYCOM data

Refer to the instructions found in `README_HYCOM.md`.

## Environment setup

It is highly recommended to use [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
to set up the Python environment, since OceanParcels is only actively updated on conda-forge.

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

## Saving netCDF files for vector fields

These should be saved in the `data/field_netcdfs/` directory. Formats and more information about the files themselves can be found in `data/field_netcdfs/README.md`.

## Using the notebooks

### Running simulations

1. Set up config files in `parcels_configs/`.
	- **(More information on creating config files can be
	found in the README in the `parcels_configs/` folder)**
2. Choose config files in the `parcels_regions.ipynb` notebook for the OceanParcels simulation. This
notebook outputs a new netcdf file with particle simulation data and creates a sequence of snapshots
of particle movement stitch together into a gif.
	- The notebook uses [ImageMagick](https://imagemagick.org/index.php) to stitch images into a gif

### Saving netcdf files from Thredds

Use the `access_thredds.ipynb` notebook to save a specified region from the Thredds server. Note
that this will be required if you want to gapfill the Thredds data.

### Gapfilling

Use `gapfilling.ipynb` to gapfill a netcdf file. Instructions are in the notebook.

#### Additional MATLAB dependency

The gapfilling notebook relies on the MATLAB engine to smooth and fill the data, so follow the
instructions in the notebook, or just install the
[MATLAB python engine](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html),
following the instructions from the link

## Last notes

any other dependencies (mostly extra netcdf and .mat files) can be downloaded from the
[Microsoft Teams Trajectories topic files](https://ucsdcloud.sharepoint.com/:f:/r/sites/HFRdataanalysis/Shared%20Documents/Trajectories/parcels_westcoast?csf=1&web=1&e=sOQuyY) (`parcels_westcoast`).

Or maybe I'll stick them onto a google drive idk
