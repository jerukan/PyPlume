# parcels_westcoast

A collection of notebooks with the purpose of taking ocean current data from the
[HFRNet Thredds Data Server](https://hfrnet-tds.ucsd.edu/) (and other files) and using
[OceanParcels](https://oceanparcels.org/) to simulate particle movement.

## Environment setup

It is highly recommended to use [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
to set up the Python environment, since OceanParcels is only actively updated on conda-forge.

If you are on macOS, the easiest way to install the dependencies is with the spec file.
```shell
conda create --name <env> --file spec-file-osx-64.txt
```

If you are any other platform, you can use the other environment YAML, though it's not guranteed
to work.
```shell
conda env create -f environment.yml
```

To make the Conda environment useable in Jupyter, just run
```shell
conda activate <env-name>
python -m ipykernel install --user --name <env-name> --display-name "Display name"
```

## Using the notebooks

### Running simulations

1. Set up config files in `parcels_configs/`.
	- **(More information on creating config files can be
	found in the README in the `parcels_cofigs/` folder)**
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

any other dependencies (mostly extra netcdf and .mat files) can be downloaded from the Microsoft
Teams Trajectories topic files (`parcels_westcoast`).

Or maybe I'll stick them onto a google drive idk
