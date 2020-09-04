# parcels_westcoast

A collection of notebooks with the purpose of taking ocean current data from the [HFRNet Thredds Data Server](https://hfrnet-tds.ucsd.edu/) and using [OceanParcels](https://oceanparcels.org/) to simulate particle movement.

## How to run and stuff

More details are found in the notebooks, but the overall way to run these notebooks:

### Step 1: generate netcdf files

1. Use `access_thredds` and define the data you want and the domain to retrieve. This will save netcdf files.

### Step 2: interpolate data

1. Configure `gapfilling` to choose a file to interpolate and which files to use as reference for interpolation. This will output another netcdf file (note: this might also take a while to run).

### Step 3: run parcels simulation

1. Set up config files in `parcels_configs/`.
2. Choose config files in the `parcels_regions` notebook for the OceanParcels simulation. This notebook outputs a new netcdf file with particle simulation data and creates a sequence of snapshots of particle movement to stitch together into a gif.
	- the notebook uses [ImageMagick](https://imagemagick.org/index.php) to stitch images into a gif

### Step 4?: trajectory analysis

1. Choose the correct files in `trajectory_analyze` and run it to find out when and where particles "beached". Works only with the coastline near the Tijuana River.
