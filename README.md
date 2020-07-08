# parcels_westcoast

A collection of notebooks with the purpose of taking ocean current data from the [HFRNet Thredds Data Server](https://hfrnet-tds.ucsd.edu/) and using [OceanParcels](https://oceanparcels.org/) to simulate particle movement.

## How to run and stuff

More details are found in the notebooks, but the overall way to run these notebooks:

### Step 1: generate netcdf files

##### Option 1: using auto-generated regions

1. Run `generate_regions_auto` for both the data you desire from the Thredds server. This will generate pickle files for use in `access_thredds` (note: this might take a while to run).
2. Run `access_thredds` for both all the data you got regions from in `generate_regions_auto`. This will save netcdf files for each region found.

##### Option 2: using manually defined regions

1. Use `access_thredds_man` and define the data you want and the domain to retrieve. This will save netcdf files.

### Step 2: interpolate data

3. Configure `gapfilling` to choose a file to interpolate and which files to use as reference for interpolation. This will output another netcdf file (note: this might also take a while to run).

### Step 3: run parcels simulation

4. Configure `parcels_regions` to choose which netcdf files to use for the OceanParcels simulation. This notebook creates a sequence of pictures of particle movement to stitch together into a gif.
	- the notebook uses [ImageMagick](https://imagemagick.org/index.php) to stitch images into a gif
