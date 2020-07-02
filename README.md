# parcels_westcoast

A collection of notebooks with the purpose of taking ocean current data from the [HFRNet Thredds Data Server](https://hfrnet-tds.ucsd.edu/) and using [OceanParcels](https://oceanparcels.org/) to simulate particle movement.

## How to run and stuff

The actual simulations are done specifically on the 2 km resolution data, with help from the 6 km resolution data for data interpolation.

More details are found in the notebooks, but the overall way to run these notebooks:

1. Run `generate_thredds_regions` for both the 2 km and 6 km resolution data from the Thredds server. This will generate pickle files for use in other notebooks (note: this might take a while to run).
2. Run `access_thredds` for both the 2 km and 6 km resolution data from the Thredds server. This depends on the pickle files from `generate_thredds_regions` and generates netcdf files for use with later notebooks.
3. Run `gapfilling` to generate interpolated data for the 2 km region (note: this might also take a while to run).
4. Run `run_netcdf_parcels` for the actual OceanParcels simulations. This notebook creates a sequence of pictures of particle movement to stitch together into a gif.
	- to stitch together the images, I used [ImageMagick](https://imagemagick.org/index.php)
