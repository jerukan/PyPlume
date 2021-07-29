# parcels_westcoast

A collection of notebooks with the purpose of taking ocean current data from the
[HFRNet Thredds Data Server](https://hfrnet-tds.ucsd.edu/) and using
[OceanParcels](https://oceanparcels.org/) to simulate particle movement.

## Running simulations

1. Set up config files in `parcels_configs/`. (More information on creating config files can be
found in the README in the `parcels_cofigs/` folder)
2. Choose config files in the `parcels_regions` notebook for the OceanParcels simulation. This
notebook outputs a new netcdf file with particle simulation data and creates a sequence of snapshots
of particle movement stitch together into a gif.
	- the notebook uses [ImageMagick](https://imagemagick.org/index.php) to stitch images into a gif

### notes

any other dependencies (mostly extra netcdf and .mat files) can be downloaded from the Microsoft Teams Trajectories topic files (`parcels_westcoast`).
