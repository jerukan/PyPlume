# Parcels simulation configuration

This is where the configs go.

example layout:

```json
{
	// make this unique to avoid any accidental overriding
	"name": "name_of_config",
	"netcdf_path": {
		"type": "<file> or <thredds>",
		"path": "path/to/file/or/url.nc",
		// start of thredds only settings
		"time_range": ["2020-06-16T21", "2020-06-23T21"],
		"lat_range": [32.524, 32.75],
		"lon_range": [-117.32, -117.09]
		// end of thredds only settings
	},
	// whether to save the plots
	// set this to false if you are doing post-processing and plotting somewhere else like MATLAB
	"save_snapshots": true,
	"parcels_config": {
		// The DIRECTORY to save the particle netcdf file to
		"save_dir_pfile": "/Volumes/T7/Documents/Programs/scripps-cordc/parcels_westcoast/particledata",
		// Can be either of the built-ins: JITParticle, ScipyParticle
		// or a custom one wrote in parcels_kernels.py
		"particle_type": "ThreddsParticle",
		// List of kernels to add, can be built ins like AdvectionRK4
		// or custom ones in parcels_kernels.py. Feel free to add new ones to the file.
		"kernels": ["AdvectionRK4", "DeleteOOB", "RandomWalk"],
		// The start and end time of the simulation. START and END are special keywords that
		// denote the start and end times of the used netcdf respectively.
		// Integers will be treated as delta time in hours. For example, [6, "END"] will run the
		// simulation from 6 hours before the end until the end
		"time_range": ["2020-07-16T00", "END"],
		// The number of times to release the particles. If repeat_dt is 0 or less, this parameter
		// is ignored.
		"repetitions": 5,
		// The time between particle spawn release in seconds. If this is positive and repetitions
		// is 0 or less, the simulation will attempt to release at the said rate until the end of
		// the simulation
		"repeat_dt": 14400,
		// spawn points can be either loaded in straight as a 2d list of (lat, lon) pairs
		"spawn_points": [
			[32.551707, -117.138],
			[32.557, -117.138]
		],
		// or they can be loaded in from a .mat file
		// the .mat file must contain 2 variables representing longitude and latitude
		// dimension of the matrices don't matter since the data is flattened
		"spawn_points": {
			"path": "matlab/glist90zj_pts_position_spawn.mat",
			"lat_var": "yf",
			"lon_var": "xf"
		},
		// dt of the simulation in seconds
		"simulation_dt": 300,
		// how often data is recorded to the particle file in seconds
		"snapshot_interval": 10800,
	},
	// this plotting config is only relevant when save_snapshots is true
	// if save_snapshots is false, this entire thing can just be null
	"plotting_config": {
		// the domain to show in the plot. if null, uses the netcdf file's domain.
		"shown_domain": {
			"S": 32.525,
			"N": 32.7,
			"W": -117.27,
			"E": -117.09
		},
		// path to save the plots to
		"save_dir_snapshots": "/Volumes/T7/Documents/Programs/scripps-cordc/parcels_westcoast/snapshots",
		// loads a coastline to detect collisions
		"coastline": {
			"path": "matlab/coastline.mat",
			"lat_var": "latz0",
			"lon_var": "lonz0"
		},
		// features to add to the plots (detailed coastline, a tracked location, etc). check the
		// bottom of parcels_analysis.py for the available sets of features. details for each
		// feature are found in plot_features.py
		"plot_feature_set": "tj_plume_tracker",
		// draw land or not?
		"draw_coasts": false
	}
}
```
