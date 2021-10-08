# Parcels simulation configuration

This is where the configs go.

example layout:

```yaml
{
	# make this unique to avoid any accidental overriding
	"name": "name_of_config",
	# information about the ocean vector field that is loaded into a Parcels field
	# all U and V components MUST be in m/s
	"netcdf_path": {
		"type": "<file> or <thredds>",
		"path": "path/to/file/or/url.nc",
		### start of thredds only settings ###
		"time_range": ["2020-06-16T21", "2020-06-23T21"],
		"lat_range": [32.524, 32.75],
		"lon_range": [-117.32, -117.09],
		### end of thredds only settings ###
		# if there is a need to use any built in gapfillers, specify them here
		# check gapfilling.py for available gapfillers and arguments
		"gapfill_steps": [
			{
				"name": "InterpolationStep",
				"args": {
					"references": ["USWC_2KM_HOURLY", "USWC_6KM_HOURLY"]
				}
			}
		],
		# specify wind data to modify the behavior of particle advection
		"wind_data": {
			# uniform or vector field wind datasets can be passed in
			# currently only supports inputting NetCDF files with U and V components in m/s
			# if the wind data is in different units or measured by heading + magnitude, you must
			# convert the information externally
			# constant wind fields per timestamp and wind vector fields are supported
			"path": "path/to/wind/data",
			# the percentage of wind velocity (as a decimal) to be used to modify the ocean current
			# vector field
			"ratio": 0.03,
			# if true, for each point in the ocean current vector field, the closest wind data will
			# be added directly to the ocean current vector
			# (NOT IMPLEMENTED) if false, the particles will use the wind data in kernels
			# during the simulation instead of modifying the ocean currents
			"add_to_field_directly": true
		}
	},
	"parcels_config": {
		# The DIRECTORY to save the particle netcdf file to
		"save_dir_pfile": "/Volumes/T7/Documents/Programs/scripps-cordc/parcels_westcoast/particledata",
		# Can be either of the built-ins: JITParticle, ScipyParticle
		# or a custom one wrote in parcels_kernels.py
		"particle_type": "ThreddsParticle",
		# List of kernels to add, can be built ins like AdvectionRK4
		# or custom ones in parcels_kernels.py. Feel free to add new ones to the file.
		"kernels": ["AdvectionRK4", "DeleteOOB", "RandomWalk"],
		# Determines how to handle boundary conditions. By default, the zero-vectors at coastlines
		# are interpolated linearly. Other options are "freeslip" and "partialslip".
		"boundary": "freeslip",
		# The start and end time of the simulation. START and END are special keywords that
		# denote the start and end times of the used NetCDF respectively.
		### IMPORTANT NOTE ###
		# if particles are scheduled to released individually, START will use the
		# earliest time a particle is released instead of the start of the NetCDF. It is highly
		# recommended to use START to avoid out of bounds errors.
		######################
		# Integers will be treated as delta time in hours. For example, [6, "END"] will run the
		# simulation from 6 hours before the end until the end
		"time_range": ["2020-07-16T00", "END"],
		### repetition settings ###
		# The number of times to release the particles. If repeat_dt is 0 or less, this parameter
		# is ignored.
		"repetitions": 5,  # default -1
		# The time between particle spawn release in seconds. If this is positive and repetitions
		# is 0 or less, the simulation will attempt to release at the said rate until the end of
		# the simulation
		"repeat_dt": 14400,  # default -1
		# Number of times per release to spawn particles from the defined spawn points.
		# There is only a visual difference if there is randomness to particle movement.
		"instances_per_spawn": 50,  # default 1
		# spawn points can be either loaded in straight as a 2d list of (lat, lon) pairs
		### end of repetition settings ###
		"spawn_points": [
			[32.551707, -117.138],
			[32.557, -117.138],
			# for more complex release types and settings, you can specify individual information
			# about how to spawn a particular point
			{
				# optional label for particle. doesn't do anything right now lol
				"label": "RADARSAT2",
				# [lat, lon] pair
				"point": [33.6495, -118.1079],
				# when to release the particle. defaults to the start of the simulation.
				# can be set to START if you really want to
				"release": "2021-10-01T21:00",
				### spawn repetition settings ###
				# if any of these settings are missing or null, they automatically default to the
				# outer settings defined above
				"repetitions": 5,  # default: from outer settings
				"repeat_dt": 14400,  # default: from outer settings
				"instances_per_spawn": 50,  # default: from outer settings
				### end of spawn repetition settings ###
				# optional, specifies if a single point should be spawned in a pattern instead
				# this example spawns particles in a 3x3 grid around specified point
				# maybe there will be documentation on this lol
				"pattern": {
					"type": "grid",
					"args": {
						"size": 3,
						"gapsize": 0.03
					}
				}
			}
		],
		# or they can be loaded in from a .mat file
		# the .mat file must contain 2 variables representing longitude and latitude
		# dimension of the matrices don't matter since the data is flattened
		"spawn_points": {
			"path": "matlab/glist90zj_pts_position_spawn.mat",
			# if the mat file has more variables than just the two lists, I highly suggest
			# specifying these variables to avoid unwanted behavior.
			"lat_var": "yf",
			"lon_var": "xf"
		},
		# dt of the simulation in seconds
		"simulation_dt": 300,
		# how often data is recorded to the particle file in seconds
		"snapshot_interval": 10800,
	},
	# settings to modify the resulting ParticleFile from the simulation
	"postprocess_config": {
		# loads a coastline to detect collisions and delete particles that collide
		"coastline": {
			"path": "matlab/coastline.mat",
			"lat_var": "latz0",
			"lon_var": "lonz0"
		},
		# loads a buoy path to calculate deviation from the path for each particle
		# the variable will be called "buoy_distances" in the netcdf file
		"buoy": {
			"path": "buoy_data/wavebuoy_704-02.csv"
		}
	},
	# whether to save the plots
	# set this to false if you are doing post-processing and plotting somewhere else like MATLAB
	"save_snapshots": true,
	# this plotting config is only relevant when save_snapshots is true
	# if save_snapshots is false, this entire thing can just be null
	"plotting_config": {
		# the domain to show in the plot. if null, uses the netcdf file's domain.
		"shown_domain": {
			"S": 32.525,
			"N": 32.7,
			"W": -117.27,
			"E": -117.09
		},
		# path to save the plots to
		"save_dir_snapshots": "/Volumes/T7/Documents/Programs/scripps-cordc/parcels_westcoast/snapshots",
		# features to add to the plots (detailed coastline, a tracked location, etc). check the
		# bottom of parcels_analysis.py for the available sets of features. details for each
		# feature are found in plot_features.py
		"plot_feature_set": "tj_plume_tracker",
		# draw land or not?
		"draw_coasts": false
	}
}
```
