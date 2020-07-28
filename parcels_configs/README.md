# Parcels simulation configuration

This is where the configs go.

example layout:

```json
{
	"name": "name_of_config",
	"netcdf_path": "path/to/data.nc",
	"resolution": 1,
	"parcels_config": {
		"time_range": ["2020-07-16T00", "END"],
		"repeat_dt": 14400,
		"particles_per_dt": 10,
		"max_variation": 0.0015,
		"spawn_points": [
			[32.551707, -117.138],
			[32.557, -117.138]
		],
		"simulation_dt": 300,
		"snapshot_interval": 10800,
		"save_snapshots": true,
		"shown_domain": {
			"S": 32.5335,
		    "N": 32.6,
		    "W": -117.19,
		    "E": -117.1174
		}
	}
}
```

information about the variables:

- `resolution`: for the data being used, 1, 2, or 6 km
- `time_range` (optional): timeframe of the data to run the simulation on
	- dates provided will round down to the nearest hour
	- `"START"` as the first time will use the first time of the data
	- `"END"` as the second time will use the last time of the data
	- an integer as one of the elements will be treated as a timedelta (in hours)
- `repeat_dt`: time to wait between releasing batches of particles (in seconds)
	- if the value is 0 or less, only one batch of particles will be released at the starting time
- `particles_per_dt`: number of particles to release after every interval of `repeat_dt`
- `max_variation`: the max amount the longitude and latitude of a particle will be randomized by
- `spawn_points`: list of [latitude, longitude] pairs to choose from as a spawn location
- `simulation_dt`: the actual dt used by parcels for the simulation (in seconds)
- `snapshot_interval`: time to wait between taking a snapshot of the simulation to save as an image in seconds
- `shown_domain` (optional): basically a cropping option
