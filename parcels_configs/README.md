# Parcels simulation configuration

This is where the configs go.

example layout:

```json
{
	"name": "name_of_config",
	"netcdf_path": "path/to/data.nc",
	"parcels_config": {
		"time_range": ["2020-07-16T00", "END"],
		"repeat_dt": 14400,
		"repetitions": 5,
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

- `name`: This will determine the filenames and folders stuff saves to. Make sure it's unique!
- `time_range`: Timeframe of the data to run the simulation on
	- `"START"` as the first time will use the first time of the data
	- `"END"` as the second time will use the last time of the data
	- An integer as one of the elements will be treated as a timedelta (in hours)
		- For example, `[6, "END"]` will run the simulation from 6 hours before the end until the
		end
- `repeat_dt`: Time to wait between releasing batches of particles (in seconds)
	- If the value is 0 or less, only one batch of particles will be released at the starting time
- `repetitions`: Number of times to spawn the cluster of particles
	- If `repeat_dt` is 0 or less, this parameter will have no effect
- `spawn_points`: List of [latitude, longitude] pairs to choose from as a spawn location
	- You can also put a path to a .mat file, but it's hardcoded right now, so don't.
- `simulation_dt`: The actual dt used by parcels for the simulation (in seconds)
- `snapshot_interval`: Time to wait between taking a snapshot of the simulation to save as an image
in seconds (also determines how often the ParticleFile saves information)
- `save_snapshots`: True to save the plots
- `shown_domain`: Basically a cropping option, set to null for default domain
