# Parcels simulation configuration

This is where the configs go.

example layout:

```json
{
	"name": "name_of_config",
	"netcdf_path": "path/to/data.nc",
	"resolution": 1,
	"parcels_config": {
		"repeat_dt": 14400,
		"particles_per_dt": 10,
		"max_variation": 0.0015,
		"spawn_points": [
			[32.551707, -117.138],
			[32.557, -117.138]
		],
		"simulation_dt": 300,
		"snapshot_interval": 10800,
		"save_snapshots": true
	}
}
```

information about the variables:

- `resolution`: for the data being used, 1, 2, or 6 km
- `repeat_dt`: time to wait between releasing batches of particles (in seconds)
- `particles_per_dt`: number of particles to release after every interval of `repeat_dt`
- `max_variation`: the max amount the longitude and latitude of a particle will be randomized by
- `spawn_points`: list of [latitude, longitude] pairs to choose from as a spawn location
- `simulation_dt`: the actual dt used by parcels for the simulation (in seconds)
- `snapshot_interval`: time to wait between taking a snapshot of the simulation to save as an image in seconds
