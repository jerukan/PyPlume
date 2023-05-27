# Parcels simulation configuration

This is where the configs go.

**Note many of the options in the config are optional.**

example layout:

```yaml
# this section should contain all information about the surface
# current datasets you want to run simulations on
# you can have multiple ocean datasets to compare different models
ocean_data:
  - name: dataset name
    data: url or path to surface current data
    # spatial domain of data to extract: South, North, West, East
    domain:
      S: 38.162201
      N: 41.520008
      W: -75.709632
      E: -69.723751
    # time range of data to extract: [start time, end time]
    time_range: [2021-08-21T12:00, 2021-08-23T18:00]
    # define how to perform spatial gapfilling of the data
    gapfill_steps:
      # example here using penalized least squares filling
      - path: pyplume.gapfilling.DCTPLS
        args:
          mask: path to mask
    # null, partialslip, or freeslip
    boundary_condition: freeslip
    # specify wind data to modify the behavior of particle advection
    # currently only supports a single timeseries of wind to apply
    # uniformly across the surface currents
    wind:
      data: path to data
      # ratio of wind speed to add to surface currents
      ratio: 0.15
      # if direction of wind in data is incoming or outgoing
      incoming: False
  # include other datasets for comparison if needed
  - name: another dataset
parcels_config:
  # The DIRECTORY to save the particle netcdf file to
  save_dir: directory to save to
  # Can be either of the built-ins: JITParticle, ScipyParticle
  # or a custom one wrote in kernels.py
  particle_type: pyplume.kernels.ThreddsParticle
  # List of kernels to add, can be built ins like AdvectionRK4
  # or custom ones in kernels.py. Feel free to add new ones to the file.
  kernels:
    - pyplume.kernels.AdvectionRK4BorderCheck
    - pyplume.kernels.AgeParticle
    - pyplume.kernels.RandomWalk5cm
    - pyplume.kernels.DeleteAfter3Days
  # The start and end time of the simulation. START and END are special keywords that
  # denote the start and end times of the used NetCDF respectively.
  ###### IMPORTANT NOTE ######
  # if particles are scheduled to released individually, START will use the
  # earliest time a particle is released instead of the start of the NetCDF. It is highly
  # recommended to use START to avoid out of bounds errors.
  ### END OF IMPORTANT NOTE ###
  # Integers will be treated as delta time in hours. For example, [6, "END"] will run the
  # simulation from 6 hours before the end until the end
  time_range: [2020-07-16T00:00, END]
  ########## repetition settings ##########
  # The number of times to release the particles. If repeat_dt is 0 or less, this parameter
  # is ignored.
  repetitions: 5  # default -1
  # The time between particle spawn release in seconds. If this is positive and repetitions
  # is 0 or less, the simulation will attempt to release at the said rate until the end of
  # the simulation
  repeat_dt: 14400  # default -1
  # Number of times per release to spawn particles from the defined spawn points.
  # There is only a visual difference if there is randomness to particle movement.
  instances_per_spawn: 50  # default 1
  # spawn points can be either loaded in straight as a 2d list of (lat, lon) pairs
  ####### end of repetition settings #######
  spawn_points:
    - [32.551707, -117.138]
    - [32.557, -117.138]
    # for more complex release types and settings, you can specify individual information
    # about how to spawn a particular point
      # optional label for particle. doesn't do anything right now lol
    - label: RADARSAT2
      # [lat, lon] pair
      point: [33.6495, -118.1079],
      # when to release the particle. defaults to the start of the simulation.
      # can be set to START if you really want to
      release: 2021-10-01T21:00
      ######## spawn repetition settings ########
      # if any of these settings are missing or null, they automatically default to the
      # outer settings defined above
      repetitions: 5  # default: from outer settings
      repeat_dt: 14400  # default: from outer settings
      instances_per_spawn: 50  # default: from outer settings
      ##### end of spawn repetition settings #####
      # optional, specifies if a single point should be spawned in a pattern instead
      # this example spawns particles in a 3x3 grid around specified point
      # maybe there will be documentation on this lol
      pattern:
        type: grid
        size: 3
        gapsize: 0.03
  # or they can be loaded in from a .mat file
  # the .mat file must contain 2 variables representing longitude and latitude
  # dimension of the matrices don't matter since the data is flattened
  spawn_points:
    path: matlab/glist90zj_pts_position_spawn.mat
    # if the mat file has more variables than just the two lists, I highly suggest
    # specifying these variables to avoid unwanted behavior.
    lat_var: yf
    lon_var: xf
  # dt of the simulation in seconds
  simulation_dt: 300
  # how often data is recorded to the particle file in seconds
  snapshot_interval: 10800
# settings to modify the resulting ParticleFile from the simulation
postprocess_config:
  # loads a coastline to detect collisions and delete particles that collide
  coastline:
    path: matlab/coastline.mat
    lat_var: latz0
    lon_var: lonz0
  # loads a buoy path to calculate deviation from the path for each particle
  # the variable will be called "buoy_distances" in the netcdf file
  buoy
    path: buoy_data/wavebuoy_704-02.csv
# if the plotting config is empty, no plots are generated
plotting_config:
  # list the types of plots you want to generate
  # can be found in pyplume/resultplots.py
  plots:
    - type: pyplume.resultplots.ParticlePlot
      label: particleplots
      coastline: true
      draw_currents: true
      domain:
        S: 33
        N: 34
        W: -118.5
        E: -117.4
```
