# netcdf file storage

All netCDF files containing ocean current information should go here

## Format

The minimum variables needed to run the simulation, and names of said variables/coordinates.

![nc example](/images/example_nc_format.png)

It's fine if there are more variables as long as these exist.

## Units

Where latitude/longitude are the usualy degrees, U and V are in m/s, and time is some properly decoded datetime that can be read by numpy.
