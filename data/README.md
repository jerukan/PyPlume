# Data format information

## Ocean surface current data

All data for ocean surface currents should be stored as [netCDF](https://www.unidata.ucar.edu/software/netcdf/) files.

### Required coordinates and variables

At minimum, coordinates for time, latitude, and longitude are required.

Data variables for eastward and westward components of surface currents are also required.

Below is an example of the minimum information required in a surface current data file.

![nc example](/images/example_nc_format.png)

PyPlume can attempt to guess what each coordinate and data variable name mean, but ambiguous variable names may require the user to specify which name corresponds to time, latitude, eastward velocity, etc.

**Each of the variables are required to be in specific units:**

- time: is some properly decoded datetime that can be read by numpy.
- latitude/longitude: decimal degrees are required
- U and V: m/s
