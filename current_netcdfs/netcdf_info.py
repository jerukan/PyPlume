"""
Displays stuff about some ocean current netcdf data.
Note that this file is in the directory where netcdf files are stored.
"""
import sys
import xarray as xr

if __name__ == "__main__":
    path = sys.argv[1]
    xrds = xr.open_dataset(path)
    print(xrds["time"])
    print("----------------------------------------")
    print(xrds["lat"])
    print("----------------------------------------")
    print(xrds["lon"])
