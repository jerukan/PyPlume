import json
import numpy as np
import xarray as xr
from parcels import FieldSet, ParticleSet, JITParticle

DATA_6KM = 6
DATA_2KM = 2
DATA_1KM = 1

filename_dict = {
    DATA_6KM: "west_coast_6km_hourly",
    DATA_2KM: "west_coast_2km_hourly",
    DATA_1KM: "west_coast_1km_hourly"
}


def xr_dataset_to_fieldset(xrds, copy=True, mesh="spherical"):
    """
    Creates a parcels FieldSet with an ocean current xarray Dataset.

    Args:
        xrds (xr.Dataset)
        copy (bool)
        mesh (str): spherical or flat
    """
    if copy:
        ds = xrds.copy(deep=True)
    else:
        ds = xrds
    return FieldSet.from_xarray_dataset(
            ds,
            dict(U="u", V="v"),
            dict(lat="lat", lon="lon", time="time"),
            mesh=mesh
        )


def get_file_info(name, path, res, parcels_cfg=None):
    """
    Reads from a netcdf file containing ocean current data.

    Args:
        name (str): whatever name the data should be labeled as.
        path (str): path to the netcdf file.
        res (int): resolution of the data.
        parcels_cfg (dict): a dictionary of parameters used for configuring Parcels simulations.

    Returns:
        dict: contains almost all useful information related to the data.
    """
    xrds = xr.open_dataset(path)
    # spherical mesh
    fs = xr_dataset_to_fieldset(xrds)
    # flat mesh
    fs_flat = xr_dataset_to_fieldset(xrds, mesh="flat")
    xrds.close()
    lat = xrds["lat"].values
    lon = xrds["lon"].values
    return dict(
        name=name,
        path=path,
        res=res,
        xrds=xrds,  # xarray Dataset
        fs=fs,
        fs_flat=fs_flat,
        timerng=(xrds["time"].min().values, xrds["time"].max().values),
        timerng_secs=fs.gridset.dimrange("time"),
        lat=lat,
        lon=lon,
        domain={
            "S": lat.min(),
            "N": lat.max(),
            "W": lon.min(),
            "E": lon.max(),
        },  # mainly for use with showing a FieldSet and restricting domain
        cfg=parcels_cfg
    )


def show_particles(fs, lats, lons):
    """
    Quick and dirty way to graph a collection of particles using ParticleSet.show()

    Args:
        fs (parcels.FieldSet)
        lats (array-like): 1-d array of particle latitude values
        lons (array-like): 1-d array of particle longitude values
    """
    pset = ParticleSet(fs, pclass=JITParticle, lon=lons, lat=lats)
    pset.show()


def load_config(path):
    """
    Returns a json file as a dict.

    Args:
        path (str)

    Returns:
        dict: data pulled from the json specified.
    """
    with open(path) as f:
        config = json.load(f)
    # TODO do some config verification here
    return config


def conv_to_dataarray(arr, darr_ref):
    """
    Takes in some array and converts it to a labelled xarray DataArray.

    Args:
        arr (array-like)
        darr_ref (xr.DataArray): only used to label coordinates, dimensions, and metadata.
            Must have the same shape as arr.
    """
    return xr.DataArray(arr, coords=darr_ref.coords, dims=darr_ref.dims, attrs=darr_ref.attrs)


def generate_mask(data):
    """
    Generates a boolean mask signifying which points in the data are invalid.

    Args:
        data (np.ndarray): an array with the shape of (time, lat, lon).

    Returns:
        np.ndarray: boolean mask with the same shape as data.
    """
    mask = np.zeros(data.shape, dtype=bool)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            point = data.T[j][i]
            nan_vals = np.isnan(point)
            # if the point at (lat, lon) contains real data and nan values
            # mark those points as invalid
            if not nan_vals.all():
                mask.T[j][i][:] = np.where(nan_vals.flatten(), 1, 0)
    return mask
