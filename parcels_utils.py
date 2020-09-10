"""
A collection of methods wrapping OceanParcels functionalities.
"""
import numpy as np
from parcels import FieldSet
import xarray as xr


def arrays_to_particlefilenc(time, lat, lon):
    """
    Generates an xarray dataset in the same format ParticleFile saves as
    given several lists.

    Does not include data variable z or metadata.

    Args:
        times (np.ndarray[np.datetime64]): 2d array
        lats (np.ndarray[float]): 2d array
        lons (np.ndarray[float]): 2d array
    """
    lat = np.array(lat, dtype=np.float32)
    lon = np.array(lon, dtype=np.float32)
    trajectory = np.repeat(
        np.arange(time.shape[0])[np.newaxis, :].T,
        time.shape[1],
        axis=1
    )
    ds = xr.Dataset(
        {
            "trajectory": (["traj", "obs"], trajectory),
            "time": (["traj", "obs"], time),
            "lat": (["traj", "obs"], lat),
            "lon": (["traj", "obs"], lon),
        }
    )
    return ds


def xr_dataset_to_fieldset(xrds, copy=True, mesh="spherical", u_key="u", v_key="v"):
    """
    Creates a parcels FieldSet with an ocean current xarray Dataset.
    copy is true by default since Parcels has a habit of turning nan values into 0s.

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
            dict(U=u_key, V=v_key),
            dict(lat="lat", lon="lon", time="time"),
            mesh=mesh
        )


def get_file_info(path, res, name=None, parcels_cfg=None):
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
    if name is None:
        name = path
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


def reload_file_fs(file_info):
    file_info["fs"] = xr_dataset_to_fieldset(file_info["xrds"])
    file_info["fs_flat"] = xr_dataset_to_fieldset(file_info["xrds"], mesh="flat")
