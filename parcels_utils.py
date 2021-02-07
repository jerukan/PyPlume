"""
A collection of methods wrapping OceanParcels functionalities.
"""
from pathlib import Path

import numpy as np
from parcels import FieldSet
import pandas as pd
import xarray as xr


def arrays_to_particleds(time, lat, lon):
    """
    Generates an xarray dataset in the same format ParticleFile saves as
    given several lists.

    Does not include data variable z or metadata.

    Args:
        times (np.ndarray[np.datetime64]): 2d array
        lats (np.ndarray[float]): 2d array
        lons (np.ndarray[float]): 2d array
        
    Returns:
        xr.Dataset
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


def buoycsv_to_particleds(csv_path):
    """
    Generates an xarray dataset in the same format ParticleFile saves as
    given a path to a csv file containing wave buoy data.

    Does not include data variable z or metadata.

    Args:
        path (path-like): path to csv file
    
    Returns:
        xr.Dataset
    """
    data = pd.read_csv(csv_path)
    # just in case for some reason it isn't already sorted by time
    data = data.sort_values("timestamp")
    times = np.empty((1, data.shape[0]), dtype="datetime64[s]")
    lats = np.zeros((1, data.shape[0]), dtype=np.float32)
    lons = np.zeros((1, data.shape[0]), dtype=np.float32)
    for i in range(data.shape[0]):
        row = data.iloc[i]
        times[0, i] = np.datetime64(int(row["timestamp"]), "s")
        lats[0, i] = row["latitude"]
        lons[0, i] = row["longitude"]
    return arrays_to_particleds(times, lats, lons)
    

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
    Use HFRGrid instead of this.

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


class HFRGrid:
    """
    Wraps information relating to ocean current data given some dataset.
    Replaces get_file_info
    """

    def __init__(self, dataset, resolution):
        """
        Reads from a netcdf file containing ocean current data.

        Args:
            dataset (path-like or xr.Dataset): represents the netcdf ocean current data.
            resolution (int): resolution of the data (500m, 1km, 2km, or 6km)
        """
        self.resolution = resolution
        if isinstance(dataset, (Path, str)):
            self.path = dataset
            with xr.open_dataset(dataset) as ds:
                self.xrds = ds
        elif isinstance(dataset, xr.Dataset):
            self.path = None
            self.xrds = dataset
        else:
            raise TypeError(f"dataset is not a path or xarray dataset")
        # spherical mesh
        self.fieldset = xr_dataset_to_fieldset(self.xrds)
        self.fieldset.check_complete()
        # flat mesh
        self.fieldset_flat = xr_dataset_to_fieldset(self.xrds, mesh="flat")
        self.fieldset_flat.check_complete()

    def reload_fieldsets(self):
        """
        Updates the parcels fieldset if the ocean current dataset has been changed.
        """
        self.fieldset = xr_dataset_to_fieldset(self.xrds)
        self.fieldset_flat = xr_dataset_to_fieldset(self.xrds, mesh="flat")

    def get_coords(self):
        """
        Returns:
            (times, latitudes, longitudes)
        """
        return self.xrds["time"].values, self.xrds["lat"].values, self.xrds["lon"].values

    def get_domain(self):
        """
        Returns:
            dict
        """
        _, lats, lons = self.get_coords()
        return {
            "S": lats.min(),
            "N": lats.max(),
            "W": lons.min(),
            "E": lons.max(),
        }  # mainly for use with showing a FieldSet and restricting domain