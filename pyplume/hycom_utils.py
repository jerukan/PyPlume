"""
Adding function to interprety HYCOM

With the new refactoring, this file should not be needed
anymore, but this will still be kept.
"""
import datetime
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from parcels import FieldSet, plotting
import scipy.spatial
import xarray as xr

import plotting
import utils


def xr_dataset_to_fieldset(xrds, copy=True, raw=True, mesh="spherical") -> FieldSet:
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
    if raw:
        fieldset = FieldSet.from_data(
            {"U": ds["water_u"].values, "V": ds["water_v"].values},
            {"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
            mesh=mesh
        )
    else:
        fieldset = FieldSet.from_xarray_dataset(
                ds,
                dict(U="water_u", V="water_v"),
                dict(lat="lat", lon="lon", time="time"),
                mesh=mesh
            )
    fieldset.check_complete()
    return fieldset




class HYCOMGrid:
    """
    Wraps information relating to ocean current data given some dataset.
    Replaces get_file_info.

    TODO generate the mask of where data should be available
    """
    def __init__(self, dataset, init_fs=True):
        """
        Reads from a netcdf file containing ocean current data.

        Args:
            dataset (path-like or xr.Dataset): represents the netcdf ocean current data.
        """
#         print("hello")
        if isinstance(dataset, (Path, str)):
            self.path = dataset
            with xr.open_dataset(dataset,decode_times=True) as ds:
                self.xrds = ds
        elif isinstance(dataset, xr.Dataset):
            self.path = None
            self.xrds = dataset
        else:
            raise TypeError(f"{dataset} is not a path or xarray dataset")
        self.times = self.xrds["time"].values
        self.lats = self.xrds["lat"].values
        self.lons = self.xrds["lon"].values
        self.timeKDTree = scipy.spatial.KDTree(np.array([self.times]).T)
        self.latKDTree = scipy.spatial.KDTree(np.array([self.lats]).T)
        self.lonKDTree = scipy.spatial.KDTree(np.array([self.lons]).T)
        if init_fs:
            self.prep_fieldsets()
        else:
            self.fieldset = None
            self.fieldset_flat = None
        # for caching
        self.u = None
        self.v = None
        


    def prep_fieldsets(self):
        # spherical mesh
        self.fieldset = xr_dataset_to_fieldset(self.xrds)
        # flat mesh
        self.fieldset_flat = xr_dataset_to_fieldset(self.xrds, mesh="flat")

    def get_coords(self) -> tuple:
        """
        Returns:
            (times, latitudes, longitudes)
        """
        return self.times, self.lats, self.lons

    def get_domain(self) -> dict:
        """
        Returns:
            dict
        """
        return {
            "S": self.lats[0],
            "N": self.lats[-1],
            "W": self.lons[0],
            "E": self.lons[-1],
        }  # mainly for use with showing a FieldSet and restricting domain

    def get_closest_index(self, t=None, lat=None, lon=None):
        """
        Args:
            t (np.datetime64): time
            lat (float)
            lon (float)

        Returns:
            (time index, lat index, lon index)
        """
        return (self.timeKDTree.query([t])[1] if t is not None else None,
            self.latKDTree.query([lat])[1] if lat is not None else None,
            self.lonKDTree.query([lon])[1] if lon is not None else None)

    def get_closest_current(self, t, lat, lon):
        """
        Args:
            t (np.datetime64 or int): time (or the index directly)
            lat (float)
            lon (float)

        returns:
            (u component, v component)
        """
        if not isinstance(t, (int, np.integer)):
            if t < self.times.min() or t > self.times.max():
                print("Warning: time is out of bounds", file=sys.stderr)
        if lat < self.lats.min() or lat > self.lats.max():
            print("Warning: latitude is out of bounds", file=sys.stderr)
        if lon < self.lons.min() or lon > self.lons.max():
            print("Warning: latitude is out of bounds", file=sys.stderr)
        if isinstance(t, (int, np.integer)):
            t_idx = t
            _, lat_idx, lon_idx = self.get_closest_index(None, lat, lon)
        else:
            t_idx, lat_idx, lon_idx = self.get_closest_index(t, lat, lon)
        # cache the whole array because isel is slow when doing it individually
        if self.u is None:
            self.u = self.xrds["u"].values
        if self.v is None:
            self.v = self.xrds["v"].values
        return self.u[t_idx, lat_idx, lon_idx], self.v[t_idx, lat_idx, lon_idx]

    def get_fs_current(self, t, lat, lon, flat=True):
        """
        Gets the current information at a position from the fieldset instead of from the
        dataset.
        TODO: support for datetime64

        Args:
            t (float): time relative to the fieldset
            lat (float)
            lon (float)
            flat (bool): if true, use the flat fieldset. otherwise use the spherical fieldset
        """
        if flat:
            return self.fieldset_flat.U[t, 0, lat, lon], self.fieldset_flat.V[t, 0, lat, lon]
        return self.fieldset.U[t, 0, lat, lon], self.fieldset.V[t, 0, lat, lon]
    
    
    