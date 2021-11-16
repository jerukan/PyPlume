"""
A collection of methods wrapping OceanParcels functionalities.
"""
import sys

import numpy as np
import pandas as pd
from parcels import FieldSet, Field, VectorField
import scipy.spatial
import xarray as xr

import src.thredds_utils as thredds_utils
import src.utils as utils


def arrays_to_particleds(time, lat, lon) -> xr.Dataset:
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


def buoycsv_to_particleds(csv_path) -> xr.Dataset:
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


def convert_lon_readings(lons):
    """
    Converts a list of longitude values going from 0 to 360 to a scale that goes from
    -180 to 180. Probably applies to latitude too.
    """
    convert_idx = np.where(lons > 180)
    new_lons = -(180 - (lons - 180))
    copied = np.zeros(lons.shape)
    copied[:] = lons[:]
    copied[convert_idx] = new_lons[convert_idx]
    return copied


def clean_erddap_ds(ds):
    """
    Converts a dataset from ERDDAP to fit with the existing framework.
    """
    clean_ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    # convert longitude values to -180 to 180 range (instead of 360)
    new_lon = convert_lon_readings(clean_ds["lon"].values)
    clean_ds = clean_ds.assign_coords({"lon": new_lon})
    # retrieve only surface currents
    clean_ds = clean_ds.sel(depth=0.0)
    return clean_ds


def rename_dataset_vars(path):
    """
    Renames variable/coord keys in an NetCDF ocean current dataset.
    If the data has a depth dimension, it will be removed.

    Args:
        path (path-like or xr.Dataset)
    """
    MAPPINGS = {
        "depth": {"depth", "z"},
        "lat": {"lat", "latitude", "y"},
        "lon": {"lon", "longitude", "long", "x"},
        "time": {"time", "t"},
        "U": {"u", "water_u"},
        "V": {"v", "water_v"}
    }
    if isinstance(path, xr.Dataset):
        ds = path
    else:
        with xr.open_dataset(path) as opened:
            ds = opened
    rename_map = {}
    for var in ds.variables.keys():
        for match in MAPPINGS.keys():
            if var.lower() in MAPPINGS[match]:
                rename_map[var] = match
    ds = ds.rename(rename_map)
    if "depth" in ds["U"].dims:
        ds["U"] = ds["U"].sel(depth=0)
    if "depth" in ds["V"].dims:
        ds["V"] = ds["V"].sel(depth=0)
    return ds


def xr_dataset_to_fieldset(xrds, copy=True, raw=True, complete=True, **kwargs) -> FieldSet:
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
            {"U": ds["U"].values, "V": ds["V"].values},
            {"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
            **kwargs
        )
    else:
        fieldset = FieldSet.from_xarray_dataset(
                ds,
                dict(U="U", V="V"),
                dict(lat="lat", lon="lon", time="time"),
                **kwargs
            )
    if complete:
        fieldset.check_complete()
    return fieldset


def read_netcdf_info(**netcdf_cfg):
    cfg = utils.get_path_cfg(netcdf_cfg)
    if "type" not in cfg or cfg["type"] == "file":
        # if type not specified or something, just try to open it
        with xr.open_dataset(netcdf_cfg["path"]) as ds:
            return ds
    if cfg["type"] == "thredds":
        return thredds_utils.get_thredds_dataset(netcdf_cfg["path"], netcdf_cfg["time_range"],
            netcdf_cfg["lat_range"], netcdf_cfg["lon_range"], inclusive=True)


class HFRGrid:
    """
    Wraps information relating to ocean current data given some dataset.
    Replaces get_file_info.

    TODO generate the mask of where data should be available
    """
    def __init__(self, dataset, init_fs=True, fields=None, fs_kwargs=None):
        """
        Reads from a netcdf file containing ocean current data.

        Args:
            dataset (path-like or xr.Dataset): represents the netcdf ocean current data.
            fields (list[parcels.Field])
        """
        self.xrds = rename_dataset_vars(utils.open_ds_if_path(dataset))
        self.fields = fields
        self.times = self.xrds["time"].values
        self.lats = self.xrds["lat"].values
        self.lons = self.xrds["lon"].values
        self.timeKDTree = scipy.spatial.KDTree(np.array([self.times]).T)
        self.latKDTree = scipy.spatial.KDTree(np.array([self.lats]).T)
        self.lonKDTree = scipy.spatial.KDTree(np.array([self.lons]).T)
        self.fs_kwargs = fs_kwargs if fs_kwargs is not None else {}
        if init_fs:
            self.prep_fieldsets(**self.fs_kwargs)
        else:
            self.fieldset = None
            self.fieldset_flat = None
        # for caching
        self.u = None
        self.v = None
        self.modified = False

    def modify_with_wind(self, dataset, ratio=1.0):
        """
        Directly modify the ocean vector dataset and update the fieldsets.

        Args:
            dataset (xr.Dataset)
            ratio (float): percentage of how much of the wind vectors to add to the ocean currents
        """
        if len(dataset["U"].shape) == 1:
            # time only dimension
            for i, t in enumerate(self.xrds["time"]):
                wind_uv = dataset.sel(time=t.values, method="nearest")
                wu = wind_uv["U"].values.item()
                wv = wind_uv["V"].values.item()
                self.xrds["U"][i] += wu * ratio
                self.xrds["V"][i] += wv * ratio
            self.prep_fieldsets(**self.fs_kwargs)
            self.modified = True
        elif len(dataset["U"].shape) == 3:
            print("Ocean current vector modifications with wind vectors must be done"
                " individually. This may take a while.", file=sys.stderr)
            # assume dataset has renamed time, lat, lon dimensions
            # oh god why
            for i, t in enumerate(self.xrds["time"]):
                for j, lat in enumerate(self.xrds["lat"]):
                    for k, lon in enumerate(self.xrds["lon"]):
                        wind_uv = dataset.sel(
                            time=t.values, lat=lat.values, lon=lon.values, method="nearest"
                        )
                        wu = wind_uv["U"].values.item()
                        wv = wind_uv["V"].values.item()
                        self.xrds["U"][i, j, k] += wu * ratio
                        self.xrds["V"][i, j, k] += wv * ratio
            self.prep_fieldsets(**self.fs_kwargs)
            self.modified = True
        else:
            raise ValueError("dataset vectors don't have a dimension of 1 or 3")

    def add_field(self, fieldset, field, name=None):
        if isinstance(field, VectorField):
            fieldset.add_vector_field(field)
        elif isinstance(field, Field):
            fieldset.add_field(field, name=name)
        else:
            raise TypeError(f"{field} is not a valid field or vector field")

    def prep_fieldsets(self, **kwargs):
        if self.fields is not None:
            # spherical mesh
            kwargs["mesh"] = "spherical"
            self.fieldset = xr_dataset_to_fieldset(self.xrds, complete=False, **kwargs)
            [self.add_field(self.fieldset, fld) for fld in self.fields]
            self.fieldset.check_complete()
            # flat mesh
            kwargs["mesh"] = "flat"
            self.fieldset_flat = xr_dataset_to_fieldset(self.xrds, complete=False, **kwargs)
            [self.add_field(self.fieldset, fld) for fld in self.fields]
            self.fieldset_flat.check_complete()
        else:
            # spherical mesh
            kwargs["mesh"] = "spherical"
            self.fieldset = xr_dataset_to_fieldset(self.xrds, **kwargs)
            # flat mesh
            kwargs["mesh"] = "flat"
            self.fieldset_flat = xr_dataset_to_fieldset(self.xrds, **kwargs)

    def get_coords(self) -> tuple:
        """
        Returns:
            (times, latitudes, longitudes)
        """
        return self.times, self.lats, self.lons

    def get_domain(self, dtype='float32') -> dict:
        """
        Args:
            dtype: specify what type to convert the coordinates to. The data is normally stored in
             float64, but parcels converts them to float32, causing some rounding errors and domain
             errors in specific cases.

        Returns:
            dict
        """
        return {
            "S": self.lats.astype(dtype)[0],
            "N": self.lats.astype(dtype)[-1],
            "W": self.lons.astype(dtype)[0],
            "E": self.lons.astype(dtype)[-1],
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
        self.check_cache()
        return self.u[t_idx, lat_idx, lon_idx], self.v[t_idx, lat_idx, lon_idx]
    
    def check_cache(self):
        # cache the whole array because isel is slow when doing it individually
        if self.u is None or self.modified:
            self.u = self.xrds["U"].values
        if self.v is None or self.modified:
            self.v = self.xrds["V"].values
        self.modified = False

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


class BuoyPath:
    def __init__(self, lats, lons, times):
        """Sorted by time"""
        self.lats = np.array(lats)
        self.lons = np.array(lons)
        self.times = np.array(times)

    def in_time_bounds(self, time):
        return time >= self.times[0] and time <= self.times[-1]

    def get_interped_point(self, time):
        """
        If a timestamp is between two buoy timestamps, linearly interpolate the position of the
        buoy to get the position at the input time.
        Other words, assumes buoy moves at constant speed between those 2 points.
        """
        if not self.in_time_bounds(time):
            raise ValueError(f"{time} is out of bounds of the buoy path")
        for i, t in enumerate(self.times):
            if time < t:
                idx = i - 1
                break
        if time == self.times[idx]:
            return self.lats[idx], self.lons[idx]
        start = self.times[idx]
        end = self.times[idx + 1]
        seconds = (time - start) / np.timedelta64(1, "s")
        seconds_total = (end - start) / np.timedelta64(1, "s")
        lat = self.lats[i] + (self.lats[i + 1] - self.lats[i]) * (seconds / seconds_total)
        lon = self.lons[i] + (self.lons[i + 1] - self.lons[i]) * (seconds / seconds_total)
        return lat, lon

    @classmethod
    def from_csv(cls, path, lat_key="latitude", lon_key="longitude", time_key="timestamp"):
        """
        Reads buoy position data from a csv
        """
        df = pd.read_csv(path)
        # just in case for some reason it isn't already sorted by time
        df = df.sort_values(time_key)
        return cls(
            df[lat_key].values, df[lon_key].values,
            df[time_key].values.astype("datetime64[s]")
        )
