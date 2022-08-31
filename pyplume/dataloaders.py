from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from parcels import FieldSet, Field, VectorField
from parcels.tools.converters import GeographicPolar, Geographic
import scipy.spatial
import xarray as xr

import pyplume.utils as utils


VAR_MAPPINGS_DEFAULT = {
    "depth": {"depth", "z"},
    "lat": {"lat", "latitude", "y"},
    "lon": {"lon", "longitude", "long", "x"},
    "time": {"time", "t"},
    "U": {"u", "water_u"},
    "V": {"v", "water_v"}
}
# controls xarray dask chunks, needed for large datasets
CHUNK_SIZE_DEFAULT = "100MB"


def open_dataset(filename_or_obj, *args, **kwargs):
    """
    Wrapper for xr.open_dataset that wraps the method in a with block in order to close the file.
    """
    with xr.open_dataset(filename_or_obj, *args, **kwargs) as ds:
        return ds


@dataclass
class DatasetInfo:
    id: str
    name: str
    url: str


class DataSource:
    def __init__(
        self, id=None, name=None, available_datasets=None, load_method=None
    ):
        """
        Args:
            load_method (str -> xr.Dataset)
        """
        if id is None: raise TypeError("NoneType received")
        self.id = id
        self.name = name if name is not None else "Data source"
        self.available_datasets = available_datasets if available_datasets is not None else []
        self.load_method = load_method if load_method is not None else open_dataset

    def get_dataset_by_id(self, id) -> DatasetInfo:
        with_id = list(filter(lambda ds: ds.id == id, self.available_datasets))
        if len(with_id) == 0: return None
        return with_id[0]

    def load_source(self, src):
        """
        Loads a dataset from some source, and processes it so it is a standard format
        for the simulation to read.

        TODO verify correct data variables are in the dataset

        Args:
            src (str or path-like): the id to the data, or the url/path to the data

        Returns:
            xr.Dataset: a dataset in the standardized format
        """
        ds_info = self.get_dataset_by_id(src)
        if ds_info is not None:
            logging.info(f"Loading data type {ds_info.id} from {ds_info.url}")
            return self.load_method(ds_info.url)
        logging.info(f"Loading dataset from {src}")
        try:
            return self.load_method(src)
        except ValueError as e:
            # xarray loading failures are pretty vague, and are always ValueErrors. We give a bit
            # more info on them
            raise RuntimeError(f"Something went wrong with loading {src}") from e


DEFAULT_DATASOURCE = DataSource(id="default", name="Default data source")


def parse_time_chunk_size(time_chunk_size):
    if time_chunk_size is not None:
        return {"time": time_chunk_size}
    return None


def get_simple_load_method(mappings=None, drop_vars=None, time_chunk_size=None):
    mappings = mappings if mappings is not None else dict()
    drop_vars = drop_vars if drop_vars is not None else set()
    time_chunks = parse_time_chunk_size(time_chunk_size)
    def new_load_method(src):
        ds = open_dataset(src, chunks=time_chunks, drop_variables=drop_vars)
        return rename_dataset_vars(ds, mappings)
    return new_load_method


def rename_dataset_vars(src, mappings=None):
    """
    Renames variable/coord keys in an NetCDF ocean current dataset.

    Args:
        src (path-like or xr.Dataset)
        mappings (dict): format:
            {
                "standardized_var_name": {"other", "possible", "names"},
                ...
            }
    """
    if mappings is None: mappings = VAR_MAPPINGS_DEFAULT
    if isinstance(src, xr.Dataset):
        ds = src
    else:
        ds = open_dataset(src)
    rename_map = {}
    for var in ds.variables.keys():
        for match in mappings.keys():
            if var.lower() in mappings[match]:
                rename_map[var] = match
    ds = ds.rename(rename_map)
    return ds


def drop_depth(ds):
    """
    Depth will still be a coordinate in the whole dataset, but the U and V
    velocity data will not have depth information anymore.

    Args:
        ds (xr.Dataset): standardized vector field
    """
    if "depth" in ds["U"].dims:
        ds["U"] = ds["U"].sel(depth=0)
    if "depth" in ds["V"].dims:
        ds["V"] = ds["V"].sel(depth=0)
    return ds


def get_time_slice(time_range, inclusive=False, ref_coords=None, precision="h"):
    time_range = list(time_range)
    time_range[0] = np.datetime64(time_range[0])
    time_range[1] = np.datetime64(time_range[1])
    if time_range[0] == time_range[1]:
        return slice(np.datetime64(time_range[0], precision),
                     np.datetime64(time_range[1], precision) + np.timedelta64(1, precision))
    if len(time_range) == 2:
        if inclusive:
            if ref_coords is not None:
                time_range = utils.include_coord_range(time_range, ref_coords)
        return slice(np.datetime64(time_range[0]), np.datetime64(time_range[1]))
    if len(time_range) == 3:
        # step size is an integer, precision may vary
        if inclusive:
            interval = time_range[2]
            if ref_coords is not None:
                time_range = utils.include_coord_range(time_range[0:2], ref_coords)
            time_range = (time_range[0], time_range[1], interval)
        return slice(np.datetime64(time_range[0]), np.datetime64(time_range[1]), time_range[2])
    raise ValueError("time_range is not a proper value")


def get_latest_span(delta):
    # GMT, data recorded hourly
    time_now = np.datetime64("now", "h")
    return (time_now - delta, time_now)


def check_bounds(dataset, lat_range, lon_range, time_range):
    times = dataset["time"].values
    lats = dataset["lat"].values
    lons = dataset["lon"].values
    span_checker = lambda rng, coords: rng[0] <= coords[0] or rng[1] >= coords[-1]
    if span_checker(time_range, times):
        print("Timespan reaches the min/max of the range")
    if span_checker(lat_range, lats):
        print("Latitude span reaches the min/max of the range")
    if span_checker(lon_range, lons):
        print("Longitude span reaches the min/max of the range")


def slice_dataset(ds, time_range=None, lat_range=None, lon_range=None,
        inclusive=False, padding=0.0) -> xr.Dataset:
    """
    Params:
        ds (xr.Dataset): formatted dataset
        time_range (np.datetime64, np.datetime64[, int]): (start, stop[, interval])
        lat_range (float, float)
        lon_range (float, float)
        inclusive (bool): set to true to keep the endpoints of all the ranges provided
        padding (float): lat and lon padding

    Returns:
        xr.Dataset
    """
    if lat_range is None:
        lat_range = (ds["lat"].min(), ds["lat"].max())
    else:
        if inclusive:
            lat_range = utils.include_coord_range(lat_range, ds["lat"].values)
        lat_range = (lat_range[0] - padding, lat_range[1] + padding)
    if lon_range is None:
        lon_range = (ds["lon"].min(), ds["lon"].max())
    else:
        if inclusive:
            lon_range = utils.include_coord_range(lon_range, ds["lon"].values)
        lon_range = (lon_range[0] - padding, lon_range[1] + padding)
    if time_range is None:
        time_slice = slice(ds["time"].min(), ds["time"].max())
    else:
        if not isinstance(time_range, slice):
            if time_range[0] == "START":
                time_range = (ds["time"].values[0], time_range[1])
            if time_range[1] == "END":
                time_range = (time_range[0], ds["time"].values[-1])
            time_slice = get_time_slice(time_range, inclusive=inclusive, ref_coords=ds["time"].values)
        else:
            time_slice = time_range
    check_bounds(ds, lat_range, lon_range, (time_slice.start, time_slice.stop))
    dataset_start = ds["time"].values[0]
    if time_slice.start >= np.datetime64("now") or time_slice.stop <= dataset_start:
        raise ValueError("Desired time range is out of range for the dataset")
    sliced_data = ds.sel(
        time=time_slice,
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    if len(sliced_data["time"]) == 0:
        raise ValueError("No timestamps inside given time interval")
    return sliced_data


class DataLoader:
    def __init__(
        self, dataset, datasource=None, time_range=None, lat_range=None, lon_range=None,
        inclusive=True, **_
    ):
        self.time_range = time_range
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.inclusive = inclusive
        if datasource is None: self.datasource = DEFAULT_DATASOURCE
        elif isinstance(datasource, str):
            module_str = ".".join(datasource.split(".")[:-1])
            var_str = datasource.split(".")[-1]
            module = importlib.import_module(module_str)
            self.datasource = getattr(module, var_str)
        else: self.datasource = datasource
        if isinstance(dataset, xr.Dataset): self.full_dataset = dataset
        elif isinstance(dataset, (str, Path)):
            self.full_dataset = self.datasource.load_source(dataset)
        else: raise TypeError("data is not a valid type")
        self.dataset = slice_dataset(
            self.full_dataset, time_range=time_range, lat_range=lat_range, lon_range=lon_range,
            inclusive=inclusive
        )

    def get_mask(self, num_samples=None):
        """
        Generate a mask from the data signifying which coordinates should have data and which ones
        shouldn't.

        Returns:
            array-like: Boolean mask of shape (lat, lon). True signifies data should exist, False
             signifies it shouldn't.
        """
        size = self.full_dataset["time"].size
        step = size // num_samples
        mask = slice_dataset(
            self.full_dataset, lat_range=self.lat_range, lon_range=self.lon_range,
            inclusive=self.inclusive
        ).isel(time=slice(0, size, step))
        return ~utils.generate_mask_no_data(mask)


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


def dataset_to_fieldset(ds, copy=True, raw=True, complete=True, boundary_condition=None, **kwargs) -> FieldSet:
    """
    Creates a parcels FieldSet with an ocean current xarray Dataset.
    copy is true by default since Parcels has a habit of turning nan values into 0s.

    Args:
        ds (xr.Dataset)
        copy (bool)
        raw (bool): if True, all the data is immediately loaded
        complete (bool)
        mesh (str): spherical or flat
        boundary_condition
        kwargs: keyword arguments to pass into FieldSet creation
    """

    if isinstance(boundary_condition, str):
        kwargs.pop("interp_method", None)
        if boundary_condition.lower() in ("free", "freeslip"):
            interp_method = {"U": "freeslip", "V": "freeslip"}
        elif boundary_condition.lower() in ("partial", "partialslip"):
            interp_method = {"U": "partialslip", "V": "partialslip"}
        else:
            raise ValueError(f"Invalid boundary_condition: {boundary_condition}")
    else:
        interp_method = kwargs.pop("interp_method", "linear")
    if copy:
        ds = ds.copy(deep=True)
    else:
        ds = ds
    if raw:
        fieldset = FieldSet.from_data(
            {"U": ds["U"].values, "V": ds["V"].values},
            {"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
            interp_method=interp_method, **kwargs
        )
    else:
        fieldset = FieldSet.from_xarray_dataset(
                ds,
                dict(U="U", V="V"),
                dict(lat="lat", lon="lon", time="time"),
                interp_method=interp_method, **kwargs
            )
    if complete:
        fieldset.check_complete()
    return fieldset


def dataset_to_vectorfield(ds, u_name, v_name, uv_name) -> VectorField:
    fu = Field.from_xarray(
        ds["U"], u_name, dict(lat="lat", lon="lon", time="time"), interp_method="nearest"
    )
    fu.units = GeographicPolar()
    fv = Field.from_xarray(
        ds["V"], v_name, dict(lat="lat", lon="lon", time="time"), interp_method="nearest"
    )
    fv.units = Geographic()
    return VectorField(uv_name, fu, fv)


class SurfaceGrid:
    """
    Wraps information relating to ocean current data given some dataset.

    TODO generate the mask of where data should be available
    """
    def __init__(self, dataset, init_fs=True, other_fields=None, boundary_condition=None, **fs_kwargs):
        """
        Reads from a netcdf file containing ocean current data.

        Args:
            dataset (xr.Dataset): formatted netcdf data
            fields (list[parcels.Field])
        """
        self.ds = dataset
        self.other_fields = other_fields
        self.times = self.ds["time"].values
        self.lats = self.ds["lat"].values
        self.lons = self.ds["lon"].values
        self.timeKDTree = scipy.spatial.KDTree(np.array([self.times]).T)
        self.latKDTree = scipy.spatial.KDTree(np.array([self.lats]).T)
        self.lonKDTree = scipy.spatial.KDTree(np.array([self.lons]).T)
        self.boundary_condition = boundary_condition
        self.fs_kwargs = fs_kwargs if fs_kwargs is not None else {}
        if init_fs:
            self.prep_fieldsets(boundary_condition=self.boundary_condition, **self.fs_kwargs)
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
            for i, t in enumerate(self.ds["time"]):
                wind_uv = dataset.sel(time=t.values, method="nearest")
                wu = wind_uv["U"].values.item()
                wv = wind_uv["V"].values.item()
                self.ds["U"][i] += wu * ratio
                self.ds["V"][i] += wv * ratio
            self.prep_fieldsets(**self.fs_kwargs)
            self.modified = True
        elif len(dataset["U"].shape) == 3:
            print("Ocean current vector modifications with wind vectors must be done"
                " individually. This may take a while.", file=sys.stderr)
            # assume dataset has renamed time, lat, lon dimensions
            # oh god why
            for i, t in enumerate(self.ds["time"]):
                for j, lat in enumerate(self.ds["lat"]):
                    for k, lon in enumerate(self.ds["lon"]):
                        wind_uv = dataset.sel(
                            time=t.values, lat=lat.values, lon=lon.values, method="nearest"
                        )
                        wu = wind_uv["U"].values.item()
                        wv = wind_uv["V"].values.item()
                        self.ds["U"][i, j, k] += wu * ratio
                        self.ds["V"][i, j, k] += wv * ratio
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

    def prep_fieldsets(self, boundary_condition=None, **kwargs):
        """
        Args:
            kwargs: keyword arguments to pass into FieldSet creation
        """
        kwargs.pop("mesh", None)
        if self.other_fields is not None:
            # spherical mesh
            self.fieldset = dataset_to_fieldset(
                self.ds, complete=False, boundary_condition=boundary_condition, mesh="spherical",
                **kwargs
            )
            [self.add_field(self.fieldset, fld) for fld in self.other_fields]
            self.fieldset.check_complete()
            # flat mesh
            self.fieldset_flat = dataset_to_fieldset(
                self.ds, complete=False, boundary_condition=boundary_condition, mesh="flat",
                **kwargs
            )
            [self.add_field(self.fieldset, fld) for fld in self.other_fields]
            self.fieldset_flat.check_complete()
        else:
            # spherical mesh
            self.fieldset = dataset_to_fieldset(
                self.ds, boundary_condition=boundary_condition, mesh="spherical", **kwargs
            )
            # flat mesh
            self.fieldset_flat = dataset_to_fieldset(
                self.ds, boundary_condition=boundary_condition, mesh="flat", **kwargs
            )

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
            self.u = self.ds["U"].values
        if self.v is None or self.modified:
            self.v = self.ds["V"].values
        self.modified = False

    def get_fs_vector(self, t, lat, lon, flat=True):
        """
        Gets the flow vector information at a position from the fieldset instead of from the
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

    @classmethod
    def from_url_or_path(cls, path, dssrc: DataSource, **kwargs):
        """
        Args:
            kwargs: keyword arguments for SurfaceGrid
        """
        ds = dssrc.load_source(path)
        return cls(ds, **kwargs)


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
