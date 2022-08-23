from dataclasses import dataclass
import logging

import numpy as np
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
        self.load_method = load_method if load_method is not None else lambda src: xr.open_dataset(src)

    def get_dataset_by_id(self, id) -> DatasetInfo:
        with_id = list(filter(lambda ds: ds.id == id, self.available_datasets))
        if len(with_id) == 0: return None
        return with_id[0]

    def load_source(self, src):
        """
        Loads a dataset from some source, and processes it so it is a standard format
        for the simulation to read.

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
        return self.load_method(src)


def parse_time_chunk_size(time_chunk_size):
    if time_chunk_size is not None:
        return {"time": time_chunk_size}
    return None


def get_simple_load_method(mappings=None, drop_vars=None, time_chunk_size=None):
    mappings = mappings if mappings is not None else dict()
    drop_vars = drop_vars if drop_vars is not None else set()
    time_chunks = parse_time_chunk_size(time_chunk_size)
    def new_load_method(src):
        ds = xr.open_dataset(src, chunks=time_chunks, drop_variables=drop_vars)
        return rename_dataset_vars(ds, mappings)
    return new_load_method


def rename_dataset_vars(src, mappings):
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
    if isinstance(src, xr.Dataset):
        ds = src
    else:
        with xr.open_dataset(src) as opened:
            ds = opened
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
        # step size is an integer in hours
        if inclusive:
            interval = time_range[2]
            time_range = utils.include_coord_range(time_range[0:2], ref_coords)
            time_range = (time_range[0], time_range[1], interval)
        return slice(np.datetime64(time_range[0]), np.datetime64(time_range[1]), time_range[2])


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
