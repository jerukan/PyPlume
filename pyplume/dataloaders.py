import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from parcels import FieldSet, Field, VectorField
from parcels.tools.converters import GeographicPolar, Geographic
import scipy.spatial
import xarray as xr

from pyplume import get_logger
import pyplume.utils as utils

logger = get_logger(__name__)


def load_pos_from_dict(data, lat_key=None, lon_key=None, infer_keys=True):
    """
    Gets positional data (longitude and latitude) arrays from a dictionary.

    Keys for either of the position can be defined or guessed naively.

    Args:
        data (dict)
        lat_key (str): Explicitly defines which key represents latitude values.
        lon_key (str): Explicitly defines which key represents longitude values.
        infer_keys (bool): If True, will attempt to naively guess which key in the
            data represents latitude or longitude.

    Returns:
        latitudes
        longitudes
    """
    possible_lat_keys = {"y", "lat", "lats", "latitude", "latitudes"}
    possible_lon_keys = {"x", "lon", "lons", "longitude", "longitudes"}

    def guess_key(keys, possibilities):
        guessed = None
        for key in keys:
            if key.lower() in possibilities:
                guessed = key
                logger.info(f"Guessed key as {key}")
                return guessed
        if guessed is None:
            for possib in possibilities:
                for key in keys:
                    if key.lower()[: len(possib)] == possib:
                        guessed = key
                        logger.info(f"Guessed key as {key}")
                        return guessed
        raise IndexError(f"No key could be guessed from keys {keys}")

    if lat_key is None and infer_keys:
        lat_key = guess_key(data.keys(), possible_lat_keys)
    if lon_key is None and infer_keys:
        lon_key = guess_key(data.keys(), possible_lon_keys)
    return data[lat_key], data[lon_key]


def load_pts_mat(path, lat_key=None, lon_key=None, del_nan=False):
    """
    Loads points from a mat file.
    Only points where both lat and lon are non-nan are returned.

    Args:
        path: path to mat file

    Returns:
        np.ndarray: [lats], [lons]
    """
    mat_data = scipy.io.loadmat(path)
    yf, xf = load_pos_from_dict(
        mat_data, lat_key=lat_key, lon_key=lon_key, infer_keys=True
    )
    xf = np.ravel(xf)
    yf = np.ravel(yf)
    if del_nan:
        # filter out nan values
        non_nan = (~np.isnan(xf)) & (~np.isnan(yf))
        xf = xf[np.where(non_nan)]
        yf = yf[np.where(non_nan)]
    return yf, xf


def load_geo_points(data, **kwargs):
    """
    Loads a collection of (lat, lon) points from a given data configuration. Each different file
    type will have different ways of loading and different required parameters.

    .mat file arguments:
        lat_key: variable in the mat file representing the array of latitude values
        lon_key: variable in the mat file representing the array of longitude values

    .npy file arguments:
        lat_key
        lon_key

    Args:
        data: actual data or path to data

    Returns:
        lats (array): flattened
        lons (array): flattened
    """
    if isinstance(data, (np.ndarray, list)):
        return utils.get_points(np.array(data), dim=2)
    if isinstance(data, (str, Path)):
        path = data
        ext = os.path.splitext(path)[1]
        if ext == ".mat":
            lats, lons = load_pts_mat(path, **kwargs)
            return lats, lons
        if ext == ".npy":
            npdata = np.load(path)
            if isinstance(npdata, dict):
                lats, lons = load_pos_from_dict(data, **kwargs)
                return np.ravel(lats), np.ravel(lons)
            return utils.get_points(npdata, dim=2)
        raise ValueError(f"Invalid extension {ext}")
    raise TypeError(f"Invalid data type {type(data)}")


def load_timeseries_data(data, **kwargs):
    """
    Timeseries data as a xarray dataset.

    The time axis is assumed to be named "time".

    Returns:
        xr.Dataset
    """
    if isinstance(data, (str, Path)):
        path = data
        ext = os.path.splitext(path)[1]
        if ext == ".mat":
            data_vars = {}
            mat_data = scipy.io.loadmat(path)
            time = mat_data["time"].flatten()
            del mat_data["time"]
            for key, val in mat_data.items():
                if isinstance(val, np.ndarray):
                    val = val.flatten()
                    if val.shape == time.shape:
                        data_vars[key] = (["time"], val)
            ds = xr.Dataset(data_vars=data_vars, coords={"time": time})
            return ds
        if ext in (".nc", ".nc4"):
            return xr.open_dataset(path)
        if ext in (".txt", ".csv"):
            data_vars = {}
            sep = kwargs.get("sep", None)
            df = pd.read_csv(path, sep=sep)
            time = df["time"]
            df = df.drop(columns=("time"))
            for colname in df.columns:
                col = df[colname]
                if val.shape == time.shape:
                    data_vars[colname] = (["time"], col)
            ds = xr.Dataset(data_vars=data_vars, coords={"time": time})
            return ds
        raise ValueError(f"Invalid extension {ext}")
    raise TypeError(f"Invalid data type {type(data)}")


def _remove_redundant_maps(mapping):
    mapping_copy = {}
    for k, v in mapping.items():
        if k != v:
            mapping_copy[k] = v
    return mapping_copy


WIND_MAPPINGS = {
    "dir": {"dir", "direction", "ang", "angle"},
    "mag": {"mag", "spd", "speed", "magnitude"},
    "U": {"u"},
    "V": {"v"},
}


def guess_wind_keys(keys, exclude=None):
    if exclude is None:
        exclude = []
    mappings = {}
    checked = set()
    for key in keys:
        for target, possible in WIND_MAPPINGS.items():
            if (
                target not in exclude
                and key.lower() in possible
                and target not in checked
            ):
                mappings[target] = key
                checked.add(target)
    return _remove_redundant_maps(mappings)


def load_wind_dataset(data, **kwargs):
    incoming = kwargs.get("incoming", True)
    degrees = kwargs.get("degrees", True)
    bearing = kwargs.get("bearing", False)
    if isinstance(data, xr.Dataset):
        ds = data
    else:
        ds = load_timeseries_data(data, **kwargs)
    key_mappings = guess_wind_keys(ds.data_vars)
    inv_map = {v: k for k, v in key_mappings.items()}
    ds = ds.rename_vars(name_dict=inv_map)
    if ("dir" in ds.data_vars or "mag" in ds.data_vars) and (
        "U" in ds.data_vars or "V" in ds.data_vars
    ):
        raise ValueError(
            "It is ambiguous if both polar and cartesian velocity information are provided in the wind dataset."
        )
    if "dir" in ds.data_vars:
        if degrees:
            dirs = np.deg2rad(ds["dir"])
        else:
            dirs = ds["dir"]
        if bearing:
            dirs = (np.pi / 2) - dirs
        ds["U"] = ds["mag"] * np.cos(dirs)
        ds["V"] = ds["mag"] * np.sin(dirs)
        if incoming:
            ds["U"] = -ds["U"]
            ds["V"] = -ds["V"]
    return ds


VAR_MAPPINGS_DEFAULT = {
    "depth": {"depth", "z"},
    "lat": {"lat", "latitude", "y"},
    "lon": {"lon", "longitude", "long", "x"},
    "time": {"time", "t"},
    "U": {"u", "water_u"},
    "V": {"v", "water_v"},
}
COORD_MAPPINGS = {
    "depth": {"depth", "z"},
    "lat": {"lat", "latitude", "y"},
    "lon": {"lon", "longitude", "long", "x"},
    "time": {"time", "t"},
}
# controls xarray dask chunks, needed for large datasets
CHUNK_SIZE_DEFAULT = "100MB"


def guess_ocean_datavars(keys, exclude=None):
    if exclude is None:
        exclude = []
    mappings = {}
    # look for keys in the format of "u total", "v total"
    # or just find keys that are "u" or "v"
    # total currents
    for key in keys:
        if "U" not in exclude:
            if ("u" in key.lower() and "tot" in key.lower()) or ("u" == key.lower()):
                mappings["U"] = key
        if "V" not in exclude:
            if ("v" in key.lower() and "tot" in key.lower()) or ("v" == key.lower()):
                mappings["V"] = key

    # attempt to look for uv keys, assume they look like "usomething" or "vsomething"
    def find_containing(target):
        found = []
        for key in keys:
            if target in key.lower():
                found.append(key)
        return found

    for found in ("U", "V"):
        if found not in exclude and found not in mappings.keys():
            possible = find_containing(found.lower())
            if len(possible) < 1:
                raise ValueError(
                    f"No column for '{found}' data found in {keys}.\nSpecify the U and V data keys with 'u_key' and 'v_key'!"
                )
            if len(possible) > 1:
                raise ValueError(
                    f"Column for '{found}' data ambiguous in {keys}.\nSpecify the U and V data keys with 'u_key' and 'v_key'!"
                )
            mappings[found] = possible[0]
    return _remove_redundant_maps(mappings)


def guess_ocean_coords(keys, exclude=None):
    if exclude is None:
        exclude = []
    mappings = {}
    checked = set()
    for key in keys:
        for target, possible in COORD_MAPPINGS.items():
            if (
                target not in exclude
                and key.lower() in possible
                and target not in checked
            ):
                mappings[target] = key
                checked.add(target)
    return _remove_redundant_maps(mappings)


def parse_time_chunk_size(time_chunk_size):
    if time_chunk_size is not None:
        return {"time": time_chunk_size}
    return None


class SimpleLoad:
    def __init__(self, mappings=None, drop_vars=None, time_chunk_size=None):
        self.mappings = mappings if mappings is not None else dict()
        self.drop_vars = drop_vars if drop_vars is not None else set()
        self.time_chunks = parse_time_chunk_size(time_chunk_size)

    def __call__(self, src):
        ds = xr.open_dataset(
            src, chunks=self.time_chunks, drop_variables=self.drop_vars
        )
        return rename_dataset_vars(ds, self.mappings)


class DefaultLoad:
    def __init__(
        self,
        u_key=None,
        v_key=None,
        time_key=None,
        lat_key=None,
        lon_key=None,
        depth_key=None,
        drop_vars=None,
        time_chunk_size=None,
    ):
        self.drop_vars = drop_vars
        if time_chunk_size is None:
            time_chunk_size = CHUNK_SIZE_DEFAULT
        self.time_chunks = parse_time_chunk_size(time_chunk_size)
        self.uv_map = {"U": u_key, "V": v_key}
        self.coord_map = {
            "time": time_key,
            "lat": lat_key,
            "lon": lon_key,
            "depth": depth_key,
        }

    def __call__(self, src):
        if isinstance(src, xr.Dataset):
            ds = src
        else:
            try:
                ds = xr.open_dataset(
                    src, chunks=self.time_chunks, drop_variables=self.drop_vars
                )
            except ValueError as e:
                errmsg = str(e)
                if "decode time units" in errmsg.lower():
                    raise ValueError(
                        "There was an issue with decoding times in one of the variables. Drop any unnecessary time variables with 'drop_vars'!"
                    ) from e
                if "did not find a match" in errmsg.lower():
                    raise ValueError(
                        f"Could not open {src}. Are you opening a NetCDF file and is the path/url correct?"
                    ) from e
                raise e
            except ZeroDivisionError as e:
                raise ValueError("Could not chunk dataset. There could be a missing coordinate in your data!") from e
        datavar_map = {}
        for key, val in self.uv_map.items():
            if val is not None:
                datavar_map[key] = val
        guessed_datavars = guess_ocean_datavars(
            ds.data_vars, exclude=datavar_map.keys()
        )
        for key, val in guessed_datavars.items():
            datavar_map[key] = val
        datavar_map = _remove_redundant_maps(datavar_map)
        inv_datavar_map = {v: k for k, v in datavar_map.items()}

        coord_map = {}
        for key, val in self.coord_map.items():
            if val is not None:
                coord_map[key] = val
        guessed_coords = guess_ocean_coords(ds.coords, exclude=coord_map.keys())
        for key, val in guessed_coords.items():
            coord_map[key] = val
        coord_map = _remove_redundant_maps(coord_map)
        inv_coord_map = {v: k for k, v in coord_map.items()}

        ds = ds.rename(inv_datavar_map)
        ds = ds.rename(inv_coord_map)
        return ds


def rename_dataset_vars(ds, mappings=None):
    """
    Renames variable/coord keys in an NetCDF ocean current dataset.

    Args:
        src (xr.Dataset)
        mappings (dict): format:
            {
                "standardized_var_name": {"other", "possible", "names"},
                ...
            }
    """
    if mappings is None:
        mappings = VAR_MAPPINGS_DEFAULT
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
        ds["U"] = ds["U"].isel(depth=0)
    if "depth" in ds["V"].dims:
        ds["V"] = ds["V"].isel(depth=0)
    if "depth" in ds.dims:
        ds = ds.drop_dims("depth")
    return ds


def replace_inf_with_nan(ds):
    whereinf = np.where(np.isinf(ds["U"]))
    ds["U"][whereinf] = np.nan
    ds["V"][whereinf] = np.nan
    return ds


def get_time_slice(time_range, inclusive=False, ref_coords=None, precision="h"):
    time_range = list(time_range)
    time_range[0] = np.datetime64(time_range[0])
    time_range[1] = np.datetime64(time_range[1])
    if time_range[0] == time_range[1]:
        return slice(
            np.datetime64(time_range[0], precision),
            np.datetime64(time_range[1], precision) + np.timedelta64(1, precision),
        )
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
        return slice(
            np.datetime64(time_range[0]), np.datetime64(time_range[1]), time_range[2]
        )
    raise ValueError("time_range is not a proper value")


def get_latest_span(delta):
    # GMT, data recorded hourly
    time_now = np.datetime64("now", "h")
    return (time_now - delta, time_now)


def slice_dataset(
    ds, time_range=None, lat_range=None, lon_range=None, inclusive=False
) -> xr.Dataset:
    """
    Params:
        ds (xr.Dataset): formatted dataset
        time_range (np.datetime64, np.datetime64[, int]): (start, stop[, interval])
        lat_range (float, float)
        lon_range (float, float)
        inclusive (bool): set to true to keep the endpoints of all the ranges provided

    Returns:
        xr.Dataset
    """
    sel_args = {}
    if lat_range is not None:
        lat_out_of_range = (
            lat_range[0] < ds["lat"].min() or lat_range[1] > ds["lat"].max()
        )
        if lat_out_of_range:
            warnings.warn(
                "A latitude value in the defined domain is out of range of the dataset."
            )
        if inclusive:
            lat_range = utils.include_coord_range(lat_range, ds["lat"].values)
        sel_args["lat"] = slice(lat_range[0], lat_range[1])
    if lon_range is not None:
        # for latitude values, 0-360 and -180-180 are both commonly used, so we have to
        # account for them
        lon_coords = ds["lon"].values
        range_360 = lon_range[0] > 180 or lon_range[1] > 180
        coords_360 = np.any(ds["lon"] > 180)
        # we only convert the range since converting coordinate points often breaks the
        # order of the points
        if coords_360 and not range_360:
            lon_range = (
                utils.convert180to360(lon_range[0]),
                utils.convert180to360(lon_range[1]),
            )
        if not coords_360 and range_360:
            lon_range = (
                utils.convert360to180(lon_range[0]),
                utils.convert360to180(lon_range[1]),
            )
        lon_out_of_range = (
            lon_range[0] < ds["lon"].min() or lon_range[1] > ds["lon"].max()
        )
        if lon_out_of_range:
            warnings.warn(
                "A longitude value in the defined domain is out of range of the dataset."
            )
        if inclusive:
            lon_range = utils.include_coord_range(lon_range, lon_coords)
        sel_args["lon"] = slice(lon_range[0], lon_range[1])
    if time_range is not None:
        if not isinstance(time_range, slice):
            if time_range[0] == "START":
                time_range = (ds["time"].values[0], time_range[1])
            if time_range[1] == "END":
                time_range = (time_range[0], ds["time"].values[-1])
            time_slice = get_time_slice(
                time_range, inclusive=inclusive, ref_coords=ds["time"].values
            )
        else:
            time_slice = time_range
        sel_args["time"] = time_slice
    sliced_data = ds.sel(**sel_args)
    if len(sliced_data["time"]) == 0:
        raise ValueError("No timestamps inside given time interval")
    return sliced_data


class DataLoader:
    """
    Utility class that handles the loading of datasets from files or data servers.
    """

    def __init__(
        self,
        dataset,
        load_method=None,
        domain=None,
        time_range=None,
        lat_range=None,
        lon_range=None,
        inclusive=True,
        **kwargs,
    ):
        """
        Loads a dataset locally given some queries or constraints. If the original dataset
        is too large, the constraints may be neccessary.

        Args:
            load_method (str -> xr.Dataset): If the dataset is in a drastically different
                format can cannot be loaded normally, a custom method can be defined.
            inclusive (bool): If True, will attempt to slice the dataset in away to
                keep the endpoints of the ranges included.
        """
        self.time_range = time_range
        if domain is not None and (lat_range is not None or lon_range is not None):
            raise ValueError(
                "Cannot define both domain and lat/lon ranges at the same time. Use one or the other!"
            )
        if domain is not None:
            self.lat_range = [domain["S"], domain["N"]]
            self.lon_range = [domain["W"], domain["E"]]
        else:
            self.lat_range = lat_range
            self.lon_range = lon_range
        self.inclusive = inclusive
        self.load_method = (
            load_method if load_method is not None else DefaultLoad(**kwargs)
        )
        if isinstance(dataset, xr.Dataset):
            self.full_dataset = dataset
        elif isinstance(dataset, (str, Path)):
            self.full_dataset = self.load_method(dataset)
        else:
            raise TypeError("data is not a valid type")
        required_coords = ("time", "lat", "lon")
        required_datavars = ("U", "V")
        for req in required_coords:
            if req not in self.full_dataset.coords:
                raise ValueError(
                    f"Coordinate {req} not in dataset. Rename or add it to the dataset!"
                )
        for req in required_datavars:
            if req not in self.full_dataset.data_vars:
                raise ValueError(
                    f"Variable {req} not in dataset. Rename or add it to the dataset!"
                )
        self.dataset = slice_dataset(
            self.full_dataset,
            time_range=self.time_range,
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            inclusive=self.inclusive,
        )
        if self.dataset.nbytes > 1e9:
            gigs = self.dataset.nbytes / 1e9
            warnings.warn(
                f"The dataset is over a gigabyte ({gigs} gigabytes). Make sure you are working with the right subset of data!"
            )
        self.dataset.load()
        self.dataset = replace_inf_with_nan(drop_depth(self.dataset))

    def __repr__(self):
        return repr(self.dataset)

    def _repr_html_(self):
        return self.dataset._repr_html_()

    def __str__(self):
        return str(self.dataset)

    def get_mask(self, num_samples=None):
        """
        Generate a mask from the data signifying which coordinates should have data and which ones
        shouldn't.

        Returns:
            array-like: Boolean mask of shape (lat, lon). True signifies data should exist, False
             signifies it shouldn't.
        """
        size = self.full_dataset["time"].size
        if num_samples is None or num_samples <= 0:
            time_slice = slice(0, size)
        else:
            step = size // num_samples
            time_slice = slice(0, size, step)
        sample_ds = slice_dataset(
            self.full_dataset.isel(time=time_slice),
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            inclusive=self.inclusive,
        )
        mask = ~utils.generate_mask_no_data(sample_ds["U"].values)
        logger.info(f"Generated mask for {self}")
        return mask

    def save(self, path):
        logger.info(f"{self.dataset.nbytes / 1024 / 1024} megabytes to save for {self}")
        result = self.dataset.to_netcdf(path)
        logger.info(f"Finished save for {self}")
        return result

    def save_mask(self, path, num_samples=None):
        mask = self.get_mask(num_samples=num_samples)
        logger.info(f"{mask.nbytes / 1024 / 1024} megabytes to save for mask {self}")
        with open(path, "wb") as f:
            result = np.save(f, mask)
            logger.info(f"Finished save for {self}")
            return result

    def close(self):
        self.full_dataset.close()
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


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
        np.arange(time.shape[0])[np.newaxis, :].T, time.shape[1], axis=1
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


def dataset_to_fieldset(
    ds, copy=True, raw=True, complete=True, boundary_condition=None, **kwargs
) -> FieldSet:
    """
    Creates a parcels FieldSet with an ocean current xarray Dataset.
    copy is true by default since Parcels has a habit of turning nan values into 0s.

    Args:
        ds (xr.Dataset)
        copy (bool): If True, pass a copy of the dataset into the fieldset instead since
            the fieldset modifies dataset values directly.
        raw (bool): If True, all the data is immediately loaded
        complete (bool): If True, Parcels will do a check for fieldset completeness
        mesh (str): 'spherical' or 'flat'
        boundary_condition (str): If needed, specify freeslip or partialslip. Otherwise,
            linear by default.
        **kwargs: keyword arguments to pass into FieldSet creation
    """

    if isinstance(boundary_condition, str):
        if "interp_method" in kwargs:
            del kwargs["interp_method"]
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
    if raw:
        fieldset = FieldSet.from_data(
            {"U": ds["U"].values, "V": ds["V"].values},
            {
                "time": ds["time"].values,
                "lat": ds["lat"].values,
                "lon": ds["lon"].values,
            },
            interp_method=interp_method,
            **kwargs,
        )
    else:
        fieldset = FieldSet.from_xarray_dataset(
            ds,
            dict(U="U", V="V"),
            dict(lat="lat", lon="lon", time="time"),
            interp_method=interp_method,
            **kwargs,
        )
    if complete:
        fieldset.check_complete()
    return fieldset


def dataset_to_vectorfield(
    ds, u_name, v_name, uv_name, interp_method="nearest"
) -> VectorField:
    fu = Field.from_xarray(
        ds["U"],
        u_name,
        dict(lat="lat", lon="lon", time="time"),
        interp_method=interp_method,
    )
    fu.units = GeographicPolar()
    fv = Field.from_xarray(
        ds["V"],
        v_name,
        dict(lat="lat", lon="lon", time="time"),
        interp_method=interp_method,
    )
    fv.units = Geographic()
    return VectorField(uv_name, fu, fv)


class SurfaceGrid:
    """
    Wraps information relating to ocean current data given some dataset.

    TODO generate the mask of where data should be available
    """

    def __init__(
        self,
        dataset,
        init_fs=True,
        other_fields=None,
        boundary_condition=None,
        **fs_kwargs,
    ):
        """
        Reads from a netcdf file containing ocean current data.

        Args:
            dataset (path-like or xr.Dataset): formatted netcdf data or path to it
            fields (list[parcels.Field])
        """
        if isinstance(dataset, (str, Path)):
            dataset = DataLoader(dataset).dataset
        self.dataset = dataset
        self.other_fields = other_fields
        self.times = self.dataset["time"].values
        self.lats = self.dataset["lat"].values
        self.lons = self.dataset["lon"].values
        self.lon_360 = np.any(self.lons > 180)
        self.timeKDTree = scipy.spatial.KDTree(np.array([self.times]).T)
        self.latKDTree = scipy.spatial.KDTree(np.array([self.lats]).T)
        self.lonKDTree = scipy.spatial.KDTree(np.array([self.lons]).T)
        self.boundary_condition = boundary_condition
        self.fs_kwargs = fs_kwargs if fs_kwargs is not None else {}
        if init_fs:
            self.prep_fieldsets(
                boundary_condition=self.boundary_condition, **self.fs_kwargs
            )
        else:
            self.fieldset = None
            self.fieldset_flat = None
        # for caching
        self.u = None
        self.v = None
        self.modified = False

    def modify_with_wind(self, dataset, ratio):
        """
        Directly modify the ocean vector dataset and update the fieldsets.

        Args:
            dataset (xr.Dataset)
            ratio (float): percentage of how much of the wind vectors to add to the ocean currents
        """
        if len(dataset["U"].shape) == 1:
            # time only dimension
            for i, t in enumerate(self.dataset["time"]):
                wind_uv = dataset.sel(time=t.values, method="nearest")
                wu = wind_uv["U"].values.item()
                wv = wind_uv["V"].values.item()
                self.dataset["U"][i] += wu * ratio
                self.dataset["V"][i] += wv * ratio
            self.prep_fieldsets(**self.fs_kwargs)
            self.modified = True
        elif len(dataset["U"].shape) == 3:
            logger.info(
                "Ocean current vector modifications with wind vectors must be done"
                " individually. This may take a while.",
                file=sys.stderr,
            )
            # assume dataset has renamed time, lat, lon dimensions
            # oh god why
            for i, t in enumerate(self.dataset["time"]):
                for j, lat in enumerate(self.dataset["lat"]):
                    for k, lon in enumerate(self.dataset["lon"]):
                        wind_uv = dataset.sel(
                            time=t.values,
                            lat=lat.values,
                            lon=lon.values,
                            method="nearest",
                        )
                        wu = wind_uv["U"].values.item()
                        wv = wind_uv["V"].values.item()
                        self.dataset["U"][i, j, k] += wu * ratio
                        self.dataset["V"][i, j, k] += wv * ratio
            self.prep_fieldsets(**self.fs_kwargs)
            self.modified = True
        else:
            raise ValueError("dataset vectors don't have a dimension of 1 or 3")

    def add_field_to_fieldset(self, fieldset, field, name=None):
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
        logger.info(
            f"Loading dataset of size {self.dataset.nbytes / 1e6} MB with shape {self.dataset['U'].shape} into fieldset"
        )
        if self.other_fields is not None:
            # spherical mesh
            self.fieldset = dataset_to_fieldset(
                self.dataset,
                complete=False,
                boundary_condition=boundary_condition,
                mesh="spherical",
                **kwargs,
            )
            # flat mesh
            self.fieldset_flat = dataset_to_fieldset(
                self.dataset,
                complete=False,
                boundary_condition=boundary_condition,
                mesh="flat",
                **kwargs,
            )
            for fld in self.other_fields:
                self.add_field_to_fieldset(self.fieldset, fld)
                self.add_field_to_fieldset(self.fieldset_flat, fld)
            self.fieldset.check_complete()
            self.fieldset_flat.check_complete()
        else:
            # spherical mesh
            self.fieldset = dataset_to_fieldset(
                self.dataset,
                boundary_condition=boundary_condition,
                mesh="spherical",
                **kwargs,
            )
            # flat mesh
            self.fieldset_flat = dataset_to_fieldset(
                self.dataset,
                boundary_condition=boundary_condition,
                mesh="flat",
                **kwargs,
            )

    def get_coords(self) -> tuple:
        """
        Returns:
            (times, latitudes, longitudes)
        """
        return self.times, self.lats, self.lons

    def get_domain(self, dtype="float32") -> dict:
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
            lon (float): converted between -180 to 180 and 0 to 360 range if needed

        Returns:
            (time index, lat index, lon index)
            note that any of the indices may be None
        """
        if lon is not None:
            if self.lon_360:
                lon = utils.convert180to360(lon)
            else:
                lon = utils.convert360to180(lon)
        return (
            self.timeKDTree.query([t])[1] if t is not None else None,
            self.latKDTree.query([lat])[1] if lat is not None else None,
            self.lonKDTree.query([lon])[1] if lon is not None else None,
        )

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
                logger.info("Warning: time is out of bounds")
        if lat < self.lats.min() or lat > self.lats.max():
            logger.info("Warning: latitude is out of bounds")
        if lon < self.lons.min() or lon > self.lons.max():
            logger.info("Warning: latitude is out of bounds")
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
            self.u = self.dataset["U"].values
        if self.v is None or self.modified:
            self.v = self.dataset["V"].values
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
            return (
                self.fieldset_flat.U[t, 0, lat, lon],
                self.fieldset_flat.V[t, 0, lat, lon],
            )
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
        lat = self.lats[i] + (self.lats[i + 1] - self.lats[i]) * (
            seconds / seconds_total
        )
        lon = self.lons[i] + (self.lons[i + 1] - self.lons[i]) * (
            seconds / seconds_total
        )
        return lat, lon

    @classmethod
    def from_csv(
        cls, path, lat_key="latitude", lon_key="longitude", time_key="timestamp"
    ):
        """
        Reads buoy position data from a csv
        """
        df = pd.read_csv(path)
        # just in case for some reason it isn't already sorted by time
        df = df.sort_values(time_key)
        return cls(
            df[lat_key].values,
            df[lon_key].values,
            df[time_key].values.astype("datetime64[s]"),
        )
