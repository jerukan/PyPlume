"""
Just some useful methods.
"""
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

DATA_6KM = 6
DATA_2KM = 2
DATA_1KM = 1

filename_dict = {
    DATA_6KM: "west_coast_6km_hourly",
    DATA_2KM: "west_coast_2km_hourly",
    DATA_1KM: "west_coast_1km_hourly"
}


def euc_dist(vec1, vec2):
    """
    Args:
        vec1 (array-like)
        vec2 (array-like)
    """
    # convert to np array if vectors aren't already
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2)
    return np.sqrt(((vec1 - vec2) ** 2).sum())


def haversine(lat1, lat2, lon1, lon2):
    """
    lol look at all this MATH
    how does this even work
    """
    R = 6378.137  # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d * 1000  # meters


def create_path(path_str):
    """
    Args:
        path_str (str)

    Returns:
        Path
    """
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def generate_mask_invalid(data):
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


def generate_mask_none(data):
    """
    Generates a boolean mask signifying which points in the data don't have data.

    Args:
        data (np.ndarray): an array with the shape of (time, lat, lon).

    Returns:
        np.ndarray: boolean mask with the same shape as data.
    """
    mask = np.zeros(data.shape, dtype=bool)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            point = data.T[j][i]
            if np.isnan(point).all():
                mask.T[j][i][:] = True
    return mask


def add_noise(arr, max_var, repeat=None):
    if repeat is None:
        var_arr = np.random.random(arr.shape)
        var_arr = (var_arr * 2 - 1) * max_var
        return arr + var_arr
    shp = np.ones(len(arr.shape) + 1, dtype=int)
    shp[0] = repeat
    rep_arr = np.tile(arr, shp)
    var_arr = np.random.random(rep_arr.shape)
    var_arr = (var_arr * 2 - 1) * max_var
    return rep_arr + var_arr
