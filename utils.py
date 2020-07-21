"""
Just some useful methods.
"""
import json
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
