"""Just some useful methods."""
import glob
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import scipy.io
import xarray as xr


from pyplume import get_logger


logger = get_logger(__name__)


def import_attr(path):
    module_str = ".".join(path.split(".")[:-1])
    var_str = path.split(".")[-1]
    module = importlib.import_module(module_str)
    return getattr(module, var_str)


def haversine(lat1, lat2, lon1, lon2):
    """
    Calculates the haversine distance between two points.
    """
    R = 6378.137  # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(lat1 * math.pi / 180) * np.cos(lat2 * math.pi / 180) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(a ** (1 / 2), (1 - a) ** (1 / 2))
    d = R * c
    return d * 1000  # meters


def get_dir(path_str, iterate=False):
    """
    Returns a Path to a directory. A new directory is created if the
    given path doesn't exist.

    Args:
        path_str (str)
        iterate (bool): if true, create new folders of the same path with
            numbers appended to the end if that path already exists.

    Returns:
        Path
    """
    path = Path(path_str)
    path_base = str(path)
    num = 0
    while iterate and path.is_dir():
        path = Path(path_base + f"-{num}")
        num += 1
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        pass
    return path


def delete_all_pngs(dir_path):
    """Deletes every simulation generated image"""
    pngs = glob.glob(os.path.join(dir_path, "*.png"))
    for png in pngs:
        if re.search(r"snap_\D*\d+\.png$", png) is None:
            raise Exception(f"Non-plot images founud in folder {dir_path}")
        os.remove(png)


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
    Generates a boolean mask signifying which points in the data are invalid. (don't have data
    but normally should have)

    Args:
        data (array-like): an array with the shape of (time, lat, lon).

    Returns:
        array-like: boolean mask with the same shape as data.
    """
    mask_none = np.tile(generate_mask_no_data(data), (data.shape[0], 1, 1))
    mask = np.isnan(data)
    mask[mask_none] = False
    return mask


def generate_mask_no_data(data):
    """
    Generates a boolean mask signifying which coordinates in the data shouldn't have data and which
    ones should.

    Args:
        data (array-like): an array with the shape of (time, lat, lon).

    Returns:
        array-like: Boolean mask with the shape of (lat, lon). True values signify data shouldn't
         exist, False values signify they should.
    """
    if isinstance(data, xr.Dataset): data = data.to_array()[0]
    if len(data.shape) != 3:
        raise ValueError(f"Incorrect data dimension: expected 3, actual {len(data.shape)}")
    nan_data = ~np.isnan(data)
    return nan_data.sum(axis=0) == 0


def create_gif(delay, images_path, out_path):
    """
    Use regex with images_path
    """
    magick_sp = subprocess.Popen(
        [
            "magick", "-delay", str(delay), images_path, out_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = magick_sp.communicate()
    logger.info((stdout, stderr))


def include_coord_range(coord_rng, ref_coords):
    """
    Takes a range of two values and changes the desired range to include the two original
    values given reference coordinates if a slice were to happen with the range.

    Args:
        coord_rng (value1, value2): where value1 < value2
        ref_coords: must be sorted ascending

    Returns:
        (float1, float2): where float1 <= coord_rng[0] and
            float2 >= coord_rng[-1]
    """
    if ref_coords[0] > coord_rng[0]:
        start = coord_rng[0]
    else:
        index_min = np.where(ref_coords <= coord_rng[0])[0][-1]
        start = ref_coords[index_min]
    if ref_coords[-1] < coord_rng[1]:
        end = coord_rng[1]
    else:
        index_max = np.where(ref_coords >= coord_rng[-1])[0][0]
        end = ref_coords[index_max]
    return start, end


def expand_time_rng(time_rng, precision="h"):
    """
    Floors the start time and ceils the end time according to the precision specified.

    Args:
        time_rng (np.datetime64, np.datetime64)
        precision (str)
    
    Returns:
        (np.datetime64, np.datetime64)
    """
    start_time = np.datetime64(time_rng[0], precision) - np.timedelta64(1, precision)
    end_time = np.datetime64(time_rng[1], precision) + np.timedelta64(1, precision)
    return start_time, end_time


def load_pts_mat(path, lat_ind=None, lon_ind=None, del_nan=False):
    """
    Loads points from a pts mat from the TJ Plume Tracker.
    Only points where both lat and lon are non-nan are returned.

    Args:
        path: path to mat file

    Returns:
        np.ndarray: [lats], [lons]
    """
    mat_data = scipy.io.loadmat(path)
    # guess keys for latitude and longitude, this is not robust at all
    if lat_ind is None:
        for key in mat_data.keys():
            if "y" in key.lower() or "lat" in key.lower():
                lat_ind = key
                logger.info(f"Guessed latitude key as {key}")
                break
    if lat_ind is None:
        raise IndexError(f"No latitude or y key could be guessed in {path}")
    if lon_ind is None:
        for key in mat_data.keys():
            if "x" in key.lower() or "lon" in key.lower():
                lon_ind = key
                logger.info(f"Guessed longitude key as {key}")
                break
    if lon_ind is None:
        raise IndexError(f"No longitude or x key could be guessed in {path}")
    xf = mat_data[lon_ind].flatten()
    yf = mat_data[lat_ind].flatten()
    if del_nan:
        # filter out nan values
        non_nan = (~np.isnan(xf)) & (~np.isnan(yf))
        xf = xf[np.where(non_nan)]
        yf = yf[np.where(non_nan)]
    return yf, xf


def load_geo_points(path, **kwargs):
    """
    Loads a collection of (lat, lon) points from a given data configuration. Each different file
    type will have different ways of loading and different required parameters.

    .mat file requirements:
        lat_var: variable in the mat file representing the array of latitude values
        lon_var: variable in the mat file representing the array of longitude values
    """
    ext = os.path.splitext(path)[1]
    if ext == ".mat":
        lat_var = kwargs.get("lat_var", None)
        lon_var = kwargs.get("lon_var", None)
        lats, lons = load_pts_mat(path, lat_ind=lat_var, lon_ind=lon_var)
        return lats, lons
    raise ValueError(f"Invalid extension {ext}")


def get_path_cfg(path, **kwargs):
    """Utility to check if just a path was passed in or a path and its kwargs."""
    if isinstance(path, (str, Path)):
        return {"path": path, **kwargs}
    if isinstance(path, dict):
        if "path" in path:
            return {**path, **kwargs}
        raise KeyError(f"key 'path' not in path config")
    raise TypeError(f"{path} is not a proper path or config")
