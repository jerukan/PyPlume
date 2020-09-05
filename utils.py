"""
Just some useful methods.
"""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import scipy.io
import xarray as xr

DATA_6KM = 6
DATA_2KM = 2
DATA_1KM = 1

filename_dict = {
    DATA_6KM: "west_coast_6km_hourly",
    DATA_2KM: "west_coast_2km_hourly",
    DATA_1KM: "west_coast_1km_hourly"
}

PARCELS_CONFIGS_DIR = Path("parcels_configs")
CURRENT_NETCDF_DIR = Path("current_netcdfs")
PARTICLE_NETCDF_DIR = Path("particledata")
MATLAB_DIR = Path("matlab")
PICUTRE_DIR = Path("snapshots")


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


def create_path(path_str, iterate=False):
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
            point = data[:, i, j]
            nan_vals = np.isnan(point)
            # if the point at (lat, lon) contains real data and nan values
            # mark those points as invalid
            if not nan_vals.all():
                mask[:, i, j] = np.where(nan_vals.flatten(), 1, 0)
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
            point = data[:, i, j]
            if np.isnan(point).all():
                mask[:, i, j] = True
    return mask


def add_noise(arr, max_var, repeat=None):
    """
    Randomly varies the values in an array.
    """
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


def create_gif(delay, images_path, out_path):
    magick_sp = subprocess.Popen(
        [
            "magick", "-delay", str(delay), images_path, out_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = magick_sp.communicate()
    print((stdout, stderr))


def expand_coord_rng(coord_rng, ref_coords):
    """
    Takes a range of two numbers and expands the range to the
    next biggest coordinates defined in ref_coords.

    Args:
        ref_coords: must be sorted ascending

    Returns:
        (float1, float2): where float1 <= coord_rng[0] and
            float2 >= coord_rng[-1]
    """
    index_min = np.where(ref_coords <= coord_rng[0])[0][-1]
    index_max = np.where(ref_coords >= coord_rng[-1])[0][0]
    return ref_coords[index_min], ref_coords[index_max]


def load_pts_mat(path, lat_ind, lon_ind):
    """
    Loads points from a pts mat from the TJ Plume Tracker.

    Args:
        path: path to mat file

    Returns:
        np.ndarray: [[lats], [lons]]
    """
    mat_data = scipy.io.loadmat(path)
    xf = mat_data[lon_ind]
    yf = mat_data[lat_ind]
    # filter out nan values
    xf = xf[np.where(~np.isnan(xf))].flatten()
    yf = yf[np.where(~np.isnan(yf))].flatten()
    assert len(xf) == len(yf)
    points = np.empty((2, len(xf)))
    # points are read as (lat, lon), so reverse order
    points[0] = yf
    points[1] = xf
    return points
