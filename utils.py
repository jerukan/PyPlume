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

FILES_ROOT = Path("/Volumes/T7/Documents/Programs/scripps-cordc/parcels_westcoast")
PARCELS_CONFIGS_DIR = Path("parcels_configs")
CURRENT_NETCDF_DIR = Path("current_netcdfs")
PARTICLE_NETCDF_DIR = Path("particledata")
WAVEBUOY_DATA_DIR = Path("buoy_data")
MATLAB_DIR = Path("matlab")
PICUTRE_DIR = Path("snapshots")


def line_seg(x1, y1, x2, y2):
    """Creates information needed to represent a linear line segment."""
    slope = (y1 - y2) / (x1 - x2) if x1 - x2 != 0 else np.nan
    return dict(
        x1=x1,  # endpoint 1 x
        y1=y1,  # endpoint 1 y
        x2=x2,  # endpoint 2 x
        y2=y2,  # endpoint 2 y
        dom=(x1, x2) if x1 <= x2 else (x2, x1),  # domain
        rng=(y1, y2) if y1 <= y2 else (y2, y1),  # range
        # check for vertical line
        slope=slope,
        y_int=y1 - slope * x1 if not np.isnan(slope) else np.nan
    )


def valid_point(x, y, line):
    """Checks if a point on a line segment is inside its domain/range"""
    in_dom = line["dom"][0] <= x <= line["dom"][1]
    in_range = line["rng"][0] <= y <= line["rng"][1]
    return in_dom and in_range


def intersection_info(x, y, line):
    """
    Given a point and a line, return the xy coordinate of the closest point to the line.

    Returns:
        intersection x, intersection y
    """
    # vertical line
    if np.isnan(line["slope"]):
        return line["x1"], y
    if line["slope"] == 0:
        return x, line["y1"]
    norm_slope = -1 / line["slope"]
    slope_d = norm_slope - line["slope"]
    int_d = (line["slope"] * -line["x1"] + line["y1"]) - (norm_slope * -x + y)
    x_int = int_d / slope_d
    y_int = norm_slope * (x_int - x) + y
    return x_int, y_int


def segment_intersection(seg1, seg2):
    # vertical line cases
    if np.isnan(seg1["slope"]):
        if np.isnan(seg2["slope"]):
            if seg1["x1"] == seg2["x2"]:
                return np.inf, np.inf
            return np.nan, np.nan
        x, y = seg1["x1"], seg2["slope"] * seg1["x1"] + seg2["y_int"]
        if valid_point(x, y, seg1) and valid_point(x, y, seg2):
            return x, y
        return np.nan, np.nan
    if np.isnan(seg2["slope"]):
        x, y = seg2["x1"], seg1["slope"] * seg2["x1"] + seg1["y_int"]
        if valid_point(x, y, seg1) and valid_point(x, y, seg2):
            return x, y
        return np.nan, np.nan
    if seg1["slope"] == seg2["slope"]:
        if seg1["y_int"] == seg2["y_int"]:
            return np.inf, np.inf
        return np.nan, np.nan
    slope_d = seg2["slope"] - seg1["slope"]
    int_d = (seg1["slope"] * -seg1["x1"] + seg1["y1"]) - (seg2["slope"] * -seg2["x1"] + seg2["y1"])
    x = int_d / slope_d
    y = seg2["slope"] * (x - seg2["x1"]) + seg2["y1"]
    if valid_point(x, y, seg1) and valid_point(x, y, seg2):
        return x, y
    return np.nan, np.nan


class Piecewise2d:
    def __init__(self, x, y):
        self.points = np.array([x, y]).T
        self.x = x
        self.y = y
        self.kdtree = scipy.spatial.KDTree(self.points)

    def rect_check(self, other):
        return (min(other.x) <= max(self.x) and min(self.x) <= max(other.x)) and \
            (min(other.y) <= max(self.y) and min(self.y) <= max(other.y))

    def get_intersections(self, other):
        intersections = []
        for i in range(len(self.points) - 1):
            for j in range(len(other.points) - 1):
                seg1 = line_seg(self.x[i], self.y[i], self.x[i + 1], self.y[i + 1])
                seg2 = line_seg(other.x[j], other.y[j], other.x[j + 1], other.y[j + 1])
                x, y = segment_intersection(seg1, seg2)
                if not np.isnan(x):
                    intersections.append([x, y])
        return np.array(intersections)


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
    Calculates the haversine distance between two points.
    """
    R = 6378.137  # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(lat1 * math.pi / 180) * np.cos(lat2 * math.pi / 180) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(a ** (1 / 2), (1 - a) ** (1 / 2))
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
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
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
    Generates a boolean mask signifying which points in the data are invalid. (don't have data
    but normally should have)

    Args:
        data (np.ndarray): an array with the shape of (time, lat, lon).

    Returns:
        np.ndarray: boolean mask with the same shape as data.
    """
    mask_none = np.tile(generate_mask_none(data), (data.shape[0], 1, 1))
    mask = np.isnan(data)
    mask[mask_none] = False
    return mask


def generate_mask_none(data):
    """
    Generates a boolean mask signifying which points in the data don't have data and never should.

    Args:
        data (np.ndarray): an array with the shape of (time, lat, lon).

    Returns:
        np.ndarray: boolean mask with the shape of (lat, lon).
    """
    nan_data = ~np.isnan(data)
    return nan_data.sum(axis=0) == 0


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
    print((stdout, stderr))


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


def load_pts_mat(path, lat_ind, lon_ind):
    """
    Loads points from a pts mat from the TJ Plume Tracker.
    Only points where both lat and lon are non-nan are returned.

    Args:
        path: path to mat file

    Returns:
        np.ndarray: [lats], [lons]
    """
    mat_data = scipy.io.loadmat(path)
    xf = mat_data[lon_ind].flatten()
    yf = mat_data[lat_ind].flatten()
    # filter out nan values
    non_nan = (~np.isnan(xf)) & (~np.isnan(yf))
    xf = xf[np.where(non_nan)]
    yf = yf[np.where(non_nan)]
    return yf, xf
