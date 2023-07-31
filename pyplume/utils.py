"""Just some useful methods."""
import glob
import importlib
import math
import os
from pathlib import Path
import warnings

import imageio
import numpy as np
import scipy.io
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import xarray as xr

from pyplume import get_logger


logger = get_logger(__name__)


def import_attr(path):
    module_str = ".".join(path.split(".")[:-1])
    var_str = path.split(".")[-1]
    module = importlib.import_module(module_str)
    return getattr(module, var_str)


def get_points(points, dim=2, transpose=None):
    """
    Given N points of dimension d, the data can either be passed in as an (N,d) or (d,N)
    dimensional array.

    Can also pass in a single (d,1) point.

    Args:
        points
        dim
        transposed: None to infer data format, False to not transpose data, True to transpose data.
    """
    points = np.array(points)
    if len(points.shape) == 1:
        points = np.array([points])
    if len(points.shape) > 2:
        raise ValueError(f"Incorrect points dimension {points.shape}")
    if transpose is None:
        # guess
        if points.shape[0] == points.shape[1] and points.shape[0] == dim:
            # if the points happen to be (d,d), just guess data was passed in as collection of pairs
            warnings.warn(
                f"Shape of points is ambiguous: {points}. Will transpose by default."
            )
            return (points.T[d] for d in range(dim))
        if points.shape[1] == dim:
            # assume (n, d)
            return (points.T[d] for d in range(dim))
        if points.shape[0] == dim:
            # assume (d, n)
            return (points[d] for d in range(dim))
    if transpose:
        return (points.T[d] for d in range(dim))
    return (points[d] for d in range(dim))


def haversine(lat1, lat2, lon1, lon2):
    """
    Calculates the haversine distance between two points.
    """
    R = 6378.137  # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(lat1 * math.pi / 180) * np.cos(
        lat2 * math.pi / 180
    ) * np.sin(dLon / 2) * np.sin(dLon / 2)
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
    """Deletes every png in a folder (non-recursive)."""
    pngs = glob.glob(os.path.join(dir_path, "*.png"))
    for png in pngs:
        os.remove(png)


def conv_to_dataarray(arr, darr_ref):
    """
    Takes in some array and converts it to a labelled xarray DataArray.

    Args:
        arr (array-like)
        darr_ref (xr.DataArray): only used to label coordinates, dimensions, and metadata.
            Must have the same shape as arr.
    """
    return xr.DataArray(
        arr, coords=darr_ref.coords, dims=darr_ref.dims, attrs=darr_ref.attrs
    )


def generate_mask_invalid(data):
    """
    Generates a boolean mask signifying which points in the data are invalid (don't have data
    but normally should have).

    In other words, if at a particular position x for timestep t it is nan, but at timestep s point
    x has a value, position x at timestep t is considered invalid.

    If position y is nan for all timesteps, it's not considered invalid (data shouldn't exist)
    in the first place.

    Time is considered to be the first dimension.

    Args:
        data (array-like): an array with the shape of (time, space...).

    Returns:
        array-like: boolean mask with the same shape as data. True signifies invalid point.
    """
    dataisnan = np.isnan(data)
    return dataisnan & np.any(~dataisnan, axis=0)


def generate_mask_no_data(data, tile=False):
    """
    Generates a boolean mask signifying which coordinates in the data shouldn't have data and which
    ones should.

    Args:
        data (array-like): An array with the shape of (time, space...).
        tile (bool): If True, the mask will match the original shape of the data. Otherwise, it
            is the shape at a single timestep.

    Returns:
        array-like: Boolean mask with the shape of (lat, lon). True values signify data shouldn't
         exist, False values signify they should.
    """
    mask = np.all(np.isnan(data), axis=0)
    if tile:
        return np.tile(mask, (data.shape[0], 1, 1))
    return mask


def include_coord_range(coord_rng, ref_coords):
    """
    Takes a range of two values and changes the desired range to include the two original
    values given reference coordinates if a slice were to happen with the range.

    Args:
        coord_rng (value1, value2): where value1 < value2
        ref_coords: must be sorted ascending

    Returns:
        (value1, value2): where value1 <= coord_rng[0] and
            value2 >= coord_rng[-1]
    """
    coords_sorted = np.all(ref_coords[:-1] <= ref_coords[1:])
    if not coords_sorted:
        raise ValueError("Coordinates of dataset are unsorted, this will cause problems.")
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


def wrap_in_kwarg(obj, merge_dict=True, key=None, **kwargs):
    if isinstance(obj, dict) and merge_dict:
        return {**obj, **kwargs}
    return {key: obj, **kwargs}


def get_path_cfg(path, **kwargs):
    """Utility to check if just a path was passed in or a path and its kwargs."""
    if isinstance(path, (str, Path)):
        return {"path": path, **kwargs}
    if isinstance(path, dict):
        if "path" in path:
            return {**path, **kwargs}
        raise KeyError("key 'path' not in path config")
    raise TypeError(f"{path} is not a proper path or config")


def generate_gif(img_paths, gif_path, frame_duration=None):
    """
    Args:
        img_paths: List of input image paths to combine into a
            gif. Order is preserved.
        frame_duration: In milliseconds
    """
    if frame_duration is None:
        frame_duration = 500
    imgs = [imageio.imread(inpath) for inpath in img_paths]
    imageio.mimsave(gif_path, imgs, duration=frame_duration, loop=0)
    return gif_path


def convert360to180(val):
    return ((val + 180) % 360) - 180


def convert180to360(val):
    return val % 360


class GeoPointCollection:
    def __init__(self, lats, lons, connected=False):
        self.lats = np.array(lats)
        self.lons = np.array(lons)
        self.points = np.array([lats, lons]).T
        # TODO: scipy is causing kernel crashes in jupyter
        self.kdtree = scipy.spatial.KDTree(self.points)
        if connected:
            self.segments = LineString(np.array([self.lons, self.lats]).T)
        else:
            self.segments = None

    def count_near(self, lats, lons, track_dist):
        """
        Counts the number of particles close to each point in this feature.

        Args:
            lats: particle lats
            lons: particle lons

        Returns:
            np.ndarray: array with length equal to the number of points in this feature.
                Each index represents the number of particles within tracking distance
                of that point.
        """
        lats = np.array(lats)
        lons = np.array(lons)
        counts = np.zeros(len(self.lats))
        for i, point in enumerate(self.points):
            close = haversine(lats, point[0], lons, point[1]) <= track_dist
            counts[i] += close.sum()
        return counts

    def get_closest_dists(self, lats, lons):
        """
        Given a lats, lons point, return the on this feature closest to the point.
        If segments is true, it will consider all the line segments too.
        """
        lats = np.array(lats)
        lons = np.array(lons)
        if self.segments is not None:
            dists = np.full(len(lats), np.nan)
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                point = Point(lon, lat)
                # check distances to line segments
                if self.segments is not None:
                    seg_closest, _ = nearest_points(self.segments, point)
                    dists[i] = haversine(point.y, seg_closest.y, point.x, seg_closest.x)
            return dists
        # check distance to closest point
        closest_idxs = self.kdtree.query(np.array([lats, lons]).T)[1]
        pnts = self.points[(closest_idxs)]
        return haversine(lats, pnts.T[0], lons, pnts.T[1])

    def get_all_dists(self, lats, lons):
        """
        Returns a 2-d array where each row is each input particle's distance is to
        a point in this feature.

        Args:
            lats: particle lats
            lons: particle lons
        """
        # an inefficient python loop implementation
        dists = np.zeros((len(self.lats), len(lats)), dtype=np.float64)
        for i in range(dists.shape[0]):
            for j in range(dists.shape[1]):
                dists[i][j] = haversine(self.lats[i], lats[j], self.lons[i], lons[j])
        return dists
