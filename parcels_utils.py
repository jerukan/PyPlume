"""
A collection of methods wrapping OceanParcels functionalities.
"""
import datetime
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from parcels import FieldSet, plotting
import scipy.spatial
import xarray as xr

import plot_utils
import utils


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


def xr_dataset_to_fieldset(xrds, copy=True, raw=True, mesh="spherical") -> FieldSet:
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
            {"U": ds["u"].values, "V": ds["v"].values},
            {"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
            mesh=mesh
        )
    else:
        fieldset = FieldSet.from_xarray_dataset(
                ds,
                dict(U="u", V="v"),
                dict(lat="lat", lon="lon", time="time"),
                mesh=mesh
            )
    fieldset.check_complete()
    return fieldset


class HFRGrid:
    """
    Wraps information relating to ocean current data given some dataset.
    Replaces get_file_info.

    TODO generate the mask of where data should be available
    """
    def __init__(self, dataset, init_fs=True):
        """
        Reads from a netcdf file containing ocean current data.

        Args:
            dataset (path-like or xr.Dataset): represents the netcdf ocean current data.
        """
        if isinstance(dataset, (Path, str)):
            self.path = dataset
            with xr.open_dataset(dataset) as ds:
                self.xrds = ds
        elif isinstance(dataset, xr.Dataset):
            self.path = None
            self.xrds = dataset
        else:
            raise TypeError(f"{dataset} is not a path or xarray dataset")
        self.times = self.xrds["time"].values
        self.lats = self.xrds["lat"].values
        self.lons = self.xrds["lon"].values
        self.timeKDTree = scipy.spatial.KDTree(np.array([self.times]).T)
        self.latKDTree = scipy.spatial.KDTree(np.array([self.lats]).T)
        self.lonKDTree = scipy.spatial.KDTree(np.array([self.lons]).T)
        if init_fs:
            self.prep_fieldsets()
        else:
            self.fieldset = None
            self.fieldset_flat = None
        # for caching
        self.u = None
        self.v = None


    def prep_fieldsets(self):
        # spherical mesh
        self.fieldset = xr_dataset_to_fieldset(self.xrds)
        # flat mesh
        self.fieldset_flat = xr_dataset_to_fieldset(self.xrds, mesh="flat")

    def get_coords(self) -> tuple:
        """
        Returns:
            (times, latitudes, longitudes)
        """
        return self.times, self.lats, self.lons

    def get_domain(self) -> dict:
        """
        Returns:
            dict
        """
        return {
            "S": self.lats[0],
            "N": self.lats[-1],
            "W": self.lons[0],
            "E": self.lons[-1],
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
        # cache the whole array because isel is slow when doing it individually
        if self.u is None:
            self.u = self.xrds["u"].values
        if self.v is None:
            self.v = self.xrds["v"].values
        return self.u[t_idx, lat_idx, lon_idx], self.v[t_idx, lat_idx, lon_idx]

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


def line_seg(x1, y1, x2, y2):
    return dict(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        dom=(x1, x2) if x1 <= x2 else (x2, x1),
        rng=(y1, y2) if y1 <= y2 else (y2, y1),
        # check for vertical line
        slope=(y1 - y2) / (x1 - x2) if x1 - x2 != 0 else np.nan
    )


def valid_point(x, y, line):
    in_dom = line["dom"][0] <= x <= line["dom"][1]
    in_range = line["rng"][0] <= y <= line["rng"][1]
    return in_dom and in_range


def intersection_info(x, y, line):
    """
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


class ParticlePlotFeature:
    """
    Represents additional points to plot and maybe track on top of the particles from a
    Parcels simulation.
    """
    def __init__(self, lats, lons, labels=None, segments=False, track_dist=0):
        self.lats = lats
        self.lons = lons
        self.points = np.array([lats, lons]).T
        self.kdtree = scipy.spatial.KDTree(self.points)
        self.labels = labels
        if self.labels is not None:
            if len(self.labels) != len(self.lats):
                raise ValueError("Labels must be the same length as lats/lons")
        if segments:
            self.segments = np.empty(len(self.lats) - 1, dtype=dict)
            for i in range(0, len(self.points) - 1):
                self.segments[i] = line_seg(self.lons[i], self.lats[i], self.lons[i + 1], self.lats[i + 1])
        else:
            self.segments = None
        self.track_dist = track_dist

    def count_near(self, p_lats, p_lons):
        counts = np.zeros(len(self.lats))
        for p_point in zip(p_lats, p_lons):
            for i, point in enumerate(self.points):
                if utils.haversine(p_point[0], point[0], p_point[1], point[1]) <= self.track_dist:
                    counts[i] += 1
        return counts

    def get_closest_dist(self, lat, lon):
        least_dist = np.inf
        closest_idx = self.kdtree.query([lat, lon])[1]
        # check distances to line segments
        if self.segments is not None:
            seg_check = []
            if closest_idx < len(self.points) - 1:
                seg_check.append(self.segments[closest_idx])
            if closest_idx > 0:
                seg_check.append(self.segments[closest_idx] - 1)
            for seg in seg_check:
                lon_int, lat_int = intersection_info(lon, lat, seg)
                if valid_point(lon_int, lat_int, seg):
                    dist = utils.haversine(lat, lat_int, lon, lon_int)
                    least_dist = dist if dist < least_dist else least_dist
        # check distance to closest point
        pnt = self.points[closest_idx]
        dist = utils.haversine(lat, pnt[0], lon, pnt[1])
        least_dist = dist if dist < least_dist else least_dist
        return least_dist

    @classmethod
    def get_sd_stations(cls, path=None):
        # TODO there are constants here, move them
        if path is None:
            path = utils.MATLAB_DIR / "wq_stposition.mat"
        station_mat = scipy.io.loadmat(path)
        station_lats = station_mat["ywq"].flatten()
        station_lons = station_mat["xwq"].flatten()
        station_names = np.array([
            "Coronado (North Island)",
            "Silver Strand",
            "Silver Strand Beach",
            "Carnation Ave.",
            "Imperial Beach Pier",
            "Cortez Ave.",
            "End of Seacoast Dr.",
            "3/4 mi. N. of TJ River Mouth",
            "Tijuana River Mouth",
            "Monument Rd.",
            "Board Fence",
            "Mexico"
        ])
        return cls(station_lats, station_lons, labels=station_names, track_dist=500)


class ParticleResult:
    """
    Wraps the output of a particle file to make visualizing and analyzing the results easier.
    Can also use an HFRGrid if the ocean currents are also needed.
    """
    def __init__(self, dataset):
        if isinstance(dataset, (Path, str)):
            self.path = dataset
            with xr.open_dataset(dataset) as ds:
                self.xrds = ds
        elif isinstance(dataset, xr.Dataset):
            self.path = None
            self.xrds = dataset
        else:
            raise TypeError(f"{dataset} is not a path or xarray dataset")
        self.lats = self.xrds["lat"].values
        self.lons = self.xrds["lon"].values
        self.traj = self.xrds["trajectory"].values
        self.times = self.xrds["time"].values
        # not part of Ak4 kernel
        self.lifetimes = self.xrds["lifetime"].values
        self.spawntimes = self.xrds["spawntime"].values
        
        self.grid = None
        self.plot_features = []

    def add_grid(self, grid: HFRGrid):
        self.grid = grid
        gtimes, glats, glons = grid.get_coords()
        # check if particle set is in-bounds of the given grid
        if np.nanmin(self.times) < gtimes.min() or np.nanmax(self.times) > gtimes.max():
            raise ValueError("Time out of bounds")
        if np.nanmin(self.lats) < glats.min() or np.nanmax(self.lats) > glats.max():
            raise ValueError("Latitude out of bounds")
        if np.nanmin(self.lons) < glons.min() or np.nanmax(self.lons) > glons.max():
            raise ValueError("Longitude out of bounds")

    def add_plot_feature(self, feature: ParticlePlotFeature, station=False):
        self.plot_features.append((feature, station))

    def plot_feature(self, t, feature, station, ax):
        if station:
            curr_lats = self.lats[:, t]
            curr_lons = self.lons[:, t]
            counts = feature.count_near(curr_lats, curr_lons)
            ax.scatter(feature.lons[counts == 0], feature.lats[counts == 0])
            ax.scatter(feature.lons[counts > 0], feature.lats[counts > 0])
        else:
            ax.scatter(feature.lons, feature.lats)
            if feature.segments is not None:
                ax.plot(feature.lons, feature.lats)
        return ax

    def plot_at_t(self, t):
        if self.grid is None:
            domain = {
                "W": np.nanmin(self.lons),
                "E": np.nanmax(self.lons),
                "S": np.nanmin(self.lats),
                "N": np.nanmax(self.lats),
            }
            domain = plot_utils.pad_domain(domain, 0.0005)
        else:
            domain = self.grid.get_domain()
        max_life = np.nanmax(self.lifetimes)
        timestamp = self.times[-1][t]
        if self.grid is None:
            fig, ax = plot_utils.get_carree_axis(domain, land=True)
            plot_utils.get_carree_gl(ax)
        else:
            show_time = int((timestamp - self.grid.times[0]) / np.timedelta64(1, "s"))
            if show_time < 0:
                raise ValueError("Particle simulation time domain goes out of bounds")
            _, fig, ax, _ = plotting.plotfield(field=self.grid.fieldset.UV, show_time=show_time,
                                            domain=domain, land=True, vmin=0, vmax=0.6,
                                            titlestr="Particles and ")
        non_nan = ~np.isnan(self.lons[:, t])
        ax.scatter(
            self.lons[:, t][non_nan], self.lats[:, t][non_nan], c=self.lifetimes[:, t][non_nan],
            edgecolor="k", vmin=0, vmax=max_life, s=40
        )
        for feature, station in self.plot_features:
            self.plot_feature(t, feature, station, ax)
        return fig, ax
