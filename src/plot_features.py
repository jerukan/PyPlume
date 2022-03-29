"""
Collection of classes that represent plotted features in a given simulation.

If you want to add additional stuff to the generated plots, this is the place to do so.
Create a new class that extends ParticlePlotFeature to customize what is plotted on top of
the particle frames.

These features represent additional information to the simulation on top of the already plotted
particle movements.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from src.constants import *
from src.parcels_utils import BuoyPath
import src.utils as utils


class ParticlePlotFeature:
    """
    Represents additional points to plot and maybe track on top of the particles from a
    Parcels simulation.
    """
    def __init__(self, lats, lons, labels=None, segments=False, track_dist=0, color=None):
        self.lats = np.array(lats)
        self.lons = np.array(lons)
        self.points = np.array([lats, lons]).T
        self.kdtree = scipy.spatial.KDTree(self.points)
        self.labels = labels
        if self.labels is not None:
            if len(self.labels) != len(self.lats):
                raise ValueError("Labels must be the same length as lats/lons")
        if segments:
            self.segments = LineString(np.array([self.lons, self.lats]).T)
        else:
            self.segments = None
        self.track_dist = track_dist
        self.color = color

    def count_near(self, lats, lons, **kwargs):
        """
        Counts the number of particles close to each point in this feature.

        Args:
            lats: particle lats
            lons: particle lons

        Returns:
            np.ndarray: array with length equal to the number of points in this feature. each index
             represents the number of particles within tracking distance of that point.
        """
        lats = np.array(lats)
        lons = np.array(lons)
        counts = np.zeros(len(self.lats))
        for i, point in enumerate(self.points):
            close = utils.haversine(lats, point[0], lons, point[1]) <= self.track_dist
            counts[i] += close.sum()
        return counts

    def get_closest_dists(self, lats, lons, **kwargs):
        """
        Given a lats, lons point, return the on this feature closest to the point. If segments is
        true, it will consider all the line segments too.
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
                    dists[i] = utils.haversine(point.y, seg_closest.y, point.x, seg_closest.x)
            return dists
        # check distance to closest point
        closest_idxs = self.kdtree.query(np.array([lats, lons]).T)[1]
        pnts = self.points[(closest_idxs)]
        return utils.haversine(lats, pnts.T[0], lons, pnts.T[1])

    def get_all_dists(self, lats, lons, **kwargs):
        """
        Yes this will be inefficient
        Returns a 2-d array where each row is each input particle's distance is to a point
        in this feature

        Args:
            lats: particle lats
            lons: particle lons
        """
        dists = np.empty((len(self.lats), len(lats)), dtype=np.float64)
        for i in range(len(dists)):
            for j in range(len(dists[i])):
                dists[i][j] = utils.haversine(self.lats[i], lats[j], self.lons[i], lons[j])
        return dists

    def plot_on_frame(self, fig, ax, lats, lons, *args, **kwargs):
        """
        Plots onto a frame plot, with information on particles at that time passed in

        Args:
            ax: the axes that the simulation plot was already drawn on
            lats: particle lats
            lons: particle lons
        """
        if self.segments is not None:
            ax.plot(self.lons, self.lats, c=self.color)
        else:
            ax.scatter(self.lons, self.lats, c=self.color)

    def generate_info_table(self, lats, lons, *args, **kwargs):
        """
        Generates an entirely new optional plot that could display any kind of additional info
        about this particular feature.

        Args:
            lats: particle lats
            lons: particle lons
        """
        return None, None

    @classmethod
    def get_sd_coastline(cls, path=None, track_dist=100):
        """A simplified SD coastline"""
        if path is None:
            path = utils.MATLAB_DIR / SD_COASTLINE_FILENAME
        if not os.path.exists(path):
            print(f"{path} does not exist", file=sys.stderr)
            return None
        lats, lons = utils.load_pts_mat(path, "latz0", "lonz0")
        return cls(lats, lons, segments=True, track_dist=track_dist)


class NanSeparatedFeature(ParticlePlotFeature):
    """
    A feature containing multiple line segments where nans separate each collection of segments.
    """
    def __init__(self, lats, lons, **kwargs):
        super().__init__(lats, lons, segments=False, **kwargs)

    def plot_on_frame(self, fig, ax, lats, lons, *args, **kwargs):
        lat_borders = np.split(self.lats, np.where(np.isnan(self.lats))[0])
        lon_borders = np.split(self.lons, np.where(np.isnan(self.lons))[0])
        for i in range(len(lat_borders)):
            ax.plot(lon_borders[i], lat_borders[i], c=self.color)

    @classmethod
    def get_sd_full_coastline(cls, path=None):
        """Gets the full detailed Tijuana coastline. Don't try get_all_dists"""
        if path is None:
            path = utils.MATLAB_DIR / SD_FULL_COASTLINE_FILENAME
        if not os.path.exists(path):
            print(f"{path} does not exist", file=sys.stderr)
            return None
        points = scipy.io.loadmat(path)["OR2Mex"]
        lats = points.T[1]
        lons = points.T[0]
        lat_borders = np.split(lats, np.where(np.isnan(lats))[0][1:])
        lon_borders = np.split(lons, np.where(np.isnan(lons))[0][1:])
        lats_all = []
        lons_all = []
        for idx in SD_FULL_TIJUANA_IDXS:
            lats_all.extend(lat_borders[idx])
            lons_all.extend(lon_borders[idx])
        inst = cls(lats_all, lons_all, track_dist=0)
        inst.color = "k"
        return inst


class StationFeature(ParticlePlotFeature):
    """
    Plots points that represent stations, where each is uniquely named and tracks how many
    particles are within tracking distance. When plotted on a frame, they change colors based
    on whether particles are near.

    The table they generate will show how many particles are near each station.
    """
    def __init__(self, lats, lons, labels, **kwargs):
        """Labels is required"""
        super().__init__(lats, lons, labels=labels, segments=False, **kwargs)

    def plot_on_frame(self, fig, ax, lats, lons, *args, **kwargs):
        """Any point with points near them are colored red, otherwise they are blue."""
        counts = self.count_near(lats, lons)
        ax.scatter(
            self.lons[counts == 0], self.lats[counts == 0], c="b", s=60, edgecolor="k"
        )
        ax.scatter(
            self.lons[counts > 0], self.lats[counts > 0], c="r", s=60, edgecolor="k"
        )

    def generate_info_table(self, lats, lons, *args, **kwargs):
        """
        Creates a table where each row contains information on each station and how many particles
        are nearby that point.
        """
        colors = np.full((len(self.lats), 4), "white", dtype=object)
        counts = self.count_near(lats, lons).astype(np.uint32)
        for i in range(len(self.lats)):
            if counts[i] > 0:
                colors[i, :] = "lightcoral"
        plume_pot = np.where(counts > 0, "YES", "NO")
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_axis_off()
        ax.table(
            cellText=np.array([np.arange(len(counts)) + 1, self.labels, counts, plume_pot]).T,
            cellColours=colors,
            colLabels=["Station ID", "Station Name", "Particle Count", "Plume Potential"],
            loc="center"
        ).auto_set_column_width(col=[0, 1, 2, 3, 4])
        ax.axis('tight')
        # fig.set_size_inches(7.17, 4)
        return fig, ax

    @classmethod
    def get_sd_stations(cls, path=None, track_dist=500):
        """Gets the stations in the SD area from the mat file."""
        if path is None:
            path = utils.MATLAB_DIR / SD_STATION_FILENAME
        if not os.path.exists(path):
            print(f"{path} does not exist", file=sys.stderr)
            return None
        lats, lons = utils.load_pts_mat(path, "ywq", "xwq")
        return cls(lats, lons, SD_STATION_NAMES, track_dist=track_dist)


class LatTrackedPointFeature(ParticlePlotFeature):
    """A single point that tracks how northward/southward the particles around it are."""
    def __init__(self, lat, lon, xlim=None, ymax=None, show=True, **kwargs):
        super().__init__([lat], [lon], **kwargs)
        self.xlim = xlim
        self.ymax = ymax
        self.show = show

    def plot_on_frame(self, fig, ax, lats, lons, *args, **kwargs):
        if self.show:
            super().plot_on_frame(fig, ax, lats, lons, *args, **kwargs)

    def generate_info_table(self, lats, lons, *args, **kwargs):
        """
        Generates a histogram showing the distribution of meridional distances from the single
        point.
        """
        dists = self.get_all_dists(lats, lons)[0]
        north = lats > self.lats[0]
        dists[north] = -dists[north]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(dists / 1000, density=True)
        ax.set_xlim(self.xlim)
        if self.ymax is not None:
            ax.set_ylim([0, self.ymax])
        fig.canvas.draw()
        # matplotlib uses a funny hyphen that doesn't work
        labels = [abs(float(item.get_text().replace("−", "-"))) for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        plt.figtext(0.5, -0.01, '(North) ------ Distance from point (km) ------ (South)', horizontalalignment='center') 
        fig.set_size_inches(6.1, 2.5)
        return fig, ax

    @classmethod
    def get_tijuana_mouth(cls):
        return cls(TIJUANA_MOUTH_POSITION[0], TIJUANA_MOUTH_POSITION[1], xlim=[-16, 4], ymax=0.1, show=False)


class NearcoastDensityFeature(ParticlePlotFeature):
    """A single point that tracks how northward/southward the particles around it are."""
    def __init__(self, origin, stations, coastline, xlim=None, ymax=None, **kwargs):
        """
        origin is a single point
        Assume point collections are [[lats], [lons]]

        track_dist: max distance to be considered as a nearcoast particle
        """
        self.origin_lat = origin[0]
        self.origin_lon = origin[1]
        super().__init__([self.origin_lat], [self.origin_lon], **kwargs)
        self.station_lats = stations[0]
        self.station_lons = stations[1]
        self.coast_lats = coastline[0]
        self.coast_lons = coastline[1]
        self.coastline = LineString(np.array([self.coast_lons, self.coast_lats]).T)
        self.xlim = xlim
        self.ymax = ymax

    def generate_info_table(self, lats, lons, *args, **kwargs):
        """
        Generates a histogram showing the distribution of meridional distances from the single
        point.
        """
        coast_dists = np.empty(len(lats))
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            _, coast_nearest = nearest_points(Point(lon, lat), self.coastline)
            coast_dists[i] = utils.haversine(coast_nearest.y, lat, coast_nearest.x, lon)
        dists = self.get_all_dists(lats, lons)[0]
        station_dists = self.get_all_dists(self.station_lats, self.station_lons)[0]
        # things north of the origin will appear on the left
        # calculate station distances
        stations_north = self.station_lats > self.lats[0]
        station_dists[stations_north] = -station_dists[stations_north]
        station_dists /= 1000
        # find which particles are north relative to origin and set them negative
        north = lats > self.lats[0]
        dists[north] = -dists[north]
        dists /= 1000
        nearcoast = coast_dists <= self.track_dist
        if self.xlim is None:
            xlim = [dists[nearcoast].min(), dists[nearcoast].max()]
        else:
            xlim = self.xlim
        # hack to prevent non-nearcoast particles from showing
        dists[~nearcoast] = xlim[1] + 1
        fig = plt.figure()
        ax = fig.add_subplot()
        bins = np.linspace(xlim[0], xlim[1], 30)
        bins = np.append(bins, self.xlim[1] + 1)
        ax.hist(dists, bins=bins, density=True)
        ax.scatter(x=station_dists, y=np.full(station_dists.shape, 0.01), c='k', edgecolor='y', zorder=1000)
        ax.set_xlim(xlim)
        if self.ymax is not None:
            ax.set_ylim([0, self.ymax])
        fig.canvas.draw()
        # matplotlib uses a funny hyphen that doesn't work
        labels = [abs(float(item.get_text().replace("−", "-"))) for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        plt.figtext(0.5, -0.01, '(North) ------ Distance from point (km) ------ (South)', horizontalalignment='center') 
        fig.set_size_inches(6.1, 2.5)
        return fig, ax

    @classmethod
    def get_tijuana_mouth(cls, path=None):
        if path is None:
            path = utils.MATLAB_DIR / SD_STATION_FILENAME
        if not os.path.exists(path):
            print(f"{path} does not exist", file=sys.stderr)
            return None
        st_lats, st_lons = utils.load_pts_mat(path, "ywq", "xwq")
        path = utils.MATLAB_DIR / SD_COASTLINE_FILENAME
        c_lats, c_lons = utils.load_pts_mat(path, "latz0", "lonz0")
        return cls(
            [TIJUANA_MOUTH_POSITION[0], TIJUANA_MOUTH_POSITION[1]],
            [st_lats, st_lons],
            [c_lats, c_lons],
            xlim=[-16, 4], ymax=1, track_dist=900
        )


class BuoyPathFeature(ParticlePlotFeature):
    def __init__(self, buoy_path: BuoyPath, backstep_delta=None, backstep_count=0):
        self.buoy_path = buoy_path
        self.backstep_count = backstep_count
        self.backstep_delta = backstep_delta
        super().__init__(buoy_path.lats, buoy_path.lons, labels=buoy_path.times, segments=True)

    def plot_on_frame(self, fig, ax, lats, lons, *args, **kwargs):
        time = kwargs.get("time", None)
        if time is None:
            return super().plot_on_frame(fig, ax, lats, lons, *args, **kwargs)
        b_lats = []
        b_lons = []
        for i in range(self.backstep_count + 1):
            curr_time = time - i * self.backstep_delta
            if not self.buoy_path.in_time_bounds(curr_time):
                break
            b_lat, b_lon = self.buoy_path.get_interped_point(curr_time)
            b_lats.append(b_lat)
            b_lons.append(b_lon)
        ax.plot(b_lons, b_lats)

    def get_closest_dists(self, lats, lons, **kwargs):
        time = kwargs.get("time", None)
        if time is None:
            return super().get_closest_dists(lats, lons, **kwargs)
        buoy_lat, buoy_lon = self.buoy_path.get_interped_point(time)
        return utils.haversine(lats, buoy_lat, lons, buoy_lon)

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(BuoyPath.from_csv(**utils.get_path_cfg(path)), **kwargs)


class WindVectorFeature(ParticlePlotFeature):
    def plot_on_frame(self, fig, ax, lats, lons, *args, **kwargs):
        if "wind" not in kwargs:
            return
        wind_u, wind_v = kwargs["wind"]  # tuple of u, v
        wind_ax = fig.add_axes([0.1, 0, 0.1, 0.1])
