"""
Collection of classes that represent plotted features in a given simulation.
These features represent additional information to the simulation on top of the already plotted
particle movements.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from constants import *
import utils


class ParticlePlotFeature:
    """
    Represents additional points to plot and maybe track on top of the particles from a
    Parcels simulation.
    """
    def __init__(self, lats, lons, labels=None, segments=False, track_dist=0, color=None):
        self.lats = lats
        self.lons = lons
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

    def count_near(self, lats, lons):
        """
        Counts the number of particles close to each point in this feature.

        Args:
            lats: particle lats
            lons: particle lons

        Returns:
            np.ndarray: array with length equal to the number of points in this feature. each index
             represents the number of particles within tracking distance of that point.
        """
        counts = np.zeros(len(self.lats))
        for i, point in enumerate(self.points):
            close = utils.haversine(lats, point[0], lons, point[1]) <= self.track_dist
            counts[i] += close.sum()
        return counts

    def get_closest_dist(self, lat, lon):
        """
        Given a (lat, lon) point, return the on this feature closest to the point. If segments is
        true, it will consider all the line segments too.
        """
        point = Point(lon, lat)
        # check distances to line segments
        if self.segments is not None:
            seg_closest, _ = nearest_points(self.segments, point)
            return utils.haversine(point.y, seg_closest.y, point.x, seg_closest.x)
        # check distance to closest point
        closest_idx = self.kdtree.query([lat, lon])[1]
        pnt = self.points[closest_idx]
        return utils.haversine(lat, pnt[0], lon, pnt[1])

    def get_all_dists(self, lats, lons):
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

    def plot_on_frame(self, ax, lats, lons, *args, **kwargs):
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
        """
        return None, None

    @classmethod
    def get_sd_coastline(cls, path=None, track_dist=100):
        if path is None:
            path = utils.MATLAB_DIR / SD_COASTLINE_FILENAME
        lats, lons = utils.load_pts_mat(path, "latz0", "lonz0")
        return cls(lats, lons, segments=True, track_dist=track_dist)


class NanSeparatedFeature(ParticlePlotFeature):
    """
    A feature containing multiple line segments where nans separate each collection of segments.
    """
    def plot_on_frame(self, ax, lats, lons, *args, **kwargs):
        lat_borders = np.split(self.lats, np.where(np.isnan(self.lats))[0])
        lon_borders = np.split(self.lons, np.where(np.isnan(self.lons))[0])
        for i in range(len(lat_borders)):
            ax.plot(lon_borders[i], lat_borders[i], c=self.color)

    @classmethod
    def get_sd_full_coastline(cls, path=None):
        if path is None:
            path = utils.MATLAB_DIR / SD_FULL_COASTLINE_FILENAME
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
        inst = cls(lats_all, lons_all, segments=True, track_dist=0)
        inst.color = "k"
        return inst


class StationFeature(ParticlePlotFeature):
    """
    Plots points that represent stations, where each is uniquely named and tracks how many
    particles are within tracking distance. When plotted on a frame, they change colors based
    on whether particles are near.

    The table they generate will show how many particles are near each station.
    """
    def __init__(self, lats, lons, labels, track_dist=0):
        super().__init__(lats, lons, labels=labels, segments=False, track_dist=track_dist)

    def plot_on_frame(self, ax, lats, lons, *args, **kwargs):
        counts = self.count_near(lats, lons)
        ax.scatter(
            self.lons[counts == 0], self.lats[counts == 0], c="b", s=60, edgecolor="k"
        )
        ax.scatter(
            self.lons[counts > 0], self.lats[counts > 0], c="r", s=60, edgecolor="k"
        )

    def generate_info_table(self, lats, lons, *args, **kwargs):
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
        if path is None:
            path = utils.MATLAB_DIR / SD_STATION_FILENAME
        lats, lons = utils.load_pts_mat(path, "ywq", "xwq")
        return cls(lats, lons, SD_STATION_NAMES, track_dist=track_dist)


class LatTrackedPointFeature(ParticlePlotFeature):
    """A single point that tracks how northward/southward the particles around it are."""
    def __init__(self, lat, lon, xlim=None, ymax=None, show=True, **kwargs):
        super().__init__([lat], [lon], **kwargs)
        self.xlim = xlim
        self.ymax = ymax
        self.show = show

    def plot_on_frame(self, ax, lats, lons, *args, **kwargs):
        if self.show:
            super().plot_on_frame(ax, lats, lons, *args, **kwargs)

    def generate_info_table(self, lats, lons, *args, **kwargs):
        dists = self.get_all_dists(lats, lons)[0]
        north = lats < self.lats[0]
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
