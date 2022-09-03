"""
Collection of classes that represent plotted features in a given simulation.

If you want to add additional stuff to the generated plots, this is the place to do so.
Create a new class that extends ParticlePlotFeature to customize what is plotted on top of
the particle frames.

These features represent additional information to the simulation on top of the already plotted
particle movements.
"""
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from pyplume.constants import *
from pyplume.dataloaders import BuoyPath
import pyplume.utils as utils


class PlotFeature:
    """
    Generic class to represent a feature that should be plotted on a plot of a simulation frame.
    """
    def add_to_plot(self, fig, ax, t, lats, lons, **kwargs):
        """
        Plots onto a simulation frame plot, with information on particles at that time passed in.

        Args:
            ax: the axes that the simulation plot was already drawn on
            t: timestamp
            lats: particle lats
            lons: particle lons

        Returns:
            fig, ax
        """
        return fig, ax

    def generate_external_plot(self, t, lats, lons, **kwargs):
        """
        Generates an entirely new optional plot that could display any kind of additional info
        about this particular feature.

        Args:
            t: timestamp
            lats: particle lats
            lons: particle lons

        Returns:
            fig, ax
        """
        return None, None

    @classmethod
    def load_from_external(cls, **kwargs):
        raise NotImplementedError()


class ParticlePlotFeature(PlotFeature):
    """
    Plots particles per frame.
    """
    def __init__(self, particle_size=4):
        self.particle_size = particle_size

    def add_to_plot(self, fig, ax, t, lats, lons, lifetimes=None, lifetime_max=None, **kwargs):
        sc = ax.scatter(lons, lats, c=lifetimes, edgecolor="k", vmin=0, vmax=lifetime_max, s=self.particle_size)
        if lifetimes is not None:
            cbar_ax = fig.add_axes([0.1, 0, 0.1, 0.1])
            plt.colorbar(sc, cax=cbar_ax)
            posn = ax.get_position()
            cbar_ax.set_position([posn.x0 + posn.width + 0.14, posn.y0, 0.04, posn.height])
            cbar_ax.get_yaxis().labelpad = 13
            # super jank label the other colorbar since it's in plotting.plotfield
            cbar_ax.set_ylabel("Age (days)\n\n\n\n\n\nVelocity (m/s)", rotation=270)
        return fig, ax


class ScatterPlotFeature(PlotFeature):
    """
    Represents additional points to plot and track in addition to the particles from a
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

    def add_to_plot(self, fig, ax, t, lats, lons, **kwargs):
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
        return fig, ax

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

    @classmethod
    def load_from_external(cls, path, track_dist=100, **kwargs):
        """
        Loads from mat files only right now.
        """
        lats, lons = utils.load_pts_mat(path)
        return cls(lats, lons, segments=True, track_dist=track_dist, **kwargs)


class NanSeparatedFeature(ScatterPlotFeature):
    """
    A feature containing multiple line segments where nans separate each collection of segments.
    """
    def __init__(self, lats, lons, **kwargs):
        super().__init__(lats, lons, segments=False, **kwargs)

    def add_to_plot(self, fig, ax, t, lats, lons, **kwargs):
        lat_borders = np.split(self.lats, np.where(np.isnan(self.lats))[0])
        lon_borders = np.split(self.lons, np.where(np.isnan(self.lons))[0])
        for i in range(len(lat_borders)):
            ax.plot(lon_borders[i], lat_borders[i], c=self.color)
        return fig, ax

    @classmethod
    def load_from_external(cls, path, **kwargs):
        lats, lons = utils.load_pts_mat(path, del_nan=False)
        inst = cls(lats, lons, track_dist=0, **kwargs)
        return inst


class StationFeature(ScatterPlotFeature):
    """
    Plots points that represent stations, where each is uniquely named and tracks how many
    particles are within tracking distance. When plotted on a frame, they change colors based
    on whether particles are near.

    The table they generate will show how many particles are near each station.
    """
    def __init__(self, lats, lons, labels=None, **kwargs):
        if labels is None:
            labels = [f"Station {i}" for i in range(len(lats))]
        super().__init__(lats, lons, labels=labels, segments=False, **kwargs)

    def add_to_plot(self, fig, ax, t, lats, lons, **kwargs):
        """Any point with points near them are colored red, otherwise they are blue."""
        counts = self.count_near(lats, lons)
        ax.scatter(
            self.lons[counts == 0], self.lats[counts == 0], c="b", s=60, edgecolor="k"
        )
        ax.scatter(
            self.lons[counts > 0], self.lats[counts > 0], c="r", s=60, edgecolor="k"
        )
        return fig, ax

    def generate_external_plot(self, t, lats, lons, **kwargs):
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
    def load_from_external(cls, path, labels=None, track_dist=1000, **kwargs):
        lats, lons = utils.load_pts_mat(path, "ywq", "xwq")
        return cls(lats, lons, labels=labels, track_dist=track_dist, **kwargs)


class LatTrackedPointFeature(ScatterPlotFeature):
    """A single point that tracks how northward/southward the particles around it are."""
    def __init__(self, lat, lon, xlim=None, ymax=None, show=True, **kwargs):
        super().__init__([lat], [lon], **kwargs)
        self.xlim = xlim
        self.ymax = ymax
        self.show = show

    def add_to_plot(self, t, fig, ax, lats, lons, **kwargs):
        if self.show:
            return super().add_to_plot(fig, ax, lats, lons, **kwargs)
        return fig, ax

    def generate_external_plot(self, t, lats, lons, **kwargs):
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
        return cls(TIJUANA_MOUTH_POSITION[0], TIJUANA_MOUTH_POSITION[1], xlim=[-16, 4], ymax=0.1, show=False, color="y")


class NearcoastDensityFeature(ScatterPlotFeature):
    """Tracks which particles are within a certain distance to the coastline."""
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

    def generate_external_plot(self, t, lats, lons, **kwargs):
        """
        Generates a histogram showing the distribution of distances of particles within the
        threshold distance to the coastline. Distribution of particles is binned by meridional
        distance to the origin.
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
    def load_from_external(cls, origin, stations, coastline, **kwargs):
        st_lats, st_lons = utils.load_pts_mat(stations)
        c_lats, c_lons = utils.load_pts_mat(coastline)
        "xlim=[-16, 4], ymax=1, track_dist=900"
        return cls(
            [origin[0], origin[1]],
            [st_lats, st_lons],
            [c_lats, c_lons],
            **kwargs
        )


class BuoyPathFeature(ScatterPlotFeature):
    def __init__(self, buoy_path: BuoyPath, backstep_delta=None, backstep_count=0):
        self.buoy_path = buoy_path
        self.backstep_count = backstep_count
        self.backstep_delta = backstep_delta
        super().__init__(buoy_path.lats, buoy_path.lons, labels=buoy_path.times, segments=True)

    def add_to_plot(self, fig, ax, t, lats, lons, **kwargs):
        b_lats = []
        b_lons = []
        for i in range(self.backstep_count + 1):
            curr_time = t - i * self.backstep_delta
            if not self.buoy_path.in_time_bounds(curr_time):
                break
            b_lat, b_lon = self.buoy_path.get_interped_point(curr_time)
            b_lats.append(b_lat)
            b_lons.append(b_lon)
        ax.plot(b_lons, b_lats)
        return fig, ax

    def get_closest_dists(self, t, lats, lons, **kwargs):
        buoy_lat, buoy_lon = self.buoy_path.get_interped_point(t)
        return utils.haversine(lats, buoy_lat, lons, buoy_lon)

    @classmethod
    def load_from_external(cls, path, **kwargs):
        return cls(BuoyPath.from_csv(**utils.get_path_cfg(path)), **kwargs)


class WindVectorFeature(ScatterPlotFeature):
    def add_to_plot(self, fig, ax, t, lats, lons, **kwargs):
        if "wind" not in kwargs:
            return
        wind_u, wind_v = kwargs["wind"]  # tuple of u, v
        wind_ax = fig.add_axes([0.1, 0, 0.1, 0.1])


def construct_features_from_configs(*feature_configs):
    features = []
    labels = []
    for feature_args in feature_configs:
        feature_class = utils.import_attr(feature_args["path"])
        features.append(feature_class.load_from_external(**feature_args["args"]))
        labels.append(feature_args.get("label", None))
    return features, labels
