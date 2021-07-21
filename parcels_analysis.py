from pathlib import Path
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parcels import plotting
import scipy.spatial
import xarray as xr

from constants import *
from parcels_utils import HFRGrid
import plot_utils
import utils


def line_seg(x1, y1, x2, y2):
    """Creates information needed to represent a linear line segment."""
    return dict(
        x1=x1,  # endpoint 1 x
        y1=y1,  # endpoint 1 y
        x2=x2,  # endpoint 2 x
        y2=y2,  # endpoint 2 y
        dom=(x1, x2) if x1 <= x2 else (x2, x1),  # domain
        rng=(y1, y2) if y1 <= y2 else (y2, y1),  # range
        # check for vertical line
        slope=(y1 - y2) / (x1 - x2) if x1 - x2 != 0 else np.nan
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
            self.segments = np.empty(len(self.lats) - 1, dtype=dict)
            for i in range(0, len(self.points) - 1):
                self.segments[i] = line_seg(self.lons[i], self.lats[i], self.lons[i + 1], self.lats[i + 1])
        else:
            self.segments = None
        self.track_dist = track_dist
        self.color = color

    def count_near(self, p_lats, p_lons):
        counts = np.zeros(len(self.lats))
        for i, point in enumerate(self.points):
            close = utils.haversine(p_lats, point[0], p_lons, point[1]) <= self.track_dist
            counts[i] += close.sum()
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
                seg_check.append(self.segments[closest_idx - 1])
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
        """Plots onto a frame plot, with information on particles at that time passed in"""
        ax.scatter(self.lons, self.lats, c=self.color)
        if self.segments is not None:
            ax.plot(self.lons, self.lats, c=self.color)

    def generate_info_table(self, lats, lons, *args, **kwargs):
        return None, None

    @classmethod
    def get_sd_coastline(cls, path=None, track_dist=100):
        if path is None:
            path = utils.MATLAB_DIR / SD_COASTLINE_FILENAME
        lats, lons = utils.load_pts_mat(path, "latz0", "lonz0")
        return cls(lats, lons, segments=True, track_dist=track_dist)


class StationFeature(ParticlePlotFeature):
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
        return cls(lats, lons, labels=SD_STATION_NAMES, track_dist=track_dist)


class LatTrackedPointFeature(ParticlePlotFeature):
    def __init__(self, lat, lon, xlim=None, ymax=None, **kwargs):
        super().__init__([lat], [lon], **kwargs)
        self.xlim = xlim
        self.ymax = ymax

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
        return cls(TIJUANA_MOUTH_POSITION[0], TIJUANA_MOUTH_POSITION[1], xlim=[-16, 4], ymax=0.1)


class TimedFrame:
    """Class that stores information about a single simulation plot"""
    def __init__(self, time, path, **kwargs):
        self.time = time
        self.path = path
        # path to other plots that display other information about the frame
        self.paths_inf = kwargs

    def __repr__(self):
        return f"([{self.path}] at [{self.time}])"


class ParticleResult:
    """
    Wraps the output of a particle file to make visualizing and analyzing the results easier.
    Can also use an HFRGrid if the ocean currents are also needed.
    """
    def __init__(self, dataset, cfg=None):
        self.cfg = cfg
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
        self.time_grid = self.xrds["time"].values
        times_unique = np.unique(self.time_grid)
        self.times = np.sort(times_unique[~np.isnan(times_unique)])
        # not part of Ak4 kernel
        self.lifetimes = self.xrds["lifetime"].values
        self.spawntimes = self.xrds["spawntime"].values
        
        self.max_life = np.nanmax(self.lifetimes)
        self.grid = None
        self.frames = None
        self.plot_features = {}

    def add_grid(self, grid: HFRGrid):
        self.grid = grid
        gtimes, _, _ = grid.get_coords()
        # check if particle set is in-bounds of the given grid
        if np.nanmin(self.times) < gtimes.min() or np.nanmax(self.times) > gtimes.max():
            raise ValueError("Time out of bounds")

    def add_plot_feature(self, feature: ParticlePlotFeature, name=None):
        if name is None:
            # just give a name
            self.plot_features[hash(feature)] = feature
        else:
            self.plot_features[name] = feature

    def plot_feature(self, t, feature: ParticlePlotFeature, ax, feat_info=True):
        mask = self.time_grid == t
        curr_lats = self.lats[mask]
        curr_lons = self.lons[mask]
        age_max = np.nanmax(self.lifetimes) / 86400
        feature.plot_on_frame(ax, curr_lats, curr_lons)
        if feat_info:
            fig_inf, ax_inf = feature.generate_info_table(
                curr_lats, curr_lons, lifetimes=self.lifetimes[mask], age_max=age_max
            )
            return fig_inf, ax_inf
        return None, None

    def plot_at_t(self, t, domain=None, feat_info="all"):
        if self.grid is None and domain is None:
            domain = {
                "W": np.nanmin(self.lons),
                "E": np.nanmax(self.lons),
                "S": np.nanmin(self.lats),
                "N": np.nanmax(self.lats),
            }
            domain = plot_utils.pad_domain(domain, 0.0005)
        elif self.grid is not None and domain is None:
            domain = self.grid.get_domain()
        if self.grid is None:
            fig, ax = plot_utils.get_carree_axis(domain, land=True)
            plot_utils.get_carree_gl(ax)
        else:
            show_time = int((t - self.grid.times[0]) / np.timedelta64(1, "s"))
            if show_time < 0:
                raise ValueError("Particle simulation time domain goes out of bounds")
            _, fig, ax, _ = plotting.plotfield(
                field=self.grid.fieldset.UV, show_time=show_time, domain=domain, land=True, vmin=0,
                vmax=0.6, titlestr="Particles and "
            )
        mask = self.time_grid == t
        ax.scatter(
            self.lons[mask], self.lats[mask], c=self.lifetimes[mask] / 86400, edgecolor="k", vmin=0,
            vmax=self.max_life / 86400, s=25
        )
        figs = {}
        axs = {}
        for name, feature in self.plot_features.items():
            fig_inf, ax_inf = self.plot_feature(
                t, feature, ax, feat_info=(feat_info == "all" or name in feat_info)
            )
            figs[name] = fig_inf
            axs[name] = ax_inf
        return fig, ax, figs, axs

    def on_plot_generated(self, savefile, savefile_infs, i, t, total):
        pass

    def generate_all_plots(self, save_dir, filename=None, figsize=None, domain=None, feat_info="all"):
        """
        Generates plots and then saves them

        Args:
            feat_info (list or str): 'all' to plot everything, list of names to choose which
             features to generate their own plots for
        """
        utils.create_path(save_dir)
        frames = []
        if self.cfg is not None:
            total_plots = int((self.times[-1] - self.times[0]) / np.timedelta64(1, "s") / self.cfg["snapshot_interval"]) + 1
            t = self.times[0]
            i = 0
            while t <= self.times[-1]:
                fig, _, figs, _ = self.plot_at_t(t, domain=domain, feat_info=feat_info)
                savefile = os.path.join(
                    save_dir, f"snap_{i}.png" if filename is None else f"{filename}_{i}.png"
                )
                plot_utils.draw_plt(savefile=savefile, fig=fig, figsize=figsize)
                savefile_infs = {}
                for name, fig_inf in figs.items():
                    if fig_inf is not None and (feat_info == "all" or name in feat_info):
                        savefile_inf = os.path.join(
                            save_dir,
                            f"snap_{name}_{i}.png" if filename is None else f"{filename}_{name}_{i}.png"
                        )
                        savefile_infs[name] = savefile_inf
                        plot_utils.draw_plt(savefile=savefile_inf, fig=fig_inf, figsize=figsize)
                frames.append(TimedFrame(t, savefile, **savefile_infs))
                self.on_plot_generated(savefile, savefile_infs, i, t, total_plots)
                i += 1
                t += np.timedelta64(self.cfg["snapshot_interval"], "s")
        else:
            for i in range(len(self.times)):
                fig, _, figs, _ = self.plot_at_t(self.times[i], domain=domain, feat_info=feat_info)
                savefile = os.path.join(
                    save_dir, f"snap_{i}.png" if filename is None else f"{filename}_{i}.png"
                )
                plot_utils.draw_plt(savefile=savefile, fig=fig, figsize=figsize)
                savefile_infs = {}
                for name, fig_inf in figs.items():
                    if fig_inf is not None and (feat_info == "all" or name in feat_info):
                        savefile_inf = os.path.join(
                            save_dir,
                            f"snap_{name}_{i}.png" if filename is None else f"{filename}_{name}_{i}.png"
                        )
                        savefile_infs[name] = savefile_inf
                        plot_utils.draw_plt(savefile=savefile_inf, fig=fig_inf, figsize=figsize)
                frames.append(TimedFrame(self.times[i], savefile, **savefile_infs))
                self.on_plot_generated(savefile, savefile_infs, i, self.times[i], len(self.times))
        self.frames = frames
        return frames

    def generate_gif(self, gif_path, gif_delay=25):
        input_paths = [str(frame.path) for frame in self.frames]
        sp_in = ["magick", "-delay", str(gif_delay)] + input_paths
        sp_in.append(str(gif_path))
        magick_sp = subprocess.Popen(
            sp_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = magick_sp.communicate()
        print(f"magick ouptput: {(stdout, stderr)}", file=sys.stderr)
        return gif_path
