"""
Everything in this file is related to processing the NetCDF file created by a Parcels particle
file.
"""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
from shapely.geometry import LineString
import xarray as xr

from pyplume.constants import *
from pyplume.parcels_utils import SurfaceGrid
from pyplume.plot_features import *
import pyplume.plot_utils as plot_utils
import pyplume.utils as utils


class TimedFrame:
    """Class that stores information about a single simulation plot"""
    def __init__(self, time, path, lats, lons, ages=None, **kwargs):
        self.time = time
        self.path = path
        self.lats = list(lats)
        self.lons = list(lons)
        # optional data if it doesn't exist
        self.ages = [] if ages is None else list(ages)
        # path to other plots that display other features about the frame
        self.paths_feat = kwargs

    def __repr__(self):
        return f"([{self.path}] at [{self.time}])"


class ParticleResult:
    """
    Wraps the output of a particle file to make visualizing and analyzing the results easier.
    Can also use an SurfaceGrid if the ocean currents are also wanted in the plot.

    NOTE this currently only works with simulations with ThreddsParticle particle classes.
    """
    def __init__(self, dataset, cfg=None):
        """
        Args:
            dataset: path to ParticleFile or just the dataset itself
            cfg: the parcels config passed into the main simulation
        """
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
        # assumed to be in data_vars: trajectory, time, lat, lon, z
        self.data_vars = {}
        self.non_vars = {}  # data variables with different dimensions than the dataset's
        self.shape = self.xrds["trajectory"].shape  # use trajectory var as reference
        for var, arr in self.xrds.variables.items():
            arr = arr.values
            if self.shape == arr.shape:
                self.data_vars[var] = arr
            else:
                self.non_vars[var] = arr
        # some particles die in a timestamp between the time intervals, and leave a specific
        # time that probably isn't included in most of the other particles
        # we take the unique timestamps across all particles
        times_unique = np.unique(self.data_vars["time"])
        self.times = np.sort(times_unique[~np.isnan(times_unique)])

        self.grid = None
        self.frames = None
        self.plot_features = {}
        self.coastline = None

    def add_coastline(self, lats, lons):
        """Adds a single coastline for processing collisions"""
        self.coastline = LineString(np.array([lons, lats]).T)

    def process_coastline_collisions(self):
        """
        Checks when each particle has collided with a coastline and removes all instances of the
        particle after the time of collision.
        Does not modify the original file (it shouldn't at least).
        """
        if self.coastline is None:
            raise AttributeError("Coastline is not defined yet")
        for i in range(self.data_vars["lat"].shape[0]):
            nan_where = np.where(np.isnan(self.data_vars["lat"][i]))[0]
            # LineString can't handle nan values, filter them out
            if len(nan_where) > 0 and nan_where[0] > 1:
                trajectory = LineString(np.array([self.data_vars["lon"][i][:nan_where[0]], self.data_vars["lat"][i][:nan_where[0]]]).T)
            elif len(nan_where) == 0:
                trajectory = LineString(np.array([self.data_vars["lon"][i], self.data_vars["lat"][i]]).T)
            else:
                # found an all nan particle (somehow)
                continue
            if trajectory.intersects(self.coastline):
                # the entire trajectory intersects with the coastline, find which timestamp it
                # crosses and delete all data after that
                for j in range(1, self.data_vars["lat"].shape[1]):
                    if np.isnan(self.data_vars["lon"][i, j]):
                        break
                    part_seg = LineString([(self.data_vars["lon"][i, j - 1], self.data_vars["lat"][i, j - 1]), (self.data_vars["lon"][i, j], self.data_vars["lat"][i, j])])
                    if self.coastline.intersects(part_seg):
                        for var in self.data_vars.keys():
                            if np.issubdtype(self.data_vars[var].dtype, np.datetime64):
                                self.data_vars[var][i, j:] = np.datetime64("NaT")
                            else:
                                self.data_vars[var][i, j:] = np.nan
                        break

    def add_grid(self, grid: SurfaceGrid):
        """Adds a SurfaceGrid to draw the currents on the plots."""
        self.grid = grid
        gtimes, _, _ = grid.get_coords()
        # check if particle set is in-bounds of the given grid
        if np.nanmin(self.data_vars["time"]) < gtimes.min() or np.nanmax(self.data_vars["time"]) > gtimes.max():
            raise ValueError("Time out of bounds")

    def add_plot_feature(self, feature: ParticlePlotFeature, name=None):
        if name is None:
            # just give a unique name
            self.plot_features[hash(feature)] = feature
        else:
            self.plot_features[name] = feature

    def plot_feature(self, t: np.datetime64, feature: ParticlePlotFeature, fig, ax, feat_info=True):
        """Plots a feature at a given time."""
        mask = self.data_vars["time"] == t
        curr_lats = self.data_vars["lat"][mask]
        curr_lons = self.data_vars["lon"][mask]
        ages = self.data_vars["lifetime"][mask] / 86400 if "lifetime" in self.data_vars else None
        # TODO cache this
        max_age = np.nanmax(self.data_vars["lifetime"]) / 86400  if "lifetime" in self.data_vars else None
        feature.plot_on_frame(fig, ax, curr_lats, curr_lons, time=t)
        if feat_info:
            fig_feat, ax_feat = feature.generate_info_table(
                curr_lats, curr_lons, lifetimes=ages, age_max=max_age
            )
            return fig_feat, ax_feat
        return None, None

    def plot_at_t(self, t, domain=None, feat_info="all", land=True):
        """
        Create figures of the simulation at a particular time.
        TODO when drawing land, prioritize coastline instead of using cartopy

        Args:
            t (int or np.datetime64): the int will index the time list
            feat_info (list): set of features to draw (their names), or 'all' to draw every feature
        """
        if isinstance(t, int):
            t = self.times[t]
        mask = self.data_vars["time"] == t
        ages = self.data_vars["lifetime"][mask] / 86400 if "lifetime" in self.data_vars else None
        # TODO cache this
        max_age = np.nanmax(self.data_vars["lifetime"]) / 86400  if "lifetime" in self.data_vars else None
        fig, ax = plot_utils.plot_particles(
            self.data_vars["lat"][mask], self.data_vars["lon"][mask], ages=ages, time=t,
            grid=self.grid, domain=domain, land=land, max_age=max_age
        )

        figs = {}
        axs = {}
        # get feature plots
        for name, feature in self.plot_features.items():
            fig_feat, ax_feat = self.plot_feature(
                t, feature, fig, ax, feat_info=(feat_info == "all" or name in feat_info)
            )
            figs[name] = fig_feat
            axs[name] = ax_feat
        return fig, ax, figs, axs

    # TODO finish
    def plot_trajectory(self, idxs, domain=None, land=True):
        if not isinstance(idxs, list):
            idxs = [idxs]
        plot_utils.draw_trajectories

    def on_plot_generated(self, savefile, savefile_feats, i, t, total):
        """
        An overridable hook just in case you want something to happen between plot generations
        in generate_all_plots or something.
        This was definitely not made just for celery progress tracking.
        """
        pass

    def save_at_t(self, t, i, save_dir, filename, figsize, domain, feat_info, land):
        """Generate and save plots at a timestamp, given a bunch of information."""
        fig, _, figs, _ = self.plot_at_t(t, domain=domain, feat_info=feat_info, land=land)
        savefile = os.path.join(
            save_dir, f"snap_{i}.png" if filename is None else f"{filename}_{i}.png"
        )
        plot_utils.draw_plt(savefile=savefile, fig=fig, figsize=figsize)
        savefile_feats = {}
        # plot and save every desired feature
        for name, fig_feat in figs.items():
            if fig_feat is not None and (feat_info == "all" or name in feat_info):
                savefile_feat = os.path.join(
                    save_dir,
                    f"snap_{name}_{i}.png" if filename is None else f"{filename}_{name}_{i}.png"
                )
                savefile_feats[name] = savefile_feat
                plot_utils.draw_plt(savefile=savefile_feat, fig=fig_feat, figsize=figsize)
        lats, lons = self.get_points_at_t(t)
        mask = self.data_vars["time"] == t  # lol idk just do it again
        ages = None
        if "lifetime" in self.data_vars:
            ages = self.data_vars["lifetime"][mask]
        self.frames.append(TimedFrame(t, savefile, lats, lons, ages=ages, **savefile_feats))
        return savefile, savefile_feats

    def generate_all_plots(
        self, save_dir, filename=None, figsize=None, domain=None, feat_info="all", land=True,
        clear_folder=False
    ):
        """
        Generates plots and then saves them

        Args:
            feat_info (list or str): 'all' to plot everything, list of names to choose which
             features to generate their own plots for
        """
        utils.create_path(save_dir)
        if clear_folder:
            utils.delete_all_pngs(save_dir)
        self.frames = []
        if self.cfg is not None:
            # The delta time between each snapshot is defined in the parcels config. This lets us avoid
            # the in-between timestamps where a single particle gets deleted.
            total_plots = int((self.times[-1] - self.times[0]) / np.timedelta64(1, "s") / self.cfg["snapshot_interval"]) + 1
            t = self.times[0]
            i = 0
            while t <= self.times[-1]:
                savefile, savefile_feats = self.save_at_t(
                    t, i, save_dir, filename, figsize, domain, feat_info, land
                )
                self.on_plot_generated(savefile, savefile_feats, i, t, total_plots)
                i += 1
                t += np.timedelta64(self.cfg["snapshot_interval"], "s")
        else:
            # If the delta time between each snapshot is unknown, we'll just use the unique times
            # from the particle files.
            for i in range(len(self.times)):
                savefile, savefile_feats = self.save_at_t(
                    self.times[i], i, save_dir, filename, figsize, domain, feat_info, land
                )
                self.on_plot_generated(savefile, savefile_feats, i, self.times[i], len(self.times))
        return self.frames

    def generate_all_positions(self):
        self.frames = []
        if self.cfg is not None:
            t = self.times[0]
            i = 0
            while t <= self.times[-1]:
                lats, lons = self.get_points_at_t(t)
                mask = self.data_vars["time"] == t  # lol idk just do it again
                ages = None
                if "lifetime" in self.data_vars:
                    ages = self.data_vars["lifetime"][mask]
                self.frames.append(TimedFrame(t, None, lats, lons, ages=ages))
                i += 1
                t += np.timedelta64(self.cfg["snapshot_interval"], "s")
        else:
            # If the delta time between each snapshot is unknown, we'll just use the unique times
            # from the particle files.
            for i in range(len(self.times)):
                lats, lons = self.get_points_at_t(self.times[i])
                mask = self.data_vars["time"] == t  # lol idk just do it again
                ages = None
                if "lifetime" in self.data_vars:
                    ages = self.data_vars["lifetime"][mask]
                self.frames.append(TimedFrame(self.times[i], None, lats, lons, ages=ages))
        return self.frames

    def generate_gif(self, gif_path, gif_delay=25):
        """Uses imagemagick to generate a gif of the main simulation plot."""
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

    def write_feature_dists(self, feat_names):
        for feat_name in feat_names:
            feat = self.plot_features[feat_name]
            dists = np.zeros(self.shape, dtype=float)
            for time in self.times:
                mask = self.data_vars["time"] == time
                dists_t = feat.get_closest_dists(
                    self.data_vars["lat"][mask], self.data_vars["lon"][mask], time=time
                )
                dists[mask] = dists_t
            self.data_vars[f"{feat_name}_distances"] = dists

    def write_data(self, path=None, override=False):
        """
        Write postprocessed data to a new netcdf file.
        If path is unspecified, it will override the original path the ParticleFile was
        saved to (if a path was passed in).
        """
        if self.path is not None and path is None and not override:
            if os.path.isfile(path):
                raise FileExistsError(f"{path} already exists. Set override=True.")
            raise FileExistsError(f"Unspecified path will override {path}. Set override=True.")
        if self.path is None and path is None:
            raise ValueError("No path specified and no previous path was passed in.")
        new_ds = xr.Dataset(
            data_vars={var: (self.xrds.dims, arr) for var, arr in self.data_vars.items()},
            coords=self.xrds.coords, attrs=self.xrds.attrs
        )
        new_ds.to_netcdf(path=self.path if path is None else path)

    def get_points_at_t(self, t: np.datetime64):
        mask = self.data_vars["time"] == t
        return self.data_vars["lat"][mask], self.data_vars["lon"][mask]


PLOT_FEATURE_SETS = {
    "tj_plume_tracker": {
        "coast": NanSeparatedFeature.get_sd_full_coastline(),
        "station": StationFeature.get_sd_stations(),
        "mouth": NearcoastDensityFeature.get_tijuana_mouth()
    }
}


def add_feature_set_to_result(result: ParticleResult, set_name: str):
    for key, value in PLOT_FEATURE_SETS[set_name].items():
        result.add_plot_feature(value, name=key)
