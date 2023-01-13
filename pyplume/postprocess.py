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

from pyplume import get_logger, plotting, utils
from pyplume.constants import *
from pyplume.dataloaders import SurfaceGrid
from pyplume.plot_features import *


logger = get_logger(__name__)


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
    # for the main plot that draws the particles per frame
    MAIN_PARTICLE_PLOT_NAME = "particles"
    counter = 0

    def __init__(self, src, sim_result_dir=None, snapshot_interval=None):
        """
        Args:
            src: path to ParticleFile or just the dataset itself
        """
        self.sim_result_dir = Path(sim_result_dir) if sim_result_dir is not None else None
        self.snapshot_interval = snapshot_interval
        if isinstance(src, (Path, str)):
            self.path = Path(src)
            self.ds = xr.open_dataset(src)
            self.ds.close()
        elif isinstance(src, xr.Dataset):
            self.path = None
            self.ds = src
        else:
            raise TypeError(f"{src} is not a path or xarray dataset")
        if self.sim_result_dir is None and self.path is not None:
            # assume particle file is already in the results directory
            self.sim_result_dir = self.path.parent
        # assumed to be in data_vars: trajectory, time, lat, lon, z
        # these variables are generated from default particles in Parcels
        self.data_vars = {}
        # data variables with different dimensions than the dataset's
        self.non_vars = {}
        self.shape = self.ds["trajectory"].shape  # use trajectory var as reference
        for var, arr in self.ds.variables.items():
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
        self.plot_features = {
            ParticleResult.MAIN_PARTICLE_PLOT_NAME: ParticlePlotFeature(particle_size=20)
        }
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

    def add_plot_feature(self, feature: PlotFeature, label=None):
        if label is None:
            self.plot_features[f"{feature.__class__.__name__}_{ParticleResult.counter}"] = feature
            ParticleResult.counter += 1
        else:
            self.plot_features[label] = feature

    def plot_feature(self, feature: PlotFeature, fig, ax, t: np.datetime64, lats, lons, lifetimes, lifetime_max):
        """Plots a feature at a given time."""
        feature.add_to_plot(fig, ax, t, lats, lons)
        fig_feat, ax_feat = feature.generate_external_plot(
            t, lats, lons, lifetimes=lifetimes, lifetime_max=lifetime_max
        )
        return fig_feat, ax_feat

    def plot_at_t(self, t, domain=None, land=True, lifetime_max=None):
        """
        Create figures of the simulation at a particular time.
        TODO when drawing land, prioritize coastline instead of using cartopy

        Args:
            t (int or np.datetime64): the int will index the time list
        """
        if isinstance(t, int):
            t = self.times[t]
        mask = self.data_vars["time"] == t
        curr_lats = self.data_vars["lat"][mask]
        curr_lons = self.data_vars["lon"][mask]
        lifetimes = self.data_vars["lifetime"][mask] / 86400 if "lifetime" in self.data_vars else None
        fig, ax = plotting.plot_field(time=t, grid=self.grid, domain=domain, land=land)

        figs = {}
        axs = {}
        # get feature plots
        for name, feature in self.plot_features.items():
            fig_feat, ax_feat = self.plot_feature(feature, fig, ax, t, curr_lats, curr_lons, lifetimes, lifetime_max)
            figs[name] = fig_feat
            axs[name] = ax_feat
        return fig, ax, figs, axs

    # TODO finish
    def plot_trajectory(self, idxs, domain=None, land=True):
        if not isinstance(idxs, list):
            idxs = [idxs]
        plotting.draw_trajectories

    def save_at_t(self, t, i, save_dir, figsize, domain, land):
        """Generate and save plots at a timestamp, given a bunch of information."""
        lifetime_max = np.nanmax(self.data_vars["lifetime"]) / 86400  if "lifetime" in self.data_vars else None
        fig, _, figs, _ = self.plot_at_t(t, domain=domain, land=land, lifetime_max=lifetime_max)
        savefile = utils.get_dir(save_dir / ParticleResult.MAIN_PARTICLE_PLOT_NAME) / f"simframe_{i}.png"
        plotting.draw_plt(savefile=savefile, fig=fig, figsize=figsize)
        savefile_feats = {}
        # plot and save every desired feature
        for name, fig_feat in figs.items():
            if fig_feat is not None:
                savefile_feat = utils.get_dir(save_dir / name) / f"simframe_{name}_{i}.png"
                savefile_feats[name] = savefile_feat
                plotting.draw_plt(savefile=savefile_feat, fig=fig_feat, figsize=figsize)
        lats, lons = self.get_positions_time(t, query="at")
        mask = self.data_vars["time"] == t  # lol idk just do it again
        ages = None
        if "lifetime" in self.data_vars:
            ages = self.data_vars["lifetime"][mask]
        self.frames.append(TimedFrame(t, savefile, lats, lons, ages=ages, **savefile_feats))
        return savefile, savefile_feats

    def generate_all_plots(self, figsize=None, domain=None, land=True, clear_folder=False):
        """
        Generates plots and then saves them.
        """
        if self.sim_result_dir is None:
            raise ValueError("Please specify a path for sim_result_dir to save the plots")
        save_dir = self.sim_result_dir / "plots"
        utils.get_dir(save_dir)
        if clear_folder:
            utils.delete_all_pngs(save_dir)
        self.frames = []
        if self.snapshot_interval is not None:
            # The delta time between each snapshot is defined in the parcels config. This lets us avoid
            # the in-between timestamps where a single particle gets deleted.
            total_plots = int((self.times[-1] - self.times[0]) / np.timedelta64(1, "s") / self.snapshot_interval) + 1
            t = self.times[0]
            i = 0
            while t <= self.times[-1]:
                savefile, savefile_feats = self.save_at_t(
                    t, i, save_dir, figsize, domain, land
                )
                i += 1
                t += np.timedelta64(self.snapshot_interval, "s")
        else:
            # If the delta time between each snapshot is unknown, we'll just use the unique times
            # from the particle files.
            for i in range(len(self.times)):
                savefile, savefile_feats = self.save_at_t(
                    self.times[i], i, save_dir, figsize, domain, land
                )
        return self.frames

    def generate_all_positions(self):
        self.frames = []
        if self.snapshot_interval is not None:
            t = self.times[0]
            i = 0
            while t <= self.times[-1]:
                lats, lons = self.get_positions_time(t, query="at")
                mask = self.data_vars["time"] == t  # lol idk just do it again
                ages = None
                if "lifetime" in self.data_vars:
                    ages = self.data_vars["lifetime"][mask]
                self.frames.append(TimedFrame(t, None, lats, lons, ages=ages))
                i += 1
                t += np.timedelta64(self.snapshot_interval, "s")
        else:
            # If the delta time between each snapshot is unknown, we'll just use the unique times
            # from the particle files.
            for i in range(len(self.times)):
                lats, lons = self.get_positions_time(self.times[i], query="at")
                mask = self.data_vars["time"] == t  # lol idk just do it again
                ages = None
                if "lifetime" in self.data_vars:
                    ages = self.data_vars["lifetime"][mask]
                self.frames.append(TimedFrame(self.times[i], None, lats, lons, ages=ages))
        return self.frames

    def generate_gif(self, gif_delay=25):
        """Uses imagemagick to generate a gif of the main simulation plot."""
        gif_path = self.sim_result_dir / f"{ParticleResult.MAIN_PARTICLE_PLOT_NAME}.gif"
        input_paths = [str(frame.path) for frame in self.frames]
        utils.generate_gif(input_paths, gif_path, gif_delay=gif_delay)
        for feat in self.frames[0].paths_feat.keys():
            feat_gif_path = self.sim_result_dir / f"{feat}.gif"
            feat_input_paths = [str(frame.paths_feat[feat]) for frame in self.frames]
            utils.generate_gif(feat_input_paths, feat_gif_path, gif_delay=gif_delay)

    def write_feature_dists(self, feat_names):
        for feat_name in feat_names:
            feat = self.plot_features[feat_name]
            if not isinstance(feat, ScatterPlotFeature): continue
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
            data_vars={var: (self.ds.dims, arr) for var, arr in self.data_vars.items()},
            coords=self.ds.coords, attrs=self.ds.attrs
        )
        new_ds.to_netcdf(path=self.path if path is None else path)

    def get_filtered_data_time(self, t: np.datetime64, query="at"):
        valid_queries = ("at", "before", "after")
        if query not in valid_queries:
            raise ValueError(f"Invalid query '{query}'. Valid queries are {valid_queries}")
        if query == "at":
            mask = self.data_vars["time"] == t
        elif query == "before":
            mask = self.data_vars["time"] <= t
        elif query == "after":
            mask = self.data_vars["time"] >= t
        filtered = {}
        for dvar in self.data_vars.keys():
            filtered[dvar] = self.data_vars[dvar][mask]
        return filtered

    def get_positions_time(self, t: np.datetime64, query="at"):
        filtered = self.get_filtered_data_time(t, query=query)
        return filtered["lat"], filtered["lon"]


class ParticleResultComparer:
    def __init__(self, *particleresults):
        self.particleresults = particleresults
