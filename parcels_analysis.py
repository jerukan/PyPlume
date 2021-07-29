from pathlib import Path
import os
import subprocess
import sys

import numpy as np
from parcels import plotting
from shapely.geometry import LineString
import xarray as xr

from constants import *
from parcels_utils import HFRGrid
from plot_features import *
import plot_utils
import utils


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
        self.coastline = None

    def add_coastline(self, lats, lons):
        self.coastline = LineString(np.array([lons, lats]).T)

    def process_coastline_collisions(self):
        """
        Checks when each particle has collided with a coastline and removes all instances of the
        particle after the time of collision.
        """
        if self.coastline is None:
            raise AttributeError("Coastline is not defined yet")
        for i in range(self.lats.shape[0]):
            nan_where = np.where(np.isnan(self.lats[i]))[0]
            # LineString can't handle nan values, filter them out
            if len(nan_where) > 0 and nan_where[0] > 1:
                trajectory = LineString(np.array([self.lons[i][:nan_where[0]], self.lats[i][:nan_where[0]]]).T)
            elif len(nan_where) == 0:
                trajectory = LineString(np.array([self.lons[i], self.lats[i]]).T)
            else:
                # found an all nan particle (somehow)
                continue
            if trajectory.intersects(self.coastline):
                for j in range(1, self.lats.shape[1]):
                    if np.isnan(self.lons[i, j]):
                        break
                    part_seg = LineString([(self.lons[i, j - 1], self.lats[i, j - 1]), (self.lons[i, j], self.lats[i, j])])
                    if self.coastline.intersects(part_seg):
                        self.lats[i, j:] = np.nan
                        self.lons[i, j:] = np.nan
                        self.traj[i, j:] = np.nan
                        self.time_grid[i, j:] = np.datetime64("NaT")
                        self.lifetimes[i, j:] = np.nan
                        self.spawntimes[i, j:] = np.nan
                        break

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

    def plot_at_t(self, t, domain=None, feat_info="all", land=False):
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
            fig, ax = plot_utils.get_carree_axis(domain, land=land)
            plot_utils.get_carree_gl(ax)
        else:
            show_time = int((t - self.grid.times[0]) / np.timedelta64(1, "s"))
            if show_time < 0:
                raise ValueError("Particle simulation time domain goes out of bounds")
            _, fig, ax, _ = plotting.plotfield(
                field=self.grid.fieldset.UV, show_time=show_time, domain=domain, land=land, vmin=0,
                vmax=0.6, titlestr="Particles and "
            )
        mask = self.time_grid == t
        ax.scatter(
            self.lons[mask], self.lats[mask], c=self.lifetimes[mask] / 86400, edgecolor="k", vmin=0,
            vmax=self.max_life / 86400, s=20
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
        utils.delete_all_pngs(save_dir)
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


PLOT_FEATURE_SETS = {
    "tj_plume_tracker": {
        "coast": NanSeparatedFeature.get_sd_full_coastline(),
        "station": StationFeature.get_sd_stations(),
        "mouth": LatTrackedPointFeature.get_tijuana_mouth()
    }
}


def add_feature_set_to_result(result: ParticleResult, set_name: str):
    for key, value in PLOT_FEATURE_SETS[set_name].items():
        result.add_plot_feature(value, name=key)
